# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn

from src.utils.logging import get_logger
from src.utils.tensors import trunc_normal_

logger = get_logger(__name__)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class IdentityMixer(nn.Module):
    """No spatial mixing; the block's channel MLP is the transition floor."""

    def __init__(self, dim, **kwargs):
        super().__init__()
        self.proj = nn.Identity()

    def forward(self, x, **kwargs):
        return self.proj(x)


class ConvMixerOp(nn.Module):
    """Per-frame spatial mixer over the native DINO patch grid."""

    def __init__(self, dim, kernel_size=3, act_layer=nn.GELU, **kwargs):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.act = act_layer()
        self.pointwise = nn.Conv2d(dim, dim, 1)

    def forward(self, x, T=None, H_patches=None, W_patches=None, cond_tokens=0, **kwargs):
        if cond_tokens:
            raise ValueError("ConvMixerOp does not support condition tokens in the patch sequence")
        if T is None or H_patches is None or W_patches is None:
            raise ValueError("ConvMixerOp requires T, H_patches, and W_patches")
        B, N, D = x.shape
        expected = T * H_patches * W_patches
        if N != expected:
            raise ValueError(f"Expected {expected} tokens, got {N}")
        y = x.view(B, T, H_patches, W_patches, D)
        # Stage 1 deliberately mixes each frame independently; temporal signal is
        # carried by the existing context tokens and action-conditioned residuals.
        y = y.reshape(B * T, H_patches, W_patches, D).permute(0, 3, 1, 2)
        y = self.depthwise(y)
        y = self.act(y)
        y = self.pointwise(y)
        y = y.permute(0, 2, 3, 1).reshape(B, T * H_patches * W_patches, D)
        return y


class MambaMixer(nn.Module):
    """Raster-scan patch mixer; falls back to GRU if mamba_ssm is unavailable."""

    def __init__(self, dim, grid_size=16, **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size * grid_size, dim))
        self.backend = "gru_fallback"
        try:
            from mamba_ssm import Mamba

            self.scan = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
            self.backend = "mamba_ssm"
        except Exception as exc:  # pragma: no cover - depends on lab01 stack
            logger.warning(f"mamba_ssm unavailable for MGVT; using GRU fallback ({exc})")
            self.scan = nn.GRU(input_size=dim, hidden_size=dim, batch_first=True)

    def forward(self, x, T=None, H_patches=None, W_patches=None, cond_tokens=0, **kwargs):
        if cond_tokens:
            raise ValueError("MambaMixer does not support condition tokens in the patch sequence")
        if T is None or H_patches is None or W_patches is None:
            raise ValueError("MambaMixer requires T, H_patches, and W_patches")
        B, N, D = x.shape
        expected = T * H_patches * W_patches
        if N != expected:
            raise ValueError(f"Expected {expected} tokens, got {N}")
        y = x.view(B, T, H_patches * W_patches, D)
        if H_patches == self.grid_size and W_patches == self.grid_size:
            y = y + self.pos_embed[:, : H_patches * W_patches].unsqueeze(1)
        y = y.reshape(B * T, H_patches * W_patches, D)
        if self.backend == "mamba_ssm":
            y = self.scan(y)
        else:
            y, _ = self.scan(y)
        return y.reshape(B, T * H_patches * W_patches, D)


class MGVTBlock(nn.Module):
    def __init__(
        self,
        dim,
        mixer_type,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        grid_size=16,
        conv_kernel_size=3,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.grid_size = grid_size
        if mixer_type == "mlp":
            self.mixer = IdentityMixer(dim)
        elif mixer_type == "convmixer":
            self.mixer = ConvMixerOp(dim, kernel_size=conv_kernel_size, act_layer=act_layer)
        elif mixer_type == "mamba":
            self.mixer = MambaMixer(dim, grid_size=grid_size)
        else:
            raise ValueError(f"Unknown MGVT mixer_type: {mixer_type}")
        self.attn = self.mixer  # compatibility for parameter rescaling checks
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x, z, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, cond_tokens=0):
        per_step_tokens = (H_patches or self.grid_size) * (W_patches or self.grid_size) + cond_tokens
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(z).repeat_interleave(per_step_tokens, dim=1).chunk(6, dim=2)
        )
        y = self.mixer(
            modulate(self.norm1(x), shift_msa, scale_msa),
            T=T,
            H_patches=H_patches,
            W_patches=W_patches,
            cond_tokens=cond_tokens,
        )
        x = x + self.drop_path(y * gate_msa)
        x = x + self.drop_path(gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class VisionMGVT(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=1,
        embed_dim=384,
        predictor_embed_dim=384,
        depth=1,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_silu=False,
        wide_silu=True,
        is_causal=False,
        use_activation_checkpointing=False,
        local_window=(-1, -1, -1),
        use_rope=False,
        action_dim=20,
        proprio_dim=10,
        use_proprio=True,
        act_mlp=False,
        prop_mlp=False,
        init_scale_factor_adaln=0,
        proprio_encoding="feature",
        proprio_emb_dim=0,
        proprio_encoder_inpred=True,
        proprio_tokens=0,
        action_encoder_inpred=True,
        mixer_type="mlp",
        conv_kernel_size=3,
        context_window=2,
        **kwargs,
    ):
        super().__init__()
        self.predictor_embed_dim = predictor_embed_dim
        self.proprio_encoder_inpred = proprio_encoder_inpred
        self.action_encoder_inpred = action_encoder_inpred
        self.context_window = context_window
        self.predictor_embed = nn.Linear(context_window * embed_dim, predictor_embed_dim, bias=True)
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.grid_depth = num_frames // self.tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.proprio_emb_dim = proprio_emb_dim
        self.proprio_tokens = proprio_tokens
        self.use_proprio = use_proprio
        self.proprio_encoding = proprio_encoding
        self.mixer_type = mixer_type

        if self.use_proprio and self.proprio_encoding == "feature":
            self.predictor_total_embed_dim = predictor_embed_dim + proprio_emb_dim
        else:
            self.predictor_total_embed_dim = predictor_embed_dim

        if self.action_encoder_inpred:
            self.action_encoder = nn.Linear(action_dim, self.predictor_total_embed_dim, bias=True)
        if self.proprio_encoder_inpred:
            if self.proprio_encoding == "token" and self.proprio_tokens > 0:
                self.proprio_encoder = nn.Linear(proprio_dim, predictor_embed_dim, bias=True)
            elif self.proprio_encoding == "feature" and self.proprio_emb_dim > 0:
                self.proprio_encoder = nn.Linear(proprio_dim, proprio_emb_dim, bias=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList(
            [
                MGVTBlock(
                    mixer_type=mixer_type,
                    grid_size=self.grid_height,
                    dim=self.predictor_total_embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    conv_kernel_size=conv_kernel_size,
                )
                for i in range(depth)
            ]
        )
        self.predictor_norm = norm_layer(self.predictor_total_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.attn_mask = None
        self.cond_tokens = 0
        self.init_std = init_std
        self.init_scale_factor_adaln = init_scale_factor_adaln
        self.apply(self._init_weights)
        self._rescale_blocks()

        with torch.no_grad():
            for block in self.predictor_blocks:
                linear_layer = block.adaLN_modulation[1]
                if self.init_scale_factor_adaln == 0:
                    nn.init.constant_(linear_layer.weight, 0)
                else:
                    trunc_normal_(linear_layer.weight, std=self.init_std * self.init_scale_factor_adaln)
                nn.init.constant_(linear_layer.bias, 0)
        logger.info(
            f"Initialized VisionMGVT(mixer_type={mixer_type}, depth={depth}, "
            f"adaln_scale={init_scale_factor_adaln})"
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            if hasattr(layer.mixer, "pointwise"):
                rescale(layer.mixer.pointwise.weight.data, layer_id + 1)
            if hasattr(layer.mlp, "fc2"):
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def concat_obs(self, z_vis, proprio):
        return torch.cat([z_vis, proprio], dim=3)

    def _stack_context(self, x):
        # x: (B, T, V, H, W, D) -> (B, T, V, H, W, context_window * D).
        # Frame 0 is boundary-padded with itself, giving zero velocity at the
        # sequence start while exposing [s_{t-1}, s_t] for the W=2 gate.
        frames = [x]
        for k in range(1, self.context_window):
            if k < x.shape[1]:
                pad = x[:, :1].expand(-1, k, -1, -1, -1, -1)
                shifted = torch.cat([pad, x[:, :-k]], dim=1)
            else:
                shifted = x[:, :1].expand_as(x)
            frames.insert(0, shifted)
        return torch.cat(frames, dim=-1)

    def forward(self, x, actions, proprio=None):
        if self.context_window > 1:
            x = self._stack_context(x)
        x = self.predictor_embed(x)
        x = x.flatten(2, 4)
        B, T, _, D = x.shape

        if self.action_encoder_inpred:
            z = self.action_encoder(actions)
        else:
            z = actions.squeeze(2)

        if self.use_proprio and proprio is not None:
            if self.proprio_encoder_inpred:
                proprio = self.proprio_encoder(proprio).unsqueeze(2)
            if self.proprio_encoding == "token":
                x = torch.cat([proprio, x], dim=2).flatten(1, 2)
            elif self.proprio_encoding == "feature":
                x = self.concat_obs(x, proprio).flatten(1, 2)
        else:
            x = x.flatten(1, 2)

        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    z,
                    None,
                    None,
                    T=T,
                    H_patches=self.grid_height,
                    W_patches=self.grid_width,
                    use_reentrant=False,
                    cond_tokens=self.cond_tokens,
                )
            else:
                x = blk(
                    x,
                    z,
                    mask=None,
                    attn_mask=None,
                    T=T,
                    H_patches=self.grid_height,
                    W_patches=self.grid_width,
                    cond_tokens=self.cond_tokens,
                )
        x = self.predictor_norm(x)

        if self.use_proprio and proprio is not None:
            if self.proprio_encoding == "token":
                x = x.view(B, T, self.cond_tokens + self.grid_height * self.grid_width, D)
                x, proprio_features = x[:, :, self.cond_tokens :, :], x[:, :, : self.cond_tokens, :]
            elif self.proprio_encoding == "feature":
                x = x.view(B, T, self.grid_height * self.grid_width, self.predictor_total_embed_dim)
                x, proprio_features = x[:, :, :, : -self.proprio_emb_dim], x[:, :, :, -self.proprio_emb_dim :]
        else:
            x = x.view(B, T, self.grid_height * self.grid_width, self.predictor_total_embed_dim)
            proprio_features = None

        x = self.predictor_proj(x)
        return x, None, proprio_features


def vit_predictor_mgvt_mlp(**kwargs):
    return VisionMGVT(
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mixer_type="mlp",
        **kwargs,
    )


def vit_predictor_mgvt_convmixer(**kwargs):
    return VisionMGVT(
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mixer_type="convmixer",
        **kwargs,
    )


def vit_predictor_mgvt_mamba(**kwargs):
    return VisionMGVT(
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mixer_type="mamba",
        **kwargs,
    )
