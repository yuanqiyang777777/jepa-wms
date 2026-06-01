"""Dynamics-guided MGVT predictor for Stage-D1.

This module keeps the Stage-D1 decomposition explicit:

* Phi_state compresses visual/proprio context into a compact dynamics state.
* F_dyn predicts a low-dimensional action-induced trend ``d_h``.
* G_guidance turns ``d_h`` into a coarse latent transport signal.
* F_refine applies scene-conditioned residual refinement to future latents.

The implementation follows the same predictor contract as AdaLN/MGVT Stage-2:
``forward(video, action, proprio) -> (pred_video, None, pred_proprio)``.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn as nn

from src.utils.logging import get_logger
from src.utils.tensors import trunc_normal_

logger = get_logger(__name__)


def _count_linear_flops(module: nn.Module, token_count: int) -> int:
    flops = 0
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            flops += int(2 * token_count * layer.in_features * layer.out_features)
    return flops


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, depth: int, act_layer=nn.GELU):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, hidden_dim),
                    act_layer(),
                    nn.Linear(hidden_dim, dim),
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


class TemporalMambaOrGRU(nn.Module):
    """Temporal action-horizon mixer used only inside F_dyn.

    CPU tests may use the GRU fallback. Final D1 Mamba GPU rows must set
    ``require_cuda_mamba=True`` and will raise unless real ``mamba_ssm`` runs on
    CUDA tensors.
    """

    def __init__(self, dim: int, require_cuda_mamba: bool = False):
        super().__init__()
        self.backend = "gru_fallback"
        self.require_cuda_mamba = require_cuda_mamba
        try:
            from mamba_ssm import Mamba

            self.scan = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
            self.backend = "mamba_ssm"
        except Exception as exc:  # pragma: no cover - lab01 dependent
            logger.warning(f"mamba_ssm unavailable for D1 F_dyn; using GRU fallback ({exc})")
            self.scan = nn.GRU(input_size=dim, hidden_size=dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "mamba_ssm":
            if self.require_cuda_mamba and not x.is_cuda:
                raise RuntimeError("D1 Mamba F_dyn requires CUDA tensors for final GPU rows")
            y = self.scan(x)
            return y
        if self.require_cuda_mamba:
            raise RuntimeError("D1 Mamba F_dyn requires mamba_ssm; GRU fallback is disallowed")
        y, _ = self.scan(x)
        return y


class DynamicsGuidedPredictor(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=1,
        embed_dim=384,
        predictor_embed_dim=128,
        depth=1,
        num_heads=4,
        mlp_ratio=4.0,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_silu=False,
        action_dim=20,
        proprio_dim=10,
        use_proprio=True,
        proprio_encoding="feature",
        proprio_emb_dim=0,
        proprio_tokens=0,
        proprio_encoder_inpred=False,
        action_encoder_inpred=False,
        fdyn_type="mlp",
        guidance_mode="trend",
        d_h_dim=16,
        state_dim=64,
        fdyn_hidden_dim=None,
        refiner_depth=None,
        context_window=2,
        proprio_flow="dyn_refine",
        dh_ablation="none",
        sparse_top_frac=0.25,
        require_cuda_mamba=False,
        delta_p_dim=None,
        init_scale_factor_adaln=0,
        **kwargs: Any,
    ):
        super().__init__()
        if action_encoder_inpred:
            raise ValueError("DynamicsGuidedPredictor expects the existing external action encoder")
        if proprio_encoder_inpred:
            raise ValueError("DynamicsGuidedPredictor expects the existing external proprio encoder")
        if proprio_encoding not in {"feature", "token"}:
            raise ValueError(f"Unsupported proprio_encoding: {proprio_encoding}")
        if dh_ablation not in {"none", "remove", "shuffle", "random"}:
            raise ValueError(f"Unsupported dh_ablation: {dh_ablation}")
        if proprio_flow not in {"no_proprio", "dyn_only", "refine_only", "dyn_refine", "old_concat"}:
            raise ValueError(f"Unsupported proprio_flow: {proprio_flow}")

        self.predictor_embed_dim = predictor_embed_dim
        self.predictor_total_embed_dim = predictor_embed_dim + (
            proprio_emb_dim if use_proprio and proprio_encoding == "feature" else 0
        )
        self.proprio_encoder_inpred = proprio_encoder_inpred
        self.action_encoder_inpred = action_encoder_inpred
        self.proprio_encoding = proprio_encoding
        self.proprio_emb_dim = proprio_emb_dim
        self.proprio_tokens = proprio_tokens
        self.use_proprio = use_proprio
        self.proprio_flow = proprio_flow
        self.fdyn_type = fdyn_type
        self.guidance_mode = guidance_mode
        self.d_h_dim = int(d_h_dim)
        self.dh_ablation = dh_ablation
        self.context_window = int(context_window)
        self.sparse_top_frac = float(sparse_top_frac)
        self.delta_p_dim = int(delta_p_dim or 0)
        self.init_std = init_std
        self.init_scale_factor_adaln = init_scale_factor_adaln
        self.last_aux_stats: dict[str, float | str] = {}

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.grid_depth = num_frames // tubelet_size

        act_layer = nn.SiLU if use_silu else nn.GELU
        fdyn_hidden_dim = int(fdyn_hidden_dim or max(predictor_embed_dim, state_dim, 4 * self.d_h_dim))
        refiner_depth = int(depth if refiner_depth is None else refiner_depth)
        refiner_depth = max(0, refiner_depth)

        self.predictor_embed = nn.Linear(self.context_window * embed_dim, predictor_embed_dim)
        self.visual_state = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, state_dim),
            act_layer(),
            nn.Linear(state_dim, state_dim),
        )
        dyn_reads_proprio = use_proprio and proprio_flow in {"dyn_only", "dyn_refine", "old_concat"}
        prop_state_dim = proprio_emb_dim if (dyn_reads_proprio and proprio_emb_dim > 0) else 0
        self.proprio_state = (
            nn.Sequential(nn.LayerNorm(prop_state_dim), nn.Linear(prop_state_dim, state_dim), act_layer())
            if prop_state_dim > 0
            else None
        )
        dyn_reads_proprio = self.proprio_state is not None
        state_in_dim = state_dim + (state_dim if dyn_reads_proprio else 0)
        self.state_fuse = nn.Sequential(nn.LayerNorm(state_in_dim), nn.Linear(state_in_dim, state_dim), act_layer())

        self.action_norm = nn.LayerNorm(self.predictor_total_embed_dim)
        dyn_input_dim = state_dim + self.predictor_total_embed_dim
        if fdyn_type == "raw_action":
            self.f_dyn = nn.Sequential(
                nn.LayerNorm(self.predictor_total_embed_dim),
                nn.Linear(self.predictor_total_embed_dim, fdyn_hidden_dim),
                act_layer(),
                nn.Linear(fdyn_hidden_dim, self.d_h_dim),
            )
            self.temporal_mixer = None
        elif fdyn_type == "mlp":
            self.f_dyn = nn.Sequential(
                nn.LayerNorm(dyn_input_dim),
                nn.Linear(dyn_input_dim, fdyn_hidden_dim),
                act_layer(),
                nn.Linear(fdyn_hidden_dim, fdyn_hidden_dim),
                act_layer(),
                nn.Linear(fdyn_hidden_dim, self.d_h_dim),
            )
            self.temporal_mixer = None
        elif fdyn_type in {"gru", "mamba"}:
            self.dyn_in = nn.Sequential(
                nn.LayerNorm(dyn_input_dim),
                nn.Linear(dyn_input_dim, fdyn_hidden_dim),
                act_layer(),
            )
            if fdyn_type == "gru":
                self.temporal_mixer = nn.GRU(fdyn_hidden_dim, fdyn_hidden_dim, batch_first=True)
            else:
                self.temporal_mixer = TemporalMambaOrGRU(fdyn_hidden_dim, require_cuda_mamba=require_cuda_mamba)
            self.f_dyn = nn.Sequential(nn.LayerNorm(fdyn_hidden_dim), nn.Linear(fdyn_hidden_dim, self.d_h_dim))
        else:
            raise ValueError(f"Unsupported fdyn_type: {fdyn_type}")

        self.guidance = nn.Sequential(
            nn.LayerNorm(self.d_h_dim),
            nn.Linear(self.d_h_dim, predictor_embed_dim),
            act_layer(),
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
        )
        self.guidance_gate = nn.Sequential(
            nn.LayerNorm(self.d_h_dim),
            nn.Linear(self.d_h_dim, predictor_embed_dim),
            nn.Sigmoid(),
        )

        refine_in_dim = predictor_embed_dim * 2 + self.d_h_dim
        if use_proprio and proprio_flow in {"refine_only", "dyn_refine", "old_concat"} and proprio_emb_dim > 0:
            refine_in_dim += proprio_emb_dim
        self.refine_in = nn.Linear(refine_in_dim, predictor_embed_dim)
        self.refiner = ResidualMLP(
            predictor_embed_dim,
            int(predictor_embed_dim * mlp_ratio),
            depth=refiner_depth,
            act_layer=act_layer,
        )
        self.refine_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

        self.proprio_head = (
            nn.Sequential(
                nn.LayerNorm(predictor_embed_dim),
                nn.Linear(predictor_embed_dim, max(predictor_embed_dim, proprio_emb_dim)),
                act_layer(),
                nn.Linear(max(predictor_embed_dim, proprio_emb_dim), proprio_emb_dim),
            )
            if use_proprio and proprio_emb_dim > 0
            else None
        )
        self.delta_p_head = nn.Linear(self.d_h_dim, self.delta_p_dim) if self.delta_p_dim > 0 else None

        self.apply(self._init_weights)
        logger.info(
            "Initialized DynamicsGuidedPredictor("
            f"fdyn_type={fdyn_type}, guidance={guidance_mode}, d_h_dim={d_h_dim}, "
            f"proprio_flow={proprio_flow}, refiner_depth={refiner_depth})"
        )

    @property
    def mamba_backend(self) -> str | None:
        if isinstance(self.temporal_mixer, TemporalMambaOrGRU):
            return self.temporal_mixer.backend
        return None

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _stack_context(self, x: torch.Tensor) -> torch.Tensor:
        frames = [x]
        for k in range(1, self.context_window):
            if k < x.shape[1]:
                pad = x[:, :1].expand(-1, k, -1, -1, -1, -1)
                shifted = torch.cat([pad, x[:, :-k]], dim=1)
            else:
                shifted = x[:, :1].expand_as(x)
            frames.insert(0, shifted)
        return torch.cat(frames, dim=-1)

    def _pool_proprio(self, proprio: torch.Tensor | None) -> torch.Tensor | None:
        if proprio is None:
            return None
        if proprio.ndim == 4:
            return proprio.mean(dim=2)
        if proprio.ndim == 3:
            return proprio
        raise ValueError(f"Unexpected proprio shape: {tuple(proprio.shape)}")

    def _compact_state(self, x: torch.Tensor, proprio: torch.Tensor | None) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3, 4))
        if x.shape[1] > 1:
            prev = torch.cat([pooled[:, :1], pooled[:, :-1]], dim=1)
        else:
            prev = pooled
        visual_state = self.visual_state(torch.cat([pooled, pooled - prev], dim=-1))

        parts = [visual_state]
        if (
            self.use_proprio
            and self.proprio_state is not None
            and self.proprio_flow in {"dyn_only", "dyn_refine", "old_concat"}
        ):
            prop = self._pool_proprio(proprio)
            if prop is not None:
                parts.append(self.proprio_state(prop))
        return self.state_fuse(torch.cat(parts, dim=-1))

    def _action_summary(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 4:
            actions = actions.squeeze(2)
        if actions.ndim != 3:
            raise ValueError(f"Unexpected action shape: {tuple(actions.shape)}")
        return self.action_norm(actions)

    def _fdyn(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.fdyn_type == "raw_action":
            d_h = self.f_dyn(action)
        elif self.fdyn_type == "mlp":
            d_h = self.f_dyn(torch.cat([state, action], dim=-1))
        else:
            dyn_in = self.dyn_in(torch.cat([state, action], dim=-1))
            if isinstance(self.temporal_mixer, nn.GRU):
                dyn_out, _ = self.temporal_mixer(dyn_in)
            else:
                dyn_out = self.temporal_mixer(dyn_in)
            d_h = self.f_dyn(dyn_out)

        if self.dh_ablation == "remove":
            d_h = d_h * 0.0
        elif self.dh_ablation == "shuffle":
            d_h = d_h.roll(shifts=1, dims=0) if d_h.shape[0] > 1 else d_h.flip(dims=[1])
        elif self.dh_ablation == "random":
            d_h = torch.randn_like(d_h) * d_h.detach().std().clamp_min(1.0e-6) + d_h * 0.0
        return d_h

    def _apply_sparse_control(self, refined: torch.Tensor, base: torch.Tensor, d_h: torch.Tensor) -> torch.Tensor:
        if self.guidance_mode != "sparse_control":
            return refined
        B, T, N, D = refined.shape
        k = max(1, int(round(N * self.sparse_top_frac)))
        score = (refined - base).norm(dim=-1)
        idx = score.topk(k=k, dim=-1).indices
        mask = torch.zeros(B, T, N, device=refined.device, dtype=refined.dtype)
        mask.scatter_(-1, idx, 1.0)
        return base + (refined - base) * mask.unsqueeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None, torch.Tensor | None]:
        # x: B T V H W D
        x_context = self._stack_context(x)
        token_base = self.predictor_embed(x_context).flatten(2, 4)
        B, T, N, D = token_base.shape

        state = self._compact_state(x, proprio)
        action = self._action_summary(actions)
        d_h = self._fdyn(state, action)

        guidance = self.guidance(d_h).unsqueeze(2)
        gate = self.guidance_gate(d_h).unsqueeze(2)
        coarse = token_base + gate * guidance

        d_tokens = d_h.unsqueeze(2).expand(-1, -1, N, -1)
        refine_parts = [token_base, coarse, d_tokens]
        if self.use_proprio and proprio is not None and self.proprio_flow in {"refine_only", "dyn_refine", "old_concat"}:
            if proprio.ndim == 3:
                prop_tokens = proprio.unsqueeze(2).expand(-1, -1, N, -1)
            else:
                prop_tokens = proprio
            refine_parts.append(prop_tokens)

        refined = self.refine_in(torch.cat(refine_parts, dim=-1))
        refined = self.refiner(refined)
        refined = self.refine_norm(refined)
        refined = self._apply_sparse_control(refined, token_base, d_h)
        pred = self.predictor_proj(refined)

        pred_proprio = self.proprio_head(refined) if self.proprio_head is not None and proprio is not None else None
        delta_p = self.delta_p_head(d_h) if self.delta_p_head is not None else None
        with torch.no_grad():
            self.last_aux_stats = {
                "mgvt_d/d_h_mean": float(d_h.detach().mean().cpu()),
                "mgvt_d/d_h_std": float(d_h.detach().std().cpu()),
                "mgvt_d/guidance_gate_mean": float(gate.detach().mean().cpu()),
                "mgvt_d/guidance_norm": float(guidance.detach().norm(dim=-1).mean().cpu()),
                "mgvt_d/fdyn_type": self.fdyn_type,
                "mgvt_d/guidance_mode": self.guidance_mode,
            }
            if delta_p is not None:
                self.last_aux_stats["mgvt_d/delta_p_norm"] = float(delta_p.detach().norm(dim=-1).mean().cpu())
            if self.mamba_backend:
                self.last_aux_stats["mgvt_d/mamba_backend"] = self.mamba_backend
        return pred, None, pred_proprio

    def estimate_flops_per_forward(self, batch_size: int = 1, seq_len: int | None = None) -> int:
        seq_len = int(seq_len or max(1, self.grid_depth))
        tokens = batch_size * seq_len * self.grid_height * self.grid_width
        step_tokens = batch_size * seq_len
        flops = 0
        flops += _count_linear_flops(self.predictor_embed, tokens)
        flops += _count_linear_flops(self.visual_state, step_tokens)
        flops += _count_linear_flops(self.state_fuse, step_tokens)
        flops += _count_linear_flops(self.f_dyn, step_tokens)
        flops += _count_linear_flops(self.guidance, step_tokens)
        flops += _count_linear_flops(self.guidance_gate, step_tokens)
        flops += _count_linear_flops(self.refine_in, tokens)
        flops += _count_linear_flops(self.refiner, tokens)
        flops += _count_linear_flops(self.predictor_proj, tokens)
        if self.proprio_head is not None:
            flops += _count_linear_flops(self.proprio_head, tokens)
        if self.delta_p_head is not None:
            flops += _count_linear_flops(self.delta_p_head, step_tokens)
        if self.fdyn_type in {"gru", "mamba"}:
            # Conservative sequence-mixer proxy used only for matching diagnostics.
            hidden = getattr(self, "dyn_in", nn.Identity())[-2].out_features if hasattr(self, "dyn_in") else self.d_h_dim
            flops += int(6 * step_tokens * hidden * hidden)
        return int(flops)


def _make_predictor(fdyn_type: str, guidance_mode: str = "trend", **kwargs):
    return DynamicsGuidedPredictor(
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        fdyn_type=fdyn_type,
        guidance_mode=guidance_mode,
        **kwargs,
    )


def vit_predictor_mgvt_d_raw_action(**kwargs):
    return _make_predictor("raw_action", **kwargs)


def vit_predictor_mgvt_d_mlp(**kwargs):
    return _make_predictor("mlp", **kwargs)


def vit_predictor_mgvt_d_gru(**kwargs):
    return _make_predictor("gru", **kwargs)


def vit_predictor_mgvt_d_mamba(**kwargs):
    return _make_predictor("mamba", **kwargs)


def vit_predictor_mgvt_d_sparse_control(**kwargs):
    return _make_predictor("mlp", guidance_mode="sparse_control", **kwargs)
