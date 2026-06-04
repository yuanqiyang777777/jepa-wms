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
import torch.nn.functional as F
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
        self.uses_compact_state = fdyn_type != "raw_action"
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
        if self.uses_compact_state:
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
        else:
            self.visual_state = None
            self.proprio_state = None
            self.state_fuse = None

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
        if self.visual_state is None or self.state_fuse is None:
            raise RuntimeError("compact dynamics state requested for an action-only F_dyn")
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
        future_video_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None, torch.Tensor | None]:
        # x: B T V H W D
        x_context = self._stack_context(x)
        token_base = self.predictor_embed(x_context).flatten(2, 4)
        B, T, N, D = token_base.shape

        action = self._action_summary(actions)
        state = self._compact_state(x, proprio) if self.uses_compact_state else None
        d_h = self._fdyn(state, action)
        self._last_d_h = d_h

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
        if self.visual_state is not None:
            flops += _count_linear_flops(self.visual_state, step_tokens)
        if self.state_fuse is not None:
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


class D1RTrendPredictor(DynamicsGuidedPredictor):
    """D1R explicit teacher-grounded dynamics trend scaffold.

    The deployable forward path is still the D1 dynamics-guided predictor.  D1R
    adds a future-grounded teacher trend ``r_h`` and staged auxiliary losses that
    are computed inside the predictor forward when the training loop supplies
    ``future_video_features``.  That keeps DDP/autograd ownership inside the
    wrapped module while preserving the existing Stage-2 predictor contract.
    """

    VALID_STAGES = {"implicit", "r1_teacher", "r2_student", "g_refine", "oracle"}

    def __init__(
        self,
        *args,
        r_h_dim: int = 64,
        p_dyn_dim: int | None = None,
        d1r_stage: str = "g_refine",
        lambda_delta: float = 1.0,
        lambda_inv_r: float = 0.1,
        lambda_sig: float = 0.01,
        lambda_trend: float = 1.0,
        lambda_inv_d: float = 0.1,
        use_inv_d: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("fdyn_type", "mlp")
        kwargs.setdefault("d_h_dim", r_h_dim)
        kwargs.setdefault("proprio_flow", "no_proprio")
        super().__init__(*args, **kwargs)
        if d1r_stage not in self.VALID_STAGES:
            raise ValueError(f"Unsupported d1r_stage: {d1r_stage}")
        if int(r_h_dim) != self.d_h_dim:
            raise ValueError("D1R first smoke requires dim(r_h) == dim(d_h)")

        act_layer = nn.SiLU if kwargs.get("use_silu", False) else nn.GELU
        self.r_h_dim = int(r_h_dim)
        self.p_dyn_dim = int(p_dyn_dim or kwargs.get("state_dim", 64))
        self.d1r_stage = d1r_stage
        self.lambda_delta = float(lambda_delta)
        self.lambda_inv_r = float(lambda_inv_r)
        self.lambda_sig = float(lambda_sig)
        self.lambda_trend = float(lambda_trend)
        self.lambda_inv_d = float(lambda_inv_d)
        self.use_inv_d = bool(use_inv_d)

        self.p_dyn = nn.Sequential(
            nn.LayerNorm(self.predictor_embed.in_features // max(1, self.context_window)),
            nn.Linear(self.predictor_embed.in_features // max(1, self.context_window), self.p_dyn_dim),
            act_layer(),
            nn.Linear(self.p_dyn_dim, self.p_dyn_dim),
        )
        self.d1r_state_fuse = nn.Sequential(
            nn.LayerNorm(2 * self.p_dyn_dim),
            nn.Linear(2 * self.p_dyn_dim, kwargs.get("state_dim", self.p_dyn_dim)),
            act_layer(),
        )
        self.e_delta = nn.Sequential(
            nn.LayerNorm(3 * self.p_dyn_dim),
            nn.Linear(3 * self.p_dyn_dim, max(self.p_dyn_dim, self.r_h_dim)),
            act_layer(),
            nn.Linear(max(self.p_dyn_dim, self.r_h_dim), self.r_h_dim),
        )
        self.b_delta = nn.Linear(self.r_h_dim, self.p_dyn_dim)
        self.q = nn.Sequential(nn.Linear(self.d_h_dim, self.r_h_dim), nn.LayerNorm(self.r_h_dim))
        inv_in_dim = self.p_dyn_dim + self.r_h_dim
        action_out_dim = self.predictor_total_embed_dim
        self.d_inv_r = nn.Sequential(
            nn.LayerNorm(inv_in_dim),
            nn.Linear(inv_in_dim, max(inv_in_dim, action_out_dim)),
            act_layer(),
            nn.Linear(max(inv_in_dim, action_out_dim), action_out_dim),
        )
        self.d_inv_d = nn.Sequential(
            nn.LayerNorm(inv_in_dim),
            nn.Linear(inv_in_dim, max(inv_in_dim, action_out_dim)),
            act_layer(),
            nn.Linear(max(inv_in_dim, action_out_dim), action_out_dim),
        )
        self._d1r_aux_losses: dict[str, torch.Tensor] = {}
        self._d1r_aux_replace_loss = False

        self.apply(self._init_weights)
        self.configure_d1r_stage(d1r_stage)
        logger.info(
            "Initialized D1RTrendPredictor("
            f"stage={d1r_stage}, r_h_dim={self.r_h_dim}, p_dyn_dim={self.p_dyn_dim}, "
            f"use_inv_d={self.use_inv_d})"
        )

    def configure_d1r_stage(self, stage: str) -> None:
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Unsupported d1r_stage: {stage}")
        self.d1r_stage = stage
        for param in self.parameters():
            param.requires_grad = False

        def enable(module: nn.Module | None) -> None:
            if module is None:
                return
            for param in module.parameters():
                param.requires_grad = True

        teacher_modules = [self.p_dyn, self.e_delta, self.b_delta, self.d_inv_r]
        student_core_modules = [self.d1r_state_fuse, self.f_dyn]
        if hasattr(self, "dyn_in"):
            student_core_modules.append(self.dyn_in)
        if self.temporal_mixer is not None:
            student_core_modules.append(self.temporal_mixer)
        refiner_modules = [
            self.predictor_embed,
            self.guidance,
            self.guidance_gate,
            self.refine_in,
            self.refiner,
            self.refine_norm,
            self.predictor_proj,
            self.proprio_head,
        ]

        if stage == "implicit":
            for module in [self.p_dyn, *student_core_modules, *refiner_modules]:
                enable(module)
        elif stage == "r1_teacher":
            for module in teacher_modules:
                enable(module)
        elif stage == "r2_student":
            for module in [*student_core_modules, self.q]:
                enable(module)
            if self.use_inv_d:
                enable(self.d_inv_d)
        elif stage == "g_refine":
            for module in refiner_modules:
                enable(module)
        elif stage == "oracle":
            for module in [*teacher_modules, *refiner_modules]:
                enable(module)

    def _compact_state(self, x: torch.Tensor, proprio: torch.Tensor | None) -> torch.Tensor:
        del proprio
        y = self.p_dyn(x).mean(dim=(2, 3, 4))
        if y.shape[1] > 1:
            prev = torch.cat([y[:, :1], y[:, :-1]], dim=1)
        else:
            prev = y
        return self.d1r_state_fuse(torch.cat([y, y - prev], dim=-1))

    def _project_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: B T V H W D -> B T N P
        y = self.p_dyn(x)
        return y.flatten(2, 4)

    def _teacher_trend(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_source = self._project_tokens(source)
        y_target = self._project_tokens(target)
        source_pool = y_source.mean(dim=2)
        target_pool = y_target.mean(dim=2)
        delta_y = target_pool - source_pool
        r_h = self.e_delta(torch.cat([source_pool, target_pool, delta_y], dim=-1))
        return r_h, delta_y, source_pool

    def _sigreg_loss(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] < 2:
            return flat.new_zeros(())
        std = flat.std(dim=0)
        return F.relu(1.0 - std).mean()

    def _oracle_refine_from_context(self, x_context: torch.Tensor, r_h: torch.Tensor) -> torch.Tensor:
        token_base = self.predictor_embed(x_context).flatten(2, 4)
        B, T, N, _ = token_base.shape
        guidance = self.guidance(r_h).unsqueeze(2)
        gate = self.guidance_gate(r_h).unsqueeze(2)
        coarse = token_base + gate * guidance
        d_tokens = r_h.unsqueeze(2).expand(-1, -1, N, -1)
        refined = self.refine_in(torch.cat([token_base, coarse, d_tokens], dim=-1))
        refined = self.refiner(refined)
        refined = self.refine_norm(refined)
        return self.predictor_proj(refined)

    def _oracle_refine(self, source: torch.Tensor, r_h: torch.Tensor) -> torch.Tensor:
        return self._oracle_refine_from_context(self._stack_context(source), r_h)

    def oracle_predict(self, source_context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Non-deployable oracle forecast for T0 diagnostics.

        ``target`` is intentionally future-derived here.  This method is used
        only for the registered oracle upper-bound score and is not reachable
        from the deployable ``forward`` path.
        """

        source_last = source_context[:, -1:]
        target_last = target[:, -1:]
        r_h, _delta_y, _source_pool = self._teacher_trend(source_last, target_last)
        x_context = self._stack_context(source_context)[:, -1:]
        return self._oracle_refine_from_context(x_context, r_h.detach())

    def _compute_d1r_aux_losses(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        if source.shape[1] < 2:
            self._d1r_aux_losses = {}
            self._d1r_aux_replace_loss = False
            return
        source = source[:, :-1]
        target = target[:, 1:]
        if actions.ndim == 4:
            actions = actions.squeeze(2)
        actions = actions[:, :-1].detach()

        r_h, delta_y, source_pool = self._teacher_trend(source, target)
        delta_hat = self.b_delta(r_h)
        loss_delta = F.mse_loss(delta_hat, delta_y.detach())
        inv_r = self.d_inv_r(torch.cat([source_pool, r_h], dim=-1))
        loss_inv_r = F.mse_loss(inv_r, actions)
        loss_sig = self._sigreg_loss(r_h)

        losses: dict[str, torch.Tensor] = {
            "d1r/loss_delta": loss_delta,
            "d1r/loss_inv_r": loss_inv_r,
            "d1r/loss_sigreg": loss_sig,
        }
        teacher_total = (
            self.lambda_delta * loss_delta
            + self.lambda_inv_r * loss_inv_r
            + self.lambda_sig * loss_sig
        )

        d_h = getattr(self, "_last_d_h", None)
        if d_h is not None:
            d_h = d_h[:, :-1]
            q_d = self.q(d_h)
            loss_trend = F.mse_loss(q_d, r_h.detach())
            inv_d = self.d_inv_d(torch.cat([source_pool.detach(), d_h], dim=-1))
            loss_inv_d = F.mse_loss(inv_d, actions)
            losses["d1r/loss_trend"] = loss_trend
            losses["d1r/loss_inv_d"] = loss_inv_d
        else:
            loss_trend = source.new_zeros(())
            loss_inv_d = source.new_zeros(())

        if self.d1r_stage == "r1_teacher":
            losses["d1r/loss_total"] = teacher_total
            self._d1r_aux_replace_loss = True
        elif self.d1r_stage == "r2_student":
            total = self.lambda_trend * loss_trend
            if self.use_inv_d:
                total = total + self.lambda_inv_d * loss_inv_d
            losses["d1r/loss_total"] = total
            self._d1r_aux_replace_loss = True
        elif self.d1r_stage == "oracle":
            oracle_pred = self._oracle_refine(source, r_h.detach())
            oracle_target = target.flatten(2, 4).detach()
            loss_oracle = F.mse_loss(oracle_pred, oracle_target)
            losses["d1r/loss_oracle_pred"] = loss_oracle
            losses["d1r/loss_total"] = loss_oracle + teacher_total
            self._d1r_aux_replace_loss = True
        else:
            self._d1r_aux_replace_loss = False

        with torch.no_grad():
            flat_r = r_h.detach().reshape(-1, r_h.shape[-1])
            flat_d = d_h.detach().reshape(-1, d_h.shape[-1]) if d_h is not None else None
            self.last_aux_stats.update(
                {
                    "d1r/r_h_std": float(flat_r.std(dim=0).median().cpu()),
                    "d1r/r_h_norm": float(flat_r.norm(dim=-1).mean().cpu()),
                    "d1r/lambda_delta": self.lambda_delta,
                    "d1r/stage": self.d1r_stage,
                }
            )
            if flat_d is not None:
                self.last_aux_stats["d1r/d_h_norm"] = float(flat_d.norm(dim=-1).mean().cpu())

        self._d1r_aux_losses = losses

    def forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        proprio: torch.Tensor | None = None,
        future_video_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None, torch.Tensor | None]:
        pred, action_features, pred_proprio = super().forward(x, actions, proprio)
        if future_video_features is not None:
            self._compute_d1r_aux_losses(x, future_video_features, actions)
        else:
            self._d1r_aux_losses = {}
            self._d1r_aux_replace_loss = False
        return pred, action_features, pred_proprio

    def get_d1r_aux_losses(self) -> tuple[dict[str, torch.Tensor], bool]:
        return self._d1r_aux_losses, self._d1r_aux_replace_loss

    def estimate_inference_path_params(self) -> int:
        modules = [
            self.p_dyn,
            self.d1r_state_fuse,
            self.f_dyn,
            self.predictor_embed,
            self.guidance,
            self.guidance_gate,
            self.refine_in,
            self.refiner,
            self.refine_norm,
            self.predictor_proj,
        ]
        if hasattr(self, "dyn_in"):
            modules.append(self.dyn_in)
        if self.temporal_mixer is not None:
            modules.append(self.temporal_mixer)
        return int(sum(p.numel() for module in modules for p in module.parameters()))


def vit_predictor_mgvt_d1r(**kwargs):
    kwargs.setdefault("d_h_dim", kwargs.get("r_h_dim", 64))
    kwargs.setdefault("state_dim", kwargs.get("p_dyn_dim", 64))
    return D1RTrendPredictor(norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
