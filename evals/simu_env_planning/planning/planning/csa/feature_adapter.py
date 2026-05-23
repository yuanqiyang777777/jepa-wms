from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class PCAWhiteningAdapter:
    """Frozen PCA-whitening adapter for compact support-query features."""

    mean: torch.Tensor
    components: torch.Tensor
    scale: torch.Tensor
    eps: float = 1e-6
    whiten: bool = True

    @classmethod
    def fit(
        cls,
        x: torch.Tensor,
        dim: int = 64,
        eps: float = 1e-6,
        whiten: bool = True,
    ) -> "PCAWhiteningAdapter":
        x = _as_2d_float(x)
        if x.shape[0] < 2:
            raise ValueError("PCAWhiteningAdapter.fit requires at least two samples")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        mean = x.mean(dim=0)
        centered = x - mean
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        keep = min(dim, vh.shape[0])
        components = vh[:keep].contiguous()
        if whiten:
            scale = singular_values[:keep] / max(float(x.shape[0] - 1) ** 0.5, 1.0)
            scale = scale.clamp_min(eps)
        else:
            scale = torch.ones(keep, dtype=x.dtype, device=x.device)
        return cls(mean=mean, components=components, scale=scale, eps=eps, whiten=whiten)

    @property
    def out_dim(self) -> int:
        return int(self.components.shape[0])

    @property
    def in_dim(self) -> int:
        return int(self.components.shape[1])

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_2d_float(x).to(device=self.mean.device, dtype=self.mean.dtype)
        if x.shape[1] != self.in_dim:
            raise ValueError(f"Expected input dim {self.in_dim}, got {x.shape[1]}")
        projected = (x - self.mean) @ self.components.T
        return projected / self.scale

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.detach().cpu(),
            "components": self.components.detach().cpu(),
            "scale": self.scale.detach().cpu(),
            "eps": self.eps,
            "whiten": self.whiten,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "PCAWhiteningAdapter":
        return cls(
            mean=state["mean"],
            components=state["components"],
            scale=state["scale"],
            eps=float(state.get("eps", 1e-6)),
            whiten=bool(state.get("whiten", True)),
        )

    def to(self, device: torch.device | str) -> "PCAWhiteningAdapter":
        return PCAWhiteningAdapter(
            mean=self.mean.to(device),
            components=self.components.to(device),
            scale=self.scale.to(device),
            eps=self.eps,
            whiten=self.whiten,
        )


def _as_2d_float(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.detach()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    return x.to(dtype=torch.float32)

