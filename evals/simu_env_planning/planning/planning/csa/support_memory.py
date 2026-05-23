from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

from evals.simu_env_planning.planning.planning.csa.feature_adapter import PCAWhiteningAdapter


REQUIRED_METADATA_KEYS = (
    "feature_source",
    "state_reduction",
    "raw_state_dim",
    "state_feature_dim",
    "action_dim",
    "action_space",
    "frameskip",
    "action_skip",
    "pca_dim",
    "sample_seed",
)


def flatten_state_encoding(state: Any) -> torch.Tensor:
    """Flatten Tensor/TensorDict-like state encodings into [B, D]."""
    return compact_state_encoding(state, reduction="flatten")


def compact_state_encoding(state: Any, reduction: str = "mean_tokens") -> torch.Tensor:
    """Compact Tensor/TensorDict-like state encodings into [B, D].

    ``mean_tokens`` preserves the final feature dimension and averages every
    intervening time/token/spatial dimension. ``flatten`` is kept as an explicit
    debug mode for already-small state encodings.
    """
    if reduction not in {"mean_tokens", "flatten"}:
        raise ValueError(f"Unsupported state_reduction={reduction!r}; expected 'mean_tokens' or 'flatten'")
    if isinstance(state, Mapping) or hasattr(state, "keys"):
        pieces = []
        for key in ("visual", "proprio"):
            if key in state:
                pieces.append(_compact_tensor(state[key], reduction))
        if not pieces:
            raise ValueError("State mapping must contain at least one of: visual, proprio")
        return torch.cat(pieces, dim=1)
    return _compact_tensor(state, reduction)


def _compact_tensor(value: Any, reduction: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.detach()
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if reduction == "flatten":
        if value.ndim > 2:
            value = value.reshape(value.shape[0], -1)
        return value.to(dtype=torch.float32)
    if value.ndim == 2:
        return value.to(dtype=torch.float32)
    if value.ndim < 2:
        raise ValueError(f"Expected at least 2D state tensor for mean_tokens reduction, got shape {tuple(value.shape)}")
    value = value.reshape(value.shape[0], -1, value.shape[-1])
    return value.mean(dim=1).to(dtype=torch.float32)


def _complete_metadata(
    metadata: Dict[str, Any],
    *,
    feature_source: str,
    state_reduction: str,
    raw_state_dim: int,
    state_feature_dim: int,
    action_dim: int,
    action_space: str,
    frameskip: int,
    action_skip: int,
    pca_dim: int,
    sample_seed: int,
) -> Dict[str, Any]:
    completed = dict(metadata)
    completed.update(
        {
            "feature_source": completed.get("feature_source", feature_source),
            "state_reduction": completed.get("state_reduction", state_reduction),
            "raw_state_dim": int(completed.get("raw_state_dim", raw_state_dim)),
            "state_feature_dim": int(completed.get("state_feature_dim", state_feature_dim)),
            "action_dim": int(completed.get("action_dim", action_dim)),
            "action_space": completed.get("action_space", action_space),
            "frameskip": int(completed.get("frameskip", frameskip)),
            "action_skip": int(completed.get("action_skip", action_skip)),
            "pca_dim": int(completed.get("pca_dim", pca_dim)),
            "sample_seed": int(completed.get("sample_seed", sample_seed)),
        }
    )
    return completed


def _require_metadata(metadata: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"SupportMemory metadata missing required keys: {missing}")


@dataclass
class SupportMemory:
    """Real state-action support memory for CSA pilot diagnostics."""

    state_features: torch.Tensor
    actions: torch.Tensor
    adapter: PCAWhiteningAdapter | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    action_mean: torch.Tensor | None = None
    action_std: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.state_features = flatten_state_encoding(self.state_features)
        self.actions = self._as_action_2d(self.actions)
        if self.state_features.shape[0] != self.actions.shape[0]:
            raise ValueError(
                f"state/action memory size mismatch: {self.state_features.shape[0]} vs {self.actions.shape[0]}"
            )
        if self.action_mean is None:
            self.action_mean = self.actions.mean(dim=0)
        if self.action_std is None:
            self.action_std = self.actions.std(dim=0, unbiased=False).clamp_min(1e-6)
        else:
            self.action_std = self.action_std.clamp_min(1e-6)
        if self.metadata:
            _require_metadata(self.metadata)

    @classmethod
    def from_tensors(
        cls,
        states: Any,
        actions: torch.Tensor,
        pca_dim: int = 64,
        max_memory: int | None = 200000,
        state_reduction: str = "mean_tokens",
        pca_fit_samples: int | None = 5000,
        sample_seed: int = 0,
        action_space: str = "model_normalized",
        feature_source: str = "pre_encoded",
        frameskip: int = 0,
        action_skip: int = 1,
        adapter: PCAWhiteningAdapter | None = None,
        generator: torch.Generator | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> "SupportMemory":
        flat_states = compact_state_encoding(states, reduction=state_reduction)
        flat_actions = cls._as_action_2d(actions)
        if flat_states.shape[0] != flat_actions.shape[0]:
            raise ValueError(f"states/actions length mismatch: {flat_states.shape[0]} vs {flat_actions.shape[0]}")

        if max_memory is not None and flat_states.shape[0] > max_memory:
            if generator is None:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(sample_seed))
            idx = torch.randperm(flat_states.shape[0], generator=generator)[:max_memory]
            flat_states = flat_states[idx]
            flat_actions = flat_actions[idx]

        if pca_fit_samples is not None and pca_fit_samples > 0 and flat_states.shape[0] > pca_fit_samples:
            fit_generator = torch.Generator(device="cpu")
            fit_generator.manual_seed(int(sample_seed) + 1)
            fit_idx = torch.randperm(flat_states.shape[0], generator=fit_generator)[:pca_fit_samples]
            fit_states = flat_states[fit_idx]
        else:
            fit_states = flat_states

        if adapter is None:
            adapter = PCAWhiteningAdapter.fit(fit_states, dim=pca_dim)
        state_features = adapter.transform(flat_states)
        completed_metadata = _complete_metadata(
            metadata or {},
            feature_source=feature_source,
            state_reduction=state_reduction,
            raw_state_dim=flat_states.shape[1],
            state_feature_dim=state_features.shape[1],
            action_dim=flat_actions.shape[1],
            action_space=action_space,
            frameskip=frameskip,
            action_skip=action_skip,
            pca_dim=pca_dim,
            sample_seed=sample_seed,
        )
        return cls(
            state_features=state_features,
            actions=flat_actions,
            adapter=adapter,
            metadata=completed_metadata,
        )

    @staticmethod
    def _as_action_2d(actions: torch.Tensor) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        actions = actions.detach()
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        elif actions.ndim > 2:
            actions = actions.reshape(-1, actions.shape[-1])
        return actions.to(dtype=torch.float32)

    @property
    def num_items(self) -> int:
        return int(self.state_features.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def state_dim(self) -> int:
        return int(self.state_features.shape[1])

    def transform_query_state(self, states: Any, raw_state: bool = True, state_reduction: str | None = None) -> torch.Tensor:
        reduction = state_reduction or self.metadata.get("state_reduction", "mean_tokens")
        flat = compact_state_encoding(states, reduction=reduction)
        if raw_state and self.adapter is not None:
            return self.adapter.transform(flat)
        return flat

    def raw_query_dim(self, states: Any, state_reduction: str | None = None) -> int:
        reduction = state_reduction or self.metadata.get("state_reduction", "mean_tokens")
        return int(compact_state_encoding(states, reduction=reduction).shape[1])

    def validate_for_runtime(
        self,
        *,
        raw_state_dim: int,
        action_dim: int,
        action_space: str,
        state_reduction: str,
        frameskip: int,
        action_skip: int,
    ) -> None:
        _require_metadata(self.metadata)
        expected = {
            "raw_state_dim": int(raw_state_dim),
            "action_dim": int(action_dim),
            "action_space": action_space,
            "state_reduction": state_reduction,
            "frameskip": int(frameskip),
            "action_skip": int(action_skip),
        }
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ValueError(
                    f"SupportMemory metadata {key} mismatch: expected runtime {value!r}, "
                    f"memory has {self.metadata.get(key)!r}"
                )

    def normalized_actions(self) -> torch.Tensor:
        return (self.actions - self.action_mean) / self.action_std

    def normalize_query_actions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = self._as_action_2d(actions).to(device=self.actions.device, dtype=self.actions.dtype)
        return (actions - self.action_mean.to(actions.device)) / self.action_std.to(actions.device)

    def to(self, device: torch.device | str) -> "SupportMemory":
        return SupportMemory(
            state_features=self.state_features.to(device),
            actions=self.actions.to(device),
            adapter=self.adapter.to(device) if self.adapter is not None else None,
            metadata=dict(self.metadata),
            action_mean=self.action_mean.to(device),
            action_std=self.action_std.to(device),
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "state_features": self.state_features.detach().cpu(),
            "actions": self.actions.detach().cpu(),
            "adapter": self.adapter.state_dict() if self.adapter is not None else None,
            "metadata": dict(self.metadata),
            "action_mean": self.action_mean.detach().cpu(),
            "action_std": self.action_std.detach().cpu(),
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "SupportMemory":
        state = torch.load(path, map_location=map_location, weights_only=False)
        adapter_state = state.get("adapter")
        adapter = PCAWhiteningAdapter.from_state_dict(adapter_state) if adapter_state is not None else None
        return cls(
            state_features=state["state_features"],
            actions=state["actions"],
            adapter=adapter,
            metadata=state.get("metadata", {}),
            action_mean=state.get("action_mean"),
            action_std=state.get("action_std"),
        )
