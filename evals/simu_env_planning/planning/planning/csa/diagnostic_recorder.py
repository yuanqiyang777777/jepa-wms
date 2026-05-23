# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""Per-episode diagnostic dump writer for the CSA-MPC full diagnostic suite.

The recorder is deliberately *dumb*: during instrumented vanilla CEM/MPC eval it
only accumulates, per replan step, the pooled imagined-rollout latents, the
pooled real executed-rollout latents, the selected action plan, and two terminal
goal costs (planner-predicted vs. measured-on-real). At episode end it writes one
``.pt`` file.

All support scoring (``ConditionalSupportScorer``) and the diagnostics 0-4 run
*offline* on these dumps (see ``analysis.py``), so the support memory, ``k``,
``beta`` and every diagnostic threshold can be changed without re-running eval --
which matters because a single MW eval is multi-hour.

Schema (one ``.pt`` per episode)::

    {
      "schema_version": int,
      "episode_id": int,
      "env": str,
      "episode_success": int,                  # 1 if the episode reached the goal
      "goal_state": Tensor[D_state] | None,    # pooled goal latent
      "metadata": dict,                        # env / frameskip / action_skip / ckpt ...
      "num_steps": int,
      "steps": [
        {
          "replan_idx": int,
          "decision_state": Tensor[D_state],      # pooled real obs latent = q_0 state
          "imagined_states": Tensor[H, D_state],  # pooled imagined rollout x_hat_1..x_hat_H
          "real_states": Tensor[H_real, D_state], # pooled real executed rollout
          "selected_actions": Tensor[H, A],       # selected plan, model-normalized actions
          "predicted_terminal_cost": float,       # planner objective units
          "real_terminal_cost": float,            # same objective units, on real terminal
          "state_dist": float,                    # real env-space distance to goal after chunk
          "step_success": int,
        }, ...
      ],
    }

Per-depth query indexing used by ``analysis.py`` (``H`` = plan length)::

    q_0 = (decision_state,            selected_actions[0])      # t=0, real observation
    q_d = (imagined_states[d - 1],    selected_actions[d])      # t>=1, imagined state
    rollout_error[d] = dist(imagined_states[d - 1], real_states[d - 1])  # d = 1..H
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch

SCHEMA_VERSION = 1
DUMP_FILENAME = "csa_diag_dump.pt"


def _cpu(value: Any) -> torch.Tensor | None:
    """Detach an array-like to a contiguous CPU float32 tensor."""
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


@dataclass
class DiagnosticRecorder:
    """Accumulates per-replan-step CSA diagnostic arrays for a single episode."""

    episode_id: int
    env: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    goal_state: torch.Tensor | None = None
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.goal_state = _cpu(self.goal_state)

    def record_step(
        self,
        *,
        replan_idx: int,
        decision_state: torch.Tensor,
        imagined_states: torch.Tensor,
        real_states: torch.Tensor,
        selected_actions: torch.Tensor,
        predicted_terminal_cost: float,
        real_terminal_cost: float,
        state_dist: float,
        step_success: bool,
    ) -> None:
        """Store one replan step. All tensors are moved to CPU float32.

        ``decision_state`` is the pooled real observation latent at the replan
        boundary (the strictly-reliable t=0 query state). ``imagined_states`` is
        the pooled imagined rollout the planner committed to; ``real_states`` is
        the pooled latent of what the environment actually produced when the
        selected plan was executed (may be shorter than the imagined rollout if
        the episode terminated mid-chunk).
        """
        self.steps.append(
            {
                "replan_idx": int(replan_idx),
                "decision_state": _cpu(decision_state),
                "imagined_states": _cpu(imagined_states),
                "real_states": _cpu(real_states),
                "selected_actions": _cpu(selected_actions),
                "predicted_terminal_cost": _to_float(predicted_terminal_cost),
                "real_terminal_cost": _to_float(real_terminal_cost),
                "state_dist": _to_float(state_dist),
                "step_success": int(bool(step_success)),
            }
        )

    def to_dict(self, episode_success: bool) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": int(self.episode_id),
            "env": str(self.env),
            "episode_success": int(bool(episode_success)),
            "goal_state": self.goal_state,
            "metadata": dict(self.metadata),
            "num_steps": len(self.steps),
            "steps": self.steps,
        }

    def save(self, path: str | Path, episode_success: bool) -> Path:
        """Write the per-episode dump to ``path`` and return the written path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(episode_success), path)
        return path


def _to_float(value: Any) -> float:
    """Best-effort scalar coercion that tolerates tensors and None."""
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        return float(value.detach().flatten()[0].cpu().item())
    return float(value)


def load_episode_dumps(dump_dir: str | Path, filename: str = DUMP_FILENAME) -> List[Dict[str, Any]]:
    """Load every per-episode dump under ``dump_dir`` (searched recursively).

    Returns the dumps sorted by ``episode_id`` so analysis output is stable
    regardless of filesystem iteration order or DDP rank sharding.
    """
    dump_dir = Path(dump_dir)
    dumps: List[Dict[str, Any]] = []
    for path in sorted(dump_dir.rglob(filename)):
        dumps.append(torch.load(path, map_location="cpu", weights_only=False))
    dumps.sort(key=lambda d: int(d.get("episode_id", 0)))
    return dumps
