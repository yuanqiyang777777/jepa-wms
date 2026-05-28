# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange

from src.utils.logging import get_logger

from ..traj_dset import TrajDataset

log = get_logger(__name__)


class LanceTrajSlicerDataset(TrajDataset):
    """Window-level slicer over a `LanceWindowDataset` (optionally wrapped in `TrajSubset`).

    Slice generation logic is a bit-identical replica of `TrajSlicerDataset.__init__`
    (`app/plan_common/datasets/traj_dset.py:86-122`) so the resulting `self.slices`
    list -- under the same `(num_frames, frameskip, action_skip, generator)` -- equals
    the raw `TrajSlicerDataset.slices` list element-by-element.

    `__getitem__` calls `read_window_normalized(ep_idx, frames, action_frames)`
    on the underlying `LanceWindowDataset` directly -- no full-episode read.
    """

    def __init__(
        self,
        dataset,  # LanceWindowDataset OR TrajSubset wrapping one
        num_frames: int,
        frameskip: int = 1,
        action_skip: int = 1,
        process_actions: str = "concat",
        generator: Optional[torch.Generator] = None,
    ):
        self.dataset = dataset
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.action_skip = action_skip
        self.process_actions = process_actions

        # ---- Slice generation (bit-identical to TrajSlicerDataset.__init__) ----
        self.slices: list[tuple[int, int, int]] = []
        for i in range(len(self.dataset)):
            T = int(self.dataset.get_seq_length(i))
            if T - num_frames < 0:
                log.info(f"Ignored short sequence #{i}: len={T}, num_frames={num_frames}")
            else:
                self.slices += [
                    (i, start, start + num_frames * self.frameskip)
                    for start in range(T - num_frames * frameskip + 1)
                ]
        order = torch.randperm(len(self.slices), generator=generator).tolist()
        self.slices = [self.slices[i] for i in order]

        # ---- Dim attributes (bit-identical to TrajSlicerDataset.__init__) ----
        self.proprio_dim = self.dataset.proprio_dim
        if self.process_actions == "concat":
            if self.frameskip < self.action_skip:
                self.action_dim = self.dataset.action_dim
            else:
                self.action_dim = self.dataset.action_dim * (self.frameskip // self.action_skip)
        else:
            self.action_dim = self.dataset.action_dim
        self.state_dim = self.dataset.state_dim

    def get_seq_length(self, idx: int) -> int:
        return self.num_frames

    def __len__(self) -> int:
        return len(self.slices)

    def _resolve(self, i: int) -> tuple[int, "object"]:
        """If `self.dataset` is a TrajSubset, translate subset-local i -> underlying ep_idx
        and return the underlying LanceWindowDataset; else pass through.
        """
        if hasattr(self.dataset, "indices") and hasattr(self.dataset, "dataset"):
            ep_idx = int(self.dataset.indices[i])
            underlying = self.dataset.dataset
        else:
            ep_idx = int(i)
            underlying = self.dataset
        return ep_idx, underlying

    def __getitem__(self, idx, **kwargs):  # **kwargs to absorb subtask=None forwarded by TrajSubset
        i, start, end = self.slices[idx]
        ep_idx, underlying = self._resolve(i)

        # Frames for visual/proprio/state (slice indices follow Python slice convention [start, end))
        step_indices = list(range(start, end, self.frameskip))
        # Frames for action -- may have different stride.
        action_step_indices = list(range(start, end, self.action_skip))

        out = underlying.read_window_normalized(
            ep_idx=ep_idx,
            step_indices=step_indices,
            action_step_indices=action_step_indices,
        )

        # ---- Visual: /255.0 + rearrange + transform (bit-identical to point_maze_dset.py:get_frames) ----
        visual = out["visual"] / 255.0  # uint8 -> float promotion, matches raw
        visual = rearrange(visual, "T H W C -> T C H W")
        if getattr(underlying, "transform", None) is not None:
            visual = underlying.transform(visual)
        obs = {"visual": visual, "proprio": out["proprio"]}

        # ---- State / reward (mirror TrajSlicerDataset.__getitem__ exactly) ----
        state = out["state"]
        # PointMaze raw flow: reward is None; TrajSlicerDataset substitutes zeros sized to FULL act.
        # After slicing, length is (end - start) // frameskip == num_frames.
        reward = torch.zeros(self.num_frames, dtype=torch.float32)
        # state can never be None here (Lance always stores it); kept for safety symmetry.
        if state is None:
            state = torch.zeros(self.num_frames, dtype=torch.float32)

        # ---- Action processing (bit-identical to TrajSlicerDataset.__getitem__:146-153) ----
        act = out["action"]
        if self.frameskip < self.action_skip:
            act = rearrange(
                act,
                "(n f) d -> n (f d)",
                n=self.num_frames * self.frameskip // self.action_skip,
            )
        else:
            if self.process_actions == "concat":
                act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)
            elif self.process_actions == "sum":
                act = rearrange(act, "(n f) d -> n f d", n=self.num_frames)
                act = act.sum(dim=1)

        return (obs, act, state, reward)
