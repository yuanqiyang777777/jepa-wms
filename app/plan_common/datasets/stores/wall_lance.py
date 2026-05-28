# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from einops import rearrange

from src.utils.logging import get_logger

from ..traj_dset import split_traj_datasets
from .lance_slicer import LanceTrajSlicerDataset
from .lance_window import LanceWindowDataset

log = get_logger(__name__)


class WallLanceWindowDataset(LanceWindowDataset):
    """Lance-backed Wall dataset.

    Differs from the generic `LanceWindowDataset` in one way:
      * `__getitem__` injects per-episode `fix_door_location` and `fix_wall_location`
        into the 5-tuple's env_info, sourced from `swm_metadata.json` (only frame-0
        values are stored -- matches raw `WallDataset.get_frames` which only reads
        `door_locations[idx, frames][0]`, see wall_dset.py:112).

    The training-time slicer (`LanceTrajSlicerDataset`) does not touch env_info,
    so this override only affects the full-episode eval/planning code path.
    """

    def __init__(self, lance_uri: str, image_codec: Optional[str] = None, **kwargs):
        super().__init__(lance_uri=lance_uri, image_codec=image_codec, **kwargs)
        self.visual_layout = "thwc"
        meta = self.store.metadata
        if "door_locations_frame0" not in meta or "wall_locations_frame0" not in meta:
            raise ValueError(
                f"Wall Lance dataset at {self.lance_uri} is missing "
                "door_locations_frame0 / wall_locations_frame0 in swm_metadata.json. "
                "Re-convert with the current writer_wall.convert_wall_to_lance."
            )
        self._door_locations_frame0 = torch.tensor(meta["door_locations_frame0"], dtype=torch.float32)
        self._wall_locations_frame0 = torch.tensor(meta["wall_locations_frame0"], dtype=torch.float32)
        if self._door_locations_frame0.shape[0] != self._n_episodes:
            raise ValueError(
                f"door_locations_frame0 length {self._door_locations_frame0.shape[0]} "
                f"!= num_episodes {self._n_episodes}"
            )

    def __getitem__(self, idx, **kwargs):
        """Full-episode read with Wall-specific env_info."""
        T = int(self.get_seq_length(idx))
        frames = list(range(T))
        out = self.read_window_normalized(ep_idx=idx, step_indices=frames, action_step_indices=frames)
        visual = out["visual"].float() / 255.0
        if self.transform:
            visual = self.transform(visual)
        obs = {"visual": visual, "proprio": out["proprio"]}
        env_info = {
            "fix_door_location": self._door_locations_frame0[idx],
            "fix_wall_location": self._wall_locations_frame0[idx],
        }
        return obs, out["action"], out["state"], None, env_info


def load_wall_lance_slice_train_val(
    transform: Optional[Callable] = None,
    lance_uri: str = "",
    image_codec: Optional[str] = None,
    n_rollout: Optional[int] = None,
    normalize_action: bool = False,
    split_ratio: float = 0.8,
    split_mode: str = "random",
    num_hist: int = 0,
    num_pred: int = 0,
    num_frames_val: Optional[int] = None,
    frameskip: int = 1,
    action_skip: int = 1,
    traj_subset: bool = True,
    random_seed: int = 42,
    process_actions: str = "concat",
    dset_fraction: float = 1.0,
):
    """Lance-backed counterpart to `load_wall_slice_train_val`.

    Phase 1.b supports split_mode="random" only (verified to be the production default
    in all configs/vjepa_wm/wall_sweep/*.yaml). Other split_modes raise NotImplementedError.
    """
    if not lance_uri:
        raise ValueError("load_wall_lance_slice_train_val: lance_uri must be a non-empty path")
    if split_mode != "random":
        raise NotImplementedError(
            f"load_wall_lance_slice_train_val: split_mode={split_mode!r} not supported in Phase 1.b "
            f"(only 'random' is implemented; production configs use it exclusively)"
        )

    dset = WallLanceWindowDataset(
        lance_uri=lance_uri,
        image_codec=image_codec,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        dset_fraction=dset_fraction,
    )

    train_set, val_set = split_traj_datasets(
        dset,
        train_fraction=split_ratio,
        random_seed=random_seed,
        traj_subset=traj_subset,
    )

    num_frames = num_hist + num_pred
    train_slices = LanceTrajSlicerDataset(
        train_set,
        num_frames=num_frames,
        frameskip=frameskip,
        action_skip=action_skip,
        process_actions=process_actions,
        generator=torch.Generator().manual_seed(random_seed),
    )
    val_slices = LanceTrajSlicerDataset(
        val_set,
        num_frames=num_frames_val if num_frames_val else num_frames,
        frameskip=frameskip,
        action_skip=action_skip,
        process_actions=process_actions,
        generator=torch.Generator().manual_seed(random_seed),
    )

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": train_set, "valid": val_set}
    return datasets, traj_dset
