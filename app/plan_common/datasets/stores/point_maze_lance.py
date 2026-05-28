# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Callable, Optional

import torch

from src.utils.logging import get_logger

from ..traj_dset import split_traj_datasets
from .lance_slicer import LanceTrajSlicerDataset
from .lance_window import LanceWindowDataset

log = get_logger(__name__)


def load_point_maze_lance_slice_train_val(
    transform: Optional[Callable] = None,
    lance_uri: str = "",
    image_codec: Optional[str] = None,
    n_rollout: Optional[int] = None,
    normalize_action: bool = False,
    split_ratio: float = 0.8,
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
    """Lance-backed counterpart to `load_point_maze_slice_train_val`.

    Produces the same `(datasets, traj_dset)` tuple shape so the
    `DataLoader` assembly in `utils.init_data` is unchanged. Slice generation
    and train/val split use the same seed -> identical
    `(ep_idx, start, end)` tuples as the raw flow under the same args.
    """
    if not lance_uri:
        raise ValueError("load_point_maze_lance_slice_train_val: lance_uri must be a non-empty path")

    dset = LanceWindowDataset(
        lance_uri=lance_uri,
        image_codec=image_codec,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        dset_fraction=dset_fraction,
    )

    # Identical split logic to traj_dset.get_train_val_sliced (lines 200-205).
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
