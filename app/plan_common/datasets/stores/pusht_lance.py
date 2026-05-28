# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from einops import rearrange

from ..pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD, STATE_MEAN, STATE_STD
from .lance_slicer import LanceTrajSlicerDataset
from .lance_window import LanceWindowDataset


class PushTLanceWindowDataset(LanceWindowDataset):
    """Lance-backed PushT split dataset.

    PushT uses hard-coded normalization constants in `pusht_dset.py`; this class
    imports those constants rather than recomputing data-derived statistics.
    """

    def __init__(
        self,
        lance_uri: str,
        image_codec: Optional[str] = None,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        action_scale: float = 100.0,
        with_velocity: bool = True,
        dset_fraction: float = 1.0,
    ):
        super().__init__(
            lance_uri=lance_uri,
            image_codec=image_codec,
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=False,
            action_scale=action_scale,
            dset_fraction=dset_fraction,
        )
        meta = self.store.metadata
        self.action_scale = float(meta.get("action_scale", action_scale))
        self.with_velocity = bool(with_velocity)
        if bool(meta.get("with_velocity", True)) != self.with_velocity:
            raise ValueError(
                f"PushT Lance with_velocity mismatch for {lance_uri}: "
                f"metadata={meta.get('with_velocity')} requested={with_velocity}"
            )
        self.shapes = list(meta.get("shapes", ["T"] * self._n_episodes))[: self._n_episodes]
        self.normalize_action = normalize_action

        if normalize_action:
            self.action_mean = ACTION_MEAN[: self.action_dim]
            self.action_std = ACTION_STD[: self.action_dim]
            self.state_mean = STATE_MEAN[: self.state_dim]
            self.state_std = STATE_STD[: self.state_dim]
            self.proprio_mean = PROPRIO_MEAN[: self.proprio_dim]
            self.proprio_std = PROPRIO_STD[: self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

    def __getitem__(self, idx, **kwargs):
        T = int(self.get_seq_length(idx))
        frames = list(range(T))
        out = self.read_window_normalized(ep_idx=idx, step_indices=frames, action_step_indices=frames)
        visual = out["visual"].float() / 255.0
        visual = rearrange(visual, "T H W C -> T C H W")
        if self.transform:
            visual = self.transform(visual)
        obs = {"visual": visual, "proprio": out["proprio"]}
        return obs, out["action"], out["state"], None, {"shape": self.shapes[idx]}


def load_pusht_lance_slice_train_val(
    transform: Optional[Callable] = None,
    lance_uri: str = "",
    image_codec: Optional[str] = None,
    n_rollout: Optional[int] = None,
    normalize_action: bool = True,
    split_ratio: float = 0.8,
    num_hist: int = 0,
    num_pred: int = 0,
    num_frames_val: Optional[int] = None,
    frameskip: int = 1,
    action_skip: int = 1,
    with_velocity: bool = True,
    random_seed: int = 42,
    process_actions: str = "concat",
    dset_fraction: float = 1.0,
):
    del split_ratio  # raw PushT uses explicit train/val folders, not a random split.
    root = Path(lance_uri)
    train_dset = PushTLanceWindowDataset(
        lance_uri=str(root / "train.lance"),
        image_codec=image_codec,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        with_velocity=with_velocity,
        dset_fraction=dset_fraction,
    )
    val_dset = PushTLanceWindowDataset(
        lance_uri=str(root / "val.lance"),
        image_codec=image_codec,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        with_velocity=with_velocity,
        dset_fraction=dset_fraction,
    )
    if train_dset.store.codec != val_dset.store.codec:
        raise ValueError(
            f"PushT Lance codec mismatch: train={train_dset.store.codec} at {root / 'train.lance'} "
            f"val={val_dset.store.codec} at {root / 'val.lance'}"
        )

    num_frames = num_hist + num_pred
    train_slices = LanceTrajSlicerDataset(
        train_dset,
        num_frames,
        frameskip,
        action_skip,
        generator=torch.Generator().manual_seed(random_seed),
        process_actions=process_actions,
    )
    val_slices = LanceTrajSlicerDataset(
        val_dset,
        num_frames_val if num_frames_val else num_frames,
        frameskip,
        action_skip,
        generator=torch.Generator().manual_seed(random_seed),
        process_actions=process_actions,
    )
    return {"train": train_slices, "valid": val_slices}, {"train": train_dset, "valid": val_dset}
