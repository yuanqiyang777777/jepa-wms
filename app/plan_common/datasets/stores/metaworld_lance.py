# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Callable, Optional

import torch
from einops import rearrange

from ..traj_dset import split_traj_datasets
from .lance_slicer import LanceTrajSlicerDataset
from .lance_window import LanceWindowDataset


class MetaworldLanceWindowDataset(LanceWindowDataset):
    """Lance-backed Metaworld HF dataset with metadata-level task filtering."""

    def __init__(
        self,
        lance_uri: str,
        image_codec: Optional[str] = None,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
        filter_tasks: Optional[list[str]] = None,
        with_reward: bool = True,
        dset_fraction: float = 1.0,
    ):
        super().__init__(
            lance_uri=lance_uri,
            image_codec=image_codec,
            n_rollout=None,
            transform=transform,
            normalize_action=False,
            action_scale=action_scale,
            dset_fraction=1.0,
        )
        meta = self.store.metadata
        all_tasks = list(meta.get("tasks_per_episode", []))
        if len(all_tasks) != len(self.store.seq_lengths):
            raise ValueError(
                f"Metaworld Lance metadata at {lance_uri} has {len(all_tasks)} tasks "
                f"for {len(self.store.seq_lengths)} episodes"
            )

        allowed = set(filter_tasks) if filter_tasks is not None else None
        filtered_global = [i for i, task in enumerate(all_tasks) if allowed is None or task in allowed]
        if n_rollout is not None:
            filtered_global = filtered_global[: min(n_rollout, len(filtered_global))]
        if dset_fraction < 1.0:
            filtered_global = filtered_global[: max(1, int(len(filtered_global) * dset_fraction))]
        if not filtered_global:
            raise ValueError(
                f"Metaworld Lance dataset at {lance_uri} has no episodes after filter_tasks={filter_tasks!r}"
            )

        self._filtered_to_global_ep = filtered_global
        self._tasks = [all_tasks[i] for i in filtered_global]
        self._n_episodes = len(filtered_global)
        self.seq_lengths = torch.tensor([self.store.seq_lengths[i] for i in filtered_global], dtype=torch.long)
        self.with_reward = with_reward
        self.action_scale = float(meta.get("action_scale", action_scale))

        all_actions = self.store.read_full_column("action") / self.action_scale
        all_states = self.store.read_full_column("state")
        all_proprios = self.store.read_full_column("proprio")
        rows = []
        cum = self.store._cum
        for global_ep in filtered_global:
            rows.extend(range(cum[global_ep], cum[global_ep + 1]))
        if rows:
            row_idx = torch.tensor(rows, dtype=torch.long)
            all_actions = all_actions[row_idx]
            all_states = all_states[row_idx]
            all_proprios = all_proprios[row_idx]

        self.action_dim = all_actions.shape[-1]
        self.state_dim = all_states.shape[-1]
        self.proprio_dim = all_proprios.shape[-1]
        self.normalize_action = normalize_action
        if normalize_action:
            self.action_mean = torch.mean(all_actions, dim=0)
            self.action_std = torch.std(all_actions, dim=0)
            self.state_mean = torch.mean(all_states, dim=0)
            self.state_std = torch.std(all_states, dim=0)
            self.proprio_mean = torch.mean(all_proprios, dim=0)
            self.proprio_std = torch.std(all_proprios, dim=0)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

    def _global_ep(self, ep_idx: int) -> int:
        return int(self._filtered_to_global_ep[ep_idx])

    def read_window_normalized(self, ep_idx: int, step_indices, action_step_indices=None) -> dict:
        raw = self.store.read_window(self._global_ep(ep_idx), step_indices, action_step_indices)
        action = raw["action"] / self.action_scale
        action = (action - self.action_mean) / self.action_std
        proprio = (raw["proprio"] - self.proprio_mean) / self.proprio_std
        out = {
            "visual": raw["visual"],
            "proprio": proprio,
            "state": raw["state"],
            "action": action,
        }
        if self.with_reward and "reward" in raw:
            out["reward"] = raw["reward"]
        return out

    def __getitem__(self, idx, **kwargs):
        T = int(self.get_seq_length(idx))
        frames = list(range(T))
        out = self.read_window_normalized(ep_idx=idx, step_indices=frames, action_step_indices=frames)
        visual = out["visual"].float() / 255.0
        visual = rearrange(visual, "T H W C -> T C H W")
        if self.transform:
            visual = self.transform(visual)
        obs = {"visual": visual, "proprio": out["proprio"]}
        reward = out.get("reward") if self.with_reward else None
        return obs, out["action"], out["state"], reward, {"task": self._tasks[idx]}


def load_metaworld_hf_lance_slice_train_val(
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
    filter_tasks: Optional[list[str]] = None,
    random_seed: int = 42,
    with_reward: bool = False,
    process_actions: str = "concat",
    dset_fraction: float = 1.0,
):
    dset = MetaworldLanceWindowDataset(
        lance_uri=lance_uri,
        image_codec=image_codec,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        filter_tasks=filter_tasks,
        with_reward=with_reward,
        dset_fraction=dset_fraction,
    )
    dset_train, dset_val = split_traj_datasets(
        dset,
        train_fraction=split_ratio,
        random_seed=random_seed,
        traj_subset=traj_subset,
    )
    num_frames = num_hist + num_pred
    train_slices = LanceTrajSlicerDataset(
        dset_train,
        num_frames,
        frameskip,
        action_skip,
        generator=torch.Generator().manual_seed(random_seed),
        process_actions=process_actions,
    )
    val_slices = LanceTrajSlicerDataset(
        dset_val,
        num_frames_val if num_frames_val else num_frames,
        frameskip,
        action_skip,
        generator=torch.Generator().manual_seed(random_seed),
        process_actions=process_actions,
    )
    return {"train": train_slices, "valid": val_slices}, {"train": dset_train, "valid": dset_val}
