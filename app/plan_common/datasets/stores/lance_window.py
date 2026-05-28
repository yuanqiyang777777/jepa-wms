# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from einops import rearrange

from src.utils.logging import get_logger

from ..traj_dset import TrajDataset
from .lance_store import LanceStore

log = get_logger(__name__)


class LanceWindowDataset(TrajDataset):
    """TrajDataset backed by a Lance store.

    Behaviour notes:
      * Normalisation statistics (action/state/proprio mean & std) are
        computed in `__init__` via columnar Lance scans -- NO image reads.
        These match the values that `PointMazeDataset.__init__` computes
        from the raw `.pth` tensors (same per-column algorithm).
      * `__getitem__` returns a FULL episode in the same 5-tuple as
        `PointMazeDataset`. It is kept for API parity / debugging but the
        production path (`LanceTrajSlicerDataset`) bypasses it and calls
        `store.read_window` directly to avoid full-episode reads.
      * All shape/normalisation attributes (`action_dim`, `proprio_dim`,
        `state_dim`, `action_mean`, `action_std`, `state_mean`, `state_std`,
        `proprio_mean`, `proprio_std`) are exposed as plain attributes so
        `Preprocessor` construction in `train.py` works unchanged.
    """

    def __init__(
        self,
        lance_uri: str,
        image_codec: Optional[str] = None,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
        dset_fraction: float = 1.0,
    ):
        self.lance_uri = str(lance_uri)
        self.transform = transform
        self.normalize_action = normalize_action
        self.action_scale = action_scale

        # `image_codec=None` means "use whatever swm_metadata.json says"; LanceStore raises on mismatch.
        self.store = LanceStore(self.lance_uri, image_codec=image_codec)
        seq_lengths = self.store.seq_lengths
        n = len(seq_lengths)
        if n_rollout:
            n = min(n_rollout, n)
        if dset_fraction < 1.0:
            n = max(1, int(n * dset_fraction))
            log.info(
                f"Slicing Lance PointMaze dataset from {len(seq_lengths)} to {n} samples "
                f"({dset_fraction*100:.1f}%)"
            )
        self._n_episodes = n
        self.seq_lengths = torch.tensor(seq_lengths[:n], dtype=torch.long)
        log.info(f"Opened Lance PointMaze dataset ({n} episodes) at {self.lance_uri}")

        # Columnar scan: read action/state/proprio once across all rows, no images.
        # Mirrors point_maze_dset.py:__init__ exactly: load -> divide by action_scale -> compute mean/std -> store.
        all_actions = self.store.read_full_column("action")  # (N_total_rows, D_a)
        all_states = self.store.read_full_column("state")  # (N_total_rows, D_s)
        all_proprios = self.store.read_full_column("proprio")  # (N_total_rows, D_p)

        # If dset_fraction or n_rollout reduced n, we need to drop rows beyond the truncated episodes.
        if n < len(seq_lengths):
            keep_rows = sum(seq_lengths[:n])
            all_actions = all_actions[:keep_rows]
            all_states = all_states[:keep_rows]
            all_proprios = all_proprios[:keep_rows]

        all_actions = all_actions / action_scale  # mirrors point_maze_dset.py:34

        self.action_dim = all_actions.shape[-1]
        self.state_dim = all_states.shape[-1]
        self.proprio_dim = all_proprios.shape[-1]

        if normalize_action:
            # mean/std over ALL valid (non-padded) rows -- Lance stores no padding,
            # so a plain dim=0 reduction equals raw `get_data_mean_std`.
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

    # -------- TrajDataset API --------

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def __len__(self):
        return self._n_episodes

    def get_all_actions(self):
        """Return concatenated normalized actions across episodes (for downstream stats)."""
        all_actions = self.store.read_full_column("action") / self.action_scale
        # Truncate to active episodes' rows.
        keep_rows = int(self.seq_lengths.sum().item())
        all_actions = all_actions[:keep_rows]
        all_actions = (all_actions - self.action_mean) / self.action_std
        return all_actions

    # -------- Per-window primitive (used by the slicer) --------

    def read_window_normalized(
        self,
        ep_idx: int,
        step_indices,
        action_step_indices=None,
    ) -> dict:
        """Like `LanceStore.read_window`, but:
          * applies `action_scale` division (matching raw flow),
          * applies action/proprio normalization,
          * returns torch tensors with the JEPA-WMs schema:
                {'visual': uint8 (T,H,W,C), 'proprio': (T,D_p), 'state': (T,D_s), 'action': (T_a,D_a)}
        """
        raw = self.store.read_window(ep_idx, step_indices, action_step_indices)
        action = raw["action"] / self.action_scale
        action = (action - self.action_mean) / self.action_std
        proprio = (raw["proprio"] - self.proprio_mean) / self.proprio_std
        state = raw["state"]  # raw flow does NOT normalize `state`
        visual = raw["visual"]
        return {"visual": visual, "proprio": proprio, "state": state, "action": action}

    # -------- Full-episode parity (NOT on the hot path; kept for raw-equivalence debugging) --------

    def __getitem__(self, idx, **kwargs):
        """Full-episode read -- same 5-tuple shape as PointMazeDataset.

        Production training calls go through `LanceTrajSlicerDataset` which uses
        `read_window_normalized` and never invokes this method.
        """
        T = int(self.get_seq_length(idx))
        frames = list(range(T))
        out = self.read_window_normalized(ep_idx=idx, step_indices=frames, action_step_indices=frames)
        visual = out["visual"].float() / 255.0
        visual = rearrange(visual, "T H W C -> T C H W")
        if self.transform:
            visual = self.transform(visual)
        obs = {"visual": visual, "proprio": out["proprio"]}
        return obs, out["action"], out["state"], None, {}  # env_info empty
