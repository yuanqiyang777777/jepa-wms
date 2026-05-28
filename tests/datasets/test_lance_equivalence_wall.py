# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.
#
# Bit-exact equivalence test for raw Wall vs Lance (PNG codec) backend.
#
# Acceptance:
#   * train/val slice lists identical (same seed -> same randperm)
#   * action / proprio / state / visual tensors equal at every (idx, sample)
#   * reward zeros parity (Wall has no reward)
#   * one multi-worker DataLoader iteration succeeds
#   * normalization stats match between raw and Lance (action_mean/std, proprio_mean/std, state_mean/std)
#   * env_info["fix_door_location"] / env_info["fix_wall_location"] bit-equal on the full-episode path

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

lance = pytest.importorskip("lance")
pa = pytest.importorskip("pyarrow")

from app.plan_common.datasets.wall_dset import load_wall_slice_train_val, WallDataset
from app.plan_common.datasets.stores.wall_lance import (
    WallLanceWindowDataset,
    load_wall_lance_slice_train_val,
)
from app.plan_common.datasets.stores.writer_wall import convert_wall_to_lance


# Mirrors a representative Wall sweep config: num_hist=3, num_pred=1, frameskip=5, action_skip=1.
NUM_HIST = 3
NUM_PRED = 1
FSKIP = 5
ASKIP = 1
SEED = 234
SPLIT_RATIO = 0.9


def _build_pair(tiny_wall_dir: str, tmp_path: Path, codec: str = "png"):
    lance_uri = tmp_path / "wall.lance"
    convert_wall_to_lance(
        src_dir=tiny_wall_dir,
        dst_uri=lance_uri,
        codec=codec,
        mode="overwrite",
        action_scale=1.0,
    )

    raw_dsets, raw_traj = load_wall_slice_train_val(
        transform=None,
        n_rollout=None,
        data_path=tiny_wall_dir,
        normalize_action=True,
        split_ratio=SPLIT_RATIO,
        split_mode="random",
        num_hist=NUM_HIST,
        num_pred=NUM_PRED,
        num_frames_val=None,
        frameskip=FSKIP,
        action_skip=ASKIP,
        traj_subset=True,
        random_seed=SEED,
        process_actions="concat",
        dset_fraction=1.0,
    )
    lance_dsets, lance_traj = load_wall_lance_slice_train_val(
        transform=None,
        lance_uri=str(lance_uri),
        image_codec=codec,
        n_rollout=None,
        normalize_action=True,
        split_ratio=SPLIT_RATIO,
        num_hist=NUM_HIST,
        num_pred=NUM_PRED,
        num_frames_val=None,
        frameskip=FSKIP,
        action_skip=ASKIP,
        traj_subset=True,
        random_seed=SEED,
        process_actions="concat",
        dset_fraction=1.0,
    )
    return raw_dsets, lance_dsets, raw_traj, lance_traj


def test_slice_lists_bit_identical(tiny_wall_dir, tmp_path):
    raw_dsets, lance_dsets, _, _ = _build_pair(tiny_wall_dir, tmp_path)
    assert raw_dsets["train"].slices == lance_dsets["train"].slices
    assert raw_dsets["valid"].slices == lance_dsets["valid"].slices
    assert len(lance_dsets["train"]) == len(raw_dsets["train"])
    assert len(lance_dsets["valid"]) == len(raw_dsets["valid"])


def test_per_sample_bit_exact_png(tiny_wall_dir, tmp_path):
    raw_dsets, lance_dsets, _, _ = _build_pair(tiny_wall_dir, tmp_path, codec="png")

    for split in ("train", "valid"):
        raw_s = raw_dsets[split]
        lance_s = lance_dsets[split]
        assert len(raw_s) == len(lance_s), f"{split}: len mismatch"
        for i in range(len(raw_s)):
            obs_r, act_r, state_r, rew_r = raw_s[i]
            obs_l, act_l, state_l, rew_l = lance_s[i]

            torch.testing.assert_close(act_r, act_l, rtol=0, atol=0)
            torch.testing.assert_close(state_r, state_l, rtol=0, atol=0)
            torch.testing.assert_close(obs_r["proprio"], obs_l["proprio"], rtol=0, atol=0)
            torch.testing.assert_close(obs_r["visual"], obs_l["visual"], rtol=0, atol=0)
            # Wall has no reward; both should be zero-filled.
            torch.testing.assert_close(rew_r, rew_l, rtol=0, atol=0)


def test_lance_dataloader_multiworker_smoke(tiny_wall_dir, tmp_path):
    _, lance_dsets, _, _ = _build_pair(tiny_wall_dir, tmp_path)
    train = lance_dsets["train"]
    if len(train) == 0:
        pytest.skip("Tiny Wall fixture produced empty train split")

    loader = DataLoader(
        train,
        batch_size=2,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        shuffle=False,
    )
    n_seen = 0
    for batch in loader:
        assert isinstance(batch, (tuple, list))
        n_seen += 1
        if n_seen >= 2:
            break
    assert n_seen >= 1


def test_lance_normalisation_stats_match(tiny_wall_dir, tmp_path):
    raw_dsets, lance_dsets, _, _ = _build_pair(tiny_wall_dir, tmp_path)
    # TrajSlicerDataset -> TrajSubset -> WallDataset / WallLanceWindowDataset
    raw_inner = raw_dsets["train"].dataset.dataset
    lance_inner = lance_dsets["train"].dataset.dataset

    for attr in ("action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std"):
        rv = getattr(raw_inner, attr)
        lv = getattr(lance_inner, attr)
        torch.testing.assert_close(rv, lv, rtol=0, atol=0)


def test_env_info_door_wall_locations_match(tiny_wall_dir, tmp_path):
    """Full-episode __getitem__ on raw vs Lance should agree on env_info[fix_door/wall_location].

    This is the only Wall-specific assertion beyond the PointMaze pattern.
    """
    lance_uri = tmp_path / "wall.lance"
    convert_wall_to_lance(
        src_dir=tiny_wall_dir,
        dst_uri=lance_uri,
        codec="png",
        mode="overwrite",
        action_scale=1.0,
    )

    raw_full = WallDataset(
        data_path=tiny_wall_dir,
        transform=None,
        normalize_action=True,
        action_scale=1.0,
    )
    lance_full = WallLanceWindowDataset(
        lance_uri=str(lance_uri),
        image_codec="png",
        transform=None,
        normalize_action=True,
    )
    assert len(raw_full) == len(lance_full)

    for ep_idx in range(len(raw_full)):
        _, _, _, _, env_info_r = raw_full[ep_idx]
        _, _, _, _, env_info_l = lance_full[ep_idx]
        torch.testing.assert_close(
            env_info_r["fix_door_location"], env_info_l["fix_door_location"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            env_info_r["fix_wall_location"], env_info_l["fix_wall_location"], rtol=0, atol=0
        )
