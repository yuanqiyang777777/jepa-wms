# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.
#
# Bit-exact equivalence test for raw PointMaze vs Lance (PNG codec) backend.
#
# Acceptance:
#   * train/val slice lists identical (same seed -> same randperm)
#   * action / proprio / state / visual tensors equal at every (idx, sample)
#   * one multi-worker DataLoader iteration succeeds (catches Lance handle
#     pickling issues under spawn-start workers).

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

# Allow the test to skip cleanly if lance/pyarrow aren't installed locally
# (e.g. on dev machines without the .[data-lance] extras).
lance = pytest.importorskip("lance")
pa = pytest.importorskip("pyarrow")

from app.plan_common.datasets.point_maze_dset import load_point_maze_slice_train_val
from app.plan_common.datasets.stores.point_maze_lance import (
    load_point_maze_lance_slice_train_val,
)
from app.plan_common.datasets.stores.writer_point_maze import (
    convert_point_maze_to_lance,
)


# Match the typical PointMaze training config used in
# configs/vjepa_wm/mz_sweep/*.yaml: num_hist=3, num_pred=1, frameskip=5, action_skip=1.
NUM_HIST = 3
NUM_PRED = 1
FSKIP = 5
ASKIP = 1
SEED = 234
SPLIT_RATIO = 0.9


def _build_pair(tiny_point_maze_dir: str, tmp_path: Path, codec: str = "png"):
    """Convert the tiny fixture and build matched raw + lance pipelines."""
    lance_uri = tmp_path / "point_maze.lance"
    convert_point_maze_to_lance(
        src_dir=tiny_point_maze_dir,
        dst_uri=lance_uri,
        codec=codec,
        mode="overwrite",
        action_scale=1.0,
    )

    raw_dsets, _ = load_point_maze_slice_train_val(
        transform=None,
        n_rollout=None,
        data_path=tiny_point_maze_dir,
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
    lance_dsets, _ = load_point_maze_lance_slice_train_val(
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
    return raw_dsets, lance_dsets


def test_slice_lists_bit_identical(tiny_point_maze_dir, tmp_path):
    """Keystone: same seed -> identical (ep_idx, start, end) triples for both backends."""
    raw_dsets, lance_dsets = _build_pair(tiny_point_maze_dir, tmp_path)
    assert raw_dsets["train"].slices == lance_dsets["train"].slices
    assert raw_dsets["valid"].slices == lance_dsets["valid"].slices
    # Sanity: non-empty (the tiny fixture must produce at least a handful of slices).
    assert len(raw_dsets["train"]) > 0
    assert len(raw_dsets["valid"]) >= 0  # val split may have 0 slices for very small fixtures
    assert len(lance_dsets["train"]) == len(raw_dsets["train"])
    assert len(lance_dsets["valid"]) == len(raw_dsets["valid"])


def test_per_sample_bit_exact_png(tiny_point_maze_dir, tmp_path):
    """For every (split, idx): action/proprio/state/visual tensors bit-equal across backends.

    PNG codec is lossless on uint8 images, so atol=rtol=0 is the right tolerance.
    """
    raw_dsets, lance_dsets = _build_pair(tiny_point_maze_dir, tmp_path, codec="png")

    for split in ("train", "valid"):
        raw_s, lance_s = raw_dsets[split], lance_dsets[split]
        assert len(raw_s) == len(lance_s), f"{split}: len mismatch"
        for i in range(len(raw_s)):
            r = raw_s[i]
            l = lance_s[i]
            assert len(r) == len(l), f"{split}#{i}: tuple length"
            obs_r, act_r, state_r, rew_r = r
            obs_l, act_l, state_l, rew_l = l

            # Action / state / proprio: exact match across the whole window.
            torch.testing.assert_close(act_r, act_l, rtol=0, atol=0)
            torch.testing.assert_close(state_r, state_l, rtol=0, atol=0)
            torch.testing.assert_close(obs_r["proprio"], obs_l["proprio"], rtol=0, atol=0)

            # Visual: PNG codec -> byte-identical decode -> bit-exact float tensor after /255.0.
            torch.testing.assert_close(obs_r["visual"], obs_l["visual"], rtol=0, atol=0)

            # Reward: raw returns zeros (PointMaze has no reward).
            torch.testing.assert_close(rew_r, rew_l, rtol=0, atol=0)


def test_lance_dataloader_multiworker_smoke(tiny_point_maze_dir, tmp_path):
    """A short DataLoader pass with num_workers=2 exercises the per-PID Lance handle cache."""
    _, lance_dsets = _build_pair(tiny_point_maze_dir, tmp_path, codec="png")
    train = lance_dsets["train"]
    if len(train) == 0:
        pytest.skip("Tiny fixture produced empty train split -- adjust seq_lengths upward")

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
        # batch is a tuple of (obs_dict, act, state, reward) collated.
        assert isinstance(batch, (tuple, list))
        n_seen += 1
        if n_seen >= 2:
            break
    assert n_seen >= 1


def test_lance_normalisation_stats_match(tiny_point_maze_dir, tmp_path):
    """The per-feature mean/std stored on the dataset object must match raw's values exactly."""
    raw_dsets, lance_dsets = _build_pair(tiny_point_maze_dir, tmp_path)
    # Both train splits wrap a TrajSubset whose .dataset is the underlying *Dataset.
    raw_inner = raw_dsets["train"].dataset.dataset  # TrajSlicerDataset -> TrajSubset -> PointMazeDataset
    lance_inner = lance_dsets["train"].dataset.dataset  # LanceTrajSlicerDataset -> TrajSubset -> LanceWindowDataset

    for attr in ("action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std"):
        rv = getattr(raw_inner, attr)
        lv = getattr(lance_inner, attr)
        torch.testing.assert_close(rv, lv, rtol=0, atol=0), attr
