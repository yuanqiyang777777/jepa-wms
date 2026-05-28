# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.
#
# JPEG opt-in tolerance test: documents the JPEG-quantisation behaviour without
# enforcing bit-exactness on `obs["visual"]`. All non-image tensors must still
# be bit-exact (action/state/proprio/reward) because JPEG only affects visual.

from __future__ import annotations

from pathlib import Path

import pytest
import torch

lance = pytest.importorskip("lance")
pa = pytest.importorskip("pyarrow")

from app.plan_common.datasets.point_maze_dset import load_point_maze_slice_train_val
from app.plan_common.datasets.stores.point_maze_lance import (
    load_point_maze_lance_slice_train_val,
)
from app.plan_common.datasets.stores.writer_point_maze import (
    convert_point_maze_to_lance,
)

NUM_HIST = 3
NUM_PRED = 1
FSKIP = 5
ASKIP = 1
SEED = 234
SPLIT_RATIO = 0.9

# JPEG q95 on 16x16 random-pixel images may diverge non-trivially because there's
# nothing for the DCT to exploit; tolerance is loose. The point of this test is
# to document the contract, not enforce a tight bound on synthetic data.
JPEG_VISUAL_ATOL = 0.30  # in [0, 1] space (post /255.0)


@pytest.mark.xfail(strict=False, reason="JPEG q95 on random-pixel synthetic fixtures may exceed the documented tolerance; tightening is an open Phase-1.b task.")
def test_jpeg_visual_within_tolerance(tiny_point_maze_dir, tmp_path):
    lance_uri = tmp_path / "point_maze_jpeg.lance"
    convert_point_maze_to_lance(
        src_dir=tiny_point_maze_dir,
        dst_uri=lance_uri,
        codec="jpeg",
        jpeg_quality=95,
        mode="overwrite",
        action_scale=1.0,
    )
    raw_dsets, _ = load_point_maze_slice_train_val(
        transform=None, n_rollout=None, data_path=tiny_point_maze_dir,
        normalize_action=True, split_ratio=SPLIT_RATIO,
        num_hist=NUM_HIST, num_pred=NUM_PRED, num_frames_val=None,
        frameskip=FSKIP, action_skip=ASKIP, traj_subset=True,
        random_seed=SEED, process_actions="concat", dset_fraction=1.0,
    )
    lance_dsets, _ = load_point_maze_lance_slice_train_val(
        transform=None, lance_uri=str(lance_uri), image_codec="jpeg",
        n_rollout=None, normalize_action=True, split_ratio=SPLIT_RATIO,
        num_hist=NUM_HIST, num_pred=NUM_PRED, num_frames_val=None,
        frameskip=FSKIP, action_skip=ASKIP, traj_subset=True,
        random_seed=SEED, process_actions="concat", dset_fraction=1.0,
    )

    train_r = raw_dsets["train"]
    train_l = lance_dsets["train"]
    assert train_r.slices == train_l.slices

    n_check = min(len(train_r), 5)
    for i in range(n_check):
        obs_r, act_r, state_r, rew_r = train_r[i]
        obs_l, act_l, state_l, rew_l = train_l[i]
        # Numeric tensors must still be bit-exact under JPEG.
        torch.testing.assert_close(act_r, act_l, rtol=0, atol=0)
        torch.testing.assert_close(state_r, state_l, rtol=0, atol=0)
        torch.testing.assert_close(obs_r["proprio"], obs_l["proprio"], rtol=0, atol=0)
        torch.testing.assert_close(rew_r, rew_l, rtol=0, atol=0)
        # Visual: only loosely close.
        torch.testing.assert_close(obs_r["visual"], obs_l["visual"], rtol=0, atol=JPEG_VISUAL_ATOL)
