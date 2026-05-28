# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

pytest.importorskip("lance")
pytest.importorskip("pyarrow")
pytest.importorskip("decord")

from app.plan_common.datasets.pusht_dset import (
    ACTION_MEAN,
    ACTION_STD,
    PROPRIO_MEAN,
    PROPRIO_STD,
    STATE_MEAN,
    STATE_STD,
    PushTDataset,
    load_pusht_slice_train_val,
)
from app.plan_common.datasets.stores.pusht_lance import (
    PushTLanceWindowDataset,
    load_pusht_lance_slice_train_val,
)
from app.plan_common.datasets.stores.writer_pusht import convert_pusht_to_lance


NUM_HIST = 3
NUM_PRED = 1
FSKIP = 5
ASKIP = 1
SEED = 234
SPLIT_RATIO = 0.9


def _build_pair(tiny_pusht_dir: str, tmp_path: Path, codec: str = "png"):
    lance_uri = tmp_path / "pusht.lance"
    convert_pusht_to_lance(
        src_dir=tiny_pusht_dir,
        dst_uri=lance_uri,
        codec=codec,
        mode="overwrite",
    )

    raw_dsets, raw_traj = load_pusht_slice_train_val(
        transform=None,
        n_rollout=None,
        data_path=tiny_pusht_dir,
        normalize_action=True,
        split_ratio=SPLIT_RATIO,
        num_hist=NUM_HIST,
        num_pred=NUM_PRED,
        num_frames_val=None,
        frameskip=FSKIP,
        action_skip=ASKIP,
        with_velocity=True,
        random_seed=SEED,
        process_actions="concat",
        dset_fraction=1.0,
    )
    lance_dsets, lance_traj = load_pusht_lance_slice_train_val(
        transform=None,
        lance_uri=str(lance_uri),
        image_codec=codec,
        n_rollout=None,
        normalize_action=True,
        num_hist=NUM_HIST,
        num_pred=NUM_PRED,
        num_frames_val=None,
        frameskip=FSKIP,
        action_skip=ASKIP,
        with_velocity=True,
        random_seed=SEED,
        process_actions="concat",
        dset_fraction=1.0,
    )
    return raw_dsets, lance_dsets, raw_traj, lance_traj


def test_slice_lists_bit_identical(tiny_pusht_dir, tmp_path):
    raw_dsets, lance_dsets, _, _ = _build_pair(tiny_pusht_dir, tmp_path)
    assert raw_dsets["train"].slices == lance_dsets["train"].slices
    assert raw_dsets["valid"].slices == lance_dsets["valid"].slices


def test_per_sample_bit_exact_png(tiny_pusht_dir, tmp_path):
    raw_dsets, lance_dsets, _, _ = _build_pair(tiny_pusht_dir, tmp_path, codec="png")
    for split in ("train", "valid"):
        raw_s = raw_dsets[split]
        lance_s = lance_dsets[split]
        assert len(raw_s) == len(lance_s)
        for i in range(len(raw_s)):
            obs_r, act_r, state_r, rew_r = raw_s[i]
            obs_l, act_l, state_l, rew_l = lance_s[i]
            torch.testing.assert_close(act_r, act_l, rtol=0, atol=0)
            torch.testing.assert_close(state_r, state_l, rtol=0, atol=0)
            torch.testing.assert_close(obs_r["proprio"], obs_l["proprio"], rtol=0, atol=0)
            torch.testing.assert_close(obs_r["visual"], obs_l["visual"], rtol=0, atol=0)
            torch.testing.assert_close(rew_r, rew_l, rtol=0, atol=0)


def test_lance_dataloader_multiworker_smoke(tiny_pusht_dir, tmp_path):
    _, lance_dsets, _, _ = _build_pair(tiny_pusht_dir, tmp_path)
    loader = DataLoader(
        lance_dsets["train"],
        batch_size=2,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        shuffle=False,
    )
    assert next(iter(loader)) is not None


def test_lance_normalisation_stats_match(tiny_pusht_dir, tmp_path):
    _, lance_dsets, _, _ = _build_pair(tiny_pusht_dir, tmp_path)
    lance_inner = lance_dsets["train"].dataset
    torch.testing.assert_close(lance_inner.action_mean, ACTION_MEAN, rtol=0, atol=0)
    torch.testing.assert_close(lance_inner.action_std, ACTION_STD, rtol=0, atol=0)
    torch.testing.assert_close(lance_inner.state_mean, STATE_MEAN[: lance_inner.state_dim], rtol=0, atol=0)
    torch.testing.assert_close(lance_inner.state_std, STATE_STD[: lance_inner.state_dim], rtol=0, atol=0)
    torch.testing.assert_close(lance_inner.proprio_mean, PROPRIO_MEAN[: lance_inner.proprio_dim], rtol=0, atol=0)
    torch.testing.assert_close(lance_inner.proprio_std, PROPRIO_STD[: lance_inner.proprio_dim], rtol=0, atol=0)
    assert lance_inner.proprio_dim == 4


def test_env_info_shape_matches(tiny_pusht_dir, tmp_path):
    lance_uri = tmp_path / "pusht.lance"
    convert_pusht_to_lance(tiny_pusht_dir, lance_uri, codec="png", mode="overwrite")
    raw_full = PushTDataset(data_path=str(Path(tiny_pusht_dir) / "train"), normalize_action=True, with_velocity=True)
    lance_full = PushTLanceWindowDataset(
        lance_uri=str(lance_uri / "train.lance"),
        image_codec="png",
        normalize_action=True,
        with_velocity=True,
    )
    for ep_idx in range(len(raw_full)):
        _, _, _, _, env_info_r = raw_full[ep_idx]
        _, _, _, _, env_info_l = lance_full[ep_idx]
        assert env_info_r["shape"] == env_info_l["shape"]


def test_cross_split_codec_mismatch_raises(tiny_pusht_dir, tmp_path):
    lance_uri = tmp_path / "pusht.lance"
    convert_pusht_to_lance(tiny_pusht_dir, lance_uri, codec="png", mode="overwrite")
    # Corrupt only val metadata to simulate a partial split reconvert.
    import json

    meta_path = lance_uri / "val.lance" / "swm_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["image_codec"] = "jpeg"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="codec mismatch"):
        load_pusht_lance_slice_train_val(lance_uri=str(lance_uri), num_hist=NUM_HIST, num_pred=NUM_PRED)
