# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader

pytest.importorskip("lance")
pytest.importorskip("pyarrow")

# Pytest adds tests/ to sys.path, where tests/datasets shadows HuggingFace
# `datasets`. Remove that entry before importing the production MW dataset.
tests_root = str(Path(__file__).resolve().parents[1])
sys.path = [p for p in sys.path if str(Path(p).resolve()) != tests_root]
shadow = sys.modules.get("datasets")
if shadow is not None and "tests" in str(getattr(shadow, "__file__", "")):
    sys.modules.pop("datasets", None)
pytest.importorskip("datasets")

from app.plan_common.datasets.metaworld_hf_dset import (
    MetaworldHFDataset,
    load_metaworld_hf_slice_train_val,
)
from app.plan_common.datasets.stores.metaworld_lance import (
    MetaworldLanceWindowDataset,
    load_metaworld_hf_lance_slice_train_val,
)
from app.plan_common.datasets.stores.writer_metaworld import convert_metaworld_to_lance


NUM_HIST = 3
NUM_PRED = 1
FSKIP = 5
ASKIP = 1
SEED = 234
SPLIT_RATIO = 0.9


def _build_pair(tiny_metaworld_dir: str, tmp_path: Path, codec: str = "png", filter_tasks=None, with_reward=True):
    lance_uri = tmp_path / "metaworld.lance"
    convert_metaworld_to_lance(
        src_dir=tiny_metaworld_dir,
        dst_uri=lance_uri,
        codec=codec,
        mode="overwrite",
    )
    raw_dsets, raw_traj = load_metaworld_hf_slice_train_val(
        transform=None,
        n_rollout=None,
        data_path=tiny_metaworld_dir,
        normalize_action=True,
        split_ratio=SPLIT_RATIO,
        num_hist=NUM_HIST,
        num_pred=NUM_PRED,
        num_frames_val=None,
        frameskip=FSKIP,
        action_skip=ASKIP,
        traj_subset=True,
        filter_tasks=filter_tasks,
        with_reward=with_reward,
        random_seed=SEED,
        process_actions="concat",
        dset_fraction=1.0,
    )
    lance_dsets, lance_traj = load_metaworld_hf_lance_slice_train_val(
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
        filter_tasks=filter_tasks,
        with_reward=with_reward,
        random_seed=SEED,
        process_actions="concat",
        dset_fraction=1.0,
    )
    return raw_dsets, lance_dsets, raw_traj, lance_traj


def test_slice_lists_bit_identical(tiny_metaworld_dir, tmp_path):
    raw_dsets, lance_dsets, _, _ = _build_pair(tiny_metaworld_dir, tmp_path)
    assert raw_dsets["train"].slices == lance_dsets["train"].slices
    assert raw_dsets["valid"].slices == lance_dsets["valid"].slices


def test_per_sample_bit_exact_png_with_reward(tiny_metaworld_dir, tmp_path):
    raw_dsets, lance_dsets, _, _ = _build_pair(tiny_metaworld_dir, tmp_path, with_reward=True)
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


def test_without_reward_uses_zero_fill(tiny_metaworld_dir, tmp_path):
    raw_dsets, lance_dsets, _, _ = _build_pair(tiny_metaworld_dir, tmp_path, with_reward=False)
    obs_r, act_r, state_r, rew_r = raw_dsets["train"][0]
    obs_l, act_l, state_l, rew_l = lance_dsets["train"][0]
    torch.testing.assert_close(act_r, act_l, rtol=0, atol=0)
    torch.testing.assert_close(state_r, state_l, rtol=0, atol=0)
    torch.testing.assert_close(obs_r["proprio"], obs_l["proprio"], rtol=0, atol=0)
    torch.testing.assert_close(obs_r["visual"], obs_l["visual"], rtol=0, atol=0)
    torch.testing.assert_close(rew_r, rew_l, rtol=0, atol=0)


def test_filter_tasks_keeps_raw_slice_alignment(tiny_metaworld_dir, tmp_path):
    raw_dsets, lance_dsets, raw_traj, lance_traj = _build_pair(
        tiny_metaworld_dir,
        tmp_path,
        filter_tasks=["reach"],
        with_reward=True,
    )
    assert len(raw_traj["train"]) == len(lance_traj["train"])
    assert raw_dsets["train"].slices == lance_dsets["train"].slices
    assert raw_dsets["valid"].slices == lance_dsets["valid"].slices


def test_lance_dataloader_multiworker_smoke(tiny_metaworld_dir, tmp_path):
    _, lance_dsets, _, _ = _build_pair(tiny_metaworld_dir, tmp_path)
    loader = DataLoader(
        lance_dsets["train"],
        batch_size=2,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        shuffle=False,
    )
    assert next(iter(loader)) is not None


def test_lance_normalisation_stats_match(tiny_metaworld_dir, tmp_path):
    raw_dsets, lance_dsets, _, _ = _build_pair(tiny_metaworld_dir, tmp_path)
    raw_inner = raw_dsets["train"].dataset.dataset
    lance_inner = lance_dsets["train"].dataset.dataset
    for attr in ("action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std"):
        torch.testing.assert_close(getattr(raw_inner, attr), getattr(lance_inner, attr), rtol=0, atol=0)


def test_full_episode_task_env_info(tiny_metaworld_dir, tmp_path):
    lance_uri = tmp_path / "metaworld.lance"
    convert_metaworld_to_lance(tiny_metaworld_dir, lance_uri, codec="png", mode="overwrite")
    raw_full = MetaworldHFDataset(tiny_metaworld_dir, normalize_action=True, with_reward=True)
    lance_full = MetaworldLanceWindowDataset(str(lance_uri), image_codec="png", normalize_action=True, with_reward=True)
    assert len(raw_full) == len(lance_full)
    for ep_idx in range(len(raw_full)):
        _, _, _, _, env_info = lance_full[ep_idx]
        assert env_info["task"] in {"reach", "push"}
