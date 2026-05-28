# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SEED = 234
NUM_HIST = 3
NUM_PRED = 1
FRAMESKIP = 5
ACTION_SKIP = 1
SPLIT_RATIO = 0.9
PROCESS_ACTIONS = "concat"


def _build_pair(env: str, raw_path: str, lance_uri: str):
    common = dict(
        transform=None,
        n_rollout=None,
        normalize_action=True,
        split_ratio=SPLIT_RATIO,
        num_hist=NUM_HIST,
        num_pred=NUM_PRED,
        num_frames_val=None,
        frameskip=FRAMESKIP,
        action_skip=ACTION_SKIP,
        random_seed=DEFAULT_SEED,
        process_actions=PROCESS_ACTIONS,
        dset_fraction=1.0,
    )

    if env == "maze":
        from app.plan_common.datasets.point_maze_dset import load_point_maze_slice_train_val
        from app.plan_common.datasets.stores.point_maze_lance import load_point_maze_lance_slice_train_val

        raw_dsets, _ = load_point_maze_slice_train_val(data_path=raw_path, traj_subset=True, **common)
        lance_dsets, _ = load_point_maze_lance_slice_train_val(lance_uri=lance_uri, traj_subset=True, **common)
    elif env == "wall":
        from app.plan_common.datasets.stores.wall_lance import load_wall_lance_slice_train_val
        from app.plan_common.datasets.wall_dset import load_wall_slice_train_val

        raw_dsets, _ = load_wall_slice_train_val(
            data_path=raw_path,
            split_mode="random",
            traj_subset=True,
            **common,
        )
        lance_dsets, _ = load_wall_lance_slice_train_val(lance_uri=lance_uri, traj_subset=True, **common)
    elif env == "pusht":
        from app.plan_common.datasets.pusht_dset import load_pusht_slice_train_val
        from app.plan_common.datasets.stores.pusht_lance import load_pusht_lance_slice_train_val

        raw_dsets, _ = load_pusht_slice_train_val(
            data_path=raw_path,
            with_velocity=True,
            **common,
        )
        lance_dsets, _ = load_pusht_lance_slice_train_val(
            lance_uri=lance_uri,
            with_velocity=True,
            **common,
        )
    elif env == "mw":
        from app.plan_common.datasets.metaworld_hf_dset import load_metaworld_hf_slice_train_val
        from app.plan_common.datasets.stores.metaworld_lance import load_metaworld_hf_lance_slice_train_val

        raw_dsets, _ = load_metaworld_hf_slice_train_val(
            data_path=raw_path,
            traj_subset=True,
            filter_tasks=None,
            with_reward=False,
            **common,
        )
        lance_dsets, _ = load_metaworld_hf_lance_slice_train_val(
            lance_uri=lance_uri,
            traj_subset=True,
            filter_tasks=None,
            with_reward=False,
            **common,
        )
    else:
        raise ValueError(f"Unsupported env {env!r}")
    return raw_dsets["train"], lance_dsets["train"]


def _assert_tensor_equal(name: str, raw_value: torch.Tensor, lance_value: torch.Tensor) -> None:
    try:
        torch.testing.assert_close(raw_value, lance_value, rtol=0, atol=0)
    except AssertionError as exc:
        diff = (raw_value - lance_value).abs()
        max_diff = diff.max().item() if diff.numel() else 0.0
        raise AssertionError(
            f"{name} mismatch: raw_shape={tuple(raw_value.shape)} "
            f"lance_shape={tuple(lance_value.shape)} max_abs_diff={max_diff}"
        ) from exc


def _compare_sample(raw_sample, lance_sample, sample_idx: int) -> None:
    obs_r, act_r, state_r, rew_r = raw_sample
    obs_l, act_l, state_l, rew_l = lance_sample

    _assert_tensor_equal(f"idx={sample_idx} action", act_r, act_l)
    _assert_tensor_equal(f"idx={sample_idx} state", state_r, state_l)
    _assert_tensor_equal(f"idx={sample_idx} proprio", obs_r["proprio"], obs_l["proprio"])
    _assert_tensor_equal(f"idx={sample_idx} visual", obs_r["visual"], obs_l["visual"])
    _assert_tensor_equal(f"idx={sample_idx} reward", rew_r, rew_l)


def _read_codec(env: str, lance_uri: str) -> str:
    from app.plan_common.datasets.stores.lance_store import read_metadata

    if env != "pusht":
        return read_metadata(lance_uri)["image_codec"]

    root = Path(lance_uri)
    train_codec = read_metadata(root / "train.lance")["image_codec"]
    val_codec = read_metadata(root / "val.lance")["image_codec"]
    if train_codec != val_codec:
        raise ValueError(
            f"PushT Lance codec mismatch: train={train_codec} at {root / 'train.lance'} "
            f"val={val_codec} at {root / 'val.lance'}"
        )
    return train_codec


def run_audit(env: str, raw_path: str, lance_uri: str, n_windows: int, seed: int) -> int:
    raw_train, lance_train = _build_pair(env, raw_path, lance_uri)

    if raw_train.slices != lance_train.slices:
        raise AssertionError(
            f"slice list mismatch: raw={len(raw_train.slices)} lance={len(lance_train.slices)}"
        )
    if len(raw_train) < n_windows:
        raise ValueError(f"{env}: requested {n_windows} windows but train split only has {len(raw_train)}")

    order = torch.randperm(len(raw_train), generator=torch.Generator().manual_seed(seed))[:n_windows].tolist()
    failures: list[str] = []
    for idx in order:
        try:
            _compare_sample(raw_train[idx], lance_train[idx], sample_idx=idx)
        except AssertionError as exc:
            failures.append(str(exc))
            if len(failures) >= 5:
                break

    if failures:
        print(f"FAIL {len(failures)} mismatching windows (env={env})")
        for failure in failures:
            print(f"  {failure}")
        return 1

    codec = _read_codec(env, lance_uri)
    print(f"OK {n_windows}/{n_windows} windows match (env={env}, codec={codec})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit raw vs Lance data-window equivalence on real datasets.")
    parser.add_argument("--env", choices=("maze", "wall", "pusht", "mw"), required=True)
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--lance-uri", required=True)
    parser.add_argument("--n-windows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    lance_uri = Path(args.lance_uri)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw path not found: {raw_path}")
    if not lance_uri.exists():
        raise FileNotFoundError(f"Lance URI not found: {lance_uri}")

    return run_audit(
        env=args.env,
        raw_path=str(raw_path),
        lance_uri=str(lance_uri),
        n_windows=args.n_windows,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
