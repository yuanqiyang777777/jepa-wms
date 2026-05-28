# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.
#
# Deterministic tiny PointMaze synthetic generator for tests.
#
# Produces a directory laid out exactly like the real PointMaze dataset that
# `PointMazeDataset` (app/plan_common/datasets/point_maze_dset.py) consumes:
#
#   <dst>/
#       states.pth        torch.Tensor [N, T_max, state_dim]   (padded; float)
#       actions.pth       torch.Tensor [N, T_max, action_dim]  (padded; float)
#       seq_lengths.pth   torch.Tensor [N]                     (int64)
#       obses/
#           episode_000.pth   torch.Tensor [T_full, H, W, C]   (uint8)
#           episode_001.pth
#           ...
#
# Everything is generated from `torch.manual_seed(seed)` so the output is
# bit-deterministic across runs.

from __future__ import annotations

from pathlib import Path

import torch


# Default dimensions matching PointMaze in
# app/plan_common/datasets/__init__.py:DATA_STATS["pointmaze"]:
#   action_dim=2, state_dim=4 (proprio_dim==state_dim)
DEFAULT_N_EPISODES = 8
DEFAULT_SEQ_LENGTHS = [25, 24, 23, 22, 25, 24, 23, 22]
DEFAULT_T_MAX = max(DEFAULT_SEQ_LENGTHS)
DEFAULT_STATE_DIM = 4
DEFAULT_ACTION_DIM = 2
DEFAULT_IMG_SHAPE = (16, 16, 3)  # (H, W, C); small for speed, real PointMaze is 64x64


def generate_tiny_point_maze(
    dst: str | Path,
    n_episodes: int = DEFAULT_N_EPISODES,
    seq_lengths: list[int] | None = None,
    state_dim: int = DEFAULT_STATE_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    img_shape: tuple[int, int, int] = DEFAULT_IMG_SHAPE,
    seed: int = 0,
) -> Path:
    """Generate a deterministic tiny PointMaze-shaped fixture under `dst`.

    Returns the destination `Path` (created if missing).
    """
    if seq_lengths is None:
        seq_lengths = list(DEFAULT_SEQ_LENGTHS[:n_episodes])
    assert len(seq_lengths) == n_episodes, "seq_lengths length must match n_episodes"
    T_max = max(seq_lengths)
    H, W, C = img_shape

    dst = Path(dst)
    (dst / "obses").mkdir(parents=True, exist_ok=True)

    gen = torch.Generator().manual_seed(seed)

    # ---- Padded tabular tensors ----
    # Random float values in a realistic-ish range; padding is also random but never read by the
    # raw flow (it slices with `:traj_len`) so its content doesn't affect tests.
    states = torch.randn(n_episodes, T_max, state_dim, generator=gen) * 1.0
    actions = torch.randn(n_episodes, T_max, action_dim, generator=gen) * 0.5
    seq_lengths_t = torch.tensor(seq_lengths, dtype=torch.long)

    torch.save(states, dst / "states.pth")
    torch.save(actions, dst / "actions.pth")
    torch.save(seq_lengths_t, dst / "seq_lengths.pth")

    # ---- Per-episode image tensors (uint8 THWC) ----
    for ep in range(n_episodes):
        T = seq_lengths[ep]
        # Use torch.randint with the same generator so output is fully deterministic.
        img = torch.randint(0, 256, size=(T, H, W, C), generator=gen, dtype=torch.uint8)
        torch.save(img, dst / "obses" / f"episode_{ep:03d}.pth")

    return dst


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate a tiny PointMaze-shaped fixture.")
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = generate_tiny_point_maze(args.dst, seed=args.seed)
    print(f"Wrote tiny PointMaze fixture to: {out}")
