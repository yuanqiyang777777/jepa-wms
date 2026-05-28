# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.
#
# Deterministic tiny Wall synthetic generator for tests.
#
# Produces a directory laid out exactly like the real Wall dataset that
# `WallDataset` (app/plan_common/datasets/wall_dset.py) consumes:
#
#   <dst>/
#       states.pth          (N, T, state_dim)        float
#       actions.pth         (N, T, action_dim)       float
#       door_locations.pth  (N, T, door_loc_dim)     float (only [..,0,..] consumed)
#       wall_locations.pth  (N, T, wall_loc_dim)     float (only [..,0,..] consumed)
#       obses/
#           episode_000.pth   (T, H, W, C) uint8
#           episode_001.pth
#           ...
#
# All episodes have uniform length T (Wall's invariant -- see WallDataset.get_seq_length).

from __future__ import annotations

from pathlib import Path

import torch


# Default dimensions matching Wall in
# app/plan_common/datasets/__init__.py:DATA_STATS["wall"]:
#   action_dim=2, state_dim=2 (proprio_dim==state_dim).
# door/wall location dims are not in DATA_STATS; pick something sensible (1 each is typical for Wall).
DEFAULT_N_EPISODES = 8
DEFAULT_T = 25
DEFAULT_STATE_DIM = 2
DEFAULT_ACTION_DIM = 2
DEFAULT_DOOR_LOC_DIM = 1
DEFAULT_WALL_LOC_DIM = 1
DEFAULT_IMG_SHAPE = (16, 16, 3)


def generate_tiny_wall(
    dst: str | Path,
    n_episodes: int = DEFAULT_N_EPISODES,
    T: int = DEFAULT_T,
    state_dim: int = DEFAULT_STATE_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    door_loc_dim: int = DEFAULT_DOOR_LOC_DIM,
    wall_loc_dim: int = DEFAULT_WALL_LOC_DIM,
    img_shape: tuple[int, int, int] = DEFAULT_IMG_SHAPE,
    seed: int = 0,
) -> Path:
    """Generate a deterministic tiny Wall-shaped fixture under `dst`. Returns the Path."""
    H, W, C = img_shape

    dst = Path(dst)
    (dst / "obses").mkdir(parents=True, exist_ok=True)

    gen = torch.Generator().manual_seed(seed)

    states = torch.randn(n_episodes, T, state_dim, generator=gen)
    actions = torch.randn(n_episodes, T, action_dim, generator=gen) * 0.5
    door_locations = torch.randn(n_episodes, T, door_loc_dim, generator=gen)
    wall_locations = torch.randn(n_episodes, T, wall_loc_dim, generator=gen)

    torch.save(states, dst / "states.pth")
    torch.save(actions, dst / "actions.pth")
    torch.save(door_locations, dst / "door_locations.pth")
    torch.save(wall_locations, dst / "wall_locations.pth")

    for ep in range(n_episodes):
        img = torch.randint(0, 256, size=(T, H, W, C), generator=gen, dtype=torch.uint8)
        torch.save(img, dst / "obses" / f"episode_{ep:03d}.pth")

    return dst


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate a tiny Wall-shaped fixture.")
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = generate_tiny_wall(args.dst, seed=args.seed)
    print(f"Wrote tiny Wall fixture to: {out}")
