# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

import pickle
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


DEFAULT_N_EPISODES = 6
DEFAULT_T = 25
DEFAULT_STATE_DIM = 5
DEFAULT_ACTION_DIM = 2
DEFAULT_VELOCITY_DIM = 2
DEFAULT_IMG_SHAPE = (16, 16, 3)


def _write_mp4(path: Path, frames: np.ndarray) -> None:
    imageio.mimsave(
        path,
        list(frames),
        fps=10,
        codec="libx264",
        quality=10,
        macro_block_size=1,
    )


def _generate_split(
    dst: Path,
    n_episodes: int,
    T: int,
    state_dim: int,
    action_dim: int,
    velocity_dim: int,
    img_shape: tuple[int, int, int],
    seed: int,
) -> None:
    h, w, c = img_shape
    (dst / "obses").mkdir(parents=True, exist_ok=True)

    gen = torch.Generator().manual_seed(seed)
    states = torch.randn(n_episodes, T, state_dim, generator=gen) * 10.0
    rel_actions = torch.randn(n_episodes, T, action_dim, generator=gen) * 20.0
    abs_actions = torch.randn(n_episodes, T, action_dim, generator=gen) * 20.0
    velocities = torch.randn(n_episodes, T, velocity_dim, generator=gen)
    seq_lengths = [T] * n_episodes
    shapes = ["T" if i % 2 == 0 else "L" for i in range(n_episodes)]

    torch.save(states, dst / "states.pth")
    torch.save(rel_actions, dst / "rel_actions.pth")
    torch.save(abs_actions, dst / "abs_actions.pth")
    torch.save(velocities, dst / "velocities.pth")
    with open(dst / "seq_lengths.pkl", "wb") as f:
        pickle.dump(seq_lengths, f)
    with open(dst / "shapes.pkl", "wb") as f:
        pickle.dump(shapes, f)

    rng = np.random.default_rng(seed)
    for ep in range(n_episodes):
        frames = rng.integers(0, 256, size=(T, h, w, c), dtype=np.uint8)
        _write_mp4(dst / "obses" / f"episode_{ep:03d}.mp4", frames)


def generate_tiny_pusht(
    dst: str | Path,
    n_episodes: int = DEFAULT_N_EPISODES,
    T: int = DEFAULT_T,
    state_dim: int = DEFAULT_STATE_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    velocity_dim: int = DEFAULT_VELOCITY_DIM,
    img_shape: tuple[int, int, int] = DEFAULT_IMG_SHAPE,
    seed: int = 0,
) -> Path:
    dst = Path(dst)
    _generate_split(dst / "train", n_episodes, T, state_dim, action_dim, velocity_dim, img_shape, seed)
    _generate_split(dst / "val", max(2, n_episodes // 2), T, state_dim, action_dim, velocity_dim, img_shape, seed + 10)
    return dst


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a tiny PushT-shaped fixture.")
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"Wrote tiny PushT fixture to: {generate_tiny_pusht(args.dst, seed=args.seed)}")
