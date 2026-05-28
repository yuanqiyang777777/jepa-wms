# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

import io
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_N_EPISODES = 6
DEFAULT_T = 25
DEFAULT_STATE_DIM = 6
DEFAULT_ACTION_DIM = 4
DEFAULT_IMG_SHAPE = (16, 16, 3)


def _mp4_bytes(frames: np.ndarray) -> bytes:
    buf = io.BytesIO()
    imageio.mimsave(
        buf,
        list(frames),
        format="mp4",
        fps=10,
        codec="libx264",
        quality=10,
        macro_block_size=1,
    )
    return buf.getvalue()


def generate_tiny_metaworld(
    dst: str | Path,
    n_episodes: int = DEFAULT_N_EPISODES,
    T: int = DEFAULT_T,
    state_dim: int = DEFAULT_STATE_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    img_shape: tuple[int, int, int] = DEFAULT_IMG_SHAPE,
    seed: int = 0,
) -> Path:
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    h, w, c = img_shape
    rng = np.random.default_rng(seed)

    videos = []
    states = []
    actions = []
    rewards = []
    tasks = []

    for ep in range(n_episodes):
        frames = rng.integers(0, 256, size=(T + 1, h, w, c), dtype=np.uint8)
        videos.append({"bytes": _mp4_bytes(frames), "path": None})
        states.append(rng.normal(size=(T + 1, state_dim)).astype(np.float32).tolist())
        actions.append(rng.normal(size=(T, action_dim)).astype(np.float32).tolist())
        rewards.append(rng.normal(size=(T,)).astype(np.float32).tolist())
        tasks.append("reach" if ep % 2 == 0 else "push")

    table = pa.table(
        {
            "video": pa.array(
                videos,
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "task": tasks,
        }
    )
    pq.write_table(table, dst / "data-00000-of-00001.parquet")
    return dst


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a tiny Metaworld parquet fixture.")
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"Wrote tiny Metaworld fixture to: {generate_tiny_metaworld(args.dst, seed=args.seed)}")
