# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

import datetime
import io
import json
import os
from pathlib import Path
from typing import Iterator, Optional

import imageio.v2 as imageio
import numpy as np

from src.utils.logging import get_logger

from .lance_store import METADATA_FILENAME, SUPPORTED_CODECS, _encode_image, _require_lance
from .writer_point_maze import (
    SWM_LANCE_WRITER_VERSION,
    SWM_METADATA_FORMAT_VERSION,
    WRITE_MODES,
    _sha256_file,
)

log = get_logger(__name__)


def _decode_video_bytes(video_bytes: bytes) -> np.ndarray:
    reader = imageio.get_reader(io.BytesIO(video_bytes), format="mp4")
    try:
        return np.stack([frame for frame in reader])
    finally:
        reader.close()


def _video_to_frames(video) -> np.ndarray:
    if isinstance(video, dict):
        if video.get("bytes") is not None:
            return _decode_video_bytes(video["bytes"])
        if video.get("path"):
            reader = imageio.get_reader(video["path"], format="mp4")
            try:
                return np.stack([frame for frame in reader])
            finally:
                reader.close()
    raise TypeError(
        "Metaworld Lance writer expects the parquet video column to contain "
        "a dict with 'bytes' or 'path'. Torchcodec VideoDecoder conversion is intentionally unsupported."
    )


def _build_metaworld_schema(d_p: int, d_a: int, d_s: int):
    import pyarrow as pa

    return pa.schema(
        [
            ("episode_idx", pa.int32()),
            ("step_idx", pa.int32()),
            ("task", pa.string()),
            ("image_bytes", pa.binary()),
            ("proprio", pa.list_(pa.float32(), d_p)),
            ("action", pa.list_(pa.float32(), d_a)),
            ("state", pa.list_(pa.float32(), d_s)),
            ("reward", pa.list_(pa.float32(), 1)),
            ("seq_length", pa.int32()),
        ]
    )


def _fsl(arr, list_size: int):
    import pyarrow as pa

    flat = pa.array(np.asarray(arr, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, list_size)


def _metaworld_record_batch(
    schema,
    ep_idx: int,
    task: str,
    images_uint8_thwc: np.ndarray,
    proprios: np.ndarray,
    actions: np.ndarray,
    states: np.ndarray,
    rewards: np.ndarray,
    codec: str,
    jpeg_quality: int,
):
    import pyarrow as pa

    T = actions.shape[0]
    image_blobs = [_encode_image(images_uint8_thwc[t], codec, jpeg_quality) for t in range(T)]
    arrays = [
        pa.array([ep_idx] * T, type=pa.int32()),
        pa.array(list(range(T)), type=pa.int32()),
        pa.array([task] * T, type=pa.string()),
        pa.array(image_blobs, type=pa.binary()),
        _fsl(proprios, proprios.shape[1]),
        _fsl(actions, actions.shape[1]),
        _fsl(states, states.shape[1]),
        _fsl(rewards.reshape(T, 1), 1),
        pa.array([T] * T, type=pa.int32()),
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def convert_metaworld_to_lance(
    src_dir: str | os.PathLike,
    dst_uri: str | os.PathLike,
    codec: str = "png",
    mode: str = "error",
    jpeg_quality: int = 95,
    limit: Optional[int] = None,
    action_scale: float = 1.0,
) -> dict:
    """Convert Metaworld HF-parquet data to one Lance dataset."""
    _require_lance()
    import lance  # noqa: F401
    import pyarrow as pa
    from datasets import load_dataset

    if mode == "append":
        raise NotImplementedError(
            "convert_metaworld_to_lance: mode='append' is not supported. "
            "Use mode='overwrite' or mode='error'."
        )
    if mode not in WRITE_MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {WRITE_MODES}")
    if codec not in SUPPORTED_CODECS:
        raise ValueError(f"Unsupported codec {codec!r}; expected one of {SUPPORTED_CODECS}")

    src = Path(src_dir)
    dst = Path(dst_uri)
    if dst.exists() and mode == "error":
        raise FileExistsError(f"Destination already exists: {dst}; use mode='overwrite'")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("parquet", data_dir=str(src), split="train")
    n_total = len(ds)
    n_eps = n_total if limit is None else min(limit, n_total)
    if n_eps <= 0:
        raise ValueError(f"No Metaworld episodes found at {src}")

    first = ds[0]
    first_states = np.asarray(first["states"], dtype=np.float32)
    first_actions = np.asarray(first["actions"], dtype=np.float32)
    d_s = first_states[:-1].shape[-1]
    d_a = first_actions.shape[-1]
    d_p = min(4, d_s)
    schema = _build_metaworld_schema(d_p=d_p, d_a=d_a, d_s=d_s)

    first_frames = _video_to_frames(first["video"])
    _t, h, w, c = first_frames.shape
    image_shape_chw = [int(c), int(h), int(w)]
    del first_frames

    seq_lengths: list[int] = []
    tasks: list[str] = []
    total_rows = 0

    def _iter_batches() -> Iterator["pa.RecordBatch"]:
        nonlocal total_rows
        for ep in range(n_eps):
            row = ds[ep]
            states = np.asarray(row["states"], dtype=np.float32)
            actions = np.asarray(row["actions"], dtype=np.float32)
            rewards = np.asarray(row["rewards"], dtype=np.float32)
            task = str(row["task"])
            T = int(actions.shape[0])
            frames = _video_to_frames(row["video"])
            if frames.shape[0] < T:
                raise ValueError(f"Metaworld episode {ep}: video frames {frames.shape[0]} < actions {T}")
            frames = frames[:T].astype(np.uint8, copy=False)
            states = states[:-1][:T]
            proprios = states[:, :d_p]
            rewards = rewards[:T]

            seq_lengths.append(T)
            tasks.append(task)
            batch = _metaworld_record_batch(
                schema=schema,
                ep_idx=ep,
                task=task,
                images_uint8_thwc=frames,
                proprios=proprios,
                actions=actions[:T],
                states=states,
                rewards=rewards,
                codec=codec,
                jpeg_quality=jpeg_quality,
            )
            total_rows += T
            del frames, states, actions, rewards, proprios
            yield batch

    reader = pa.RecordBatchReader.from_batches(schema, _iter_batches())
    lance.write_dataset(reader, str(dst), mode={"error": "create", "overwrite": "overwrite"}[mode])

    meta = {
        "format_version": SWM_METADATA_FORMAT_VERSION,
        "source_dataset": "metaworld",
        "source_path": str(src.resolve()),
        "image_codec": codec,
        "image_columns": ["image_bytes"],
        "image_shape": image_shape_chw,
        "num_episodes": n_eps,
        "seq_lengths": seq_lengths,
        "tasks_per_episode": tasks,
        "with_reward_column": True,
        "action_scale": float(action_scale),
        "imageio_version": getattr(imageio, "__version__", "unknown"),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "swm_lance_writer_version": SWM_LANCE_WRITER_VERSION,
    }
    parquet_files = sorted(src.glob("*.parquet"))
    if parquet_files:
        meta["sha256_parquet_first"] = _sha256_file(parquet_files[0])
    if codec == "jpeg":
        meta["jpeg_quality"] = int(jpeg_quality)
    with open(dst / METADATA_FILENAME, "w") as f:
        json.dump(meta, f, indent=2)
    return meta
