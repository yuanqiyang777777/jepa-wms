# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

import datetime
import json
import os
import pickle
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from src.utils.logging import get_logger

from .lance_store import METADATA_FILENAME, SUPPORTED_CODECS, _require_lance
from .writer_point_maze import (
    SWM_LANCE_WRITER_VERSION,
    SWM_METADATA_FORMAT_VERSION,
    WRITE_MODES,
    _build_schema,
    _episode_record_batch,
    _sha256_file,
)

log = get_logger(__name__)


def _decord_batch_to_numpy(frames) -> np.ndarray:
    if isinstance(frames, torch.Tensor):
        return frames.cpu().numpy()
    if hasattr(frames, "asnumpy"):
        return frames.asnumpy()
    return np.asarray(frames)


def convert_pusht_to_lance(
    src_dir: str | os.PathLike,
    dst_uri: str | os.PathLike,
    codec: str = "png",
    mode: str = "error",
    jpeg_quality: int = 95,
    limit: Optional[int] = None,
    action_scale: float = 100.0,
    relative: bool = True,
    with_velocity: bool = True,
) -> dict:
    """Convert PushT train/val raw folders to a split Lance dataset.

    `dst_uri` contains two Lance datasets:
      * `train.lance/`
      * `val.lance/`
    Each split directory carries its own `swm_metadata.json`.
    """
    if mode == "append":
        raise NotImplementedError(
            "convert_pusht_to_lance: mode='append' is not supported. "
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
    dst.mkdir(parents=True, exist_ok=True)

    metas = {}
    for split in ("train", "val"):
        split_src = src / split
        if not split_src.exists():
            raise FileNotFoundError(f"PushT split directory not found: {split_src}")
        metas[split] = _convert_pusht_split_to_lance(
            src_dir=split_src,
            dst_uri=dst / f"{split}.lance",
            split=split,
            codec=codec,
            mode=mode,
            jpeg_quality=jpeg_quality,
            limit=limit,
            action_scale=action_scale,
            relative=relative,
            with_velocity=with_velocity,
        )
    return metas


def _convert_pusht_split_to_lance(
    src_dir: Path,
    dst_uri: Path,
    split: str,
    codec: str,
    mode: str,
    jpeg_quality: int,
    limit: Optional[int],
    action_scale: float,
    relative: bool,
    with_velocity: bool,
) -> dict:
    _require_lance()
    import decord
    import lance  # noqa: F401
    import pyarrow as pa

    log.info(f"Converting PushT {split} {src_dir} -> Lance {dst_uri} (codec={codec}, mode={mode})")

    states_full = torch.load(src_dir / "states.pth").float()
    action_name = "rel_actions.pth" if relative else "abs_actions.pth"
    actions_full = torch.load(src_dir / action_name).float()
    with open(src_dir / "seq_lengths.pkl", "rb") as f:
        seq_lengths = [int(x) for x in pickle.load(f)]
    shapes_path = src_dir / "shapes.pkl"
    if shapes_path.exists():
        with open(shapes_path, "rb") as f:
            shapes = list(pickle.load(f))
    else:
        shapes = ["T"] * len(seq_lengths)

    proprios_full = states_full[..., :2].clone()
    if with_velocity:
        velocities = torch.load(src_dir / "velocities.pth").float()
        states_full = torch.cat([states_full, velocities], dim=-1)
        proprios_full = torch.cat([proprios_full, velocities], dim=-1)

    n_total = len(seq_lengths)
    n_eps = n_total if limit is None else min(limit, n_total)
    seq_lengths = seq_lengths[:n_eps]
    shapes = shapes[:n_eps]

    schema = _build_schema(
        d_p=proprios_full.shape[-1],
        d_a=actions_full.shape[-1],
        d_s=states_full.shape[-1],
    )

    decord.bridge.set_bridge("torch")
    first_reader = decord.VideoReader(str(src_dir / "obses" / "episode_000.mp4"), num_threads=1)
    first_frame = first_reader.get_batch([0])
    _n, h, w, c = first_frame.shape
    image_shape_chw = [int(c), int(h), int(w)]
    del first_reader, first_frame

    total_rows = 0

    def _iter_batches() -> Iterator["pa.RecordBatch"]:
        nonlocal total_rows
        for ep in range(n_eps):
            T = seq_lengths[ep]
            reader = decord.VideoReader(str(src_dir / "obses" / f"episode_{ep:03d}.mp4"), num_threads=1)
            frames = reader.get_batch(list(range(T)))
            img_np = _decord_batch_to_numpy(frames)
            if img_np.dtype != np.uint8:
                raise TypeError(f"Expected uint8 decoded PushT frames, got {img_np.dtype}")

            batch = _episode_record_batch(
                schema=schema,
                ep_idx=ep,
                T=T,
                images_uint8_thwc=img_np,
                proprios=proprios_full[ep, :T].cpu().numpy().astype(np.float32),
                actions=actions_full[ep, :T].cpu().numpy().astype(np.float32),
                states=states_full[ep, :T].cpu().numpy().astype(np.float32),
                codec=codec,
                jpeg_quality=jpeg_quality,
            )
            total_rows += T
            del reader, frames, img_np
            yield batch

    reader = pa.RecordBatchReader.from_batches(schema, _iter_batches())
    lance.write_dataset(reader, str(dst_uri), mode={"error": "create", "overwrite": "overwrite"}[mode])

    meta = {
        "format_version": SWM_METADATA_FORMAT_VERSION,
        "source_dataset": "pusht",
        "source_path": str(src_dir.resolve()),
        "split": split,
        "image_codec": codec,
        "image_columns": ["image_bytes"],
        "image_shape": image_shape_chw,
        "num_episodes": n_eps,
        "seq_lengths": seq_lengths,
        "shapes": shapes,
        "action_scale": float(action_scale),
        "relative": bool(relative),
        "with_velocity": bool(with_velocity),
        "decord_version": getattr(decord, "__version__", "unknown"),
        "sha256_actions": _sha256_file(src_dir / action_name),
        "sha256_states": _sha256_file(src_dir / "states.pth"),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "swm_lance_writer_version": SWM_LANCE_WRITER_VERSION,
    }
    if with_velocity:
        meta["sha256_velocities"] = _sha256_file(src_dir / "velocities.pth")
    if codec == "jpeg":
        meta["jpeg_quality"] = int(jpeg_quality)
    with open(dst_uri / METADATA_FILENAME, "w") as f:
        json.dump(meta, f, indent=2)
    return meta
