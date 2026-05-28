# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.
#
# PointMaze -> Lance writer (Phase 1.a, no CLI).
#
# Reads the on-disk `states.pth / actions.pth / seq_lengths.pth / obses/episode_NNN.pth`
# layout used by `point_maze_dset.py:PointMazeDataset` and produces:
#   <dst_uri>/                  - a Lance dataset directory
#   <dst_uri>/swm_metadata.json - sibling JSON with codec, shape, seq_lengths, hashes
#
# Conversion is streaming: episodes are encoded and yielded one at a time via a
# pyarrow.RecordBatchReader so peak memory stays bounded to a single episode
# regardless of total dataset size.

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from src.utils.logging import get_logger

from .lance_store import METADATA_FILENAME, SUPPORTED_CODECS, _encode_image, _require_lance

log = get_logger(__name__)


SWM_LANCE_WRITER_VERSION = "0.1.0"
SWM_METADATA_FORMAT_VERSION = 1
# Phase 1.a deliberately excludes 'append': true append needs episode_idx offset and
# metadata merge (sha256/seq_lengths/num_episodes). Until that lands, we reject it loudly.
WRITE_MODES = ("error", "overwrite")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_schema(d_p: int, d_a: int, d_s: int):
    import pyarrow as pa

    return pa.schema(
        [
            ("episode_idx", pa.int32()),
            ("step_idx", pa.int32()),
            ("image_bytes", pa.binary()),
            ("proprio", pa.list_(pa.float32(), d_p)),
            ("action", pa.list_(pa.float32(), d_a)),
            ("state", pa.list_(pa.float32(), d_s)),
            ("seq_length", pa.int32()),
        ]
    )


def _episode_record_batch(
    schema,
    ep_idx: int,
    T: int,
    images_uint8_thwc: np.ndarray,
    proprios: np.ndarray,
    actions: np.ndarray,
    states: np.ndarray,
    codec: str,
    jpeg_quality: int,
):
    import pyarrow as pa

    assert images_uint8_thwc.shape[0] == T, f"image rows {images_uint8_thwc.shape[0]} != T {T}"
    assert proprios.shape[0] == T == actions.shape[0] == states.shape[0]
    image_blobs = [_encode_image(images_uint8_thwc[t], codec, jpeg_quality) for t in range(T)]

    def _fsl(arr: np.ndarray, list_size: int):
        flat = pa.array(arr.flatten().astype("float32"), type=pa.float32())
        return pa.FixedSizeListArray.from_arrays(flat, list_size)

    arrays = [
        pa.array([ep_idx] * T, type=pa.int32()),
        pa.array(list(range(T)), type=pa.int32()),
        pa.array(image_blobs, type=pa.binary()),
        _fsl(proprios, proprios.shape[1]),
        _fsl(actions, actions.shape[1]),
        _fsl(states, states.shape[1]),
        pa.array([T] * T, type=pa.int32()),
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def convert_point_maze_to_lance(
    src_dir: str | os.PathLike,
    dst_uri: str | os.PathLike,
    codec: str = "png",
    mode: str = "error",
    jpeg_quality: int = 95,
    limit: Optional[int] = None,
    action_scale: float = 1.0,
) -> dict:
    """Convert a PointMaze raw directory to a Lance dataset.

    Args:
        src_dir: directory containing `states.pth / actions.pth / seq_lengths.pth / obses/episode_NNN.pth`
            (matches `PointMazeDataset.__init__` layout at point_maze_dset.py:28-44).
        dst_uri: destination directory for the Lance dataset (also holds swm_metadata.json).
        codec: image codec; one of {"png" (default, lossless), "raw_uint8", "jpeg"}.
        mode: one of {"error", "overwrite"}. Phase 1.a does NOT support "append" - the safe
            implementation would need cumulative episode_idx offsets and a metadata merge that
            we haven't written yet.
        jpeg_quality: JPEG quality factor (used only when codec="jpeg").
        limit: if set, only convert the first N episodes (smoke testing).
        action_scale: stored in metadata; the writer does NOT divide actions by this
            (mirrors raw: action_scale is applied in PointMazeDataset.__init__, not on disk).

    Returns:
        The metadata dict that was written to swm_metadata.json.
    """
    _require_lance()
    import lance  # noqa: F401  (already verified by _require_lance)
    import pyarrow as pa

    if codec not in SUPPORTED_CODECS:
        raise ValueError(f"Unsupported codec {codec!r}; expected one of {SUPPORTED_CODECS}")
    if mode == "append":
        raise NotImplementedError(
            "convert_point_maze_to_lance: mode='append' is not supported in Phase 1.a. "
            "Safe append needs cumulative episode_idx offsets and a swm_metadata.json merge "
            "(sha256, seq_lengths, num_episodes) that we haven't implemented yet. "
            "Use mode='overwrite' or mode='error'."
        )
    if mode not in WRITE_MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {WRITE_MODES}")

    src = Path(src_dir)
    dst = Path(dst_uri)

    if dst.exists():
        if mode == "error":
            raise FileExistsError(f"Destination already exists: {dst}; use mode='overwrite'")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Converting PointMaze {src} -> Lance {dst} (codec={codec}, mode={mode})")

    # ---- Load raw tabular (small: actions/states are ~MB total even for full PointMaze) ----
    states_full = torch.load(src / "states.pth").float()  # (N, T_max, D_s)
    actions_full = torch.load(src / "actions.pth").float()  # (N, T_max, D_a)
    seq_lengths = torch.load(src / "seq_lengths.pth")  # (N,)
    if isinstance(seq_lengths, torch.Tensor):
        seq_lengths = seq_lengths.tolist()
    else:
        seq_lengths = list(seq_lengths)
    seq_lengths = [int(L) for L in seq_lengths]

    n_eps_total = len(seq_lengths)
    n_eps = n_eps_total if limit is None else min(limit, n_eps_total)
    seq_lengths = seq_lengths[:n_eps]

    d_a = actions_full.shape[-1]
    d_s = states_full.shape[-1]
    d_p = d_s  # PointMaze: proprio = state.clone()

    schema = _build_schema(d_p=d_p, d_a=d_a, d_s=d_s)

    # ---- Probe image shape from the first episode (one-frame peek) ----
    first_img = torch.load(src / "obses" / f"episode_{0:03d}.pth")
    if isinstance(first_img, torch.Tensor):
        first_img_np = first_img.cpu().numpy() if first_img.is_floating_point() is False else first_img.cpu().numpy().astype(np.uint8)
    else:
        first_img_np = np.asarray(first_img)
    assert first_img_np.ndim == 4, f"expected per-episode image shape (T,H,W,C); got {first_img_np.shape}"
    _T0, H, W, C = first_img_np.shape
    image_shape_chw = [int(C), int(H), int(W)]
    del first_img, first_img_np  # release the peek tensor before the streaming pass

    # ---- Streaming pass: yield one episode's RecordBatch at a time ----
    total_rows = 0

    def _iter_batches() -> Iterator["pa.RecordBatch"]:
        nonlocal total_rows
        for ep in range(n_eps):
            T = seq_lengths[ep]
            proprios_ep = states_full[ep, :T].cpu().numpy().astype(np.float32)
            states_ep = states_full[ep, :T].cpu().numpy().astype(np.float32)
            actions_ep = actions_full[ep, :T].cpu().numpy().astype(np.float32)

            img_t = torch.load(src / "obses" / f"episode_{ep:03d}.pth")
            img_np = img_t.cpu().numpy() if isinstance(img_t, torch.Tensor) else np.asarray(img_t)
            if img_np.dtype != np.uint8:
                raise TypeError(
                    f"Expected uint8 images for episode {ep}, got dtype {img_np.dtype}. "
                    "Refusing to silently quantise - fix the source dataset or extend the writer."
                )
            if img_np.shape[0] < T:
                raise ValueError(
                    f"Episode {ep}: image tensor has {img_np.shape[0]} frames but seq_length is {T}"
                )
            img_np = img_np[:T]

            batch = _episode_record_batch(
                schema=schema,
                ep_idx=ep,
                T=T,
                images_uint8_thwc=img_np,
                proprios=proprios_ep,
                actions=actions_ep,
                states=states_ep,
                codec=codec,
                jpeg_quality=jpeg_quality,
            )
            total_rows += T
            if (ep + 1) % 50 == 0 or ep == n_eps - 1:
                log.info(f"   episode {ep+1}/{n_eps} encoded ({total_rows} rows so far)")
            # Drop large refs before yielding so the previous batch can be GC'd while lance writes.
            del img_t, img_np, proprios_ep, states_ep, actions_ep
            yield batch

    reader = pa.RecordBatchReader.from_batches(schema, _iter_batches())

    # ---- Write to Lance (consumes the reader incrementally) ----
    lance_mode = {"error": "create", "overwrite": "overwrite"}[mode]
    lance.write_dataset(reader, str(dst), mode=lance_mode)
    log.info(f"   Lance write complete ({total_rows} rows)")

    # ---- Sibling metadata JSON ----
    meta = {
        "format_version": SWM_METADATA_FORMAT_VERSION,
        "source_dataset": "pointmaze",
        "source_path": str(Path(src_dir).resolve()),
        "image_codec": codec,
        "image_columns": ["image_bytes"],
        "image_shape": image_shape_chw,  # (C, H, W)
        "num_episodes": n_eps,
        "seq_lengths": seq_lengths,
        "action_scale": float(action_scale),
        "sha256_actions": _sha256_file(src / "actions.pth"),
        "sha256_states": _sha256_file(src / "states.pth"),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "swm_lance_writer_version": SWM_LANCE_WRITER_VERSION,
    }
    if codec == "jpeg":
        meta["jpeg_quality"] = int(jpeg_quality)
    with open(dst / METADATA_FILENAME, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"   wrote {dst / METADATA_FILENAME}")

    return meta
