# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.
#
# Wall -> Lance writer (Phase 1.b, no CLI).
#
# Reads the on-disk layout used by `wall_dset.py:WallDataset`:
#   states.pth          (N, T, D_s)  float
#   actions.pth         (N, T, D_a)  float
#   door_locations.pth  (N, T, D_dl) float    (only frame 0 is consumed by env_info)
#   wall_locations.pth  (N, T, D_wl) float    (only frame 0)
#   obses/episode_NNN.pth   (T, H, W, C) uint8
#
# All episodes have uniform length T (= actions.shape[1]); no seq_lengths.pth file.
#
# Produces:
#   <dst_uri>/                  - Lance dataset directory (same schema as PointMaze)
#   <dst_uri>/swm_metadata.json - sibling JSON with door_locations_frame0 / wall_locations_frame0
#
# Conversion is streaming via pyarrow.RecordBatchReader (peak memory bounded to one episode).

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


def convert_wall_to_lance(
    src_dir: str | os.PathLike,
    dst_uri: str | os.PathLike,
    codec: str = "png",
    mode: str = "error",
    jpeg_quality: int = 95,
    limit: Optional[int] = None,
    action_scale: float = 1.0,
) -> dict:
    """Convert a Wall raw directory to a Lance dataset.

    Args:
        src_dir: directory containing `states.pth / actions.pth / door_locations.pth /
            wall_locations.pth / obses/episode_NNN.pth` (matches
            `WallDataset.__init__` layout at wall_dset.py:36-46).
        dst_uri: destination directory for the Lance dataset (also holds swm_metadata.json).
        codec: image codec; one of {"png" (default, lossless), "raw_uint8", "jpeg"}.
        mode: one of {"error", "overwrite"} (Phase 1.b inherits Phase 1.a's no-append policy).
        jpeg_quality: JPEG quality factor (used only when codec="jpeg").
        limit: if set, only convert the first N episodes.
        action_scale: stored in metadata; the writer does NOT divide actions by this
            (mirrors raw: action_scale is applied in WallDataset.__init__, not on disk).

    Returns:
        The metadata dict that was written to swm_metadata.json.
    """
    _require_lance()
    import lance  # noqa: F401
    import pyarrow as pa

    if codec not in SUPPORTED_CODECS:
        raise ValueError(f"Unsupported codec {codec!r}; expected one of {SUPPORTED_CODECS}")
    if mode == "append":
        raise NotImplementedError(
            "convert_wall_to_lance: mode='append' is not supported in Phase 1.b. "
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

    log.info(f"Converting Wall {src} -> Lance {dst} (codec={codec}, mode={mode})")

    # ---- Load raw tabular ----
    states_full = torch.load(src / "states.pth").float()       # (N, T, D_s)
    actions_full = torch.load(src / "actions.pth").float()      # (N, T, D_a)
    door_locations_full = torch.load(src / "door_locations.pth").float()  # (N, T, D_dl)
    wall_locations_full = torch.load(src / "wall_locations.pth").float()  # (N, T, D_wl)

    n_eps_total = states_full.shape[0]
    traj_len = actions_full.shape[1]
    n_eps = n_eps_total if limit is None else min(limit, n_eps_total)
    seq_lengths = [int(traj_len)] * n_eps  # uniform length per WallDataset.get_seq_length

    assert states_full.shape[0] == actions_full.shape[0] == door_locations_full.shape[0] == wall_locations_full.shape[0], \
        "Episode counts mismatch across states/actions/door/wall"

    d_a = actions_full.shape[-1]
    d_s = states_full.shape[-1]
    d_p = d_s  # Wall: proprio = state.clone() (wall_dset.py:39)
    d_dl = door_locations_full.shape[-1]
    d_wl = wall_locations_full.shape[-1]

    schema = _build_schema(d_p=d_p, d_a=d_a, d_s=d_s)

    # ---- Probe image shape ----
    first_img = torch.load(src / "obses" / f"episode_{0:03d}.pth")
    first_img_np = first_img.cpu().numpy() if isinstance(first_img, torch.Tensor) else np.asarray(first_img)
    assert first_img_np.ndim == 4, f"expected per-episode image shape (T,H,W,C); got {first_img_np.shape}"
    _T0, H, W, C = first_img_np.shape
    image_shape_chw = [int(C), int(H), int(W)]
    del first_img, first_img_np

    # ---- Per-episode frame-0 door/wall locations (env_info contract -- only [0] is consumed) ----
    door_locations_frame0 = door_locations_full[:n_eps, 0, :].cpu().numpy().astype(np.float32).tolist()
    wall_locations_frame0 = wall_locations_full[:n_eps, 0, :].cpu().numpy().astype(np.float32).tolist()

    # ---- Streaming pass ----
    total_rows = 0

    def _iter_batches() -> Iterator["pa.RecordBatch"]:
        nonlocal total_rows
        for ep in range(n_eps):
            T = traj_len
            proprios_ep = states_full[ep, :T].cpu().numpy().astype(np.float32)
            states_ep = states_full[ep, :T].cpu().numpy().astype(np.float32)
            actions_ep = actions_full[ep, :T].cpu().numpy().astype(np.float32)

            img_t = torch.load(src / "obses" / f"episode_{ep:03d}.pth")
            img_np = img_t.cpu().numpy() if isinstance(img_t, torch.Tensor) else np.asarray(img_t)
            if img_np.dtype != np.uint8:
                raise TypeError(
                    f"Expected uint8 images for episode {ep}, got dtype {img_np.dtype}. "
                    "Refusing to silently quantise."
                )
            if img_np.shape[0] < T:
                raise ValueError(
                    f"Episode {ep}: image tensor has {img_np.shape[0]} frames but T is {T}"
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
            del img_t, img_np, proprios_ep, states_ep, actions_ep
            yield batch

    reader = pa.RecordBatchReader.from_batches(schema, _iter_batches())

    lance_mode = {"error": "create", "overwrite": "overwrite"}[mode]
    lance.write_dataset(reader, str(dst), mode=lance_mode)
    log.info(f"   Lance write complete ({total_rows} rows)")

    # ---- Sibling metadata JSON ----
    meta = {
        "format_version": SWM_METADATA_FORMAT_VERSION,
        "source_dataset": "wall",
        "source_path": str(Path(src_dir).resolve()),
        "image_codec": codec,
        "image_columns": ["image_bytes"],
        "image_shape": image_shape_chw,
        "num_episodes": n_eps,
        "seq_lengths": seq_lengths,  # uniform; included for parity with other envs
        "traj_len": int(traj_len),
        "action_scale": float(action_scale),
        "door_locations_frame0": door_locations_frame0,  # N x D_dl
        "wall_locations_frame0": wall_locations_frame0,  # N x D_wl
        "door_location_dim": int(d_dl),
        "wall_location_dim": int(d_wl),
        "sha256_actions": _sha256_file(src / "actions.pth"),
        "sha256_states": _sha256_file(src / "states.pth"),
        "sha256_door_locations": _sha256_file(src / "door_locations.pth"),
        "sha256_wall_locations": _sha256_file(src / "wall_locations.pth"),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "swm_lance_writer_version": SWM_LANCE_WRITER_VERSION,
    }
    if codec == "jpeg":
        meta["jpeg_quality"] = int(jpeg_quality)
    with open(dst / METADATA_FILENAME, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"   wrote {dst / METADATA_FILENAME}")

    return meta
