# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.
#
# Design adapted from MIT-licensed stable-worldmodel
# (https://github.com/galilai-group/stable-worldmodel); structure and naming
# are independent of any vendored code path.

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.utils.logging import get_logger

log = get_logger(__name__)


SUPPORTED_CODECS = ("png", "raw_uint8", "jpeg")
METADATA_FILENAME = "swm_metadata.json"


def _require_lance():
    try:
        import lance  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Lance backend requested but the `lance` package is not installed. "
            "Install with: pip install -e '.[data-lance]'"
        ) from e


def _decode_image(blob: bytes, codec: str, image_shape: Sequence[int]) -> np.ndarray:
    """Decode a single image blob to a numpy uint8 array of shape (H, W, C)."""
    if codec == "raw_uint8":
        c, h, w = image_shape
        arr = np.frombuffer(blob, dtype=np.uint8).reshape(h, w, c)
        return arr
    elif codec in ("png", "jpeg"):
        from PIL import Image

        img = Image.open(io.BytesIO(blob))
        arr = np.asarray(img)
        if arr.ndim == 2:  # grayscale -> expand to 3 channels (defensive)
            arr = np.stack([arr] * 3, axis=-1)
        return arr
    else:
        raise ValueError(f"Unsupported image codec: {codec!r}; expected one of {SUPPORTED_CODECS}")


def _encode_image(arr: np.ndarray, codec: str, jpeg_quality: int = 95) -> bytes:
    """Encode an image array (H, W, C) uint8 to bytes per codec."""
    assert arr.dtype == np.uint8, f"image must be uint8, got {arr.dtype}"
    if codec == "raw_uint8":
        return arr.tobytes()
    elif codec == "png":
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", optimize=False, compress_level=1)
        return buf.getvalue()
    elif codec == "jpeg":
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=jpeg_quality)
        return buf.getvalue()
    else:
        raise ValueError(f"Unsupported image codec: {codec!r}; expected one of {SUPPORTED_CODECS}")


def read_metadata(lance_uri: str | Path) -> dict:
    """Read the sibling swm_metadata.json next to <lance_uri>."""
    meta_path = Path(lance_uri).parent / METADATA_FILENAME if Path(lance_uri).is_file() else Path(lance_uri) / METADATA_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"swm_metadata.json not found next to lance dataset at {lance_uri}. "
            f"Expected at: {meta_path}"
        )
    with open(meta_path, "r") as f:
        return json.load(f)


class LanceStore:
    """Per-worker-lazy Lance reader.

    Holds only string URI + decoded metadata on the instance; opens the
    lance.dataset handle lazily inside the worker process (keyed on PID) so
    that fork/spawn safety is preserved across DataLoader workers.

    `read_window(ep_idx, frames, action_frames)` returns a dict with
    unnormalized values; callers (LanceWindowDataset / LanceTrajSlicerDataset)
    apply normalization, transform, and post-slice rearrangement.
    """

    def __init__(self, lance_uri: str | Path, image_codec: str | None = None):
        _require_lance()
        self._uri: str = str(lance_uri)
        self._metadata: dict = read_metadata(self._uri)
        # swm_metadata.json is the AUTHORITATIVE source of the image codec. We do not let the
        # caller (e.g. init_data via config) silently override it: if the config provides a codec,
        # it must match metadata exactly. None means "use whatever metadata says".
        meta_codec = self._metadata.get("image_codec")
        if meta_codec is None:
            raise ValueError(f"swm_metadata.json at {self._uri} is missing 'image_codec'")
        if meta_codec not in SUPPORTED_CODECS:
            raise ValueError(
                f"swm_metadata.json declares unsupported codec {meta_codec!r}; "
                f"expected one of {SUPPORTED_CODECS}"
            )
        if image_codec is not None and image_codec != meta_codec:
            raise ValueError(
                f"Image codec mismatch: config requested {image_codec!r} but the dataset at "
                f"{self._uri} was written with {meta_codec!r} (per swm_metadata.json). "
                "Remove the explicit codec from the config to use the dataset's codec, "
                "or re-convert the dataset with the desired codec."
            )
        self._codec: str = meta_codec
        self._image_shape: tuple[int, int, int] = tuple(self._metadata["image_shape"])  # (C, H, W)
        self._seq_lengths: list[int] = list(self._metadata["seq_lengths"])
        # Cumulative offsets: row index of step 0 of episode i is _cum[i]; row count of episode i is _seq_lengths[i].
        self._cum: list[int] = [0]
        for L in self._seq_lengths:
            self._cum.append(self._cum[-1] + L)
        self._handle_by_pid: dict[int, object] = {}

    # ----- pickling: drop the open handle dict so spawn workers re-open lazily. -----
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle_by_pid"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def codec(self) -> str:
        return self._codec

    @property
    def image_shape_chw(self) -> tuple[int, int, int]:
        return self._image_shape

    @property
    def seq_lengths(self) -> list[int]:
        return list(self._seq_lengths)

    @property
    def num_episodes(self) -> int:
        return len(self._seq_lengths)

    def _ds(self):
        import lance

        pid = os.getpid()
        ds = self._handle_by_pid.get(pid)
        if ds is None:
            ds = lance.dataset(self._uri)
            self._handle_by_pid[pid] = ds
        return ds

    def _row_indices(self, ep_idx: int, step_indices: Sequence[int]) -> list[int]:
        base = self._cum[ep_idx]
        L = self._seq_lengths[ep_idx]
        out = []
        for s in step_indices:
            if s < 0 or s >= L:
                raise IndexError(f"step {s} out of bounds for episode {ep_idx} of length {L}")
            out.append(base + int(s))
        return out

    def read_window(
        self,
        ep_idx: int,
        step_indices: Sequence[int],
        action_step_indices: Sequence[int] | None = None,
    ) -> dict:
        """Read a window from a single episode.

        Args:
            ep_idx: episode index (in original seq_lengths order).
            step_indices: step indices for visual / proprio / state (length T).
            action_step_indices: step indices for action (length T_a). If None,
                uses step_indices.

        Returns:
            dict with keys:
                'visual':  uint8 Tensor of shape (T, H, W, C)
                'proprio': float32 Tensor of shape (T, D_p)   [UNNORMALIZED]
                'state':   float32 Tensor of shape (T, D_s)   [UNNORMALIZED]
                'action':  float32 Tensor of shape (T_a, D_a) [UNNORMALIZED, unscaled]
        """
        if action_step_indices is None:
            action_step_indices = step_indices

        ds = self._ds()
        obs_rows = self._row_indices(ep_idx, step_indices)
        act_rows = self._row_indices(ep_idx, action_step_indices)

        schema_names = set(ds.schema.names)
        obs_columns = ["image_bytes", "proprio", "state"]
        if "reward" in schema_names:
            obs_columns.append("reward")

        # Take obs columns at obs_rows; take action column at act_rows. Two scans.
        obs_table = ds.take(obs_rows, columns=obs_columns)
        act_table = ds.take(act_rows, columns=["action"])

        # Decode images.
        img_blobs = obs_table.column("image_bytes").to_pylist()
        images = np.stack(
            [_decode_image(b, self._codec, self._image_shape) for b in img_blobs],
            axis=0,
        )  # (T, H, W, C) uint8
        visual = torch.from_numpy(images)

        # Tabular columns are pyarrow.FixedSizeListArray; convert to numpy then torch.
        proprio = _arrow_2d_to_tensor(obs_table.column("proprio"))
        state = _arrow_2d_to_tensor(obs_table.column("state"))
        action = _arrow_2d_to_tensor(act_table.column("action"))

        out = {"visual": visual, "proprio": proprio, "state": state, "action": action}
        if "reward" in schema_names:
            out["reward"] = _arrow_2d_to_tensor(obs_table.column("reward"))
        return out

    def read_full_column(self, name: str) -> torch.Tensor:
        """Read a single tabular column across the entire table.

        Used by LanceWindowDataset.__init__ to compute normalization stats with
        no image reads. Returns a (N_total_rows, D) float32 tensor.
        """
        ds = self._ds()
        table = ds.to_table(columns=[name])
        return _arrow_2d_to_tensor(table.column(name))


def _arrow_2d_to_tensor(arr) -> torch.Tensor:
    """Convert a pyarrow FixedSizeListArray (or ChunkedArray of such) to a (N, D) float32 torch tensor."""
    # Handle both ChunkedArray and FixedSizeListArray.
    if hasattr(arr, "combine_chunks"):
        arr = arr.combine_chunks()
    # values has dtype float32; reshape into (N, D). Force a writable copy so
    # torch.from_numpy doesn't warn about non-writable PyArrow-backed buffers.
    values = np.ascontiguousarray(np.asarray(arr.values, dtype=np.float32))
    n = len(arr)
    d = values.size // n if n > 0 else 0
    return torch.from_numpy(values.reshape(n, d).copy())
