# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.
#
# Codec-authority contract test: swm_metadata.json is the source of truth.
# Verifies:
#   1. Passing image_codec=None to LanceStore is fine; reader uses metadata's codec.
#   2. Passing image_codec that matches metadata is fine.
#   3. Passing image_codec that differs from metadata raises ValueError.
#   4. Disabled mode='append' raises NotImplementedError.

from __future__ import annotations

from pathlib import Path

import pytest

lance = pytest.importorskip("lance")
pa = pytest.importorskip("pyarrow")

from app.plan_common.datasets.stores.lance_store import LanceStore
from app.plan_common.datasets.stores.writer_point_maze import (
    convert_point_maze_to_lance,
)


def _convert_tiny(tiny_point_maze_dir, tmp_path: Path, codec: str = "png"):
    lance_uri = tmp_path / f"point_maze_{codec}.lance"
    convert_point_maze_to_lance(
        src_dir=tiny_point_maze_dir,
        dst_uri=lance_uri,
        codec=codec,
        mode="overwrite",
        action_scale=1.0,
    )
    return lance_uri


def test_codec_none_uses_metadata(tiny_point_maze_dir, tmp_path):
    lance_uri = _convert_tiny(tiny_point_maze_dir, tmp_path, codec="png")
    store = LanceStore(str(lance_uri), image_codec=None)
    assert store.codec == "png"


def test_codec_matching_metadata_ok(tiny_point_maze_dir, tmp_path):
    lance_uri = _convert_tiny(tiny_point_maze_dir, tmp_path, codec="png")
    store = LanceStore(str(lance_uri), image_codec="png")
    assert store.codec == "png"


def test_codec_mismatch_raises(tiny_point_maze_dir, tmp_path):
    lance_uri = _convert_tiny(tiny_point_maze_dir, tmp_path, codec="png")
    with pytest.raises(ValueError, match="codec mismatch"):
        LanceStore(str(lance_uri), image_codec="jpeg")


def test_codec_mismatch_raw_uint8_vs_png(tiny_point_maze_dir, tmp_path):
    lance_uri = _convert_tiny(tiny_point_maze_dir, tmp_path, codec="raw_uint8")
    with pytest.raises(ValueError, match="codec mismatch"):
        LanceStore(str(lance_uri), image_codec="png")


def test_append_mode_rejected(tiny_point_maze_dir, tmp_path):
    """Phase 1.a explicitly forbids append; safe append needs episode_idx offset + metadata merge."""
    dst = tmp_path / "point_maze_append.lance"
    convert_point_maze_to_lance(
        src_dir=tiny_point_maze_dir,
        dst_uri=dst,
        codec="png",
        mode="overwrite",
        action_scale=1.0,
    )
    with pytest.raises(NotImplementedError, match="append"):
        convert_point_maze_to_lance(
            src_dir=tiny_point_maze_dir,
            dst_uri=dst,
            codec="png",
            mode="append",
            action_scale=1.0,
        )


def test_overwrite_replaces_dataset(tiny_point_maze_dir, tmp_path):
    """mode='overwrite' must not depend on append semantics."""
    dst = tmp_path / "point_maze_ow.lance"
    convert_point_maze_to_lance(
        src_dir=tiny_point_maze_dir, dst_uri=dst, codec="png", mode="overwrite", action_scale=1.0
    )
    # Re-running with overwrite must succeed.
    convert_point_maze_to_lance(
        src_dir=tiny_point_maze_dir, dst_uri=dst, codec="png", mode="overwrite", action_scale=1.0
    )
    store = LanceStore(str(dst))
    assert store.num_episodes == 8  # tiny fixture default


def test_error_mode_blocks_overwrite(tiny_point_maze_dir, tmp_path):
    """mode='error' (default) must refuse if destination exists."""
    dst = tmp_path / "point_maze_err.lance"
    convert_point_maze_to_lance(
        src_dir=tiny_point_maze_dir, dst_uri=dst, codec="png", mode="overwrite", action_scale=1.0
    )
    with pytest.raises(FileExistsError):
        convert_point_maze_to_lance(
            src_dir=tiny_point_maze_dir, dst_uri=dst, codec="png", mode="error", action_scale=1.0
        )
