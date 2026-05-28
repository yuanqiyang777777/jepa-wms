# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

import pytest

from ._gen_tiny_point_maze import generate_tiny_point_maze


@pytest.fixture(scope="session")
def tiny_point_maze_dir(tmp_path_factory) -> str:
    """Session-scoped tiny PointMaze fixture. Generated once per pytest invocation."""
    dst = tmp_path_factory.mktemp("point_maze_tiny")
    generate_tiny_point_maze(dst, seed=0)
    return str(dst)
