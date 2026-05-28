# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

import pytest

from ._gen_tiny_point_maze import generate_tiny_point_maze
from ._gen_tiny_metaworld import generate_tiny_metaworld
from ._gen_tiny_pusht import generate_tiny_pusht
from ._gen_tiny_wall import generate_tiny_wall


@pytest.fixture(scope="session")
def tiny_point_maze_dir(tmp_path_factory) -> str:
    """Session-scoped tiny PointMaze fixture. Generated once per pytest invocation."""
    dst = tmp_path_factory.mktemp("point_maze_tiny")
    generate_tiny_point_maze(dst, seed=0)
    return str(dst)


@pytest.fixture(scope="session")
def tiny_wall_dir(tmp_path_factory) -> str:
    """Session-scoped tiny Wall fixture. Generated once per pytest invocation."""
    dst = tmp_path_factory.mktemp("wall_tiny")
    generate_tiny_wall(dst, seed=0)
    return str(dst)


@pytest.fixture(scope="session")
def tiny_pusht_dir(tmp_path_factory) -> str:
    """Session-scoped tiny PushT fixture. Generated once per pytest invocation."""
    dst = tmp_path_factory.mktemp("pusht_tiny")
    generate_tiny_pusht(dst, seed=0)
    return str(dst)


@pytest.fixture(scope="session")
def tiny_metaworld_dir(tmp_path_factory) -> str:
    """Session-scoped tiny Metaworld fixture. Generated once per pytest invocation."""
    dst = tmp_path_factory.mktemp("metaworld_tiny")
    generate_tiny_metaworld(dst, seed=0)
    return str(dst)
