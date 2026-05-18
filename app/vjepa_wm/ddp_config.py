# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from dataclasses import dataclass
from typing import Optional

from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}
_NONE_HOOKS = {"", "none", "off", "0", "false", "no"}
_SUPPORTED_COMM_HOOKS = _NONE_HOOKS | {"bf16"}


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(f"{name} must be one of {sorted(_TRUTHY | _FALSY)}; got {raw!r}")


def _positive_int_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}")
    return value


def _comm_hook_env(name: str) -> str:
    raw = os.environ.get(name, "none")
    value = raw.strip().lower()
    if value not in _SUPPORTED_COMM_HOOKS:
        raise ValueError(f"{name} must be one of {sorted(_SUPPORTED_COMM_HOOKS)}; got {raw!r}")
    return "none" if value in _NONE_HOOKS else value


@dataclass(frozen=True)
class DDPConfig:
    static_graph: bool = False
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: Optional[int] = None
    comm_hook: str = "none"

    def ddp_kwargs(self) -> dict:
        kwargs = {
            "static_graph": self.static_graph,
            "find_unused_parameters": self.find_unused_parameters,
            "gradient_as_bucket_view": self.gradient_as_bucket_view,
            "broadcast_buffers": self.broadcast_buffers,
        }
        if self.bucket_cap_mb is not None:
            kwargs["bucket_cap_mb"] = self.bucket_cap_mb
        return kwargs

    def summary(self) -> str:
        return (
            f"static_graph={self.static_graph}, "
            f"find_unused_parameters={self.find_unused_parameters}, "
            f"gradient_as_bucket_view={self.gradient_as_bucket_view}, "
            f"broadcast_buffers={self.broadcast_buffers}, "
            f"bucket_cap_mb={self.bucket_cap_mb}, "
            f"comm_hook={self.comm_hook}"
        )


def ddp_config_from_env() -> DDPConfig:
    return DDPConfig(
        static_graph=_bool_env("JEPAWM_DDP_STATIC_GRAPH", False),
        find_unused_parameters=False,
        gradient_as_bucket_view=_bool_env("JEPAWM_DDP_GRADIENT_AS_BUCKET_VIEW", False),
        broadcast_buffers=_bool_env("JEPAWM_DDP_BROADCAST_BUFFERS", True),
        bucket_cap_mb=_positive_int_env("JEPAWM_DDP_BUCKET_CAP_MB"),
        comm_hook=_comm_hook_env("JEPAWM_DDP_COMM_HOOK"),
    )


def wrap_ddp_module(module, config: DDPConfig):
    ddp_module = DDP(module, **config.ddp_kwargs())
    if config.comm_hook == "bf16":
        ddp_module.register_comm_hook(state=None, hook=ddp_default_hooks.bf16_compress_hook)
    return ddp_module
