# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Segmented per-training-step profiler.

Off by default; enabled with the JEPAWM_PROFILE=1 environment variable, so it
needs no config change and adds no measurable overhead to normal runs.

Each section is bracketed by torch.cuda.synchronize(), so a recorded time is
the section's true wall cost -- async kernel overlap is intentionally removed
because the goal is cost *attribution*, not throughput measurement. After
JEPAWM_PROFILE_WARMUP + JEPAWM_PROFILE_STEPS training steps the averaged
breakdown is printed once on rank 0 and profiling turns itself off.

Env vars:
    JEPAWM_PROFILE         "1" to enable (default off)
    JEPAWM_PROFILE_WARMUP  steps skipped before measuring (default 20)
    JEPAWM_PROFILE_STEPS   steps averaged (default 200)

Instrumented code uses:
    profiler.start("name"); ...; profiler.stop("name")
    with profiler.section("name"): ...
    profiler.step_end(step_wall_ms)   # once per training iteration
"""

import contextlib
import os
import time

import torch

_TRUTHY = {"1", "true", "yes", "on"}


def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


class StepProfiler:
    def __init__(self):
        self.enabled = os.environ.get("JEPAWM_PROFILE", "0").strip().lower() in _TRUTHY
        self.warmup = _int_env("JEPAWM_PROFILE_WARMUP", 20)
        self.steps = _int_env("JEPAWM_PROFILE_STEPS", 200)
        self._sum = {}
        self._count = {}
        self._order = []
        self._open = {}
        self._step = 0
        self._step_total_ms = 0.0
        self._step_total_n = 0
        self._reported = False
        if self.enabled:
            print(
                f"[StepProfiler] enabled: warmup={self.warmup}, steps={self.steps} "
                "(set JEPAWM_PROFILE=0 to disable)",
                flush=True,
            )

    def _active(self):
        return (
            self.enabled
            and not self._reported
            and self.warmup <= self._step < self.warmup + self.steps
        )

    def start(self, name):
        if not self._active():
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._open[name] = time.perf_counter()

    def stop(self, name):
        if not self._active() or name not in self._open:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - self._open.pop(name)) * 1000.0
        if name not in self._sum:
            self._sum[name] = 0.0
            self._count[name] = 0
            self._order.append(name)
        self._sum[name] += dt_ms
        self._count[name] += 1

    @contextlib.contextmanager
    def section(self, name):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def step_end(self, step_wall_ms=None):
        if not self.enabled or self._reported:
            return
        if self._active() and step_wall_ms is not None:
            self._step_total_ms += step_wall_ms
            self._step_total_n += 1
        self._step += 1
        if self._step >= self.warmup + self.steps:
            self.report()

    def report(self):
        if self._reported:
            return
        self._reported = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
        if not self._order:
            print("[StepProfiler] no sections were recorded", flush=True)
            return
        avg = {n: self._sum[n] / max(self._count[n], 1) for n in self._order}
        sum_sections = sum(avg.values())
        step_total = self._step_total_ms / self._step_total_n if self._step_total_n else 0.0
        ref = step_total if step_total > 0 else sum_sections
        bar = "=" * 68
        lines = [
            "",
            bar,
            f"[StepProfiler] mean per-step breakdown over {self._count[self._order[0]]} steps",
            "-" * 68,
            f"  {'section':<20s} {'mean ms':>11s} {'% of step':>12s} {'samples':>10s}",
        ]
        for n in self._order:
            pct = 100.0 * avg[n] / ref if ref > 0 else 0.0
            lines.append(f"  {n:<20s} {avg[n]:>11.3f} {pct:>11.2f}% {self._count[n]:>10d}")
        lines.append("-" * 68)
        lines.append(f"  {'sum of sections':<20s} {sum_sections:>11.3f}")
        if step_total > 0:
            unacc = step_total - sum_sections
            lines.append(f"  {'measured step wall':<20s} {step_total:>11.3f}")
            lines.append(
                f"  {'unaccounted':<20s} {unacc:>11.3f} {100.0 * unacc / step_total:>11.2f}%"
            )
        lines.append(bar)
        print("\n".join(lines), flush=True)


profiler = StepProfiler()
