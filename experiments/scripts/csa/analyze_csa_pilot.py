#!/usr/bin/env python
"""Run the full CSA-MPC diagnostic suite (诊断 0-4) on a directory of per-episode
dumps written by the PlanEvaluator hook.

The dump layout is whatever the eval emits:
    <dump_dir>/.../ep_<N>/csa_dumps/csa_diag_dump.pt
We rely on ``load_episode_dumps`` to recurse-and-collect every match.

This script is the offline half of the diagnostic phase: changing ``k``,
``beta``, the support memory or any diagnostic threshold never requires
re-running the (multi-hour) eval — only this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.simu_env_planning.planning.planning.csa.analysis import (
    analyze_dump_dir,
    write_report_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", required=True, help="Directory containing per-episode csa_diag_dump.pt files")
    parser.add_argument("--memory-path", required=True, help="Path to support_<env>.pt built by build_support_memory.py")
    parser.add_argument("--output", required=True, help="Path to write the diagnostic JSON report")
    parser.add_argument("--state-k", type=int, default=64, help="kNN neighbors for state support score")
    parser.add_argument("--action-k", type=int, default=64, help="kNN neighbors for conditional action support score")
    parser.add_argument("--beta", type=float, default=1.0, help="Conditional-support exponent")
    parser.add_argument(
        "--gate-min-ratio",
        type=float,
        default=0.10,
        help="Advisory threshold for the 诊断 0 quadrant-fraction Go/No-Go gate (memo §6).",
    )
    parser.add_argument(
        "--state-familiar-quantile",
        type=float,
        default=0.50,
        help="Quantile defining 'state-familiar' for diagnostics 2 and 3 (default median).",
    )
    parser.add_argument(
        "--action-unsupported-quantile",
        type=float,
        default=0.75,
        help="Quantile defining 'unsupported action' for diagnostic 0 (default 0.75).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="kNN chunk size (memory/throughput knob; correctness-invariant).",
    )
    args = parser.parse_args()

    report = analyze_dump_dir(
        dump_dir=args.dump_dir,
        memory_path=args.memory_path,
        state_k=args.state_k,
        action_k=args.action_k,
        beta=args.beta,
        chunk_size=args.chunk_size,
        gate_min_ratio=args.gate_min_ratio,
        state_familiar_quantile=args.state_familiar_quantile,
        action_unsupported_quantile=args.action_unsupported_quantile,
    )
    write_report_json(report, args.output)
    print(f"Wrote diagnostic report → {args.output}")


if __name__ == "__main__":
    sys.exit(main())
