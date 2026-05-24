#!/usr/bin/env python
"""Run the CSA support-geometry sensitivity analysis for one env's dumps.

Builds 6 (PCA dim, whitening, calibration) variants of the support memory
from the raw 384-d pooled-visual + raw action tensors, re-scores all queries
from an existing dump directory under each variant, and writes a single JSON
report describing the full Q1-Q4 quadrant distribution per variant.

DOES NOT touch the existing diag-0-6 reports or raw eval dumps -- writes only
to ``--output`` (default ``<env>_geometry_sensitivity_report.json``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.simu_env_planning.planning.planning.csa.diagnostic_recorder import (
    load_episode_dumps,
)
from evals.simu_env_planning.planning.planning.csa.geometry_sensitivity import (
    DEFAULT_VARIANTS,
    run_geometry_sensitivity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="env tag, used for the report 'env' field if dumps don't carry one")
    parser.add_argument("--dump-dir", required=True, help="directory with per-episode csa_diag_dump.pt files")
    parser.add_argument("--raw-states", required=True, help="raw <env>_states.pt (pooled 384-d visual)")
    parser.add_argument("--raw-actions", required=True, help="raw <env>_actions.pt")
    parser.add_argument("--output", required=True, help="output JSON path")
    parser.add_argument("--state-k", type=int, default=64)
    parser.add_argument("--action-k", type=int, default=64)
    parser.add_argument("--state-familiar-quantile", type=float, default=0.50)
    parser.add_argument("--action-unsupported-quantile", type=float, default=0.75)
    parser.add_argument("--pca-fit-samples", type=int, default=50_000,
                        help="cap rows used to fit PCA; full memory still projected")
    parser.add_argument("--device", default="cpu",
                        help="cpu (default) or cuda:N. The sweep is small; CPU is usually faster than the GPU-launch overhead.")
    args = parser.parse_args()

    dumps = load_episode_dumps(args.dump_dir)
    if not dumps:
        sys.stderr.write(f"ERROR: no dumps under {args.dump_dir}\n")
        return 1
    # Stamp env onto every dump if missing -- some legacy dumps may lack the field.
    for d in dumps:
        d.setdefault("env", args.env)

    raw_states = torch.load(args.raw_states, map_location="cpu", weights_only=False)
    raw_actions = torch.load(args.raw_actions, map_location="cpu", weights_only=False)
    if isinstance(raw_states, dict):
        # Defensive: some legacy encoders wrap tensors in a dict.
        raw_states = raw_states.get("states", raw_states.get("state_features"))
    if isinstance(raw_actions, dict):
        raw_actions = raw_actions.get("actions")
    if raw_states is None or raw_actions is None:
        sys.stderr.write("ERROR: raw_states / raw_actions could not be unwrapped\n")
        return 1

    report = run_geometry_sensitivity(
        raw_support_states=raw_states,
        raw_support_actions=raw_actions,
        dumps=dumps,
        variants=DEFAULT_VARIANTS,
        state_k=args.state_k,
        action_k=args.action_k,
        state_familiar_quantile=args.state_familiar_quantile,
        action_unsupported_quantile=args.action_unsupported_quantile,
        pca_fit_samples=args.pca_fit_samples,
        device=args.device,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote geometry-sensitivity report -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
