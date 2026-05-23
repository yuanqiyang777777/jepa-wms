#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.simu_env_planning.planning.planning.csa.support_memory import SupportMemory


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSA-MPC support_memory.pt from pre-encoded tensors.")
    parser.add_argument("--states", required=True, help="Path to tensor or dict containing state encodings")
    parser.add_argument("--actions", required=True, help="Path to tensor containing aligned actions")
    parser.add_argument("--output", required=True, help="Path to write support_memory.pt")
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--max-memory", type=int, default=200000)
    parser.add_argument("--pca-fit-samples", type=int, default=5000)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--state-reduction", choices=["mean_tokens", "flatten"], default="mean_tokens")
    parser.add_argument("--action-space", default="model_normalized")
    parser.add_argument("--feature-source", default="pre_encoded")
    parser.add_argument("--frameskip", type=int, default=0)
    parser.add_argument("--action-skip", type=int, default=1)
    parser.add_argument("--metadata-json", default=None, help="Optional JSON object to store in metadata")
    args = parser.parse_args()

    states = torch.load(args.states, map_location="cpu", weights_only=False)
    actions = torch.load(args.actions, map_location="cpu", weights_only=False)
    metadata = json.loads(args.metadata_json) if args.metadata_json else {}
    memory = SupportMemory.from_tensors(
        states=states,
        actions=actions,
        pca_dim=args.pca_dim,
        max_memory=args.max_memory,
        state_reduction=args.state_reduction,
        pca_fit_samples=args.pca_fit_samples,
        sample_seed=args.sample_seed,
        action_space=args.action_space,
        feature_source=args.feature_source,
        frameskip=args.frameskip,
        action_skip=args.action_skip,
        metadata=metadata,
    )
    memory.save(args.output)


if __name__ == "__main__":
    main()
