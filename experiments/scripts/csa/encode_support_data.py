#!/usr/bin/env python
"""Encode a per-env training-split into (state, action) tensors for the CSA-MPC
support memory.

This is the missing companion to ``build_support_memory.py``: it reads an eval
YAML (so the visual transform + dataset registry exactly match what the
PlanEvaluator hook sees at scoring time), bootstraps the same EncPredWM model +
preprocessor as ``evals.simu_env_planning.eval.init_module``, iterates the
TRAINING slicer (not the 20-episode val split that ``make_datasets`` returns)
with single-frame slices, encodes obs through ``EncPredWM.encode``, mean-pools
to ``[B, D_state]``, and writes::

    <output_dir>/<env>_states.pt   # Tensor [N, D_state] (mean-pooled latents)
    <output_dir>/<env>_actions.pt  # Tensor [N, action_dim] (model-normalized)

Then::

    python experiments/scripts/csa/build_support_memory.py \
        --states <env>_states.pt --actions <env>_actions.pt \
        --output support_<env>.pt --pca-dim 64

produces the SupportMemory the offline analysis scorer queries.

CRITICAL — risk R2 from the diagnostic plan: the visual transform used here
MUST be the eval-time transform, otherwise ``M_real`` lives in a different
latent space than the runtime queries and every diagnostic is invalid. We
reuse ``evals.utils.make_transforms`` via the same config dict the eval reads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.plan_common.datasets.utils import init_data  # noqa: E402
from evals.simu_env_planning.eval import init_module  # noqa: E402
from evals.simu_env_planning.planning.planning.csa.support_memory import (  # noqa: E402
    compact_state_encoding,
)
from evals.utils import get_dataset_paths, make_transforms, make_inverse_transforms, Preprocessor  # noqa: E402


def _load_eval_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_env_vars(text: str) -> str:
    """Expand ``${VAR}`` from the process environment so config-file refs to
    ``${JEPAWM_CKPT}`` etc. resolve correctly during the encode pass."""
    return OmegaConf.to_container(OmegaConf.create({"v": text}), resolve=True)["v"]


def _resolve_paths_in_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve every string entry that contains ``${...}``. We only need this
    for ``checkpoint_folder``; OmegaConf does the rest implicitly."""
    if "checkpoint_folder" in cfg and isinstance(cfg["checkpoint_folder"], str):
        cfg["checkpoint_folder"] = _resolve_env_vars(cfg["checkpoint_folder"])
    return cfg


def build_train_slicer(cfg: Dict[str, Any], filter_first_episodes: int | None = None):
    """Re-implements ``evals.utils.make_datasets`` but returns the TRAIN slicer
    + the eval-time preprocessor. ``make_datasets`` discards the train split
    and forces ``filter_first_episodes=20`` (val), which is unsuitable for
    building a representative support memory.
    """
    cfgs_data = cfg["model_kwargs"]["data"]
    cfgs_data_aug = cfg["model_kwargs"]["data_aug"]

    img_size = cfgs_data["img_size"]
    transform = make_transforms(
        img_size=img_size,
        normalize=cfgs_data_aug["normalize"],
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    inverse_transform = make_inverse_transforms(img_size=img_size, **cfgs_data_aug)

    cfgs_validation = cfgs_data.get("validation", {})
    cfgs_loader = cfgs_data.get("loader", {})
    cfgs_custom = cfgs_data.get("custom", {})
    cfgs_droid = cfgs_data.get("droid", {})

    datasets = cfgs_data.get("datasets", [])
    dataset_paths = get_dataset_paths(datasets)
    val_datasets = cfgs_validation.get("val_datasets", [])
    val_dataset_paths = get_dataset_paths(val_datasets) if val_datasets else None

    excluded_keys = [
        "datasets",
        "val_datasets",
        "img_size",
        "validation",
        "loader",
        "custom",
        "droid",
    ]
    data_kwargs = {k: v for k, v in cfgs_data.items() if k not in excluded_keys}
    data_kwargs.update(cfgs_validation)
    data_kwargs.update(cfgs_loader)
    data_kwargs.update(cfgs_custom)
    data_kwargs.update(cfgs_droid)
    data_kwargs.update(
        {
            "data_paths": dataset_paths,
            "val_data_paths": val_dataset_paths,
            "transform": transform,
            "world_size": 1,
            "rank": 0,
            "num_workers": 0,
            "filter_first_episodes": filter_first_episodes,
            "output_rcasa_state": True,
            "output_rcasa_info": True,
            # Override the slicer window to a single-step pair (obs at t, action at t).
            "num_hist": 1,
            "num_pred": 0,
        }
    )

    (
        train_slicer,
        _val_slicer,
        train_traj_dset,
        _val_traj_dset,
        *_extras,
    ) = init_data(**data_kwargs)

    preprocessor = Preprocessor(
        action_mean=train_traj_dset.action_mean,
        action_std=train_traj_dset.action_std,
        state_mean=train_traj_dset.state_mean,
        state_std=train_traj_dset.state_std,
        proprio_mean=train_traj_dset.proprio_mean,
        proprio_std=train_traj_dset.proprio_std,
        transform=transform,
        inverse_transform=inverse_transform,
    )
    return train_slicer, train_traj_dset, preprocessor


def encode_train_split(
    *,
    cfg: Dict[str, Any],
    max_samples: int,
    batch_size: int,
    filter_first_episodes: int | None,
    device: torch.device,
    seed: int,
):
    """Bootstrap model + train slicer, walk the slicer, and return pooled
    ``[N, D_state]`` + ``[N, A]`` tensors. ``N <= max_samples``."""
    cfg = _resolve_paths_in_cfg(dict(cfg))  # shallow copy + resolve

    train_slicer, _train_traj_dset, preprocessor = build_train_slicer(
        cfg, filter_first_episodes=filter_first_episodes
    )

    model_kwargs = cfg["model_kwargs"]
    module_name = model_kwargs["module_name"]
    pretrain_kwargs = model_kwargs.get("pretrain_kwargs", {})
    wrapper_kwargs = model_kwargs.get("wrapper_kwargs", {})
    cfgs_data = model_kwargs["data"]
    checkpoint = model_kwargs.get("checkpoint")
    checkpoint_folder = cfg.get("checkpoint_folder", "")
    model = init_module(
        folder=checkpoint_folder,
        checkpoint=checkpoint,
        module_name=module_name,
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        cfgs_data=cfgs_data,
        device=device,
        action_dim=train_slicer.action_dim,
        proprio_dim=train_slicer.proprio_dim,
        preprocessor=preprocessor,
    )
    model.eval()

    gen = torch.Generator().manual_seed(int(seed))
    total = len(train_slicer)
    n_take = min(int(max_samples), total)
    indices = torch.randperm(total, generator=gen)[:n_take].tolist()
    print(f"Encoding {n_take}/{total} train slices from train_slicer …", flush=True)

    state_chunks = []
    action_chunks = []
    with torch.no_grad():
        for batch_start in range(0, n_take, batch_size):
            batch_idx = indices[batch_start : batch_start + batch_size]
            batch_obs_visual = []
            batch_obs_proprio = []
            batch_actions = []
            for idx in batch_idx:
                obs, act, _state, _reward = train_slicer[idx]
                # obs[visual]: (1, C, H, W), obs[proprio]: (1, P), act: (1, A)
                batch_obs_visual.append(obs["visual"])
                batch_obs_proprio.append(obs["proprio"])
                batch_actions.append(act)
            visual = torch.stack(batch_obs_visual).to(device)  # (B, 1, C, H, W)
            proprio = torch.stack(batch_obs_proprio).to(device)  # (B, 1, P)
            actions = torch.stack(batch_actions).to(device)  # (B, 1, A)

            obs_td = {"visual": visual, "proprio": proprio}
            enc = model.encode(obs_td, act=False)
            pooled = compact_state_encoding(enc, reduction="mean_tokens")  # (B, D_state)
            state_chunks.append(pooled.detach().cpu())
            action_chunks.append(actions.squeeze(1).detach().cpu())

            if (batch_start // batch_size) % 10 == 0:
                done = batch_start + len(batch_idx)
                print(f"  encoded {done}/{n_take}", flush=True)

    states = torch.cat(state_chunks, dim=0)
    actions = torch.cat(action_chunks, dim=0)
    return states, actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-config", required=True, help="Path to an eval YAML (e.g. wall/jepa-wm/...ep96_decode.yaml)")
    parser.add_argument("--env", required=True, help="Env tag for the output filenames (wall, pusht, maze, mw-reach, mw-reach-wall)")
    parser.add_argument("--output-dir", required=True, help="Directory to write <env>_states.pt + <env>_actions.pt")
    parser.add_argument("--max-samples", type=int, default=200_000, help="Cap on the number of encoded train slices")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--filter-first-episodes",
        type=int,
        default=None,
        help="If set, cap the training split to the first N episodes (otherwise all).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg_path = Path(args.eval_config)
    cfg = _load_eval_yaml(cfg_path)
    device = torch.device(args.device)

    states, actions = encode_train_split(
        cfg=cfg,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        filter_first_episodes=args.filter_first_episodes,
        device=device,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states_path = output_dir / f"{args.env}_states.pt"
    actions_path = output_dir / f"{args.env}_actions.pt"
    torch.save(states, states_path)
    torch.save(actions, actions_path)
    print(f"Wrote {states_path} (shape={tuple(states.shape)})")
    print(f"Wrote {actions_path} (shape={tuple(actions.shape)})")


if __name__ == "__main__":
    main()
