#!/usr/bin/env python
"""Encode a per-env VAL split into (state, action, residual) tensors for the
CSA-MPC local residual-calibration memory (``M_resid``).

This is the residual analogue of ``encode_support_data.py``: it bootstraps the
same EncPredWM + preprocessor as the eval pipeline (R2: shared transforms +
shared encoder), walks the held-out VAL slicer with ``num_hist=1, num_pred=1``
so each slice is one transition ``(o_t, a_t, o_{t+1})``, runs::

    x_t       = encode(o_t)
    x_hat_t+1 = unroll(x_t, [a_t])[-1]    # one-step latent prediction
    x_t+1     = encode(o_{t+1})
    e_i       = mean( (x_hat_t+1 - x_t+1) ** 2 )     # pooled-latent MSE,
                                                       # SAME metric as the dumps'
                                                       # per-depth rollout_error
                                                       # (see diagnostic_recorder.py)

and writes three aligned tensors::

    <output_dir>/<env>_residual_states.pt    # [N, D_state]   raw pooled x_t
    <output_dir>/<env>_residual_actions.pt   # [N, A]         raw action a_t
    <output_dir>/<env>_residual_errors.pt    # [N]            scalar e_i

Then the residual-correlation analysis loads (states, actions, errors), builds
geometry-variant memories (raw + PCA-64) parallel to the existing M_real
sweep, and compares ``r(q)`` against ``U_action|x`` / ``U_SA``.

CRITICAL -- R2 (encoder/transform consistency): like ``encode_support_data.py``,
this script reuses ``init_module`` and the eval config's ``make_transforms``
so that ``M_resid`` lives in the SAME latent space as the runtime queries the
dumps were recorded against; otherwise every correlation is meaningless.

VAL slicer (not TRAIN): we want real generalization residuals -- the model has
already seen the training transitions during pre-training, so building M_resid
from the training split would under-estimate ``e_i`` and bias ``r(q)`` low. The
held-out val split is the natural choice. ``make_datasets`` hard-codes
``filter_first_episodes=20`` for val; we accept that small size for now and
revisit if r(q) is too noisy on the sparser envs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from tensordict.tensordict import TensorDict

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.plan_common.datasets.utils import init_data  # noqa: E402
from evals.simu_env_planning.eval import init_module  # noqa: E402
from evals.simu_env_planning.planning.planning.csa.support_memory import (  # noqa: E402
    compact_state_encoding,
)
from evals.utils import (  # noqa: E402
    get_dataset_paths,
    make_inverse_transforms,
    make_transforms,
    Preprocessor,
)


def _load_eval_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_paths_in_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "checkpoint_folder" in cfg and isinstance(cfg["checkpoint_folder"], str):
        cfg["checkpoint_folder"] = os.path.expandvars(cfg["checkpoint_folder"])
    return cfg


def build_val_slicer(cfg: Dict[str, Any], num_pred: int = 1, filter_first_episodes: int | None = None):
    """Re-implements ``evals.utils.make_datasets`` to expose the VAL slicer +
    val_traj_dset with a 2-frame window (``num_hist=1, num_pred=num_pred``).

    Unlike ``encode_support_data.build_train_slicer``, this returns the *val*
    slicer so M_resid lives on the held-out split. ``filter_first_episodes``
    is forwarded to init_data; passing ``None`` keeps whatever val split the
    dataset registry exposes (typically ~20 episodes).
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
            # Two-frame window: t=0 context + t=1 target, one action between.
            "num_hist": 1,
            "num_pred": int(num_pred),
        }
    )

    (
        train_slicer,
        val_slicer,
        train_traj_dset,
        val_traj_dset,
        *_extras,
    ) = init_data(**data_kwargs)

    # Fall back to train_slicer if val_slicer is empty/missing (some env
    # registries only expose val via the same dataset path).
    slicer = val_slicer if val_slicer is not None and len(val_slicer) > 0 else train_slicer
    traj_dset = val_traj_dset if val_traj_dset is not None else train_traj_dset

    preprocessor = Preprocessor(
        action_mean=traj_dset.action_mean,
        action_std=traj_dset.action_std,
        state_mean=traj_dset.state_mean,
        state_std=traj_dset.state_std,
        proprio_mean=traj_dset.proprio_mean,
        proprio_std=traj_dset.proprio_std,
        transform=transform,
        inverse_transform=inverse_transform,
    )
    return slicer, traj_dset, preprocessor


def _pool_state(enc: Any) -> torch.Tensor:
    """Compact a TensorDict-or-Tensor latent to ``[B, D_state]`` using the
    same mean-tokens reduction the existing SupportMemory + dumps use."""
    return compact_state_encoding(enc, reduction="mean_tokens")


def _split_visual_proprio(enc: Any, t_start: int, t_end: int) -> Any:
    """Slice timesteps ``[t_start:t_end]`` out of a TensorDict-or-Tensor
    latent. For TensorDict with visual ``[B, T, V, H, W, D]`` + proprio
    ``[B, T, ...]`` returns a TensorDict with T sliced; for a bare Tensor
    returns the same.

    NOTE: returns a TensorDict (not a plain dict) when the input has
    visual/proprio keys -- ``EncPredWM.unroll`` only honours TensorDict's
    return-type branch, so passing a plain dict back into unroll would
    silently return None.
    """
    if isinstance(enc, dict) or hasattr(enc, "keys"):
        visual = enc["visual"][:, t_start:t_end]
        proprio = enc["proprio"][:, t_start:t_end]
        return TensorDict({"visual": visual, "proprio": proprio}, device=visual.device)
    return enc[:, t_start:t_end]


def encode_val_transitions(
    *,
    cfg: Dict[str, Any],
    max_samples: int,
    batch_size: int,
    filter_first_episodes: int | None,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bootstrap model + val slicer, walk the slicer, return three aligned
    tensors: pooled state at t, raw action at t, scalar residual e_i."""
    cfg = _resolve_paths_in_cfg(dict(cfg))

    slicer, traj_dset, preprocessor = build_val_slicer(
        cfg, num_pred=1, filter_first_episodes=filter_first_episodes
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
        action_dim=traj_dset.action_dim,
        proprio_dim=traj_dset.proprio_dim,
        preprocessor=preprocessor,
    )
    model.eval()

    gen = torch.Generator().manual_seed(int(seed))
    total = len(slicer)
    n_take = min(int(max_samples), total)
    indices = torch.randperm(total, generator=gen)[:n_take].tolist()
    print(f"Encoding {n_take}/{total} val transitions ...", flush=True)

    state_chunks: list[torch.Tensor] = []
    action_chunks: list[torch.Tensor] = []
    error_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_start in range(0, n_take, batch_size):
            batch_idx = indices[batch_start : batch_start + batch_size]
            batch_obs_visual: list[torch.Tensor] = []
            batch_obs_proprio: list[torch.Tensor] = []
            batch_actions: list[torch.Tensor] = []
            for idx in batch_idx:
                obs, act, _state, _reward = slicer[idx]
                # With num_hist=1, num_pred=1 the slicer returns two-frame
                # obs (t and t+1) and a one-step action between them.
                # obs["visual"]: (2, C, H, W); obs["proprio"]: (2, P);
                # act: (1, A).
                if obs["visual"].shape[0] < 2 or act.shape[0] < 1:
                    # Defensive: drop short slices at trajectory boundaries.
                    continue
                batch_obs_visual.append(obs["visual"][:2])
                batch_obs_proprio.append(obs["proprio"][:2])
                batch_actions.append(act[:1])
            if not batch_obs_visual:
                continue
            visual = torch.stack(batch_obs_visual).to(device)   # (B, 2, C, H, W)
            proprio = torch.stack(batch_obs_proprio).to(device) # (B, 2, P)
            actions = torch.stack(batch_actions).to(device)     # (B, 1, A)

            obs_td = {"visual": visual, "proprio": proprio}
            enc = model.encode(obs_td, act=False)
            # enc["visual"]: (B, 2, V, H, W, D); enc["proprio"]: (B, 2, ..., D)

            # Pool both frames separately (needed for the target).
            x_ctxt_pool = _pool_state(_split_visual_proprio(enc, 0, 1))   # (B, D_state)
            x_target_pool = _pool_state(_split_visual_proprio(enc, 1, 2)) # (B, D_state)

            # One-step latent prediction. unroll expects act_suffix in
            # ``[T, B, A]`` (T == prediction horizon == 1 here).
            ctxt_td = _split_visual_proprio(enc, 0, 1)
            act_suffix = actions.transpose(0, 1).contiguous()  # (1, B, A)
            try:
                pred = model.unroll(ctxt_td, act_suffix=act_suffix)
            except Exception as exc:
                print(f"unroll failed on batch_start={batch_start}: {exc!r}", flush=True)
                raise
            # unroll returns TensorDict with visual/proprio in (T+tau, B, ...)
            # The predicted next frame is the LAST timestep (index 1 here).
            pred_visual_last = pred["visual"][-1:]    # (1, B, V, H, W, D)
            pred_proprio_last = pred["proprio"][-1:]  # (1, B, ..., D)
            pred_visual_last = pred_visual_last.transpose(0, 1)    # (B, 1, ...)
            pred_proprio_last = pred_proprio_last.transpose(0, 1)  # (B, 1, ...)
            x_pred_pool = _pool_state({
                "visual": pred_visual_last,
                "proprio": pred_proprio_last,
            })  # (B, D_state)

            # Residual = pooled-latent MSE = SAME metric as
            # diagnostic_recorder dumps' rollout_error. See
            # analysis.score_dumps for the (imagined - real)**2 .mean()
            # computation per depth.
            e_i = ((x_pred_pool - x_target_pool) ** 2).mean(dim=-1)  # (B,)

            state_chunks.append(x_ctxt_pool.detach().cpu())
            action_chunks.append(actions.squeeze(1).detach().cpu())
            error_chunks.append(e_i.detach().cpu())

            if (batch_start // batch_size) % 10 == 0:
                done = batch_start + len(batch_idx)
                print(f"  encoded {done}/{n_take} (mean e_i so far = {float(torch.cat(error_chunks).mean()):.4f})",
                      flush=True)

    if not state_chunks:
        raise RuntimeError("Encoded zero residual transitions; check slicer / num_pred config.")

    states = torch.cat(state_chunks, dim=0)
    actions = torch.cat(action_chunks, dim=0)
    errors = torch.cat(error_chunks, dim=0)
    return states, actions, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-config", required=True,
                        help="Path to an eval YAML; same one used by encode_support_data.py.")
    parser.add_argument("--env", required=True,
                        help="Env tag for output filenames (wall, pusht, maze, mw-reach, mw-reach-wall).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write <env>_residual_{states,actions,errors}.pt.")
    parser.add_argument("--max-samples", type=int, default=100_000,
                        help="Cap on the number of encoded val transitions.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--filter-first-episodes", type=int, default=None,
                        help="If set, cap the val split to the first N episodes (otherwise the registry's default).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = _load_eval_yaml(Path(args.eval_config))
    device = torch.device(args.device)

    states, actions, errors = encode_val_transitions(
        cfg=cfg,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        filter_first_episodes=args.filter_first_episodes,
        device=device,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states_path = output_dir / f"{args.env}_residual_states.pt"
    actions_path = output_dir / f"{args.env}_residual_actions.pt"
    errors_path = output_dir / f"{args.env}_residual_errors.pt"
    torch.save(states, states_path)
    torch.save(actions, actions_path)
    torch.save(errors, errors_path)
    print(
        f"Wrote {states_path} (shape={tuple(states.shape)})\n"
        f"Wrote {actions_path} (shape={tuple(actions.shape)})\n"
        f"Wrote {errors_path} (shape={tuple(errors.shape)}, mean={float(errors.mean()):.4f})"
    )


if __name__ == "__main__":
    main()
