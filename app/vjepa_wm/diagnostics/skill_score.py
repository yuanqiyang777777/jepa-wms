# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MGVT persistence skill diagnostic.

This script evaluates latent prediction on the validation split and compares
the model to the copy-last-observed-frame baseline. It intentionally does not
run CEM, planning, or any Terver protocol evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn


def _as_path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def _nested_get(mapping: dict[str, Any], keys: list[str], default=None):
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_checkpoint(config: dict[str, Any], checkpoint_arg: str | None) -> tuple[Path, str]:
    if checkpoint_arg:
        ckpt = Path(checkpoint_arg).expanduser().resolve()
        return ckpt.parent, ckpt.name

    folder = config.get("checkpoint_folder") or config.get("folder")
    if not folder:
        raise ValueError("--checkpoint is required when config has neither checkpoint_folder nor folder")
    folder = Path(folder).expanduser().resolve()
    for name in ("jepa-latest.pth.tar", "checkpoint.pth.tar"):
        candidate = folder / name
        if candidate.exists():
            return folder, name
    raise FileNotFoundError(f"No checkpoint found in {folder}; pass --checkpoint explicitly")


def _build_val_loader_and_preprocessor(
    config: dict[str, Any],
    batch_size: int | None,
    num_workers: int | None,
    min_eval_frames: int | None = None,
):
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
    from app.plan_common.datasets.utils import init_data
    from src.datasets.utils.utils import get_dataset_paths

    cfgs_data = config["data"]
    cfgs_data_aug = config.get("data_aug", {})
    cfgs_validation = cfgs_data.get("validation", {})
    cfgs_loader = cfgs_data.get("loader", {})
    cfgs_custom = cfgs_data.get("custom", {})
    cfgs_droid = cfgs_data.get("droid", {})

    if min_eval_frames is not None:
        num_hist = int(cfgs_custom.get("num_hist", 1))
        cfgs_custom["num_pred"] = max(int(cfgs_custom.get("num_pred", 1)), max(0, min_eval_frames - num_hist))
        cfgs_validation["num_frames_val"] = max(int(cfgs_validation.get("num_frames_val", 0)), min_eval_frames)

    transform = make_transforms(img_size=cfgs_data.get("img_size", 224), **cfgs_data_aug)
    inverse_transform = make_inverse_transforms(img_size=cfgs_data.get("img_size", 224), **cfgs_data_aug)

    dataset_type = cfgs_data.get("dataset_type", "custom")
    datasets = cfgs_data.get("datasets", [])
    data_paths = datasets if dataset_type.lower() == "mixed_dataset" else get_dataset_paths(datasets)

    val_datasets = cfgs_validation.get("val_datasets", [])
    val_data_paths = get_dataset_paths(val_datasets) if val_datasets else None

    excluded_keys = {
        "datasets",
        "val_datasets",
        "img_size",
        "validation",
        "loader",
        "custom",
        "droid",
    }
    data_kwargs = {k: v for k, v in cfgs_data.items() if k not in excluded_keys}
    data_kwargs.update(cfgs_validation)
    data_kwargs.update(cfgs_loader)
    data_kwargs.update(cfgs_custom)
    data_kwargs.update(cfgs_droid)
    data_kwargs.update(
        {
            "data_paths": data_paths,
            "val_data_paths": val_data_paths,
            "transform": transform,
            "world_size": 1,
            "rank": 0,
            "filter_first_episodes": cfgs_custom.get("filter_first_episodes"),
        }
    )
    if batch_size is not None:
        data_kwargs["val_dataset_batch_size"] = batch_size
    if num_workers is not None:
        data_kwargs["num_workers"] = num_workers
        data_kwargs["persistent_workers"] = num_workers > 0 and bool(data_kwargs.get("persistent_workers", True))

    (
        _dataset,
        val_dataset,
        traj_dataset,
        val_traj_dataset,
        _train_loader,
        val_loader,
        _sampler,
        _viz_loader,
    ) = init_data(**data_kwargs)

    preprocessor = Preprocessor(
        action_mean=traj_dataset.action_mean,
        action_std=traj_dataset.action_std,
        state_mean=traj_dataset.state_mean,
        state_std=traj_dataset.state_std,
        proprio_mean=traj_dataset.proprio_mean,
        proprio_std=traj_dataset.proprio_std,
        transform=transform,
        inverse_transform=inverse_transform,
    )
    return val_dataset, val_traj_dataset, val_loader, traj_dataset, preprocessor


def _move_obs_to_device(obs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, dtype=torch.float32, non_blocking=True) for k, v in obs.items()}


def _count_predictor_params(model) -> int:
    predictor = getattr(model.model.predictor, "module", model.model.predictor)
    action_encoder = getattr(model.model.action_encoder, "module", model.model.action_encoder)
    proprio_encoder = getattr(model.model.proprio_encoder, "module", model.model.proprio_encoder)
    modules = [m for m in (predictor, action_encoder, proprio_encoder) if m is not None]
    return int(sum(p.numel() for module in modules for p in module.parameters()))


def _count_inference_path_params(model) -> int | None:
    predictor = getattr(model.model.predictor, "module", model.model.predictor)
    estimator = getattr(predictor, "estimate_inference_path_params", None)
    if estimator is None:
        return None
    return int(estimator())


def _count_linear_flops(module: nn.Module, token_count: int, excluded: set[int] | None = None) -> int:
    excluded = excluded or set()
    flops = 0
    for layer in module.modules():
        if isinstance(layer, nn.Linear) and id(layer) not in excluded:
            flops += int(2 * token_count * layer.in_features * layer.out_features)
    return flops


def _estimate_adaln_flops(predictor: nn.Module, batch_size: int, seq_len: int | None) -> int | None:
    blocks = getattr(predictor, "predictor_blocks", None)
    if blocks is None or not all(hasattr(block, "adaLN_modulation") for block in blocks):
        return None
    grid_height = getattr(predictor, "grid_height", None)
    grid_width = getattr(predictor, "grid_width", None)
    if grid_height is None or grid_width is None:
        return None

    seq_len = int(seq_len or max(1, getattr(predictor, "grid_depth", 1)))
    tokens = int(batch_size * seq_len * grid_height * grid_width)
    step_tokens = int(batch_size * seq_len)
    adaln_linear_ids = {
        id(layer)
        for block in blocks
        for layer in block.adaLN_modulation.modules()
        if isinstance(layer, nn.Linear)
    }

    flops = _count_linear_flops(predictor, tokens, excluded=adaln_linear_ids)
    for block in blocks:
        flops += _count_linear_flops(block.adaLN_modulation, step_tokens)
    return int(flops)


def _estimate_predictor_flops(model, batch_size: int = 1, seq_len: int | None = None) -> int | None:
    predictor = getattr(model.model.predictor, "module", model.model.predictor)
    action_encoder = getattr(model.model.action_encoder, "module", model.model.action_encoder)
    proprio_encoder = getattr(model.model.proprio_encoder, "module", model.model.proprio_encoder)
    estimate = getattr(predictor, "estimate_flops_per_forward", None)
    if estimate is not None:
        flops = int(estimate(batch_size=batch_size, seq_len=seq_len))
    else:
        flops = _estimate_adaln_flops(predictor, batch_size=batch_size, seq_len=seq_len)
        if flops is None:
            return None

    step_tokens = int(batch_size * int(seq_len or 1))
    for encoder in (action_encoder, proprio_encoder):
        if encoder is not None:
            flops += _count_linear_flops(encoder, step_tokens)
    return int(flops)


def _mamba_backend(model) -> str | None:
    predictor = getattr(model.model, "predictor", None)
    predictor = getattr(predictor, "module", predictor)
    direct_backend = getattr(predictor, "mamba_backend", None)
    if direct_backend:
        return str(direct_backend)
    blocks = getattr(predictor, "predictor_blocks", [])
    backends = []
    for block in blocks:
        mixer = getattr(block, "mixer", None)
        backend = getattr(mixer, "backend", None)
        if backend:
            backends.append(backend)
    return ",".join(sorted(set(backends))) if backends else None


def _read_train_step_time_ms(csv_path: Path | None) -> float | None:
    if csv_path is None or not csv_path.exists():
        return None
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        for candidate in ("gpu-time(ms)", "iter-time(ms)"):
            if candidate in fields:
                values = []
                for row in reader:
                    raw = row.get(candidate)
                    if raw:
                        try:
                            values.append(float(raw))
                        except ValueError:
                            pass
                return float(sum(values) / len(values)) if values else None
    return None


def _safe_skill(model_mse: float, persist_mse: float) -> float | None:
    if persist_mse <= 0.0:
        return None
    return 1.0 - (model_mse / persist_mse)


def _finalize_horizon_metrics(
    model_sums: dict[int, float],
    persist_sums: dict[int, float],
    counts: dict[int, int],
    prefix: str,
) -> dict[str, dict[str, float | None]]:
    model_key = f"{prefix}_mse_model"
    persist_key = f"{prefix}_mse_persist"
    skill_key = f"{prefix}_skill"
    out: dict[str, dict[str, float | None]] = {
        model_key: {},
        persist_key: {},
        skill_key: {},
    }
    for horizon in sorted(counts):
        key = str(horizon)
        count = counts[horizon]
        model_mse = model_sums[horizon] / count if count > 0 else None
        persist_mse = persist_sums[horizon] / count if count > 0 else None
        out[model_key][key] = model_mse
        out[persist_key][key] = persist_mse
        out[skill_key][key] = (
            _safe_skill(model_mse, persist_mse)
            if model_mse is not None and persist_mse is not None
            else None
        )
    return out


def _add_patch_mse(
    model_patch_mse: torch.Tensor,
    persist_patch_mse: torch.Tensor,
    horizon: int,
    model_sums: dict[int, float],
    persist_sums: dict[int, float],
    counts: dict[int, int],
) -> None:
    model_sums[horizon] += float(model_patch_mse.sum().detach().cpu())
    persist_sums[horizon] += float(persist_patch_mse.sum().detach().cpu())
    counts[horizon] += int(model_patch_mse.numel())


def _dilate_square_mask(mask: torch.Tensor, grid_size: int, iterations: int = 1) -> torch.Tensor:
    if iterations <= 0:
        return mask
    if mask.shape[-1] != grid_size * grid_size:
        return mask
    import torch.nn.functional as F

    flat_shape = mask.shape
    mask_2d = mask.reshape(-1, 1, grid_size, grid_size).float()
    for _ in range(iterations):
        mask_2d = F.max_pool2d(mask_2d, kernel_size=3, stride=1, padding=1)
    return mask_2d.reshape(flat_shape).bool()


def _make_moved_patch_mask(
    persist_patch_mse: torch.Tensor,
    top_frac: float = 0.20,
    min_floor: float = 0.0,
    grid_size: int | None = None,
    dilation: int = 0,
) -> torch.Tensor:
    """Model-independent moved-region mask from persistence error.

    ``persist_patch_mse`` is target-side motion relative to the observed prefix;
    it never reads model predictions.
    """
    if not 0.0 < top_frac <= 1.0:
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}")
    k = max(1, int(round(persist_patch_mse.shape[-1] * top_frac)))
    threshold = persist_patch_mse.topk(k=k, dim=-1).values[..., -1:]
    mask = persist_patch_mse >= threshold
    if min_floor > 0.0:
        mask = mask & (persist_patch_mse >= min_floor)
    if grid_size is not None:
        mask = _dilate_square_mask(mask, grid_size=grid_size, iterations=dilation)
    return mask


def _add_masked_patch_mse(
    model_patch_mse: torch.Tensor,
    persist_patch_mse: torch.Tensor,
    mask: torch.Tensor,
    horizon: int,
    model_sums: dict[int, float],
    persist_sums: dict[int, float],
    counts: dict[int, int],
    denominator_floor: float = 0.0,
) -> None:
    valid = mask & (persist_patch_mse >= denominator_floor)
    if not valid.any():
        return
    _add_patch_mse(
        model_patch_mse[valid],
        persist_patch_mse[valid],
        horizon,
        model_sums,
        persist_sums,
        counts,
    )


def _make_markdown(results: dict[str, Any]) -> str:
    if "skill_by_horizon" in results:
        rows = [
            "| Horizon | Skill | Change skill | Moved change skill | Proprio skill | MSE model | MSE persist |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for horizon in sorted(results["skill_by_horizon"], key=lambda h: int(h)):
            skill = results["skill_by_horizon"].get(horizon)
            change_skill = results["change_skill_by_horizon"].get(horizon)
            moved_skill = results.get("moved_region_change_skill_by_horizon", {}).get(horizon)
            proprio_skill = results["proprio_skill_by_horizon"].get(horizon)
            mse_model = results["mse_model_by_horizon"].get(horizon)
            mse_persist = results["mse_persist_by_horizon"].get(horizon)
            rows.append(
                "| "
                f"{horizon} | "
                f"{'n/a' if skill is None else f'{skill:.4f}'} | "
                f"{'n/a' if change_skill is None else f'{change_skill:.4f}'} | "
                f"{'n/a' if moved_skill is None else f'{moved_skill:.4f}'} | "
                f"{'n/a' if proprio_skill is None else f'{proprio_skill:.4f}'} | "
                f"{'n/a' if mse_model is None else f'{mse_model:.6g}'} | "
                f"{'n/a' if mse_persist is None else f'{mse_persist:.6g}'} |"
            )
        return "\n".join(rows) + "\n"

    skill = "n/a" if results["skill"] is None else f"{results['skill']:.4f}"
    change_skill = "n/a" if results["change_skill"] is None else f"{results['change_skill']:.4f}"
    rows = [
        "| Model | Task | Seed | MSE model | MSE persist | Skill | Change skill | Params | Step ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {results['model']} | {results['task']} | {results['seed']} | "
            f"{results['mse_model']:.6g} | {results['mse_persist']:.6g} | "
            f"{skill} | {change_skill} | "
            f"{results['param_count']} | "
            f"{results['train_step_time_ms']:.3f} |"
            if results["train_step_time_ms"] is not None
            else f"| {results['model']} | {results['task']} | {results['seed']} | "
            f"{results['mse_model']:.6g} | {results['mse_persist']:.6g} | "
            f"{skill} | {change_skill} | "
            f"{results['param_count']} | n/a |"
        ),
    ]
    return "\n".join(rows) + "\n"


@torch.no_grad()
def run_skill_score(args: argparse.Namespace) -> dict[str, Any]:
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module
    from src.utils.yaml_utils import load_yaml

    config = load_yaml(args.config)
    device = torch.device(args.device)
    checkpoint_folder, checkpoint_name = _resolve_checkpoint(config, args.checkpoint)
    ctxt_window = _nested_get(config, ["model", "rollout_cfg", "ctxt_window_train_rollout"], 2)
    prefix = args.prefix if args.prefix is not None else max(0, ctxt_window - 1)
    max_horizon = max(1, int(args.max_horizon))
    min_eval_frames = prefix + max_horizon + 1

    val_dataset, _val_traj_dataset, val_loader, traj_dataset, preprocessor = _build_val_loader_and_preprocessor(
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_eval_frames=min_eval_frames,
    )

    model = init_module(
        folder=checkpoint_folder,
        checkpoint=checkpoint_name,
        model_kwargs=config["model"],
        device=device,
        action_dim=traj_dataset.action_dim,
        proprio_dim=traj_dataset.proprio_dim,
        preprocessor=preprocessor,
        cfgs_data=config["data"],
        wrapper_kwargs={"ctxt_window": ctxt_window},
    )
    model.eval()

    model_name = _nested_get(config, ["model", "predictor", "pred_type"], "unknown")
    task = ",".join(config["data"].get("datasets", []))
    filter_tasks = _nested_get(config, ["data", "custom", "filter_tasks"], None)
    if filter_tasks:
        task = f"{task}:{','.join(filter_tasks)}"
    seed = int(config.get("meta", {}).get("seed", config["data"].get("seed", -1)))

    horizons = list(range(1, max_horizon + 1))
    mse_model_sums = {h: 0.0 for h in horizons}
    mse_persist_sums = {h: 0.0 for h in horizons}
    mse_model_top_sums = {h: 0.0 for h in horizons}
    mse_persist_top_sums = {h: 0.0 for h in horizons}
    moved_model_sums = {h: 0.0 for h in horizons}
    moved_persist_sums = {h: 0.0 for h in horizons}
    proprio_model_sums = {h: 0.0 for h in horizons}
    proprio_persist_sums = {h: 0.0 for h in horizons}
    counts = {h: 0 for h in horizons}
    top_counts = {h: 0 for h in horizons}
    moved_counts = {h: 0 for h in horizons}
    proprio_counts = {h: 0 for h in horizons}
    latent_means = []
    latent_stds = []
    deterministic_max_abs = None
    batch_times = []
    evaluated_batches = 0

    max_batches = args.max_batches if args.max_batches and args.max_batches > 0 else None
    top_k = args.top_k_patches
    if top_k <= 0:
        raise ValueError("--top-k-patches must be positive")
    moved_top_frac = float(args.moved_top_frac)
    moved_min_floor = float(args.moved_min_floor)
    moved_denominator_floor = float(args.moved_denominator_floor)
    moved_dilation = int(args.moved_dilation)
    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        obs, action, _state, _reward = batch[:4]
        obs = _move_obs_to_device(obs, device)
        action = action.to(device, dtype=torch.float32, non_blocking=True)

        start = time.perf_counter()
        # The validation loader already applies the train-time image transform.
        # Use the underlying VideoWM path to avoid double-transforming through EncPredWM.encode().
        video_features, proprio_features, action_features = model.model.encode(obs, action)
        available_horizon = min(max_horizon, video_features.shape[1] - prefix - 1)
        if available_horizon < 1:
            continue

        context_start = max(0, prefix - ctxt_window + 1)
        pred_video_features, _pred_action_features, _pred_proprio_features = model.model.forward_pred(
            video_features[:, context_start : prefix + 1],
            action_features[:, context_start : prefix + 1],
            proprio_features[:, context_start : prefix + 1] if proprio_features is not None else None,
        )
        pred_by_horizon = {
            1: pred_video_features[:, -1:],
        }
        pred_prop_by_horizon = {}
        if _pred_proprio_features is not None:
            pred_prop_by_horizon[1] = _pred_proprio_features[:, -1:]

        vid_context = torch.cat([video_features[:, : prefix + 1], pred_by_horizon[1]], dim=1)
        prop_context = None
        if proprio_features is not None and 1 in pred_prop_by_horizon:
            prop_context = torch.cat([proprio_features[:, : prefix + 1], pred_prop_by_horizon[1]], dim=1)
        act_context = action_features[:, : prefix + 1]

        for horizon in range(2, available_horizon + 1):
            act_context = torch.cat(
                [act_context, action_features[:, prefix + horizon - 1 : prefix + horizon]],
                dim=1,
            )
            next_vid_feats, _next_act_feats, next_prop_feats = model.model.forward_pred(
                vid_context[:, -ctxt_window:].detach(),
                act_context[:, -ctxt_window:],
                prop_context[:, -ctxt_window:].detach() if prop_context is not None else None,
            )
            pred_by_horizon[horizon] = next_vid_feats[:, -1:]
            vid_context = torch.cat([vid_context.detach(), pred_by_horizon[horizon]], dim=1)
            if prop_context is not None and next_prop_feats is not None:
                pred_prop_by_horizon[horizon] = next_prop_feats[:, -1:]
                prop_context = torch.cat([prop_context.detach(), pred_prop_by_horizon[horizon]], dim=1)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_times.append((time.perf_counter() - start) * 1000.0)
        evaluated_batches += 1

        if batch_idx == 0:
            video_features_2, _proprio_features_2, _action_features_2 = model.model.encode(obs, action)
            deterministic_max_abs = float((video_features - video_features_2).abs().max().detach().cpu())

        persistent_visual = video_features[:, prefix : prefix + 1]
        persistent_proprio = proprio_features[:, prefix : prefix + 1] if proprio_features is not None else None

        for horizon in range(1, available_horizon + 1):
            target = video_features[:, prefix + horizon : prefix + horizon + 1]
            pred = pred_by_horizon[horizon]
            B = pred.shape[0]
            model_patch_mse = (
                pred.reshape(B, 1, -1, pred.shape[-1])
                - target.reshape(B, 1, -1, target.shape[-1])
            ).pow(2).mean(dim=-1)
            persist_patch_mse = (
                persistent_visual.reshape(B, 1, -1, persistent_visual.shape[-1])
                - target.reshape(B, 1, -1, target.shape[-1])
            ).pow(2).mean(dim=-1)

            _add_patch_mse(model_patch_mse, persist_patch_mse, horizon, mse_model_sums, mse_persist_sums, counts)

            k = min(top_k, persist_patch_mse.shape[-1])
            top_idx = persist_patch_mse.topk(k=k, dim=-1).indices
            _add_patch_mse(
                model_patch_mse.gather(dim=-1, index=top_idx),
                persist_patch_mse.gather(dim=-1, index=top_idx),
                horizon,
                mse_model_top_sums,
                mse_persist_top_sums,
                top_counts,
            )
            moved_mask = _make_moved_patch_mask(
                persist_patch_mse,
                top_frac=moved_top_frac,
                min_floor=moved_min_floor,
                grid_size=model.model.grid_size,
                dilation=moved_dilation,
            )
            _add_masked_patch_mse(
                model_patch_mse,
                persist_patch_mse,
                moved_mask,
                horizon,
                moved_model_sums,
                moved_persist_sums,
                moved_counts,
                denominator_floor=moved_denominator_floor,
            )

            if persistent_proprio is not None and horizon in pred_prop_by_horizon:
                prop_target = proprio_features[:, prefix + horizon : prefix + horizon + 1]
                prop_pred = pred_prop_by_horizon[horizon]
                prop_model_mse = (
                    prop_pred.reshape(B, 1, -1, prop_pred.shape[-1])
                    - prop_target.reshape(B, 1, -1, prop_target.shape[-1])
                ).pow(2).mean(dim=-1)
                prop_persist_mse = (
                    persistent_proprio.reshape(B, 1, -1, persistent_proprio.shape[-1])
                    - prop_target.reshape(B, 1, -1, prop_target.shape[-1])
                ).pow(2).mean(dim=-1)
                _add_patch_mse(
                    prop_model_mse,
                    prop_persist_mse,
                    horizon,
                    proprio_model_sums,
                    proprio_persist_sums,
                    proprio_counts,
                )

        latent_means.append(float(video_features.mean().detach().cpu()))
        latent_stds.append(float(video_features.std().detach().cpu()))

    if not any(counts.values()):
        raise RuntimeError("No validation batches were evaluated")

    visual_metrics = _finalize_horizon_metrics(mse_model_sums, mse_persist_sums, counts, "visual")
    change_metrics = _finalize_horizon_metrics(mse_model_top_sums, mse_persist_top_sums, top_counts, "change")
    moved_metrics = _finalize_horizon_metrics(moved_model_sums, moved_persist_sums, moved_counts, "moved_region")
    proprio_metrics = _finalize_horizon_metrics(proprio_model_sums, proprio_persist_sums, proprio_counts, "proprio")
    h1 = "1"

    train_log_csv = _as_path(args.train_log_csv)
    results = {
        "model": model_name,
        "task": task,
        "seed": seed,
        "checkpoint": str(checkpoint_folder / checkpoint_name),
        "config": str(Path(args.config).resolve()),
        "backend": _nested_get(config, ["data", "backend"], {}),
        "num_val_windows": len(val_dataset),
        "num_batches": evaluated_batches,
        "max_horizon": max_horizon,
        "prefix": prefix,
        "top_k_patches": top_k,
        "moved_region_mask": {
            "source": "target_side_persistence_patch_mse",
            "top_frac": moved_top_frac,
            "min_floor": moved_min_floor,
            "denominator_floor": moved_denominator_floor,
            "dilation": moved_dilation,
        },
        "mse_model": visual_metrics["visual_mse_model"].get(h1),
        "mse_persist": visual_metrics["visual_mse_persist"].get(h1),
        "skill": visual_metrics["visual_skill"].get(h1),
        "change_mse_model": change_metrics["change_mse_model"].get(h1),
        "change_mse_persist": change_metrics["change_mse_persist"].get(h1),
        "change_skill": change_metrics["change_skill"].get(h1),
        "moved_region_mse_model": moved_metrics["moved_region_mse_model"].get(h1),
        "moved_region_mse_persist": moved_metrics["moved_region_mse_persist"].get(h1),
        "moved_region_change_skill": moved_metrics["moved_region_skill"].get(h1),
        "proprio_mse_model": proprio_metrics["proprio_mse_model"].get(h1),
        "proprio_mse_persist": proprio_metrics["proprio_mse_persist"].get(h1),
        "proprio_skill": proprio_metrics["proprio_skill"].get(h1),
        "mse_model_by_horizon": visual_metrics["visual_mse_model"],
        "mse_persist_by_horizon": visual_metrics["visual_mse_persist"],
        "skill_by_horizon": visual_metrics["visual_skill"],
        "change_mse_model_by_horizon": change_metrics["change_mse_model"],
        "change_mse_persist_by_horizon": change_metrics["change_mse_persist"],
        "change_skill_by_horizon": change_metrics["change_skill"],
        "moved_region_mse_model_by_horizon": moved_metrics["moved_region_mse_model"],
        "moved_region_mse_persist_by_horizon": moved_metrics["moved_region_mse_persist"],
        "moved_region_change_skill_by_horizon": moved_metrics["moved_region_skill"],
        "moved_region_counts_by_horizon": {str(k): int(v) for k, v in moved_counts.items()},
        "proprio_mse_model_by_horizon": proprio_metrics["proprio_mse_model"],
        "proprio_mse_persist_by_horizon": proprio_metrics["proprio_mse_persist"],
        "proprio_skill_by_horizon": proprio_metrics["proprio_skill"],
        "param_count": _count_predictor_params(model),
        "inference_path_param_count": _count_inference_path_params(model),
        "flops_per_forward_estimate": _estimate_predictor_flops(
            model,
            batch_size=args.batch_size or 1,
            seq_len=min_eval_frames,
        ),
        "mamba_backend": _mamba_backend(model),
        "train_step_time_ms": _read_train_step_time_ms(train_log_csv),
        "diagnostic_batch_time_ms": float(sum(batch_times) / len(batch_times)),
        "latent_mean": float(sum(latent_means) / len(latent_means)),
        "latent_std": float(sum(latent_stds) / len(latent_stds)),
        "deterministic_encode_max_abs_diff": deterministic_max_abs,
        "normalization_note": "Validation loader uses train-split dataset statistics; no val statistics are recomputed.",
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training config used for the checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to config checkpoint folder/latest.")
    parser.add_argument("--output", default=None, help="Output directory or JSON filename.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-horizon", type=int, default=6)
    parser.add_argument("--prefix", type=int, default=None, help="Observed prefix index; defaults to ctxt_window - 1.")
    parser.add_argument("--top-k-patches", type=int, default=32)
    parser.add_argument("--moved-top-frac", type=float, default=0.20)
    parser.add_argument("--moved-min-floor", type=float, default=0.0)
    parser.add_argument("--moved-denominator-floor", type=float, default=0.0)
    parser.add_argument("--moved-dilation", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-log-csv", default=None, help="Optional log_r0.csv for train step-time metadata.")
    args = parser.parse_args()

    results = run_skill_score(args)
    print(json.dumps(results, indent=2, sort_keys=True))

    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.suffix.lower() == ".json":
            json_path = output
            out_dir = output.parent
        else:
            out_dir = output
            json_path = out_dir / "skill_score.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out_dir / "skill_score.md").write_text(_make_markdown(results), encoding="utf-8")


if __name__ == "__main__":
    main()
