# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Stage-1 MGVT persistence skill diagnostic.

This script evaluates H=1 latent prediction on the validation split and compares
the model to the copy-last-frame baseline. It intentionally does not run CEM,
planning, H=4 rollout, or any Terver protocol evaluation.
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
    predictor = model.model.predictor
    action_encoder = model.model.action_encoder
    proprio_encoder = model.model.proprio_encoder
    modules = [m for m in (predictor, action_encoder, proprio_encoder) if m is not None]
    return int(sum(p.numel() for module in modules for p in module.parameters()))


def _mamba_backend(model) -> str | None:
    predictor = getattr(model.model, "predictor", None)
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


def _make_markdown(results: dict[str, Any]) -> str:
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

    val_dataset, _val_traj_dataset, val_loader, traj_dataset, preprocessor = _build_val_loader_and_preprocessor(
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
        wrapper_kwargs={"ctxt_window": _nested_get(config, ["model", "rollout_cfg", "ctxt_window_train_rollout"], 2)},
    )
    model.eval()

    model_name = _nested_get(config, ["model", "predictor", "pred_type"], "unknown")
    task = ",".join(config["data"].get("datasets", []))
    filter_tasks = _nested_get(config, ["data", "custom", "filter_tasks"], None)
    if filter_tasks:
        task = f"{task}:{','.join(filter_tasks)}"
    seed = int(config.get("meta", {}).get("seed", config["data"].get("seed", -1)))

    mse_model_sum = 0.0
    mse_persist_sum = 0.0
    mse_model_top_sum = 0.0
    mse_persist_top_sum = 0.0
    num_patch_steps = 0
    num_top_patch_steps = 0
    latent_means = []
    latent_stds = []
    deterministic_max_abs = None
    batch_times = []

    max_batches = args.max_batches if args.max_batches and args.max_batches > 0 else None
    top_k = args.top_k_patches
    if top_k <= 0:
        raise ValueError("--top-k-patches must be positive")
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
        pred_video_features, _pred_action_features, _pred_proprio_features = model.model.forward_pred(
            video_features,
            action_features,
            proprio_features,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_times.append((time.perf_counter() - start) * 1000.0)

        if batch_idx == 0:
            video_features_2, _proprio_features_2, _action_features_2 = model.model.encode(obs, action)
            deterministic_max_abs = float((video_features - video_features_2).abs().max().detach().cpu())

        current = video_features[:, :-1].reshape(video_features.shape[0], -1, video_features.shape[3] * video_features.shape[4], video_features.shape[-1])
        target = video_features[:, 1:].reshape(video_features.shape[0], -1, video_features.shape[3] * video_features.shape[4], video_features.shape[-1])
        pred = pred_video_features[:, :-1].reshape(pred_video_features.shape[0], -1, pred_video_features.shape[3] * pred_video_features.shape[4], pred_video_features.shape[-1])

        model_patch_mse = (pred - target).pow(2).mean(dim=-1)
        persist_patch_mse = (current - target).pow(2).mean(dim=-1)
        change_score = persist_patch_mse

        mse_model_sum += float(model_patch_mse.sum().detach().cpu())
        mse_persist_sum += float(persist_patch_mse.sum().detach().cpu())
        num_patch_steps += int(model_patch_mse.numel())

        k = min(top_k, change_score.shape[-1])
        top_idx = change_score.topk(k=k, dim=-1).indices
        model_top = model_patch_mse.gather(dim=-1, index=top_idx)
        persist_top = persist_patch_mse.gather(dim=-1, index=top_idx)
        mse_model_top_sum += float(model_top.sum().detach().cpu())
        mse_persist_top_sum += float(persist_top.sum().detach().cpu())
        num_top_patch_steps += int(model_top.numel())

        latent_means.append(float(video_features.mean().detach().cpu()))
        latent_stds.append(float(video_features.std().detach().cpu()))

    if num_patch_steps == 0:
        raise RuntimeError("No validation batches were evaluated")

    mse_model = mse_model_sum / num_patch_steps
    mse_persist = mse_persist_sum / num_patch_steps
    mse_model_top = mse_model_top_sum / num_top_patch_steps
    mse_persist_top = mse_persist_top_sum / num_top_patch_steps

    train_log_csv = _as_path(args.train_log_csv)
    results = {
        "model": model_name,
        "task": task,
        "seed": seed,
        "checkpoint": str(checkpoint_folder / checkpoint_name),
        "config": str(Path(args.config).resolve()),
        "backend": _nested_get(config, ["data", "backend"], {}),
        "num_val_windows": len(val_dataset),
        "num_batches": min(len(val_loader), max_batches) if max_batches is not None else len(val_loader),
        "top_k_patches": top_k,
        "mse_model": mse_model,
        "mse_persist": mse_persist,
        "skill": _safe_skill(mse_model, mse_persist),
        "change_mse_model": mse_model_top,
        "change_mse_persist": mse_persist_top,
        "change_skill": _safe_skill(mse_model_top, mse_persist_top),
        "param_count": _count_predictor_params(model),
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
    parser.add_argument("--top-k-patches", type=int, default=32)
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
