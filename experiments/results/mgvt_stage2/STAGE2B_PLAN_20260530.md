# MGVT-JEPA Stage-2b Plan

Date: 2026-05-30

Branch: `exp/20260530-mgvt-stage2`

## Objective

Run a prediction-only H=4 rollout-stability breadth scan across all five Phase-1 tasks before any CEM or Terver planning evaluation.

Stage-2b asks whether the Stage-2a H=4 stability result survives task breadth, especially on topology-sensitive Wall and Maze. It also keeps real-`mamba_ssm` Mamba as a full ablation, without restoring it as the primary method unless the five-task evidence justifies a later 3-seed recheck.

## Scope

- Tasks: PushT, Wall, Maze/PointMaze, MW-R, MW-RW.
- Backbones: AdaLN(depth=1), MGVT ConvMixer, MGVT Mamba.
- Total one-seed scan: 5 tasks x 3 backbones = 15 runs.
- Rollout: W=2, H=4 training, diagnostic H=1..6.
- Proprio: enabled, `proprio_emb_dim=16`.
- Encoder: frozen DINOv2-S/14, normalized latent targets.
- Data IO: Lance backend with `fall_back_to_raw_if_unsupported: false`.
- Out of scope: CEM, Terver planning eval, H=8, Stage-3, automatic 3-seed confirmation.

## Task Configuration

All Stage-2b configs use ImageNet normalization:

```yaml
[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
```

| Task | Dataset | Lance URI | filter_tasks | val camera | Seed |
| --- | --- | --- | --- | --- | ---: |
| PushT | `PushT` | `${JEPAWM_DSET_LANCE}/PushT.lance` | null | null | 234 |
| Wall | `Wall` | `${JEPAWM_DSET_LANCE}/Wall.lance` | null | null | 234 |
| Maze | `PointMaze` | `${JEPAWM_DSET_LANCE}/PointMaze.lance` | null | null | 234 |
| MW-R | `METAWORLD_HF` | `${JEPAWM_DSET_LANCE}/Metaworld.lance` | `[mw-reach]` | `exterior_image_2_left` | 1 |
| MW-RW | `METAWORLD_HF` | `${JEPAWM_DSET_LANCE}/Metaworld.lance` | `[mw-reach-wall]` | `exterior_image_2_left` | 1 |

Stage-2a MW-RW used `0.5/0.5` image normalization. Stage-2b reruns MW-RW with ImageNet normalization for a fair five-task comparison; do not pool Stage-2a MW-RW and Stage-2b MW-RW as identical-protocol results.

## Files

- Generated configs: `configs/vjepa_wm/mgvt_stage2b/*.yaml`
- Config generator: `experiments/scripts/gen_stage2b_configs.py`
- Launcher: `experiments/scripts/mgvt_stage2b_scan.sh`
- Wall Lance converter: `experiments/scripts/convert_wall_lance.sh`
- Contract test: `tests/configs/test_mgvt_stage2b_configs.py`

## Lance Preflight

Stage-2b must not silently fall back to raw data. Before training, verify these stores exist:

- `${JEPAWM_DSET_LANCE}/PushT.lance`
- `${JEPAWM_DSET_LANCE}/Wall.lance`
- `${JEPAWM_DSET_LANCE}/PointMaze.lance`
- `${JEPAWM_DSET_LANCE}/Metaworld.lance`

If `Wall.lance` is missing, build it with `experiments/scripts/convert_wall_lance.sh`. Wall observations are float32 tensors, so the converter uses `codec=raw_float32` by default.

## Gate 4

Primary gate: AdaLN(depth=1), evaluated per task.

PASS requires:

- `skill[h] > 0` for h=1..4
- `change_skill[h] > 0` for h=1..4
- `proprio_skill[h] > 0` for h=1..4
- no H4 collapse

For Wall and Maze, `change_skill[4]` is the key metric because static geometry can inflate raw persistence skill. High raw skill with weak moved-patch skill is a failure signature.

ConvMixer and Mamba are comparative controls, not primary gate deciders. Mamba stays an ablation at one seed. Flag Mamba for a later approval-gated 3-seed confirm only if it clearly beats AdaLN on Wall/Maze change_skill[4] or if AdaLN collapses there while Mamba does not.

## Stop Conditions

Stop after the Stage-2b Gate 4 verdict and report. Do not run CEM, planning eval, H=8, Stage-3, or 3-seed confirmation without explicit approval.

## Reporting

After completion, pull artifacts to `experiments/results/mgvt_stage2/stage2b_1seed_<date>/`, validate all 15 JSON files, then create:

- `MGVT_STAGE2B_GATE_<date>.md`
- `notes/alignment_log.md` entry
- `experiments/registry.md` entry

Lance must be described only as a validated data-IO backend, not an algorithmic contribution.
