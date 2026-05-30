# MGVT-JEPA Stage-1 Backbone Gate

Date: 2026-05-30

## Setup

- Prediction-only H=1 gate on frozen DINOv2-S/14 latents.
- Tasks: PushT and MetaWorld reach-wall.
- Backbones: `mgvt_mlp`, `mgvt_convmixer`, real `mgvt_mamba`, and `AdaLN(depth=1)`.
- Three seeds per task/backbone: 24 runs total.
- Lance is used only as the validated data-IO backend.
- Real Mamba path: `mamba_ssm==2.2.6.post3`, source-built on lab01 with CUDA 12.6 for L40 `sm_89`.
- Raw artifacts: `experiments/results/mgvt_stage1/real_mamba_20260530/`.

## Results

| Task | Model | Skill mean | Skill std | Change-skill mean | Change-skill std | Params | Step ms | Mamba backend |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PushT | AdaLN | 0.3742 | 0.0043 | 0.5885 | 0.0054 | 278376 | 46.60 | |
| PushT | mgvt_convmixer | 0.3324 | 0.0056 | 0.5546 | 0.0045 | 282295 | 38.78 | |
| PushT | mgvt_mamba | 0.3160 | 0.0031 | 0.5523 | 0.0033 | 293512 | 43.06 | mamba_ssm |
| PushT | mgvt_mlp | 0.2940 | 0.0024 | 0.4865 | 0.0043 | 282874 | 35.15 | |
| MW-RW | AdaLN | 0.2797 | 0.0044 | 0.5800 | 0.0051 | 279416 | 46.38 | |
| MW-RW | mgvt_convmixer | 0.2682 | 0.0035 | 0.5701 | 0.0077 | 283325 | 38.98 | |
| MW-RW | mgvt_mamba | 0.2305 | 0.0034 | 0.5730 | 0.0043 | 294392 | 45.70 | mamba_ssm |
| MW-RW | mgvt_mlp | 0.2485 | 0.0022 | 0.5348 | 0.0079 | 283934 | 37.20 | |

## Gate Verdict

- Gate 1 PASS on both tasks: at least one backbone exceeds `skill > 0.10` and has positive change-skill.
- Gate 2 FAIL for Mamba: real `mgvt_mamba` beats `mgvt_mlp` on PushT, but not on MW-RW, and is clearly below both `AdaLN(depth=1)` and ConvMixer on both tasks.
- Decision: retain the conditioned latent-transition idea, use `AdaLN(depth=1)` as the preferred Stage-2 candidate, and demote Mamba to an ablation. Do not start Stage 2 automatically.

## Evidence Notes

- All 24 skill-score JSON files report `deterministic_encode_max_abs_diff == 0.0`.
- The Mamba rows report `mamba_backend == "mamba_ssm"`; these are not GRU-fallback results.
- The checked-in launcher ignores only the known exit-time DataLoader worker shutdown traceback when the training command exits successfully; non-zero exits, CUDA errors, OOM, missing checkpoints, and missing CSV logs remain fatal.
