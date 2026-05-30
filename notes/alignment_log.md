# Alignment Log

## 2026-05-29 - Lance backend equivalence validation

Validated the Phase-1 Lance data backend on lab01 without a 50-epoch run. Tier A fixture tests passed (`29 passed, 1 xfailed`) and Tier C real-dataset 256-window audits passed for Maze, Wall, PushT, and MW. After independent audit, Tier B is treated as limited because the historical `log_r0.csv` files have a header/value ordering issue and should not be cited as literal loss-equivalence proof; a follow-up code fix aligns CSV headers for future runs without rewriting old logs. Full report: `experiments/results/lance_equivalence_validation_20260529.md`. Boundary: this supports data/backend equivalence and stable Lance absolute train-time estimates; raw-to-Lance mean-wall speedups require a tail-latency caveat and this does not prove Terver success-rate or long-run checkpoint equivalence.

## 2026-05-30 - MGVT-JEPA Stage-1 backbone gate

Recovered the real `mamba_ssm==2.2.6.post3` CUDA path on lab01 without changing the core torch stack (`torch 2.7.0+cu126`). The extension was source-built for L40 `sm_89`; validation confirmed CUDA forward and `MambaMixer.backend == "mamba_ssm"`. The 24-run H=1 gate completed on PushT and MetaWorld reach-wall with three seeds per backbone. Gate 1 passed on both tasks. Gate 2 failed for Mamba: AdaLN(depth=1) was the strongest model on both tasks, ConvMixer also exceeded Mamba on both tasks, and Mamba failed to beat MLP on MW-RW. Decision: preserve the conditioned latent-transition direction, use AdaLN(depth=1) as the preferred Stage-2 candidate, keep ConvMixer as the lightweight control, and demote Mamba to an ablation. Stop before Stage 2 planning work. Full report: `experiments/results/mgvt_stage1/MGVT_STAGE1_GATE_20260530.md`.
