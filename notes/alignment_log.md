# Alignment Log

## 2026-05-29 - Lance backend equivalence validation

Validated the Phase-1 Lance data backend on lab01 without a 50-epoch run. Tier A fixture tests passed (`29 passed, 1 xfailed`) and Tier C real-dataset 256-window audits passed for Maze, Wall, PushT, and MW. After independent audit, Tier B is treated as limited because the historical `log_r0.csv` files have a header/value ordering issue and should not be cited as literal loss-equivalence proof; a follow-up code fix aligns CSV headers for future runs without rewriting old logs. Full report: `experiments/results/lance_equivalence_validation_20260529.md`. Boundary: this supports data/backend equivalence and stable Lance absolute train-time estimates; raw-to-Lance mean-wall speedups require a tail-latency caveat and this does not prove Terver success-rate or long-run checkpoint equivalence.

## 2026-05-30 - MGVT-JEPA Stage-1 backbone gate

Recovered the real `mamba_ssm==2.2.6.post3` CUDA path on lab01 without changing the core torch stack (`torch 2.7.0+cu126`). The extension was source-built for L40 `sm_89`; validation confirmed CUDA forward and `MambaMixer.backend == "mamba_ssm"`. The 24-run H=1 gate completed on PushT and MetaWorld reach-wall with three seeds per backbone. Gate 1 passed on both tasks. Gate 2 failed for Mamba: AdaLN(depth=1) was the strongest model on both tasks, ConvMixer also exceeded Mamba on both tasks, and Mamba failed to beat MLP on MW-RW. Decision: preserve the conditioned latent-transition direction, use AdaLN(depth=1) as the preferred Stage-2 candidate, keep ConvMixer as the lightweight control, and demote Mamba to an ablation. Stop before Stage 2 planning work. Full report: `experiments/results/mgvt_stage1/MGVT_STAGE1_GATE_20260530.md`.

## 2026-05-30 - MGVT-JEPA Stage-2a H=4 rollout gate

Completed the one-seed Stage-2a rollout-stability probe on lab01 for PushT and MetaWorld reach-wall with AdaLN(depth=1), MGVT ConvMixer, and real-`mamba_ssm` MGVT Mamba. All six runs passed the launcher health check and produced finite H=1..6 diagnostic metrics with deterministic DINO encode diff 0.0. Gate 3 passed for the AdaLN(depth=1) candidate on both tasks: PushT H4 visual/change/proprio skill = 0.5458/0.6982/0.9943; MW-RW H4 = 0.7141/0.8480/0.9538. Decision: Stage-2a passes; next step should be Stage-2b expansion of the winning backbone to the remaining Phase-1 tasks before any CEM or Terver planning evaluation. Lance remains IO-only evidence. Full report: `experiments/results/mgvt_stage2/stage2a_1seed_20260530/MGVT_STAGE2A_GATE_20260530.md`.
