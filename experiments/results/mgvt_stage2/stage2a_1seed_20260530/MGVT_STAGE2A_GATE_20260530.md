# MGVT-JEPA Stage-2a H=4 Rollout Gate

Date: 2026-05-30

Branch: `exp/20260530-mgvt-stage2`

Remote run root: `$JEPAWM_LOGS/mgvt_stage2_1seed_20260530`

Local artifacts: `experiments/results/mgvt_stage2/stage2a_1seed_20260530/`

## Protocol

- Prediction-only Stage-2a gate; no CEM, no planning eval, no H=8, no Stage-2b.
- Tasks: PushT and MetaWorld reach-wall.
- Backbones: AdaLN(depth=1), MGVT ConvMixer, MGVT Mamba.
- Rollout: W=2 context, H=4 training, diagnostic horizons H=1..6.
- Data IO: validated Lance backend only; this is an IO acceleration, not an algorithmic claim.
- Seeds: PushT seed 234; MW-RW seed 1.

## Verification

- 6/6 `skill_score.json` files present.
- All visual `skill[h]`, `change_skill[h]`, and `proprio_skill[h]` values are finite for H=1..6.
- `deterministic_encode_max_abs_diff == 0.0` for all runs.
- Mamba runs used real `mamba_ssm`, not the GRU fallback.
- All six training runs passed the launcher health check; no OOM/CUDA/training-health failure markers were found.

## Gate 3 Verdict

Gate 3 evaluates AdaLN(depth=1) per task. PASS requires positive `skill[h]`, `change_skill[h]`, and `proprio_skill[h]` for H=1..4, with no collapse at H=4.

| Task | Gate 3 | H4 visual skill | H4 change skill | H4 proprio skill |
| --- | --- | ---: | ---: | ---: |
| PushT | PASS | 0.5458 | 0.6982 | 0.9943 |
| MW-RW | PASS | 0.7141 | 0.8480 | 0.9538 |

Decision: Stage-2a passes for AdaLN(depth=1) on the two-task probe. The next reasonable step is Stage-2b: expand the winning backbone to the remaining Phase-1 tasks before any CEM/planning evaluation.

## H4 Backbone Snapshot

| Task | Model | H4 visual skill | H4 change skill | H4 proprio skill | Params | Step ms | Mamba backend |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PushT | AdaLN | 0.5458 | 0.6982 | 0.9943 | 343480 | 264.81 |  |
| PushT | mgvt_convmixer | 0.5302 | 0.6748 | 0.9895 | 336279 | 95.23 |  |
| PushT | mgvt_mamba | 0.5509 | 0.7022 | 0.9969 | 362136 | 110.74 | mamba_ssm |
| MW-RW | AdaLN | 0.7141 | 0.8480 | 0.9538 | 344680 | 302.47 |  |
| MW-RW | mgvt_convmixer | 0.5609 | 0.7279 | 0.6601 | 337469 | 95.37 |  |
| MW-RW | mgvt_mamba | 0.5920 | 0.7583 | 0.7329 | 363176 | 112.18 | mamba_ssm |

## Notes

- This is a one-seed stability gate, not a final planning result.
- H=5..6 were diagnostic extrapolation only and were not used as hard gate criteria.
- Mamba remains an ablation because Stage-1 Gate 2 failed for Mamba. The Stage-2a numbers show Mamba is viable on H=4 rollout, but not enough to reverse the Stage-1 backbone decision.

Full aggregate tables are in:

- `summary/mgvt_stage2_horizon_summary.md`
- `summary/mgvt_stage2_horizon_summary.csv`
- `summary/mgvt_stage2_horizon_raw.csv`
