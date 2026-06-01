# MGVT-D Stage-D1-A Smoke Matrix

Date: 2026-06-02

Status: `SMOKE PASS WITH ONE RECORDED RETRY`. This is plumbing evidence only. All metrics below are from `QUICK_DEBUG=1` two-epoch runs and must not be used as method-quality evidence.

## Scope

- Task/seed for the new narrow matrix: Wall seed 234.
- Already covered mamba smoke from retry6: Wall seed 234 and PushT seed 234, real `mamba_ssm`, commit `6072b00`.
- New code fix after raw-action failure: `36d41b6 Fix raw D1 guidance unused state params`.
- No CEM, no planning eval, no H=8, no Stage-3, no full D1-A confirmation.

## Validation

- Local D1 tests after the raw-action fix: `31 passed, 3 warnings`.
- Config generation dry run: `would write 105 Stage-D1 configs`.
- Remote launcher syntax: `bash -n experiments/scripts/mgvt_d1_scan.sh` passed at `36d41b6`.
- Successful smoke roots all have `scan_end.txt`, checkpoint, `log_r0.csv`, `skill_score.json`, and `probes/*/mgvt_d1_probes.json`.
- Successful smoke logs reject `traceback`, `runtimeerror`, and standalone `nan`.
- Remote checkpoint existence was verified for all six successful smoke score JSONs.

## Recorded Failure And Retry

| Root | Commit | Status | Reason | Follow-up |
| --- | --- | --- | --- | --- |
| `mgvt_d1_smoke_20260602_raw_action_guidance` | `6072b00c004e879821b022b966ecfebed402cb2c` | FAIL | DDP unused-parameter error in `raw_action_guidance`; action-only F_dyn created compact visual/proprio state modules that did not feed the loss. Error files: `20260602_mgvt_d1a_wall_raw_action_guidance_seed234\launch.log` | Kept the failed run, fixed code in `36d41b6`, reran as `_retry1`. |

## Successful Smoke Metrics

| Variant | Task | Seed | Model | Commit | Backend | Skill H4 | Change H4 | Moved H4 | Proprio H4 | Params | FLOPs | Root |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `adaln_param_match` | Wall | 234 | `AdaLN` | `36d41b6` | `` | -3.778693 | -0.043033 | -1.575345 | -1.232787 | 852704 | n/a | `mgvt_d1_smoke_20260602_adaln_param_match` |
| `gru_trend_dim16` | Wall | 234 | `mgvt_d_gru` | `36d41b6` | `` | -3.339961 | -0.019002 | -1.409088 | -4.847129 | 551968 | 1369473024 | `mgvt_d1_smoke_20260602_gru_trend_dim16` |
| `mamba_trend_dim16` | PushT | 234 | `mgvt_d_mamba` | `6072b00` | `mamba_ssm` | -2.197798 | -0.110760 | -0.839849 | 0.378269 | 569408 | 1369473024 | `mgvt_d1_smoke_20260602_retry6` |
| `mamba_trend_dim16` | Wall | 234 | `mgvt_d_mamba` | `6072b00` | `mamba_ssm` | -3.182400 | -0.003070 | -1.331408 | -4.938977 | 569376 | 1369473024 | `mgvt_d1_smoke_20260602_retry6` |
| `mlp_trend_dim16` | Wall | 234 | `mgvt_d_mlp` | `36d41b6` | `` | -2.805464 | 0.034135 | -1.152890 | -4.876315 | 469152 | 1369374720 | `mgvt_d1_smoke_20260602_mlp_trend_dim16` |
| `raw_action_guidance` | Wall | 234 | `mgvt_d_raw_action` | `36d41b6` | `` | -3.047870 | -0.006135 | -1.286418 | -5.144991 | 379776 | 1367998464 | `mgvt_d1_smoke_20260602_raw_action_guidance_retry1` |

## Interpretation

- The D1-A scaffold now launches, trains briefly, checkpoints, evaluates H1-H6 skill diagnostics, and writes probe summaries for all planned D1-A smoke variants.
- `raw_action_guidance` required a real code fix; the old failed root is retained and the successful rerun uses `_retry1`, so this is not a silent reroll.
- Negative QUICK_DEBUG metrics are expected and not meaningful for architecture quality; they only verify plumbing and DDP graph coverage.
- `adaln_param_match` has no FLOP estimate in the current skill-score output; this remains a full-run review flag for fair param/FLOP accounting.

## Next Gate

Before launching D1-A full confirmation, Claude should review `36d41b6`, this smoke matrix, and `D1_BASELINE_FREEZE_20260602.md`. Full confirmation remains blocked until that review says `REVIEW PASS` or an explicitly accepted `PASS-WITH-FLAGS`.
