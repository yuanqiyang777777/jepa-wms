# Run naming convention

Every experiment run in this project must have a unique `run_id` that follows this format. This file is the single source of truth for the convention. All launcher scripts must use it; all stored artifacts must use it; all memory / journal entries must reference it.

Launcher scripts live in the fork at `jepa-wms/experiments/scripts/` and are run from the lab01 repo root after `git pull`. Local pulled-back results live at the workspace root `E:\jepa-wms\experiments\results\`, outside the fork.

## Format

```
{date}_{phase}_{env}_{method}_{key_hp}_seed{N}
```

| Field      | Allowed values                                                | Example         |
|------------|---------------------------------------------------------------|-----------------|
| `date`     | `YYYYMMDD` (run launch date in lab01 local time)              | `20260518`      |
| `phase`    | `smoke` \| `baseline` \| `pilot` \| `main` \| `ablation`      | `pilot`         |
| `env`      | `pusht` \| `wall` \| `maze` \| `mwr` \| `mwrw` \| `granular`  | `pusht`         |
| `method`   | `terver` (vanilla baseline) \| `mfl` \| `mfl1p` \| `vwl` \| `spe` \| `tkm` \| (other variant) | `mfl`     |
| `key_hp`   | the 1-2 hyperparameters that distinguish this run from others in the same sweep, joined by `_`. Use compact form (`e0.5` for eta=0.5, `g1` for gamma=1) | `e0.5_g1`       |
| `seed`     | non-negative int                                              | `seed0`         |

For baseline reproduction runs with no method-specific hyperparameters, omit `key_hp`:

```
{date}_baseline_pusht_terver_seed{N}
```

**Full example**: `20260518_pilot_pusht_mfl_e0.5_g1_seed0`

## Hyperparameter shorthand

| Symbol | Compact | Notes                                |
|--------|---------|--------------------------------------|
| eta    | `e`     | MFL mix weight, e.g. `e0`, `e0.5`, `e1` |
| gamma  | `g`     | MFL focal sharpness, e.g. `g0.5`, `g1`, `g2` |
| alpha  | `a`     | proprioception loss weight, e.g. `a0.1` |
| W      | `w`     | training context length              |
| W^p    | `wp`    | planning context length              |
| H      | `h`     | planning horizon                     |
| (others) | use 1-2 lowercase letters, document here on first use |     |

## Examples

```
# Smoke test
20260516_smoke_pusht_terver_seed0

# Baseline reproduction (no method-specific HPs)
20260518_baseline_pusht_terver_seed0
20260518_baseline_pusht_terver_seed1
20260518_baseline_pusht_terver_seed2

# Pilot grid (eta, gamma sweep)
20260520_pilot_pusht_mfl_e0_g1_seed0
20260520_pilot_pusht_mfl_e0.5_g1_seed0
20260520_pilot_pusht_mfl_e1_g1_seed0
20260520_pilot_pusht_mfl_e0.5_g2_seed0

# Main eval (selected HPs from pilot, multiple seeds)
20260605_main_wall_mfl_e0.5_g1_seed0
20260605_main_wall_mfl_e0.5_g1_seed1
20260605_main_wall_mfl_e0.5_g1_seed2

# Ablation
20260615_ablation_pusht_mfl_predside_e0.5_g1_seed0
```

## Anti-examples (DO NOT do)

```
pusht_run1
mfl_test
20260520_pilot_pusht_mfl_seed0
2026-05-18_pilot_pusht
my_amazing_run_v3
```

## Where `run_id` must appear

When you launch a run, set it once and propagate everywhere:

```bash
RUN_ID="20260520_pilot_pusht_mfl_e0.5_g1_seed0"
export RUN_ID
```

Use in:

- remote output dir: `$JEPAWM_LOGS/$RUN_ID`
- local pull dir: workspace-root `experiments/results/$RUN_ID/`
- alignment log entry
- registry entry
- memory note, if entered into memory

## Why this format

- **Date first**: `ls` and `glob` give chronological order for free.
- **Phase before env**: filtering all pilot runs is `*pilot*`; all PushT runs is `*pusht*`.
- **method+hp inline**: greppable for sweep analysis.
- **seed last**: easy to identify all seeds of one config.
- **No spaces, no special chars**: works in every shell, filesystem, and W&B backend.

## When changing this convention

If the format ever needs to evolve, edit this file in a single commit, mention every existing place that needs to be updated, and add an entry to `notes/alignment_log.md`. Do not silently introduce a second naming style.
