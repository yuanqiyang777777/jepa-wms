from __future__ import annotations

import csv
import math
from pathlib import Path


def test_profile_step_exposes_timing_controls():
    script = Path("experiments/scripts/profile_step.sh").read_text()

    assert "NUM_WORKERS=" in script
    assert "BACKEND_KIND=" in script
    assert "LANCE_URI=" in script
    assert "SCRIPT_DIR=" in script
    assert 'cd "$REPO_ROOT"' in script
    assert 'cfg["data"]["loader"]["num_workers"] = int(num_workers)' in script
    assert 'backend["fall_back_to_raw_if_unsupported"] = False' in script
    assert 'datasets != ["PointMaze"]' in script


def test_phase1_launcher_and_lance_converter_are_available():
    launcher = Path("experiments/scripts/run_phase1_timing_profile.sh")
    converter = Path("experiments/scripts/convert_pointmaze_lance.sh")

    assert launcher.exists()
    assert converter.exists()

    launcher_text = launcher.read_text()
    assert "20260528_dino_wm_timing_pusht_raw_r${rep}" in launcher_text
    assert "20260528_dino_wm_timing_maze_lance_r${rep}" in launcher_text
    assert "PROFILE_WARMUP=30" in launcher_text
    assert "PROFILE_STEPS=300" in launcher_text
    assert "PROFILE_IPE=360" in launcher_text
    assert "RUN_LANCE=" in launcher_text
    assert 'PROFILE_SCRIPT="$SCRIPT_DIR/profile_step.sh"' in launcher_text
    assert 'SUMMARY_SCRIPT="$SCRIPT_DIR/summarize_training_time_profile.py"' in launcher_text
    assert 'if [ "$RUN_LANCE" = "1" ]; then' in launcher_text

    converter_text = converter.read_text()
    assert "JEPAWM_DSET/_lance_20260528" in converter_text
    assert "SCRIPT_DIR=" in converter_text
    assert 'cd "$REPO_ROOT"' in converter_text
    assert "convert_point_maze_to_lance" in converter_text
    assert "codec=codec" in converter_text
    assert "mode=mode" in converter_text


def test_training_time_summary_parses_run_dir(tmp_path):
    run_dir = tmp_path / "20260528_dino_wm_timing_maze_raw_r1"
    run_dir.mkdir()

    with (run_dir / "log_r0.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "itr", "gpu-time(ms)", "iter-time(ms)", "loss"])
        for itr in range(40):
            writer.writerow([1, itr, 100.0 + itr, 200.0 + itr, 1.0])

    (run_dir / "launch.log").write_text(
        "\n".join(
            [
                "RUN_ID=20260528_dino_wm_timing_maze_raw_r1",
                "BASE_CONFIG=configs/vjepa_wm/mz_sweep/mz.yaml",
                "BACKEND_KIND=raw",
                "LANCE_URI=unset",
                "NUM_WORKERS=16",
                "PROFILE_WARMUP=30",
                "PROFILE_STEPS=10",
                "DEVICES=cuda:0 cuda:1 (world_size=2)",
                "BATCH_SIZE=32",
                "Iterations per epoch: 360 (dataset size: 759)",
                "[StepProfiler] mean per-step breakdown over 10 steps",
                "  section                  mean ms    % of step    samples",
                "  data_fetch                12.500        5.00%         10",
                "  data_to_device             2.000        0.80%         10",
                "  measured step wall       250.000",
            ]
        )
    )

    from experiments.scripts.summarize_training_time_profile import summarize_run

    summary = summarize_run(run_dir, warmup=30, measured_steps=10)

    assert summary["run_id"] == "20260528_dino_wm_timing_maze_raw_r1"
    assert summary["env"] == "maze"
    assert summary["backend"] == "raw"
    assert summary["rows_used"] == 10
    assert math.isclose(summary["iter_median_ms"], 234.5)
    assert math.isclose(summary["gpu_mean_ms"], 134.5)
    assert math.isclose(summary["data_fetch_ms"], 12.5)
    assert summary["dataset_size"] == 759
    assert math.isclose(summary["epoch_estimate_min"], 759 * 234.5 / 1000 / 60)
