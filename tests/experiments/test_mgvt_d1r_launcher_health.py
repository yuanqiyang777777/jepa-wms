from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _health_check_source() -> str:
    script = Path("experiments/scripts/mgvt_d1r_scan.sh").read_text(encoding="utf-8").replace("\r\n", "\n")
    start_marker = 'python - "$run_dir" "$ckpt_dir" "$cmd_status" <<\'PY\''
    start = script.index(start_marker) + len(start_marker)
    end = script.index("\nPY\n}", start)
    return script[start:end].strip()


def _write_run(
    tmp_path: Path,
    *,
    complete: bool = True,
    launch_text: str = "",
    checkpoint: bool = True,
    config: bool = True,
    last_epoch: int | None = None,
    last_itr: int | None = None,
    last_loss: str | float = 0.1,
):
    run_dir = tmp_path / "run"
    ckpt_dir = tmp_path / "ckpt"
    run_dir.mkdir()
    ckpt_dir.mkdir()
    if config:
        (run_dir / "config.yaml").write_text(
            """
optimization:
  transition_model:
    num_epochs: 10
    iterations_per_epoch: 1000
""".lstrip(),
            encoding="utf-8",
        )
    if checkpoint:
        (ckpt_dir / "jepa-latest.pth.tar").write_bytes(b"checkpoint")
    with (run_dir / "log_r0.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "itr", "loss"])
        writer.writerow([1, 0, 0.5])
        writer.writerow(
            [
                last_epoch if last_epoch is not None else (10 if complete else 9),
                last_itr if last_itr is not None else 999,
                last_loss,
            ]
        )
    (run_dir / "launch.log").write_text(launch_text, encoding="utf-8")
    return run_dir, ckpt_dir


def _run_health(run_dir: Path, ckpt_dir: Path, cmd_status: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-", str(run_dir), str(ckpt_dir), str(cmd_status)],
        input=_health_check_source(),
        text=True,
        cwd=Path.cwd(),
        capture_output=True,
        check=False,
    )


def test_d1r_health_check_allows_complete_training_with_loader_teardown_warning(tmp_path):
    launch_text = """
Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x123>
Traceback (most recent call last):
RuntimeError: DataLoader worker (pid 123) is killed by signal: Aborted.
"""
    run_dir, ckpt_dir = _write_run(tmp_path, launch_text=launch_text)

    result = _run_health(run_dir, ckpt_dir)

    assert result.returncode == 0, result.stderr
    payload = json.loads((run_dir / "health_check.json").read_text(encoding="utf-8"))
    assert payload["known_loader_shutdown_warning"] is True
    assert payload["last_epoch"] == 10
    assert payload["last_itr"] == 999
    assert payload["extra_tracebacks"] == 0


def test_d1r_health_check_rejects_unexpected_traceback(tmp_path):
    run_dir, ckpt_dir = _write_run(tmp_path, launch_text="Traceback\nRuntimeError: real failure")

    result = _run_health(run_dir, ckpt_dir)

    assert result.returncode != 0
    payload = json.loads((run_dir / "health_check.json").read_text(encoding="utf-8"))
    assert payload["known_loader_shutdown_warning"] is False
    assert any("unexpected traceback/runtimeerror" in error for error in payload["errors"])


def test_d1r_health_check_rejects_incomplete_training_even_with_loader_warning(tmp_path):
    launch_text = """
Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x123>
Traceback (most recent call last):
RuntimeError: DataLoader worker (pid 123) is killed by signal: Aborted.
"""
    run_dir, ckpt_dir = _write_run(tmp_path, complete=False, launch_text=launch_text)

    result = _run_health(run_dir, ckpt_dir)

    assert result.returncode != 0
    payload = json.loads((run_dir / "health_check.json").read_text(encoding="utf-8"))
    assert any("last_epoch=9" in error for error in payload["errors"])


def test_d1r_health_check_rejects_nonzero_command_status(tmp_path):
    launch_text = """
Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x123>
Traceback (most recent call last):
RuntimeError: DataLoader worker (pid 123) is killed by signal: Aborted.
"""
    run_dir, ckpt_dir = _write_run(tmp_path, launch_text=launch_text)

    result = _run_health(run_dir, ckpt_dir, cmd_status=1)

    assert result.returncode != 0
    payload = json.loads((run_dir / "health_check.json").read_text(encoding="utf-8"))
    assert any("non-zero" in error for error in payload["errors"])


def test_d1r_health_check_rejects_mixed_loader_and_real_runtime_error(tmp_path):
    launch_text = """
Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x123>
Traceback (most recent call last):
RuntimeError: DataLoader worker (pid 123) is killed by signal: Aborted.
RuntimeError: unrelated model failure
"""
    run_dir, ckpt_dir = _write_run(tmp_path, launch_text=launch_text)

    result = _run_health(run_dir, ckpt_dir)

    assert result.returncode != 0
    payload = json.loads((run_dir / "health_check.json").read_text(encoding="utf-8"))
    assert payload["known_loader_shutdown_warning"] is False
    assert any("unexpected RuntimeError lines" in error for error in payload["errors"])


def test_d1r_health_check_rejects_extra_non_runtime_traceback(tmp_path):
    launch_text = """
Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x123>
Traceback (most recent call last):
RuntimeError: DataLoader worker (pid 123) is killed by signal: Aborted.
Traceback (most recent call last):
ValueError: unrelated post-train failure
"""
    run_dir, ckpt_dir = _write_run(tmp_path, launch_text=launch_text)

    result = _run_health(run_dir, ckpt_dir)

    assert result.returncode != 0
    payload = json.loads((run_dir / "health_check.json").read_text(encoding="utf-8"))
    assert payload["known_loader_shutdown_warning"] is False
    assert payload["extra_tracebacks"] == 1
    assert any("not explained by DataLoader finalizer" in error for error in payload["errors"])


def test_d1r_health_check_rejects_incomplete_final_iteration(tmp_path):
    run_dir, ckpt_dir = _write_run(tmp_path, last_epoch=10, last_itr=998)

    result = _run_health(run_dir, ckpt_dir)

    assert result.returncode != 0
    payload = json.loads((run_dir / "health_check.json").read_text(encoding="utf-8"))
    assert any("last_itr=998" in error for error in payload["errors"])


def test_d1r_health_check_rejects_nonfinite_loss(tmp_path):
    run_dir, ckpt_dir = _write_run(tmp_path, last_loss="nan")

    result = _run_health(run_dir, ckpt_dir)

    assert result.returncode != 0
    payload = json.loads((run_dir / "health_check.json").read_text(encoding="utf-8"))
    assert any("finite loss" in error for error in payload["errors"])


def test_d1r_health_check_rejects_fatal_log_markers(tmp_path):
    run_dir, ckpt_dir = _write_run(tmp_path, launch_text="CUDA error\nout of memory\nloss nan")

    result = _run_health(run_dir, ckpt_dir)

    assert result.returncode != 0
    payload = json.loads((run_dir / "health_check.json").read_text(encoding="utf-8"))
    assert any("cuda error" in error.lower() for error in payload["errors"])
    assert any("out of memory" in error.lower() for error in payload["errors"])
    assert any("'nan'" in error.lower() for error in payload["errors"])


def test_d1r_health_check_rejects_missing_checkpoint_and_config(tmp_path):
    run_dir, ckpt_dir = _write_run(tmp_path, checkpoint=False, config=False)

    result = _run_health(run_dir, ckpt_dir)

    assert result.returncode != 0
    payload = json.loads((run_dir / "health_check.json").read_text(encoding="utf-8"))
    assert any("missing checkpoint" in error for error in payload["errors"])
    assert any("missing config" in error for error in payload["errors"])
