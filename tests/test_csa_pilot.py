"""Unit tests for the CSA-MPC offline diagnostic pipeline.

The eval-loop hook is exercised on lab01 (it needs a real EncPredWM + env),
not here. These tests cover the modules the offline diagnostic relies on:

  * ``PCAWhiteningAdapter`` — fit/transform parity, frozen weights.
  * ``SupportMemory`` — building/persisting/loading + metadata validation.
  * ``ConditionalSupportScorer`` — correctness of the kNN signs.
  * ``DiagnosticRecorder`` — schema, save/load round-trip.
  * ``analysis.score_dumps`` + ``analysis.analyze`` — full pipeline on synthetic dumps.
  * ``make_official_csa_eval_config.py`` — the eval YAML generator.
  * ``build_support_memory.py`` — the offline CLI that builds support_<env>.pt.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from evals.simu_env_planning.planning.config_overrides import apply_quick_debug_overrides
from evals.simu_env_planning.planning.planning.csa import support_memory
from evals.simu_env_planning.planning.planning.csa.analysis import (
    analyze,
    analyze_dump_dir,
    score_dumps,
    write_report_json,
)
from evals.simu_env_planning.planning.planning.csa.diagnostic_recorder import (
    DUMP_FILENAME,
    DiagnosticRecorder,
    load_episode_dumps,
)
from evals.simu_env_planning.planning.planning.csa.feature_adapter import PCAWhiteningAdapter
from evals.simu_env_planning.planning.planning.csa.support_memory import SupportMemory
from evals.simu_env_planning.planning.planning.csa.support_scorer import ConditionalSupportScorer


# --------------------------------------------------------------------------
# Quick-debug overrides (unchanged module; sanity test the helper).
# --------------------------------------------------------------------------

def test_quick_debug_keeps_explicit_eval_episode_count():
    cfg = SimpleNamespace(
        meta=SimpleNamespace(quick_debug=True, eval_episodes=2),
        planner=SimpleNamespace(iterations=8, num_samples=64, num_elites=8),
        logging=SimpleNamespace(tqdm_silent=True),
    )

    apply_quick_debug_overrides(cfg)

    assert cfg.meta.eval_episodes == 2
    assert cfg.planner.iterations == 2
    assert cfg.planner.num_samples == 2
    assert cfg.planner.num_elites == 2
    assert cfg.logging.tqdm_silent is False


# --------------------------------------------------------------------------
# PCA-whitening adapter.
# --------------------------------------------------------------------------

def test_pca_whitening_adapter_fit_transform_is_frozen_and_compact():
    torch.manual_seed(0)
    x = torch.randn(20, 5)

    adapter = PCAWhiteningAdapter.fit(x, dim=3)
    z1 = adapter.transform(x)
    z2 = adapter.transform(x + 0.0)

    assert z1.shape == (20, 3)
    assert torch.allclose(z1, z2)
    assert adapter.components.shape == (3, 5)
    assert adapter.scale.shape == (3,)


# --------------------------------------------------------------------------
# Support memory.
# --------------------------------------------------------------------------

def test_support_memory_builds_compact_real_memory_and_preserves_metadata():
    torch.manual_seed(0)
    states = torch.randn(10, 2, 3)
    actions = torch.randn(10, 2)

    memory = SupportMemory.from_tensors(
        states,
        actions,
        pca_dim=4,
        max_memory=6,
        metadata={"env": "toy"},
    )

    assert memory.num_items == 6
    assert memory.state_features.shape == (6, 3)
    assert memory.actions.shape == (6, 2)
    assert memory.metadata["env"] == "toy"
    assert memory.metadata["state_reduction"] == "mean_tokens"
    assert memory.adapter is not None


def test_support_memory_round_trips_support_memory_pt(tmp_path):
    torch.manual_seed(0)
    memory = SupportMemory.from_tensors(
        torch.randn(8, 5),
        torch.randn(8, 2),
        pca_dim=3,
        metadata={"source": "unit"},
    )

    path = tmp_path / "support_memory.pt"
    memory.save(path)
    loaded = SupportMemory.load(path)

    assert loaded.num_items == memory.num_items
    assert loaded.metadata["source"] == "unit"
    assert torch.allclose(loaded.state_features, memory.state_features)
    assert torch.allclose(loaded.actions, memory.actions)


def test_support_memory_rejects_action_dim_or_action_space_mismatch():
    assert hasattr(SupportMemory, "validate_for_runtime")
    memory = SupportMemory(
        state_features=torch.randn(4, 3),
        actions=torch.randn(4, 2),
        metadata={
            "feature_source": "pre_encoded",
            "state_reduction": "mean_tokens",
            "raw_state_dim": 3,
            "state_feature_dim": 3,
            "action_dim": 2,
            "action_space": "model_normalized",
            "frameskip": 5,
            "action_skip": 1,
            "pca_dim": 3,
            "sample_seed": 0,
        },
    )

    memory.validate_for_runtime(
        raw_state_dim=3,
        action_dim=2,
        action_space="model_normalized",
        state_reduction="mean_tokens",
        frameskip=5,
        action_skip=1,
    )
    try:
        memory.validate_for_runtime(
            raw_state_dim=3,
            action_dim=3,
            action_space="model_normalized",
            state_reduction="mean_tokens",
            frameskip=5,
            action_skip=1,
        )
    except ValueError as exc:
        assert "action_dim" in str(exc)
    else:
        raise AssertionError("Expected action_dim mismatch to raise")

    try:
        memory.validate_for_runtime(
            raw_state_dim=3,
            action_dim=2,
            action_space="env_denormalized",
            state_reduction="mean_tokens",
            frameskip=5,
            action_skip=1,
        )
    except ValueError as exc:
        assert "action_space" in str(exc)
    else:
        raise AssertionError("Expected action_space mismatch to raise")


def test_mean_tokens_reduction_keeps_runtime_and_memory_dims_consistent():
    assert hasattr(support_memory, "compact_state_encoding")
    states = {
        "visual": torch.randn(6, 3, 1, 2, 2, 4),
        "proprio": torch.randn(6, 3, 2, 4),
    }
    actions = torch.randn(6, 2)
    memory = SupportMemory.from_tensors(
        states,
        actions,
        pca_dim=3,
        state_reduction="mean_tokens",
        max_memory=None,
        sample_seed=7,
        metadata={"frameskip": 5, "action_skip": 1},
    )

    compact = support_memory.compact_state_encoding(states, reduction="mean_tokens")
    query = {key: value[:2] for key, value in states.items()}
    query_features = memory.transform_query_state(query, raw_state=True)

    assert compact.shape == (6, 8)
    assert memory.metadata["raw_state_dim"] == 8
    assert memory.metadata["state_feature_dim"] == 3
    assert query_features.shape == (2, 3)


# --------------------------------------------------------------------------
# Conditional support scorer.
# --------------------------------------------------------------------------

def test_conditional_support_scorer_detects_supported_and_unsupported_actions():
    state_memory = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [3.0, 3.0],
            [3.1, 3.0],
        ]
    )
    actions = torch.tensor(
        [
            [1.0, 0.0],
            [1.1, 0.0],
            [-1.0, 0.0],
            [-1.1, 0.0],
        ]
    )
    memory = SupportMemory(state_features=state_memory, actions=actions)
    scorer = ConditionalSupportScorer(memory, state_k=2, action_k=2, beta=1.0)

    supported = scorer.score(torch.tensor([[0.05, 0.0]]), torch.tensor([[1.05, 0.0]]), raw_state=False)
    unsupported = scorer.score(torch.tensor([[0.05, 0.0]]), torch.tensor([[-1.05, 0.0]]), raw_state=False)

    assert supported["state_score"].shape == (1,)
    assert supported["action_score"].item() < unsupported["action_score"].item()
    assert supported["csa_score"].item() < unsupported["csa_score"].item()


# --------------------------------------------------------------------------
# DiagnosticRecorder schema + round-trip.
# --------------------------------------------------------------------------

def test_diagnostic_recorder_round_trip(tmp_path):
    rec = DiagnosticRecorder(
        episode_id=7,
        env="wall",
        metadata={"frameskip": 5, "action_skip": 1, "ckpt_path": "jepa_wm_wall.pth.tar"},
        goal_state=torch.randn(384),
    )
    rec.record_step(
        replan_idx=0,
        decision_state=torch.randn(384),
        imagined_states=torch.randn(6, 384),
        real_states=torch.randn(6, 384),
        selected_actions=torch.randn(6, 2),
        predicted_terminal_cost=0.42,
        real_terminal_cost=0.77,
        state_dist=1.5,
        step_success=False,
    )
    rec.record_step(
        replan_idx=1,
        decision_state=torch.randn(384),
        imagined_states=torch.randn(6, 384),
        real_states=torch.randn(6, 384),
        selected_actions=torch.randn(6, 2),
        predicted_terminal_cost=0.30,
        real_terminal_cost=0.05,
        state_dist=0.3,
        step_success=True,
    )
    dump_path = tmp_path / "csa_dumps" / DUMP_FILENAME
    rec.save(dump_path, episode_success=True)

    dumps = load_episode_dumps(tmp_path)
    assert len(dumps) == 1
    payload = dumps[0]
    assert payload["episode_id"] == 7
    assert payload["env"] == "wall"
    assert payload["episode_success"] == 1
    assert payload["num_steps"] == 2
    assert payload["steps"][0]["replan_idx"] == 0
    assert payload["steps"][0]["step_success"] == 0
    assert payload["steps"][1]["step_success"] == 1
    assert payload["steps"][0]["imagined_states"].shape == (6, 384)
    assert payload["steps"][0]["real_states"].shape == (6, 384)
    assert payload["goal_state"].shape == (384,)


# --------------------------------------------------------------------------
# score_dumps + analyze on synthetic data.
# --------------------------------------------------------------------------

def _make_synthetic_dump(*, episode_id, env, episode_success, num_steps=2, depth=4, d_state=8, action_dim=2):
    """Build one well-formed dump payload (no .pt I/O)."""
    steps = []
    for k in range(num_steps):
        steps.append(
            {
                "replan_idx": k,
                "decision_state": torch.randn(d_state),
                "imagined_states": torch.randn(depth, d_state),
                "real_states": torch.randn(depth, d_state),
                "selected_actions": torch.randn(depth, action_dim),
                "predicted_terminal_cost": float(0.1 * (k + 1)),
                "real_terminal_cost": float(0.1 * (k + 1) + (0.5 if not episode_success else 0.0)),
                "state_dist": float(1.5 - 0.3 * k),
                "step_success": int(bool(episode_success and k == num_steps - 1)),
            }
        )
    return {
        "schema_version": 1,
        "episode_id": int(episode_id),
        "env": str(env),
        "episode_success": int(bool(episode_success)),
        "goal_state": torch.randn(d_state),
        "metadata": {"frameskip": 5, "action_skip": 1},
        "num_steps": num_steps,
        "steps": steps,
    }


def test_score_dumps_attaches_per_query_scores():
    torch.manual_seed(0)
    d_state, action_dim = 8, 2
    memory = SupportMemory.from_tensors(
        torch.randn(64, d_state),
        torch.randn(64, action_dim),
        pca_dim=4,
        state_reduction="flatten",
        metadata={"frameskip": 5, "action_skip": 1, "action_space": "model_normalized"},
    )
    dumps = [_make_synthetic_dump(episode_id=0, env="wall", episode_success=False, d_state=d_state, action_dim=action_dim)]

    records = score_dumps(dumps, memory, state_k=4, action_k=4, beta=1.0)

    # score_dumps iterates depth in range(n_actions). selected_actions has shape
    # (4, 2) → n_actions=4 → depths {0, 1, 2, 3}. Two replan steps → 8 queries.
    # (Note: imagined_states[H-1] is the terminal imagined state with no
    # emanating action, so it is never queried.)
    expected_queries = 2 * 4
    assert len(records) == expected_queries
    for r in records:
        assert "state_score" in r and "action_score" in r and "csa_score" in r
        assert "rollout_error" in r  # depth>=1 has real-vs-imagined MSE
        assert "depth" in r and r["depth"] in {0, 1, 2, 3}


def test_analyze_reports_all_five_diagnostics_on_synthetic_dumps():
    torch.manual_seed(1)
    d_state, action_dim = 8, 2
    memory = SupportMemory.from_tensors(
        torch.randn(64, d_state),
        torch.randn(64, action_dim),
        pca_dim=4,
        state_reduction="flatten",
        metadata={"frameskip": 5, "action_skip": 1, "action_space": "model_normalized"},
    )
    dumps = [
        _make_synthetic_dump(episode_id=0, env="wall", episode_success=False, d_state=d_state, action_dim=action_dim),
        _make_synthetic_dump(episode_id=1, env="wall", episode_success=True, d_state=d_state, action_dim=action_dim),
        _make_synthetic_dump(episode_id=2, env="wall", episode_success=False, d_state=d_state, action_dim=action_dim),
    ]

    report = analyze(dumps, memory, state_k=4, action_k=4, beta=1.0)

    assert report["env"] == "wall"
    assert report["num_episodes"] == 3
    assert report["num_failed_episodes"] == 2
    assert "diagnostic_0_failure_mode" in report
    assert "diagnostic_1_exploitation_gap" in report
    assert "diagnostic_2_conditioned_correlation" in report
    assert "diagnostic_3_buckets" in report
    assert "diagnostic_4_false_negative_by_depth" in report
    assert "go_no_go" in report
    # Diagnostic 1: exploitation gap is real - predicted. We set failures to have
    # real = pred + 0.5 → mean_gap > 0 on the failed-episode subset.
    diag1 = report["diagnostic_1_exploitation_gap"]
    assert int(diag1.get("num_replan_steps", 0)) > 0
    assert float(diag1["mean_gap"]) > 0.0


def test_analyze_dump_dir_loads_and_runs_on_tmp_dumps(tmp_path):
    torch.manual_seed(2)
    d_state, action_dim = 8, 2
    # Write 2 dumps under nested ep_<N> dirs (the convention the eval emits).
    for ep in range(2):
        payload = _make_synthetic_dump(
            episode_id=ep, env="wall", episode_success=bool(ep), d_state=d_state, action_dim=action_dim
        )
        dump_path = tmp_path / f"ep_{ep}" / "csa_dumps" / DUMP_FILENAME
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, dump_path)

    memory = SupportMemory.from_tensors(
        torch.randn(64, d_state),
        torch.randn(64, action_dim),
        pca_dim=4,
        state_reduction="flatten",
        metadata={"frameskip": 5, "action_skip": 1, "action_space": "model_normalized"},
    )
    memory_path = tmp_path / "support_wall.pt"
    memory.save(memory_path)

    report = analyze_dump_dir(
        dump_dir=tmp_path,
        memory_path=memory_path,
        state_k=4,
        action_k=4,
    )
    assert report["num_episodes"] == 2

    out_json = tmp_path / "report.json"
    write_report_json(report, out_json)
    assert json.loads(out_json.read_text())["num_episodes"] == 2


# --------------------------------------------------------------------------
# make_official_csa_eval_config.py CLI.
# --------------------------------------------------------------------------

def test_make_official_csa_eval_config_emits_eval_only_csa_block(tmp_path):
    output = tmp_path / "wall_csa_official.yaml"

    subprocess.run(
        [
            sys.executable,
            "experiments/scripts/csa/make_official_csa_eval_config.py",
            "--env",
            "wall",
            "--output",
            str(output),
            "--eval-episodes",
            "2",
            "--quick-debug",
        ],
        check=True,
    )

    cfg = yaml.safe_load(output.read_text())
    assert cfg["checkpoint_folder"] == "${JEPAWM_CKPT}"
    assert cfg["model_kwargs"]["checkpoint"] == "jepa_wm_wall.pth.tar"
    assert cfg["folder"] == "${JEPAWM_LOGS}/csa_pilot/wall_official_jepa_wm"
    assert cfg["meta"]["quick_debug"] is True
    assert cfg["meta"]["eval_episodes"] == 2
    # New design: analysis is offline -> eval config only carries `enabled` (and optional dump_dir).
    assert cfg["csa_diagnostics"]["enabled"] is True
    assert "memory_path" not in cfg["csa_diagnostics"]


def test_make_official_csa_eval_config_supports_all_five_phase1_envs(tmp_path):
    for env in ["pusht", "wall", "maze", "mw-reach", "mw-reach-wall"]:
        output = tmp_path / f"{env}.yaml"
        subprocess.run(
            [
                sys.executable,
                "experiments/scripts/csa/make_official_csa_eval_config.py",
                "--env",
                env,
                "--output",
                str(output),
                "--eval-episodes",
                "2",
            ],
            check=True,
        )
        cfg = yaml.safe_load(output.read_text())
        assert cfg["csa_diagnostics"]["enabled"] is True
        assert cfg["model_kwargs"]["checkpoint"].endswith(".pth.tar")


# --------------------------------------------------------------------------
# build_support_memory.py CLI.
# --------------------------------------------------------------------------

def test_build_support_memory_random_sampling_is_seeded_and_reproducible(tmp_path):
    states_path = tmp_path / "states.pt"
    actions_path = tmp_path / "actions.pt"
    memory_a_path = tmp_path / "support_memory_a.pt"
    memory_b_path = tmp_path / "support_memory_b.pt"
    states = torch.arange(60, dtype=torch.float32).reshape(20, 3)
    actions = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    torch.save(states, states_path)
    torch.save(actions, actions_path)

    base_cmd = [
        sys.executable,
        "experiments/scripts/csa/build_support_memory.py",
        "--states",
        str(states_path),
        "--actions",
        str(actions_path),
        "--pca-dim",
        "3",
        "--max-memory",
        "6",
        "--pca-fit-samples",
        "5",
        "--sample-seed",
        "123",
        "--state-reduction",
        "mean_tokens",
        "--metadata-json",
        '{"env":"toy","frameskip":5,"action_skip":1}',
    ]
    subprocess.run(base_cmd + ["--output", str(memory_a_path)], check=True)
    subprocess.run(base_cmd + ["--output", str(memory_b_path)], check=True)

    memory_a = SupportMemory.load(memory_a_path)
    memory_b = SupportMemory.load(memory_b_path)

    assert torch.allclose(memory_a.actions, memory_b.actions)
    assert memory_a.metadata["sample_seed"] == 123
    assert memory_a.metadata["state_reduction"] == "mean_tokens"
    assert not torch.allclose(memory_a.actions, actions[:6])
