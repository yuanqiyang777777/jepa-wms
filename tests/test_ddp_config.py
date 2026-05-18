import pytest

from app.vjepa_wm.ddp_config import ddp_config_from_env


DDP_ENV_KEYS = [
    "JEPAWM_DDP_STATIC_GRAPH",
    "JEPAWM_DDP_GRADIENT_AS_BUCKET_VIEW",
    "JEPAWM_DDP_BROADCAST_BUFFERS",
    "JEPAWM_DDP_BUCKET_CAP_MB",
    "JEPAWM_DDP_COMM_HOOK",
]


def clear_ddp_env(monkeypatch):
    for key in DDP_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_ddp_config_defaults_preserve_existing_ddp_behavior(monkeypatch):
    clear_ddp_env(monkeypatch)

    cfg = ddp_config_from_env()

    assert cfg.static_graph is False
    assert cfg.find_unused_parameters is False
    assert cfg.gradient_as_bucket_view is False
    assert cfg.broadcast_buffers is True
    assert cfg.bucket_cap_mb is None
    assert cfg.comm_hook == "none"
    assert cfg.ddp_kwargs() == {
        "static_graph": False,
        "find_unused_parameters": False,
        "gradient_as_bucket_view": False,
        "broadcast_buffers": True,
    }


def test_ddp_config_reads_safe_optimization_env(monkeypatch):
    clear_ddp_env(monkeypatch)
    monkeypatch.setenv("JEPAWM_DDP_STATIC_GRAPH", "1")
    monkeypatch.setenv("JEPAWM_DDP_GRADIENT_AS_BUCKET_VIEW", "true")
    monkeypatch.setenv("JEPAWM_DDP_BROADCAST_BUFFERS", "0")
    monkeypatch.setenv("JEPAWM_DDP_BUCKET_CAP_MB", "100")
    monkeypatch.setenv("JEPAWM_DDP_COMM_HOOK", "bf16")

    cfg = ddp_config_from_env()

    assert cfg.static_graph is True
    assert cfg.gradient_as_bucket_view is True
    assert cfg.broadcast_buffers is False
    assert cfg.bucket_cap_mb == 100
    assert cfg.comm_hook == "bf16"
    assert cfg.ddp_kwargs() == {
        "static_graph": True,
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
        "broadcast_buffers": False,
        "bucket_cap_mb": 100,
    }


@pytest.mark.parametrize(
    ("env_key", "env_value"),
    [
        ("JEPAWM_DDP_STATIC_GRAPH", "maybe"),
        ("JEPAWM_DDP_BUCKET_CAP_MB", "0"),
        ("JEPAWM_DDP_COMM_HOOK", "fp8"),
    ],
)
def test_ddp_config_rejects_invalid_env(monkeypatch, env_key, env_value):
    clear_ddp_env(monkeypatch)
    monkeypatch.setenv(env_key, env_value)

    with pytest.raises(ValueError):
        ddp_config_from_env()
