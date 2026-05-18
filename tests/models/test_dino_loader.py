from app.plan_common.models import dino


def test_dinov2_loader_prefers_existing_local_hub_cache(monkeypatch, tmp_path):
    hub_dir = tmp_path / "hub"
    cache_dir = hub_dir / "facebookresearch_dinov2_main"
    cache_dir.mkdir(parents=True)
    calls = []

    def fake_load(repo_or_dir, name, **kwargs):
        calls.append((repo_or_dir, name, kwargs))
        return object()

    monkeypatch.setattr(dino.torch.hub, "get_dir", lambda: str(hub_dir))
    monkeypatch.setattr(dino.torch.hub, "load", fake_load)

    model = dino.load_dinov2_model("dinov2_vits14")

    assert model is not None
    assert calls == [(str(cache_dir), "dinov2_vits14", {"source": "local"})]


def test_dinov2_loader_falls_back_to_remote_when_cache_missing(monkeypatch, tmp_path):
    hub_dir = tmp_path / "hub"
    hub_dir.mkdir()
    calls = []

    def fake_load(repo_or_dir, name, **kwargs):
        calls.append((repo_or_dir, name, kwargs))
        return object()

    monkeypatch.setattr(dino.torch.hub, "get_dir", lambda: str(hub_dir))
    monkeypatch.setattr(dino.torch.hub, "load", fake_load)

    model = dino.load_dinov2_model("dinov2_vits14")

    assert model is not None
    assert calls == [("facebookresearch/dinov2", "dinov2_vits14", {})]
