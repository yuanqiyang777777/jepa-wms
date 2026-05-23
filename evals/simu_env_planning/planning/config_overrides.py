def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def _cfg_set(cfg, key, value):
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def apply_quick_debug_overrides(cfg):
    meta = _cfg_get(cfg, "meta")
    if not _cfg_get(meta, "quick_debug", False):
        return

    if _cfg_get(meta, "eval_episodes", None) is None:
        _cfg_set(meta, "eval_episodes", 1)

    planner = _cfg_get(cfg, "planner")
    if planner is not None:
        _cfg_set(planner, "iterations", 2)
        _cfg_set(planner, "num_samples", 2)
        _cfg_set(planner, "num_elites", 2)

    logging_cfg = _cfg_get(cfg, "logging")
    if logging_cfg is not None:
        _cfg_set(logging_cfg, "tqdm_silent", False)
