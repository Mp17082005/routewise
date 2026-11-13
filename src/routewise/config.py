"""Config loading. Weights and thresholds live in YAML, never in code."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

PKG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PKG_ROOT / "config" / "weights.yaml"
DATA_DIR = PKG_ROOT / "data"
REPORTS_DIR = PKG_ROOT / "reports"

CRITERIA = ("cost", "time", "reliability", "risk")


@functools.lru_cache(maxsize=8)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    if "profiles" not in cfg:
        raise ValueError("config: missing 'profiles'")
    for name, w in cfg["profiles"].items():
        missing = set(CRITERIA) - set(w)
        if missing:
            raise ValueError(f"config: profile {name} missing weights {sorted(missing)}")
        extra = set(w) - set(CRITERIA)
        if extra:
            raise ValueError(f"config: profile {name} has unknown weights {sorted(extra)}")
        total = sum(w.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"config: profile {name} weights sum to {total}, expected 1.0")
        if any(v < 0 for v in w.values()):
            raise ValueError(f"config: profile {name} has a negative weight")


def profile_for(priority: str, cargo_type: str, cfg: dict[str, Any] | None = None) -> str:
    """Critical cargo overrides the priority-derived profile."""
    cfg = cfg or load_config()
    if cargo_type in cfg.get("critical_cargo_types", []):
        return "critical"
    return priority


def weights_for(profile: str, cfg: dict[str, Any] | None = None) -> dict[str, float]:
    cfg = cfg or load_config()
    try:
        return dict(cfg["profiles"][profile])
    except KeyError as exc:
        raise KeyError(f"unknown profile {profile!r}; known: {sorted(cfg['profiles'])}") from exc
