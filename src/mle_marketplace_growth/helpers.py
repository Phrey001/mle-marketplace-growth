"""Shared helper utilities."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

from dateutil.relativedelta import relativedelta
import yaml


def generate_snapshot_dates(panel_end_date: date) -> list[date]:
    """Build 12 monthly snapshots ending at panel_end_date (inclusive)."""
    snapshots = [panel_end_date + relativedelta(months=offset) for offset in range(-11, 1)]
    if snapshots[-1] != panel_end_date:
        raise ValueError("Derived monthly snapshot panel does not end on panel_end_date")
    return snapshots


def cfg_required(cfg: dict, key: str):
    """Return required config value or raise a consistent error."""
    if key not in cfg or cfg[key] is None:
        raise ValueError(f"Missing required config key in YAML: {key}")
    return cfg[key]


def load_yaml_defaults(path_value: str | None, label: str) -> dict:
    """Load YAML defaults from file; return empty dict when no path is provided."""
    if not path_value:
        return {}
    config_path = Path(path_value)
    if not config_path.exists():
        raise FileNotFoundError(f"{label} file not found: {config_path}")
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"{label} file must use .yaml or .yml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} file must contain a key-value object")
    return payload


def read_json(path: Path) -> dict:
    """Read JSON file into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list) -> None:
    """Write JSON payload to file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
