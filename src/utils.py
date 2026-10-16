"""Utility functions for loading parameters and configurations."""

from __future__ import annotations

from pathlib import Path
import yaml


def load_params(params_path: str | Path = "params.yaml") -> dict:
    """Loads experiment parameters from a YAML file."""
    path = Path(params_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
