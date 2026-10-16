"""Unit tests for the utils module."""

from __future__ import annotations

from pathlib import Path
from src.utils import load_params


def test_load_params_file_not_found(tmp_path: Path) -> None:
    """Test load_params returns an empty dict when params.yaml does not exist."""
    non_existent_path = tmp_path / "non_existent_params.yaml"
    params = load_params(non_existent_path)
    assert params == {}
