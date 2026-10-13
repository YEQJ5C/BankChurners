"""Unit tests for the environment validation module.

Covers requirement parsing, environment data collection,
report rendering, markdown file saving, and the __main__ block.
"""

import importlib.metadata
import runpy

from src.check_env import (
    collect,
    parse_requirements,
    render,
    save_report,
)


def test_parse_requirements_non_existent(tmp_path):
    """Non-existent requirements file should return empty list."""
    missing_file = tmp_path / "non_existent.txt"
    assert parse_requirements(str(missing_file)) == []


def test_parse_requirements_default_missing(monkeypatch, tmp_path):
    """Test parse_requirements default argument when missing."""
    monkeypatch.chdir(tmp_path)
    result = parse_requirements()
    assert result == []


def test_parse_requirements_parsing(tmp_path):
    """Requirements parsing should handle comments and versions."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text(
        "# This is a comment\n\n"
        "-e .\n"
        "pandas>=2.0.0\n"
        "numpy==1.24.0\n"
        "requests\n"
    )
    packages = parse_requirements(str(req_file))
    assert "bankchurners" in packages
    assert "pandas" in packages
    assert "numpy" in packages
    assert "requests" in packages
    assert "pandas>=2.0.0" not in packages


def test_collect_with_venv(monkeypatch):
    """Collect environment details when VIRTUAL_ENV is active."""
    monkeypatch.setenv("VIRTUAL_ENV", "/fake/venv")

    def mock_version(pkg):
        if pkg == "missing_pkg":
            raise importlib.metadata.PackageNotFoundError
        return "1.2.3"

    monkeypatch.setattr(importlib.metadata, "version", mock_version)
    monkeypatch.setattr(
        "src.check_env.parse_requirements",
        lambda path="requirements.txt": ["pandas", "missing_pkg"],
    )

    checks = collect()
    assert checks["virtual_env"] == "/fake/venv"
    assert checks["packages"]["pandas"] == "1.2.3"
    assert checks["packages"]["missing_pkg"] == "Not installed ❌"
    assert "python_version" in checks
    assert "tools" in checks
    assert "directories" in checks


def test_collect_without_venv(monkeypatch):
    """Collect environment details when no VIRTUAL_ENV is present."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(
        "src.check_env.parse_requirements",
        lambda path="requirements.txt": [],
    )

    checks = collect()
    assert "No active virtual environment" in checks["virtual_env"]


def test_render_report():
    """Render checks dictionary into a formatted Markdown string."""
    checks = {
        "platform": "Linux-5.15-test",
        "python_executable": "/usr/bin/python",
        "python_version": "3.10.12",
        "virtual_env": "/venv",
        "packages": {"pandas": "2.0.0"},
        "tools": {"git": True, "dvc": False},
        "directories": {"src": True, "data": False},
    }
    report = render(checks)
    assert "# BankChurners - Environment Check Report" in report
    assert "pandas" in report
    assert "✅ Available" in report
    assert "❌ NOT FOUND" in report
    assert "✅ Exists" in report
    assert "❌ Missing" in report


def test_save_report(tmp_path):
    """Save rendered report into the specified docs directory."""
    content = "# Test Report Content"
    saved_path = save_report(content, docs_dir=str(tmp_path))

    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == content


def test_check_env_main_block(monkeypatch, tmp_path):
    """Test the if __name__ == '__main__': execution block."""
    monkeypatch.setattr("sys.argv", ["check_env.py"])

    monkeypatch.setattr(
        "src.check_env.save_report",
        lambda report, docs_dir="docs":
        tmp_path / "env_report_test.md"
    )

    runpy.run_module("src.check_env", run_name="__main__")
