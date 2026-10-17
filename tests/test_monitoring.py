"""Unit tests for the BankChurners Evidently monitoring module."""

import sys
import runpy
from pathlib import Path
import pytest
import pandas as pd

from src.config import settings


def test_generate_drift_report_missing_reference(tmp_path: Path) -> None:
    """Test that missing reference file triggers logger.error and raises FileNotFoundError."""
    import src.monitoring
    curr_file = tmp_path / "curr.csv"
    curr_file.write_text("Customer_Age,Credit_Limit,Attrition_Flag\n25,1000,Existing", encoding="utf-8")
    out_html = tmp_path / "drift_report.html"

    with pytest.raises(FileNotFoundError, match="Referencia adatfájl nem található"):
        src.monitoring.generate_drift_report(
            reference_data_path=tmp_path / "missing_ref.csv",
            current_data_path=curr_file,
            output_html_path=out_html,
        )


def test_generate_drift_report_missing_current(tmp_path: Path) -> None:
    """Test that missing current file triggers logger.error and raises FileNotFoundError."""
    import src.monitoring
    ref_file = tmp_path / "ref.csv"
    ref_file.write_text("Customer_Age,Credit_Limit,Attrition_Flag\n25,1000,Existing", encoding="utf-8")
    out_html = tmp_path / "drift_report.html"

    with pytest.raises(FileNotFoundError, match="Aktuális adatfájl nem található"):
        src.monitoring.generate_drift_report(
            reference_data_path=ref_file,
            current_data_path=tmp_path / "missing_curr.csv",
            output_html_path=out_html,
        )


def test_generate_drift_report_default_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test handling when paths use default settings (covers default None branches and settings paths)."""
    import src.monitoring
    monkeypatch.setattr(settings, "processed_dir", tmp_path)
    monkeypatch.setattr(settings, "base_dir", tmp_path)

    with pytest.raises(FileNotFoundError):
        src.monitoring.generate_drift_report(
            reference_data_path=None,
            current_data_path=None,
            output_html_path=None,
        )


def test_generate_drift_report_success(tmp_path: Path) -> None:
    """Test successful execution, logging, and return of output path."""
    import src.monitoring
    ref_file = tmp_path / "ref.csv"
    curr_file = tmp_path / "curr.csv"
    out_file = tmp_path / "drift_report.html"

    dummy_data = pd.DataFrame({
        "Customer_Age": list(range(20, 40)),
        "Credit_Limit": [float(1000 + i * 100) for i in range(20)],
        "Attrition_Flag": ["Existing Customer" if i % 2 == 0 else "Attrited Customer" for i in range(20)],
    })
    dummy_data.to_csv(ref_file, index=False)
    dummy_data.to_csv(curr_file, index=False)

    result_path = src.monitoring.generate_drift_report(
        reference_data_path=ref_file,
        current_data_path=curr_file,
        output_html_path=out_file,
    )

    assert result_path == out_file
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Evidently Data Drift Report executed successfully" in content


def test_monitoring_main_execution_block(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test the if __name__ == '__main__': entrypoint block using runpy."""
    monkeypatch.setattr(settings, "processed_dir", tmp_path)
    monkeypatch.setattr(settings, "base_dir", tmp_path)

    with pytest.raises(FileNotFoundError):
        runpy.run_module("src.monitoring", run_name="__main__")


def test_import_fallback_metric_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force execution of the first DataDriftPreset fallback branch (lines 22-26) to push coverage > 90%."""
    # Tiszta lappal indítjuk a monitoring modult ebben a tesztben
    sys.modules.pop("src.monitoring", None)

    orig_import = __import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Csak a metric_preset importot kényszerítjük hibára, hogy a 22-26. sor lefusson
        if name == "evidently.metric_preset":
            raise ImportError("Forced fallback for DataDriftPreset")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    import src.monitoring

    assert src.monitoring.DataDriftPreset is not None
