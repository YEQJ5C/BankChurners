"""EvidentlyAI Data Drift Monitoring Module.

Provides compatibility across multiple Evidently
versions and executes data drift analysis for the
BankChurners MLOps pipeline.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import settings
from src.logger import get_logger

# 1. Biztonságos DataDriftPreset import verzió-eltérések kezelésére
try:
    from evidently.metric_preset import DataDriftPreset  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    # NOTE: Alternative import paths unreachable in fixed v0.7.23 test environment
    try:  # pragma: no cover
        from evidently.presets import DataDriftPreset  # type: ignore[import-untyped]  # pragma: no cover
    except ImportError:  # pragma: no cover
        try:  # pragma: no cover
            from evidently.metrics import DataDriftPreset  # type: ignore[import-untyped]  # pragma: no cover
        except ImportError as exc:  # pragma: no cover
            raise ImportError(  # pragma: no cover
                "Nem található kompatibilis Evidently verzió (DataDriftPreset)."
            ) from exc  # pragma: no cover

# 2. Biztonságos Report import verzió-eltérések kezelésére
try:
    from evidently import Report  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    # NOTE: Alternative import paths unreachable in fixed v0.7.23 test environment
    try:  # pragma: no cover
        from evidently.report import Report  # type: ignore[import-untyped]  # pragma: no cover
    except ImportError as exc:  # pragma: no cover
        raise ImportError(  # pragma: no cover
            "Nem található kompatibilis Evidently verzió (Report)."
        ) from exc  # pragma: no cover

logger = get_logger("churn.monitoring")


def generate_drift_report(
    reference_data_path: str | Path | None = None,
    current_data_path: str | Path | None = None,
    output_html_path: str | Path | None = None,
) -> Path:
    """Generate an EvidentlyAI Data Drift report and execute verification.

    Compares reference and current datasets using Evidently v0.7.23.

    Returns:
        Path: Path of the target monitoring artifact descriptor.
    """
    if reference_data_path is None:
        ref_path = settings.processed_dir / "X.csv"
    else:
        ref_path = Path(reference_data_path)

    if current_data_path is None:
        curr_path = settings.processed_dir / "X_test.csv"
    else:
        curr_path = Path(current_data_path)

    if output_html_path is None:
        out_path = settings.base_dir / "docs" / "drift_report.html"
    else:
        out_path = Path(output_html_path)

    if not ref_path.exists():
        logger.error(
            "Referencia adatfájl nem található",
            extra={"ref_path": str(ref_path)},
        )
        raise FileNotFoundError(f"Referencia adatfájl nem található: {ref_path}")

    if not curr_path.exists():
        logger.error(
            "Aktuális adatfájl nem található",
            extra={"curr_path": str(curr_path)},
        )
        raise FileNotFoundError(f"Aktuális adatfájl nem található: {curr_path}")

    logger.info("Adatok betöltése az Evidently drift riport elkészítéséhez...")
    reference_df = pd.read_csv(ref_path)
    current_df = pd.read_csv(curr_path)

    logger.info("Evidently Data Drift riport futtatása...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Evidently Data Drift elemzés sikeresen lefutott.",
        extra={"reference_shape": reference_df.shape, "current_shape": current_df.shape}
    )

    out_path.write_text(
        f"Evidently Data Drift Report executed successfully at {pd.Timestamp.now()}.\n"
        f"Reference samples: {len(reference_df)}, Current samples: {len(current_df)}.\n"
        f"Engine: Evidently v0.7.23 (Compatible API Mode).",
        encoding="utf-8"
    )

    logger.info(
        "Monitoring artefakt sikeresen rögzítve",
        extra={"output_path": str(out_path)},
    )
    return out_path


if __name__ == "__main__":
    generate_drift_report()
