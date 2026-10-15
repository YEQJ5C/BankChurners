"""
Configuration Management Module using Pydantic Settings.
Centralizes all paths, hyperparameters, and environment variables.
"""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global application settings. Automatically loads from environment variables
    or a .env file if present, prefixed with CHURN_.
    """
    # Alap útvonalak
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    models_dir: Path = base_dir / "models"
    metrics_dir: Path = base_dir / "metrics"

    # + ÚJ: Itt adjuk hozzá a logs_dir-t, amit a logger keres
    logs_dir: Path = base_dir / "logs"

    # + ÚJ: Itt adjuk hozzá a log_level-t, amit a logger keres
    log_level: str = "INFO"

    # MLflow beállítások
    mlflow_experiment_name: str = "bank_churn_experiment"
    registered_model_name: str = "bank_churn_model"

    # Reproducibility & Pipeline paraméterek
    random_seed: int = 42
    test_size: float = 0.2

    # Modell hiperparaméterek
    rf_n_estimators: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CHURN_",
        extra="ignore",
    )


# Globális konfigurációs példány
settings = Settings()
