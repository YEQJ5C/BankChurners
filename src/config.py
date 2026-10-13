"""Application configuration module using Pydantic Settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# A projekt gyökérkönyvtárának kiszámítása
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CHURN_",
        extra="ignore",
    )

    # Core project paths (fully independent and safe)
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_dir: Path = BASE_DIR / "data" / "raw"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    models_dir: Path = BASE_DIR / "models"
    logs_dir: Path = BASE_DIR / "logs"

    # Logging parameters
    log_level: str = "INFO"

    # Model parameters
    random_seed: int = 42
    registered_model_name: str = "bank_churn_model"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


# Globális konfig példány
settings = Settings()
