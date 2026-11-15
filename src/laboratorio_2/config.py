from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str = (
        "postgresql+asyncpg://novamercado:novamercado@localhost:5433/novamercado"
    )
    data_dir: Path = Path("data/raw")
    model_cache_dir: Path = Path("models")

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_model_name: str = "novamercado-forecast"
    mlflow_alias: str = "champion"
    model_reload_seconds: float = 30.0

    airflow_base_url: str = "http://localhost:8080"
    airflow_username: str = "admin"
    airflow_password: str = "admin"
    airflow_dag_id: str = "novamercado_retrain"
    airflow_request_timeout_seconds: float = 10.0
    internal_task_token: str = "local-internal-token"

    health_probe_timeout_seconds: float = 3.0
    validation_days: int = 28
    minimum_training_days: int = 42
    promotion_cost_improvement: float = 0.01
    promotion_mae_tolerance: float = 0.05

    bootstrap_on_start: bool = False
    seed_demo_data: bool = False

    @property
    def model_cache_path(self) -> Path:
        return self.model_cache_dir / f"{self.mlflow_alias}.joblib"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
