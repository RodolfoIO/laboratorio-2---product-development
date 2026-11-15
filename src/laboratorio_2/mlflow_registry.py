from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from laboratorio_2.config import Settings


@dataclass(frozen=True)
class RegisteredVersion:
    name: str
    version: str
    run_id: str
    algorithm: str


class MlflowRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any = None

    @property
    def client(self):
        if self._client is None:
            from mlflow import MlflowClient

            self._client = MlflowClient(
                tracking_uri=self.settings.mlflow_tracking_uri,
                registry_uri=self.settings.mlflow_tracking_uri,
            )
        return self._client

    def health(self) -> None:
        self.client.search_experiments(max_results=1)

    def get_alias(self) -> tuple[str, str] | None:
        from mlflow.exceptions import MlflowException

        try:
            version = self.client.get_model_version_by_alias(
                self.settings.mlflow_model_name,
                self.settings.mlflow_alias,
            )
        except MlflowException as exc:
            if exc.error_code == "RESOURCE_DOES_NOT_EXIST" or "not found" in str(exc).lower():
                return None
            raise
        return str(version.version), str(version.run_id)

    def download_alias(self, destination: Path) -> Path:
        import mlflow

        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        model_uri = (
            f"models:/{self.settings.mlflow_model_name}"
            f"@{self.settings.mlflow_alias}"
        )
        with tempfile.TemporaryDirectory(prefix="novamercado-model-") as temp_dir:
            downloaded = Path(mlflow.artifacts.download_artifacts(model_uri, dst_path=temp_dir))
            candidates = list(downloaded.rglob("*.joblib"))
            if not candidates:
                candidates = list(downloaded.rglob("model.joblib"))
            if not candidates:
                raise FileNotFoundError("El modelo registrado no contiene un archivo .joblib")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            shutil.copy2(candidates[0], temporary)
            temporary.replace(destination)
        return destination

    def log_artifact_model(
        self,
        artifact_path: Path,
        *,
        algorithm: str,
        metrics: dict[str, float],
        params: dict[str, Any],
        tags: dict[str, str],
        run_id_hint: str,
    ) -> RegisteredVersion:
        import mlflow
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment("novamercado-forecast")
        with mlflow.start_run(run_name=f"{run_id_hint}-{algorithm}") as run:
            mlflow.set_tags({**tags, "algorithm": algorithm, "retraining_run": run_id_hint})
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(artifact_path), artifact_path="model")
            model_uri = f"runs:/{run.info.run_id}/model"
            registered = mlflow.register_model(
                model_uri,
                name=self.settings.mlflow_model_name,
                tags={"algorithm": algorithm},
            )
            self.client.set_model_version_tag(
                self.settings.mlflow_model_name,
                str(registered.version),
                "algorithm",
                algorithm,
            )
            self.client.set_model_version_tag(
                self.settings.mlflow_model_name,
                str(registered.version),
                "code_revision",
                tags.get("code_revision", ""),
            )
            return RegisteredVersion(
                name=self.settings.mlflow_model_name,
                version=str(registered.version),
                run_id=run.info.run_id,
                algorithm=algorithm,
            )

    def set_champion(self, version: str) -> None:
        self.client.set_registered_model_alias(
            self.settings.mlflow_model_name,
            self.settings.mlflow_alias,
            version,
        )

    def download_version(self, version: str, destination: Path) -> Path:
        import mlflow

        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        with tempfile.TemporaryDirectory(prefix="novamercado-version-") as temp_dir:
            downloaded = Path(
                mlflow.artifacts.download_artifacts(
                    f"models:/{self.settings.mlflow_model_name}/{version}",
                    dst_path=temp_dir,
                )
            )
            candidates = list(downloaded.rglob("*.joblib"))
            if not candidates:
                raise FileNotFoundError("La version registrada no contiene un archivo .joblib")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            shutil.copy2(candidates[0], temporary)
            temporary.replace(destination)
        return destination
