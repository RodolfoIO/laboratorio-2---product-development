from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from laboratorio_2.config import Settings
from laboratorio_2.contracts import ModelInfo
from laboratorio_2.features import build_training_features
from laboratorio_2.forecasting import ForecastArtifact, LightGBMForecaster, SeasonalBaseline
from laboratorio_2.metrics import ForecastMetrics, should_promote
from laboratorio_2.mlflow_registry import MlflowRegistry, RegisteredVersion
from laboratorio_2.repository import ProductRepository

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TrainingOutcome:
    run_id: str
    promoted: bool
    version: str
    motive: str
    candidate_metrics: ForecastMetrics
    baseline_metrics: ForecastMetrics
    candidate: ForecastArtifact
    baseline: ForecastArtifact
    candidate_version: RegisteredVersion | None
    baseline_version: RegisteredVersion | None


def _data_fingerprint(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def _split_features(
    features: pd.DataFrame, validation_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(features["fecha"]).dt.date.unique())
    if len(dates) < validation_days + 14:
        raise ValueError(
            f"Se requieren al menos {validation_days + 14} dias de historia; hay {len(dates)}"
        )
    validation_start = dates[-validation_days]
    train = features[pd.to_datetime(features["fecha"]).dt.date < validation_start].copy()
    validation = features[
        pd.to_datetime(features["fecha"]).dt.date >= validation_start
    ].copy()
    if train.empty or validation.empty:
        raise ValueError("La separacion temporal dejo un conjunto vacio")
    return train, validation


class TrainingPipeline:
    def __init__(
        self,
        settings: Settings,
        repository: ProductRepository,
        registry: MlflowRegistry | None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.registry = registry

    async def train(
        self,
        run_id: str,
        *,
        log_mlflow: bool = True,
    ) -> TrainingOutcome:
        catalog = await self.repository.get_catalog()
        if not catalog.costs:
            raise ValueError("No existen catalogos ni costos importados")
        sales = await self.repository.load_sales()
        holidays = await self.repository.load_holidays()
        features = build_training_features(sales, catalog, holidays)
        train_frame, validation_frame = _split_features(
            features, self.settings.validation_days
        )
        fingerprint = _data_fingerprint(sales)
        revision = _code_revision()
        trained_through = max(pd.to_datetime(train_frame["fecha"]).dt.date)

        baseline_model = SeasonalBaseline().fit(train_frame, catalog)
        baseline_metrics, _, _ = baseline_model.evaluate(validation_frame, catalog)
        baseline = ForecastArtifact(
            algorithm="baseline",
            baseline=baseline_model,
            ml_model=LightGBMForecaster(),
            metrics=baseline_metrics.as_dict(),
            trained_through=trained_through,
            data_fingerprint=fingerprint,
            code_revision=revision,
        )

        ml_model = LightGBMForecaster().fit(train_frame, catalog)
        candidate_metrics, _, _ = ml_model.evaluate(validation_frame, catalog)
        candidate = ForecastArtifact(
            algorithm="lightgbm",
            baseline=baseline_model,
            ml_model=ml_model,
            metrics=candidate_metrics.as_dict(),
            trained_through=trained_through,
            data_fingerprint=fingerprint,
            code_revision=revision,
        )

        champion: ForecastArtifact | None = None
        champion_metrics: ForecastMetrics | None = None
        champion_version: str | None = None
        if self.registry is not None and log_mlflow:
            alias = self.registry.get_alias()
            if alias is not None:
                champion_version = alias[0]
                with tempfile.TemporaryDirectory(prefix="champion-") as temp_dir:
                    path = self.registry.download_version(
                        champion_version, Path(temp_dir) / "champion.joblib"
                    )
                    champion = joblib.load(path)
                if not isinstance(champion, ForecastArtifact):
                    raise TypeError("El campeon registrado no es un ForecastArtifact")
                champion_metrics = champion.evaluate(validation_frame, catalog)

        if champion_metrics is None:
            promote_candidate = candidate_metrics.business_cost < baseline_metrics.business_cost
            if promote_candidate:
                selected = candidate
                promoted = True
                motive = (
                    "No existia campeon; LightGBM supero al baseline con costo "
                    f"{candidate_metrics.business_cost:.2f} vs "
                    f"{baseline_metrics.business_cost:.2f}"
                )
            else:
                selected = baseline
                promoted = True
                motive = (
                    "No existia campeon; se promovio el baseline porque LightGBM no lo "
                    f"supero ({candidate_metrics.business_cost:.2f} vs "
                    f"{baseline_metrics.business_cost:.2f})"
                )
        else:
            promote_candidate, motive = should_promote(
                candidate_metrics,
                champion_metrics,
                baseline_metrics,
                cost_improvement=self.settings.promotion_cost_improvement,
                mae_tolerance=self.settings.promotion_mae_tolerance,
            )
            promoted = promote_candidate
            selected = candidate if promote_candidate else champion
        logger.info(
            "Retraining %s decision: promoted=%s champion=%s selected=%s motive=%s",
            run_id,
            promoted,
            champion_version,
            selected.algorithm,
            motive,
        )

        candidate_version: RegisteredVersion | None = None
        baseline_version: RegisteredVersion | None = None
        selected_version = champion_version
        if self.registry is not None and log_mlflow:
            with tempfile.TemporaryDirectory(prefix="train-artifacts-") as temp_dir:
                baseline_path = Path(temp_dir) / "baseline.joblib"
                candidate_path = Path(temp_dir) / "candidate.joblib"
                joblib.dump(baseline, baseline_path)
                joblib.dump(candidate, candidate_path)
                common_tags = {
                    "code_revision": revision,
                    "data_fingerprint": fingerprint,
                    "trained_through": trained_through.isoformat(),
                }
                baseline_version = self.registry.log_artifact_model(
                    baseline_path,
                    algorithm="baseline",
                    metrics=baseline_metrics.as_dict(),
                    params={"validation_days": self.settings.validation_days},
                    tags=common_tags,
                    run_id_hint=run_id,
                )
                candidate_version = self.registry.log_artifact_model(
                    candidate_path,
                    algorithm="lightgbm",
                    metrics=candidate_metrics.as_dict(),
                    params={
                        "validation_days": self.settings.validation_days,
                        "n_estimators": 250,
                        "learning_rate": 0.04,
                    },
                    tags=common_tags,
                    run_id_hint=run_id,
                )
            logger.info(
                "Retraining %s registered versions baseline=%s candidate=%s; "
                "alias before promotion=%s",
                run_id,
                baseline_version.version,
                candidate_version.version,
                self.registry.get_alias(),
            )
            if promoted or selected_version is None:
                selected_version = (
                    candidate_version.version
                    if selected.algorithm == "lightgbm"
                    else baseline_version.version
                )
                self.registry.set_champion(selected_version)
                self._cache_selected_model(selected, selected_version)
            current_alias = self.registry.get_alias()
            if current_alias is not None:
                selected_version = current_alias[0]
            logger.info(
                "Retraining %s finished with champion version %s",
                run_id,
                selected_version,
            )
        else:
            selected_version = "local"
            self._cache_selected_model(selected, selected_version)

        if selected_version is None:
            selected_version = "local"
        return TrainingOutcome(
            run_id=run_id,
            promoted=promoted,
            version=selected_version,
            motive=motive,
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
            candidate=candidate,
            baseline=baseline,
            candidate_version=candidate_version,
            baseline_version=baseline_version,
        )

    def _cache_selected_model(self, artifact: ForecastArtifact, version: str) -> None:
        directory = self.settings.model_cache_dir
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"version-{version}.joblib"
        temporary = directory / f".version-{version}.tmp"
        joblib.dump(artifact, temporary)
        os.replace(temporary, destination)
        alias_temp = directory / ".champion.tmp"
        joblib.dump(artifact, alias_temp)
        os.replace(alias_temp, self.settings.model_cache_path)
        metadata_temp = directory / ".champion-meta.tmp"
        metadata = ModelInfo(
            nombre=self.settings.mlflow_model_name,
            alias=self.settings.mlflow_alias,
            version=version,
        )
        metadata_temp.write_text(metadata.model_dump_json(), encoding="utf-8")
        os.replace(metadata_temp, directory / "champion.json")


def artifact_version_from_path(path: Path) -> str | None:
    name = path.stem
    if name.startswith("version-"):
        return name.removeprefix("version-")
    return None


def latest_date(sales: pd.DataFrame) -> date | None:
    if sales.empty:
        return None
    return max(pd.to_datetime(sales["fecha"]).dt.date)
