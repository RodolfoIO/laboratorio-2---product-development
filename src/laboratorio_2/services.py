from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from laboratorio_2.airflow_client import AirflowClient
from laboratorio_2.cleaning import clean_forecast, clean_observation
from laboratorio_2.config import Settings
from laboratorio_2.contracts import (
    FamilyPrediction,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ObservationRequest,
    ObservationResponse,
    RetrainStatusResponse,
)
from laboratorio_2.database import Database
from laboratorio_2.errors import (
    DependencyUnavailableError,
    ModelUnavailableError,
    OrchestrationError,
)
from laboratorio_2.features import PredictionFamily, build_prediction_features
from laboratorio_2.mlflow_registry import MlflowRegistry
from laboratorio_2.model_runtime import ModelManager
from laboratorio_2.repository import ProductRepository
from laboratorio_2.training import TrainingPipeline


class ForecastService:
    def __init__(
        self,
        repository: ProductRepository,
        model_manager: ModelManager,
    ) -> None:
        self.repository = repository
        self.model_manager = model_manager

    async def predict(self, request: ForecastRequest) -> ForecastResponse:
        catalog = await self.repository.get_catalog()
        clean = clean_forecast(request, catalog)
        snapshot = self.model_manager.get()
        if snapshot is None:
            raise ModelUnavailableError("No hay un modelo cargado")
        try:
            sales = await self.repository.load_sales(
                store_id=clean.tienda_id, before=clean.fecha
            )
            holidays = await self.repository.load_holidays()
        except Exception as exc:
            raise DependencyUnavailableError(
                f"La base de datos no permite registrar el pronostico: {exc}"
            ) from exc
        features = build_prediction_features(
            sales,
            catalog,
            holidays,
            store_id=clean.tienda_id,
            forecast_date=clean.fecha,
            families=[
                PredictionFamily(item.familia_id, item.en_promocion)
                for item in clean.familias
            ],
        )
        expected, suggested = snapshot.artifact.predict(features, catalog)
        predictions = [
            {
                "familia_id": family.familia_id,
                "venta_esperada": round(float(price), 2),
                "cantidad_sugerida": int(quantity),
            }
            for family, price, quantity in zip(
                clean.familias, expected, suggested, strict=True
            )
        ]
        try:
            await self.repository.save_forecast(
                clean,
                predictions,
                snapshot.info,
                request.model_dump(mode="json"),
            )
        except Exception as exc:
            raise DependencyUnavailableError(
                f"La base de datos no permite registrar el pronostico: {exc}"
            ) from exc
        return ForecastResponse(
            tienda_id=clean.tienda_id,
            fecha=clean.fecha,
            version_modelo=snapshot.info.version,
            predicciones=[FamilyPrediction(**item) for item in predictions],
            calidad=clean.quality,
        )


class ObservationService:
    def __init__(self, repository: ProductRepository) -> None:
        self.repository = repository

    async def register(self, request: ObservationRequest) -> ObservationResponse:
        catalog = await self.repository.get_catalog()
        clean = clean_observation(request, catalog)
        try:
            changed = await self.repository.save_observations(clean)
        except Exception as exc:
            raise DependencyUnavailableError(
                f"La base de datos no permite registrar la observacion: {exc}"
            ) from exc
        return ObservationResponse(registros_guardados=changed, calidad=clean.quality)


class HealthService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        mlflow_registry: MlflowRegistry,
        airflow: AirflowClient,
        model_manager: ModelManager,
    ) -> None:
        self.settings = settings
        self.database = database
        self.mlflow_registry = mlflow_registry
        self.airflow = airflow
        self.model_manager = model_manager

    async def check(self) -> tuple[HealthResponse, int]:
        if self.model_manager.info() is None:
            return HealthResponse(
                estado="degradado",
                motivo="No hay un modelo cargado",
            ), 503

        checks = [
            ("database", self.database.ping),
            ("mlflow", lambda: asyncio.to_thread(self.mlflow_registry.health)),
            ("airflow", self.airflow.health),
        ]
        results = await asyncio.gather(
            *(self._run_check(name, check) for name, check in checks)
        )
        failures = [message for name, message in results if message]
        if failures:
            return HealthResponse(estado="degradado", motivo="; ".join(failures)), 200
        return HealthResponse(estado="ok"), 200

    async def _run_check(self, name: str, check) -> tuple[str, str | None]:
        try:
            result = check()
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(
                    result, timeout=self.settings.health_probe_timeout_seconds
                )
            return name, None
        except Exception as exc:
            return name, f"{name}: {exc}"


class RetrainingService:
    def __init__(
        self,
        settings: Settings,
        repository: ProductRepository,
        airflow: AirflowClient,
        pipeline: TrainingPipeline,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.airflow = airflow
        self.pipeline = pipeline

    async def start(self) -> str:
        run_id = str(uuid4())
        await self.repository.create_retraining_run(run_id)
        try:
            await self.airflow.trigger(run_id)
        except OrchestrationError as exc:
            await self.repository.update_retraining_run(
                run_id,
                state="fallido",
                promoted=False,
                version=await self._current_version(),
                motive="No se pudo iniciar el reentrenamiento",
                error=str(exc),
            )
            raise
        return run_id

    async def status(self, run_id: str) -> RetrainStatusResponse:
        record = await self.repository.get_retraining_run(run_id)
        if record is None:
            raise KeyError(run_id)
        return RetrainStatusResponse(
            estado=record.state,
            promovido=record.promoted,
            version=record.version,
            motivo=record.motive,
        )

    async def execute(self, run_id: str) -> None:
        try:
            outcome = await self.pipeline.train(run_id, log_mlflow=True)
            await self.repository.update_retraining_run(
                run_id,
                state="exitoso",
                promoted=outcome.promoted,
                version=outcome.version,
                motive=outcome.motive,
            )
        except Exception as exc:
            await self.repository.update_retraining_run(
                run_id,
                state="fallido",
                promoted=False,
                version=await self._current_version(),
                motive="El reentrenamiento fallo; el campeon vigente no fue modificado",
                error=str(exc),
            )
            raise

    async def _current_version(self) -> str:
        if self.pipeline.registry is None:
            return ""
        try:
            alias = await asyncio.to_thread(self.pipeline.registry.get_alias)
        except Exception:
            return ""
        return alias[0] if alias is not None else ""


@dataclass
class Services:
    settings: Settings
    database: Database
    repository: ProductRepository
    mlflow_registry: MlflowRegistry
    model_manager: ModelManager
    airflow: AirflowClient
    forecast: ForecastService
    observations: ObservationService
    health: HealthService
    retraining: RetrainingService
    training_pipeline: TrainingPipeline

    async def start(self) -> None:
        await self.model_manager.start()

    async def stop(self) -> None:
        await self.model_manager.stop()
        await self.database.close()


def build_services(settings: Settings) -> Services:
    database = Database(settings)
    repository = ProductRepository(database)
    registry = MlflowRegistry(settings)
    manager = ModelManager(settings, registry)
    airflow = AirflowClient(settings)
    pipeline = TrainingPipeline(settings, repository, registry)
    return Services(
        settings=settings,
        database=database,
        repository=repository,
        mlflow_registry=registry,
        model_manager=manager,
        airflow=airflow,
        forecast=ForecastService(repository, manager),
        observations=ObservationService(repository),
        health=HealthService(settings, database, registry, airflow, manager),
        retraining=RetrainingService(settings, repository, airflow, pipeline),
        training_pipeline=pipeline,
    )


def raw_payload(value: Any) -> dict:
    return value.model_dump(mode="json")
