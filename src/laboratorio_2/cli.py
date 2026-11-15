from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import func, select

from laboratorio_2.config import Settings, get_settings
from laboratorio_2.data_import import import_data, load_published_data
from laboratorio_2.database import Database
from laboratorio_2.db_models import StoreRecord
from laboratorio_2.mlflow_registry import MlflowRegistry
from laboratorio_2.model_runtime import ModelManager
from laboratorio_2.repository import ProductRepository
from laboratorio_2.services import build_services
from laboratorio_2.training import TrainingPipeline


def _migration_config() -> Config:
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    return config


async def _wait_database(database: Database, timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            await database.ping()
            return
        except Exception:
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(2)


async def _wait_mlflow(registry: MlflowRegistry, timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            await asyncio.to_thread(registry.health)
            return
        except Exception:
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(2)


async def bootstrap(settings: Settings) -> None:
    command.upgrade(_migration_config(), "head")
    database = Database(settings)
    repository = ProductRepository(database)
    try:
        await _wait_database(database)
        async with database.session_factory() as session:
            store_count = await session.scalar(select(func.count()).select_from(StoreRecord))
        if not store_count:
            bundle = load_published_data(settings.data_dir)
            await import_data(database, bundle)

        registry = MlflowRegistry(settings)
        await _wait_mlflow(registry)
        alias = await asyncio.to_thread(registry.get_alias)
        if alias is None:
            pipeline = TrainingPipeline(settings, repository, registry)
            await pipeline.train(f"bootstrap-{uuid4()}", promote=True, log_mlflow=True)

        manager = ModelManager(settings, registry)
        await manager.refresh_once()
        if not manager.load_local() and manager.get() is None:
            raise RuntimeError("El modelo inicial no pudo cargarse en la cache local")
    finally:
        await database.close()


async def retrain(settings: Settings, run_id: str) -> None:
    services = build_services(settings)
    try:
        await services.retraining.execute(run_id)
    finally:
        await services.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="NovaMercado forecasting service")
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=("serve", "bootstrap", "retrain"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = get_settings()
    if args.command == "bootstrap":
        asyncio.run(bootstrap(settings))
        return
    if args.command == "retrain":
        if not args.run_id:
            parser.error("--run-id es obligatorio para retrain")
        asyncio.run(retrain(settings, args.run_id))
        return
    import uvicorn

    uvicorn.run("laboratorio_2.main:app", host=args.host, port=args.port)
