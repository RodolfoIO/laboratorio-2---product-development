from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib

from laboratorio_2.config import Settings
from laboratorio_2.contracts import ModelInfo
from laboratorio_2.forecasting import ForecastArtifact
from laboratorio_2.mlflow_registry import MlflowRegistry


@dataclass(frozen=True)
class ModelSnapshot:
    artifact: ForecastArtifact
    info: ModelInfo


class ModelManager:
    def __init__(self, settings: Settings, registry: MlflowRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._snapshot: ModelSnapshot | None = None
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._last_refresh_error: str | None = None

    @property
    def metadata_path(self) -> Path:
        return self.settings.model_cache_dir / "champion.json"

    def load_local(self) -> bool:
        path = self.settings.model_cache_path
        if not path.is_file():
            return False
        artifact = joblib.load(path)
        if not isinstance(artifact, ForecastArtifact):
            raise TypeError("El artefacto local no es un ForecastArtifact")
        metadata = {
            "nombre": self.settings.mlflow_model_name,
            "alias": self.settings.mlflow_alias,
            "version": "local",
        }
        if self.metadata_path.is_file():
            stored = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if "name" in stored and "nombre" not in stored:
                stored["nombre"] = stored.pop("name")
            metadata.update(stored)
        self._snapshot = ModelSnapshot(artifact, ModelInfo(**metadata))
        return True

    async def start(self) -> None:
        if self._snapshot is None:
            self.load_local()
        if self._snapshot is None:
            try:
                await asyncio.wait_for(
                    self.refresh_once(),
                    timeout=self.settings.health_probe_timeout_seconds,
                )
            except TimeoutError:
                self._last_refresh_error = "MLflow no respondio durante el arranque"
        self._task = asyncio.create_task(self._poll(), name="model-alias-poller")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _poll(self) -> None:
        while True:
            await asyncio.sleep(self.settings.model_reload_seconds)
            try:
                await self.refresh_once()
            except Exception:
                continue

    async def refresh_once(self) -> bool:
        try:
            alias = await asyncio.to_thread(self.registry.get_alias)
        except Exception as exc:
            self._last_refresh_error = str(exc)
            return False
        self._last_refresh_error = None
        if alias is None:
            return False
        version, _run_id = alias
        if self._snapshot is not None and self._snapshot.info.version == version:
            return False
        async with self._lock:
            if self._snapshot is not None and self._snapshot.info.version == version:
                return False
            self.settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.settings.model_cache_path.with_suffix(".download")
            await asyncio.to_thread(self.registry.download_alias, temporary)
            artifact = joblib.load(temporary)
            if not isinstance(artifact, ForecastArtifact):
                raise TypeError("El alias champion no contiene un ForecastArtifact")
            os.replace(temporary, self.settings.model_cache_path)
            info = ModelInfo(
                nombre=self.settings.mlflow_model_name,
                alias=self.settings.mlflow_alias,
                version=version,
            )
            metadata_temp = self.settings.model_cache_dir / ".champion-meta.tmp"
            metadata_temp.write_text(info.model_dump_json(), encoding="utf-8")
            os.replace(metadata_temp, self.metadata_path)
            self._snapshot = ModelSnapshot(artifact, info)
        return True

    def get(self) -> ModelSnapshot | None:
        return self._snapshot

    def info(self) -> ModelInfo | None:
        return self._snapshot.info if self._snapshot else None

    @property
    def last_refresh_error(self) -> str | None:
        return self._last_refresh_error
