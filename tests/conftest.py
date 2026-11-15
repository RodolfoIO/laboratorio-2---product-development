from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from laboratorio_2.catalogs import Catalog, Cost, Family, Store
from laboratorio_2.contracts import ModelInfo


@pytest.fixture
def catalog() -> Catalog:
    return Catalog(
        stores={
            5: Store(5, "Centro", "Guatemala", "hipermercado"),
            6: Store(6, "Zona 10", "Guatemala", "supermercado"),
        },
        families={
            "abarrotes": Family("abarrotes", "Abarrotes"),
            "lacteos": Family("lacteos", "Lácteos"),
        },
        costs={
            "abarrotes": Cost("abarrotes", shortage=2.0, surplus=1.0),
            "lacteos": Cost("lacteos", shortage=1.0, surplus=3.0),
        },
        min_date=date(2024, 1, 1),
        max_date=date(2026, 12, 31),
    )


@pytest.fixture
def sales_frame() -> pd.DataFrame:
    start = date(2025, 1, 1)
    rows = []
    for store_id in (5, 6):
        for family_id, base in (("abarrotes", 100), ("lacteos", 60)):
            for day in range(120):
                current = start + timedelta(days=day)
                rows.append(
                    {
                        "tienda_id": store_id,
                        "familia_id": family_id,
                        "fecha": current,
                        "ventas": base + current.weekday() * 4 + (day % 5),
                        "en_promocion": int(current.weekday() == 4),
                        "origen": "historico",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def holidays_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fecha": date(2025, 1, 15),
                "ambito": "nacional",
                "tipo": "feriado",
                "descripcion": "Feriado de prueba",
            },
            {
                "fecha": date(2025, 2, 14),
                "ambito": "Guatemala",
                "tipo": "comercial",
                "descripcion": "Fecha comercial",
            },
        ]
    )


class FakeArtifact:
    def predict(self, frame, _catalog):
        expected = [100.0 + index for index, _ in enumerate(frame.itertuples())]
        suggested = [int(value) + 5 for value in expected]
        return expected, suggested


class FakeModelManager:
    def __init__(self) -> None:
        self.info_value = ModelInfo(
            nombre="novamercado-forecast",
            alias="champion",
            version="7",
        )
        self.artifact = FakeArtifact()

    def get(self):
        return type("Snapshot", (), {"artifact": self.artifact, "info": self.info_value})()

    def info(self):
        return self.info_value


class FakeRepository:
    def __init__(self, catalog, sales):
        self.catalog = catalog
        self.sales = sales
        self.saved_forecasts = []
        self.observations = {}
        self.runs = {}

    async def get_catalog(self):
        return self.catalog

    async def load_sales(self, **_kwargs):
        return self.sales.copy()

    async def load_holidays(self):
        return pd.DataFrame(columns=["fecha", "ambito", "tipo", "descripcion"])

    async def save_forecast(self, clean, predictions, model, raw_payload):
        self.saved_forecasts.append((clean, predictions, model, raw_payload))
        return "request-id"

    async def save_observations(self, clean):
        changed = 0
        for item in clean.ventas:
            key = (clean.tienda_id, clean.fecha, item.familia_id)
            if self.observations.get(key) != item.ventas:
                self.observations[key] = item.ventas
                changed += 1
        return changed

    async def create_retraining_run(self, run_id):
        self.runs[run_id] = {"state": "en_curso"}

    async def get_retraining_run(self, run_id):
        values = self.runs.get(run_id)
        if values is None:
            return None
        return type("Run", (), values)()

    async def update_retraining_run(self, run_id, **values):
        self.runs[run_id] = {
            "state": values.get("state"),
            "promoted": values.get("promoted"),
            "version": values.get("version"),
            "motive": values.get("motive"),
        }
