from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from laboratorio_2.main import create_app
from laboratorio_2.services import (
    ForecastService,
    ObservationService,
    RetrainingService,
)


class FakeAirflow:
    def __init__(self, fail: bool = False):
        self.fail = fail

    async def trigger(self, run_id):
        if self.fail:
            from laboratorio_2.errors import OrchestrationError

            raise OrchestrationError("Airflow no disponible")


class FakeHealth:
    async def check(self):
        from laboratorio_2.contracts import HealthResponse

        return HealthResponse(estado="degradado", motivo="airflow: caido"), 200


def client_for(catalog, sales, fail_airflow: bool = False) -> tuple[TestClient, object]:
    from conftest import FakeModelManager, FakeRepository

    repository = FakeRepository(catalog, sales)
    manager = FakeModelManager()
    airflow = FakeAirflow(fail_airflow)
    retraining = RetrainingService(
        settings=SimpleNamespace(),
        repository=repository,
        airflow=airflow,
        pipeline=SimpleNamespace(),
    )
    services = SimpleNamespace(
        forecast=ForecastService(repository, manager),
        observations=ObservationService(repository),
        model_manager=manager,
        health=FakeHealth(),
        retraining=retraining,
    )
    app = create_app(services=services, start_services=False)
    return TestClient(app), repository


def test_prediction_contract_and_model_version(catalog, sales_frame) -> None:
    client, repository = client_for(catalog, sales_frame)
    response = client.post(
        "/v1/prediccion",
        json={
            "tienda_id": "5",
            "fecha": "2025-05-01",
            "familias": [
                {"familia_id": "LÁCTEOS", "en_promocion": "2"},
                {"familia_id": "abarrotes", "en_promocion": 0},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tienda_id"] == 5
    assert body["fecha"] == "2025-05-01"
    assert body["version_modelo"] == "7"
    assert len(body["predicciones"]) == 2
    assert isinstance(body["predicciones"][0]["cantidad_sugerida"], int)
    assert body["calidad"]["estado"] == "corregido"
    assert len(repository.saved_forecasts) == 1


def test_contract_rejection_has_textual_detail(catalog, sales_frame) -> None:
    client, _ = client_for(catalog, sales_frame)
    response = client.post(
        "/v1/prediccion",
        json={
            "tienda_id": 5,
            "fecha": "03/04/2025",
            "familias": [{"familia_id": "abarrotes", "en_promocion": 0}],
        },
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)
    assert "ambigua" in response.json()["detail"]

    unknown_field = client.post(
        "/v1/prediccion",
        json={
            "tienda_id": 5,
            "fecha": "2025-04-03",
            "familias": [{"familia_id": "abarrotes", "en_promocion": 0, "extra": 1}],
        },
    )
    assert unknown_field.status_code == 422
    assert isinstance(unknown_field.json()["detail"], str)


def test_observation_idempotency(catalog, sales_frame) -> None:
    client, _ = client_for(catalog, sales_frame)
    payload = {
        "tienda_id": 5,
        "fecha": "2025-05-01",
        "ventas": [{"familia_id": "abarrotes", "ventas": "105"}],
    }
    first = client.post("/v1/observaciones", json=payload)
    second = client.post("/v1/observaciones", json=payload)
    assert first.status_code == 200
    assert first.json()["registros_guardados"] == 1
    assert second.json()["registros_guardados"] == 0


def test_health_without_model_is_503(monkeypatch, catalog, sales_frame) -> None:
    client, _ = client_for(catalog, sales_frame)
    client.app.state.services.model_manager.info_value = None

    class NoModelManager:
        def info(self):
            return None

    client.app.state.services.model_manager = NoModelManager()

    class MissingModelHealth:
        async def check(self):
            from laboratorio_2.contracts import HealthResponse

            return HealthResponse(estado="degradado", motivo="No hay un modelo cargado"), 503

    client.app.state.services.health = MissingModelHealth()
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["estado"] == "degradado"


def test_degraded_health_and_admin_contract(catalog, sales_frame) -> None:
    client, _ = client_for(catalog, sales_frame)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["estado"] == "degradado"

    started = client.post("/admin/reentrenar")
    assert started.status_code == 202
    run_id = started.json()["id"]
    status = client.get(f"/admin/reentrenar/{run_id}")
    assert status.status_code == 200
    assert status.json() == {"estado": "en_curso"}
    assert client.get("/admin/reentrenar/unknown").status_code == 404


def test_admin_returns_503_when_airflow_cannot_start(catalog, sales_frame) -> None:
    client, _ = client_for(catalog, sales_frame, fail_airflow=True)
    response = client.post("/admin/reentrenar")
    assert response.status_code == 503
