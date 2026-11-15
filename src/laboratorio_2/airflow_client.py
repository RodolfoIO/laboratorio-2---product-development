from __future__ import annotations

from datetime import UTC, datetime

import httpx

from laboratorio_2.config import Settings
from laboratorio_2.errors import OrchestrationError


class AirflowClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.auth = httpx.BasicAuth(
            settings.airflow_username, settings.airflow_password
        )

    async def health(self) -> None:
        async with httpx.AsyncClient(
            base_url=self.settings.airflow_base_url.rstrip("/"),
            auth=self.auth,
            timeout=self.settings.health_probe_timeout_seconds,
        ) as client:
            response = await client.get("/api/v2/version")
            response.raise_for_status()

    async def trigger(self, run_id: str) -> None:
        payload = {
            "dag_run_id": run_id,
            "logical_date": datetime.now(UTC).isoformat(),
            "conf": {"run_id": run_id},
        }
        try:
            async with httpx.AsyncClient(
                base_url=self.settings.airflow_base_url.rstrip("/"),
                auth=self.auth,
                timeout=self.settings.airflow_request_timeout_seconds,
            ) as client:
                response = await client.post(
                    f"/api/v2/dags/{self.settings.airflow_dag_id}/dagRuns",
                    json=payload,
                )
                response.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            raise OrchestrationError(f"No se pudo iniciar Airflow: {exc}") from exc
