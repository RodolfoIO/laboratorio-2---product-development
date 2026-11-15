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

    async def _headers(self) -> dict[str, str]:
        async with httpx.AsyncClient(
            base_url=self.settings.airflow_base_url.rstrip("/"),
            timeout=self.settings.health_probe_timeout_seconds,
        ) as client:
            response = await client.post(
                "/auth/token",
                json={
                    "username": self.settings.airflow_username,
                    "password": self.settings.airflow_password,
                },
            )
            response.raise_for_status()
            return {"Authorization": f"Bearer {response.json()['access_token']}"}

    async def health(self) -> None:
        headers = await self._headers()
        async with httpx.AsyncClient(
            base_url=self.settings.airflow_base_url.rstrip("/"),
            timeout=self.settings.health_probe_timeout_seconds,
        ) as client:
            response = await client.get("/api/v2/version", headers=headers)
            response.raise_for_status()

    async def trigger(self, run_id: str) -> None:
        payload = {
            "dag_run_id": run_id,
            "logical_date": datetime.now(UTC).isoformat(),
            "conf": {"run_id": run_id},
        }
        try:
            headers = await self._headers()
            async with httpx.AsyncClient(
                base_url=self.settings.airflow_base_url.rstrip("/"),
                timeout=self.settings.airflow_request_timeout_seconds,
            ) as client:
                response = await client.post(
                    f"/api/v2/dags/{self.settings.airflow_dag_id}/dagRuns",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            raise OrchestrationError(f"No se pudo iniciar Airflow: {exc}") from exc
