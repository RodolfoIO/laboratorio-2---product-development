from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import requests
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from pendulum import datetime

INTERNAL_API_URL = os.getenv("INTERNAL_API_URL", "http://api:8000").rstrip("/")
INTERNAL_TASK_TOKEN = os.getenv("INTERNAL_TASK_TOKEN", "local-internal-token")
HEADERS = {"X-Internal-Token": INTERNAL_TASK_TOKEN}


def _run_id(context: dict[str, Any]) -> str:
    dag_run = context["dag_run"]
    configured = (dag_run.conf or {}).get("run_id")
    return str(configured or dag_run.run_id)


def validate_training_data(**context: Any) -> dict[str, int | str]:
    response = requests.post(
        f"{INTERNAL_API_URL}/internal/validar-datos",
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return {"run_id": _run_id(context), **response.json()}


def execute_retraining(**context: Any) -> None:
    run_id = _run_id(context)
    response = requests.post(
        f"{INTERNAL_API_URL}/internal/reentrenar/{run_id}",
        headers=HEADERS,
        timeout=590,
    )
    response.raise_for_status()


def mark_retraining_failed(**context: Any) -> None:
    requests.post(
        f"{INTERNAL_API_URL}/internal/reentrenar/{_run_id(context)}/fallo",
        headers=HEADERS,
        json={"error": str(context.get("exception", "fallo de Airflow"))},
        timeout=30,
    )


with DAG(
    dag_id="novamercado_retrain",
    description="Entrena, evalua y promueve el modelo de NovaMercado",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    dagrun_timeout=timedelta(minutes=10),
    default_args={"retries": 0, "on_failure_callback": mark_retraining_failed},
    tags=["novamercado", "ml", "retraining"],
) as dag:
    validate = PythonOperator(
        task_id="validar_datos",
        python_callable=validate_training_data,
    )

    retrain = PythonOperator(
        task_id="entrenar_evaluar_promover",
        python_callable=execute_retraining,
    )

    validate >> retrain
