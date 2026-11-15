from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from pendulum import datetime

from laboratorio_2.airflow_tasks import (
    execute_retraining,
    mark_retraining_failed,
    validate_training_data,
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
