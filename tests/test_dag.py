from __future__ import annotations

import pytest


def test_dag_loads_with_airflow() -> None:
    pytest.importorskip("airflow")
    from airflow.models import DagBag

    bag = DagBag(dag_folder="dags", include_examples=False)
    assert bag.dags["novamercado_retrain"].max_active_runs == 1
