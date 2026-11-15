from __future__ import annotations

import asyncio
from typing import Any

from laboratorio_2.config import get_settings
from laboratorio_2.services import build_services


def _run_id(context: dict[str, Any]) -> str:
    dag_run = context["dag_run"]
    configured = (dag_run.conf or {}).get("run_id")
    return str(configured or dag_run.run_id)


def validate_training_data(**context: Any) -> dict[str, int | str]:
    async def validate() -> dict[str, int | str]:
        services = build_services(get_settings())
        try:
            catalog = await services.repository.get_catalog()
            sales = await services.repository.load_sales()
            if not catalog.stores or not catalog.families:
                raise ValueError("No existen catalogos importados")
            if sales.empty:
                raise ValueError("No existen ventas historicas u observaciones")
            return {
                "run_id": _run_id(context),
                "stores": len(catalog.stores),
                "families": len(catalog.families),
                "rows": len(sales),
            }
        finally:
            await services.database.close()

    return asyncio.run(validate())


def execute_retraining(**context: Any) -> None:
    async def execute() -> None:
        services = build_services(get_settings())
        try:
            await services.retraining.execute(_run_id(context))
        finally:
            await services.database.close()

    asyncio.run(execute())


def mark_retraining_failed(**context: Any) -> None:
    async def mark() -> None:
        services = build_services(get_settings())
        try:
            await services.repository.update_retraining_run(
                _run_id(context),
                state="fallido",
                promoted=False,
                version="",
                motive="El DAG fallo; el campeon vigente no fue modificado",
                error=str(context.get("exception", "fallo de Airflow")),
            )
        finally:
            await services.database.close()

    asyncio.run(mark())
