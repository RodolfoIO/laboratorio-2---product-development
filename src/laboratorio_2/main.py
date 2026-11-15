from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from laboratorio_2.config import Settings, get_settings
from laboratorio_2.contracts import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfo,
    ObservationRequest,
    ObservationResponse,
    RetrainStartResponse,
    RetrainStatusResponse,
)
from laboratorio_2.errors import (
    DataContractError,
    DependencyUnavailableError,
    ModelUnavailableError,
    OrchestrationError,
)
from laboratorio_2.services import Services, build_services


def _validation_detail(exc: RequestValidationError) -> str:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        message = error.get("msg", "valor invalido")
        messages.append(f"{location}: {message}" if location else message)
    return "; ".join(messages) or "Solicitud invalida"


def create_app(
    settings: Settings | None = None,
    services: Services | None = None,
    *,
    start_services: bool = True,
) -> FastAPI:
    app_settings = settings or get_settings()
    app_services = services or build_services(app_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if start_services:
            await app_services.start()
        try:
            yield
        finally:
            if start_services:
                await app_services.stop()

    app = FastAPI(
        title="NovaMercado Forecasting Service",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.services = app_services

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": _validation_detail(exc)})

    @app.exception_handler(DataContractError)
    async def data_contract_handler(
        _request: Request, exc: DataContractError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable_handler(
        _request: Request, exc: ModelUnavailableError
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(DependencyUnavailableError)
    async def dependency_handler(
        _request: Request, exc: DependencyUnavailableError
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(OrchestrationError)
    async def orchestration_handler(
        _request: Request, exc: OrchestrationError
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.post("/v1/prediccion", response_model=ForecastResponse)
    async def predict(request: ForecastRequest) -> ForecastResponse:
        return await app_services.forecast.predict(request)

    @app.post("/v1/observaciones", response_model=ObservationResponse)
    async def register_observation(request: ObservationRequest) -> ObservationResponse:
        return await app_services.observations.register(request)

    def verify_internal_token(token: str | None) -> None:
        if token != app_settings.internal_task_token:
            raise HTTPException(status_code=403, detail="Token interno invalido")

    @app.post("/internal/validar-datos", include_in_schema=False)
    async def validate_training_data(
        x_internal_token: str | None = Header(default=None),
    ) -> dict[str, int]:
        verify_internal_token(x_internal_token)
        catalog = await app_services.repository.get_catalog()
        sales = await app_services.repository.load_sales()
        if not catalog.stores or not catalog.families or sales.empty:
            raise HTTPException(status_code=422, detail="No hay datos para entrenar")
        return {
            "tiendas": len(catalog.stores),
            "familias": len(catalog.families),
            "registros": len(sales),
        }

    @app.post("/internal/reentrenar/{run_id}", status_code=202, include_in_schema=False)
    async def execute_retraining(
        run_id: str,
        x_internal_token: str | None = Header(default=None),
    ) -> dict[str, str]:
        verify_internal_token(x_internal_token)

        def execute_in_worker() -> None:
            worker_services = build_services(app_settings)

            async def execute() -> None:
                try:
                    await worker_services.retraining.execute(run_id)
                finally:
                    await worker_services.database.close()

            asyncio.run(execute())

        await asyncio.to_thread(execute_in_worker)
        return {"id": run_id, "estado": "exitoso"}

    @app.post("/internal/reentrenar/{run_id}/fallo", include_in_schema=False)
    async def mark_retraining_failed(
        run_id: str,
        payload: dict[str, str] | None = None,
        x_internal_token: str | None = Header(default=None),
    ) -> dict[str, str]:
        verify_internal_token(x_internal_token)
        current_model = app_services.model_manager.info()
        await app_services.repository.update_retraining_run(
            run_id,
            state="fallido",
            promoted=False,
            version=current_model.version if current_model is not None else "",
            motive="El DAG fallo; el campeon vigente no fue modificado",
            error=(payload or {}).get("error", "fallo de Airflow"),
        )
        return {"id": run_id, "estado": "fallido"}

    @app.post(
        "/admin/reentrenar",
        status_code=202,
        response_model=RetrainStartResponse,
    )
    async def start_retraining() -> RetrainStartResponse:
        run_id = await app_services.retraining.start()
        return RetrainStartResponse(id=run_id)

    @app.get(
        "/admin/reentrenar/{run_id}",
        response_model=RetrainStatusResponse,
        response_model_exclude_none=True,
    )
    async def retraining_status(run_id: str) -> RetrainStatusResponse:
        try:
            return await app_services.retraining.status(run_id)
        except KeyError as exc:
            raise _not_found(f"Ejecucion desconocida: {run_id}") from exc

    @app.get("/modelo", response_model=ModelInfo)
    async def current_model() -> ModelInfo:
        info = app_services.model_manager.info()
        if info is None:
            raise ModelUnavailableError("No hay un modelo cargado")
        return info

    @app.get(
        "/health",
        response_model=HealthResponse,
        response_model_exclude_none=True,
    )
    async def health(response: Response) -> HealthResponse:
        body, status_code = await app_services.health.check()
        response.status_code = status_code
        return body

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"servicio": "NovaMercado", "estado": "disponible"}

    return app


def _not_found(detail: str) -> Exception:
    from fastapi import HTTPException

    return HTTPException(status_code=404, detail=detail)


app = create_app()
