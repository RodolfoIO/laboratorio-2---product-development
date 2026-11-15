from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RawForecastFamily(StrictModel):
    familia_id: Any
    en_promocion: Any = None


class ForecastRequest(StrictModel):
    tienda_id: Any
    fecha: Any
    familias: list[RawForecastFamily] = Field(min_length=1)


class RawObservation(StrictModel):
    familia_id: Any
    ventas: Any


class ObservationRequest(StrictModel):
    tienda_id: Any
    fecha: Any
    ventas: list[RawObservation] = Field(min_length=1)


class QualityObservation(StrictModel):
    campo: str
    accion: Literal["reparar", "advertir"]
    detalle: str


class Quality(StrictModel):
    estado: Literal["ok", "corregido", "advertencia"] = "ok"
    observaciones: list[QualityObservation] = Field(default_factory=list)

    @classmethod
    def from_observations(cls, observations: list[QualityObservation]) -> Quality:
        if not observations:
            return cls()
        state = (
            "advertencia"
            if any(item.accion == "advertir" for item in observations)
            else "corregido"
        )
        return cls(estado=state, observaciones=observations)


class FamilyPrediction(StrictModel):
    familia_id: str
    venta_esperada: float = Field(ge=0)
    cantidad_sugerida: int = Field(ge=0)


class ForecastResponse(StrictModel):
    tienda_id: int
    fecha: date
    version_modelo: str
    predicciones: list[FamilyPrediction]
    calidad: Quality


class ObservationResponse(StrictModel):
    registros_guardados: int = Field(ge=0)
    calidad: Quality


class ModelInfo(StrictModel):
    nombre: str
    alias: str
    version: str


class HealthResponse(StrictModel):
    estado: Literal["ok", "degradado"]
    motivo: str | None = None


class RetrainStartResponse(StrictModel):
    id: str


class RetrainStatusResponse(StrictModel):
    estado: Literal["en_curso", "exitoso", "fallido"]
    promovido: bool | None = None
    version: str | None = None
    motivo: str | None = None
