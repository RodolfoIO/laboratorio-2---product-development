from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from laboratorio_2.catalogs import Catalog, normalize_key
from laboratorio_2.contracts import (
    ForecastRequest,
    ObservationRequest,
    Quality,
    QualityObservation,
)
from laboratorio_2.errors import DataContractError


@dataclass(frozen=True)
class CleanFamilyRequest:
    familia_id: str
    en_promocion: int


@dataclass(frozen=True)
class CleanForecast:
    tienda_id: int
    fecha: date
    familias: tuple[CleanFamilyRequest, ...]
    quality: Quality


@dataclass(frozen=True)
class CleanFamilyObservation:
    familia_id: str
    ventas: int


@dataclass(frozen=True)
class CleanObservation:
    tienda_id: int
    fecha: date
    ventas: tuple[CleanFamilyObservation, ...]
    quality: Quality


def _text(value: object) -> str:
    if value is None:
        return ""
    normalized = normalize_key(value)
    return normalized


def normalize_date(value: object) -> tuple[date, bool, str | None]:
    if isinstance(value, datetime):
        return value.date(), True, "La fecha con hora se interpreto como fecha"
    if isinstance(value, date):
        return value, False, None
    if not isinstance(value, str):
        raise DataContractError(f"fecha invalida: {value!r}")
    candidate = value.strip()
    if not candidate:
        raise DataContractError("fecha no puede estar vacia")
    if "T" in candidate.upper() or " " in candidate:
        try:
            timestamp = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            return timestamp.date(), True, "La fecha con hora se interpreto como su dia"
        except ValueError:
            pass

    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%Y%m%d",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(candidate, fmt).date()
            return parsed, fmt != "%Y-%m-%d", (
                None if fmt == "%Y-%m-%d" else f"Fecha convertida desde {fmt}"
            )
        except ValueError:
            continue

    day_first: date | None = None
    month_first: date | None = None
    for fmt in ("%d/%m/%Y", "%m/%d/%Y"):
        try:
            parsed = datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
        if fmt.startswith("%d"):
            day_first = parsed
        else:
            month_first = parsed
    if day_first is not None and month_first is not None:
        raise DataContractError(f"fecha ambigua, no se puede reparar: {value!r}")
    if day_first is not None:
        return day_first, True, "Fecha convertida desde DD/MM/AAAA"
    if month_first is not None:
        return month_first, True, "Fecha convertida desde MM/DD/AAAA"
    raise DataContractError(f"fecha invalida: {value!r}")


def normalize_non_negative_int(
    value: object,
    *,
    field_name: str,
    missing_allowed: bool = False,
) -> tuple[int, bool, bool]:
    """Return canonical value, repaired flag and missing flag."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if missing_allowed:
            return 0, False, True
        raise DataContractError(f"{field_name} es obligatorio")
    if isinstance(value, bool):
        raise DataContractError(f"{field_name} debe ser un entero no negativo")
    repaired = False
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise DataContractError(f"{field_name} debe ser un entero no negativo")
        parsed = int(value)
        repaired = True
    elif isinstance(value, str):
        candidate = value.strip()
        try:
            parsed_decimal = _parse_localized_decimal(candidate, field_name)
        except ValueError as exc:
            raise DataContractError(str(exc)) from exc
        if parsed_decimal != parsed_decimal.to_integral_value():
            raise DataContractError(f"{field_name} debe ser un entero no negativo")
        parsed = int(parsed_decimal)
        repaired = True
    else:
        raise DataContractError(f"{field_name} debe ser un entero no negativo")
    if parsed < 0:
        raise DataContractError(f"{field_name} debe ser un entero no negativo")
    return parsed, repaired, False


def _parse_localized_decimal(candidate: str, field_name: str) -> Decimal:
    compact = candidate.replace(" ", "")
    if compact.startswith("-"):
        raise ValueError(f"{field_name} debe ser un entero no negativo")
    if re.fullmatch(r"\d{1,3}(,\d{3})+", compact):
        compact = compact.replace(",", "")
    elif re.fullmatch(r"\d+(,\d{1,2})", compact):
        compact = compact.replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", compact):
        compact = compact.replace(".", "")
    elif not re.fullmatch(r"\d+(?:\.\d+)?", compact):
        raise ValueError(f"{field_name} debe ser un entero no negativo")
    try:
        return Decimal(compact)
    except InvalidOperation as exc:
        raise ValueError(f"{field_name} debe ser un entero no negativo") from exc


def _resolve_store(catalog: Catalog, value: object) -> tuple[int, list[QualityObservation]]:
    try:
        store_id, repaired = catalog.resolve_store(value)
    except (KeyError, ValueError) as exc:
        raise DataContractError(str(exc).strip("'\"")) from exc
    observations = []
    if repaired:
        observations.append(
            QualityObservation(
                campo="tienda_id",
                accion="reparar",
                detalle=f"{value!r} se interpreto como tienda_id {store_id}",
            )
        )
    return store_id, observations


def _resolve_date(
    catalog: Catalog,
    value: object,
) -> tuple[date, list[QualityObservation]]:
    parsed, repaired, detail = normalize_date(value)
    try:
        catalog.validate_date_range(parsed)
    except ValueError as exc:
        raise DataContractError(str(exc)) from exc
    observations = []
    if repaired:
        observations.append(
            QualityObservation(
                campo="fecha",
                accion="reparar",
                detalle=detail or f"{value!r} se interpreto como {parsed.isoformat()}",
            )
        )
    return parsed, observations


def _resolve_family(
    catalog: Catalog,
    value: object,
    field_path: str,
) -> tuple[str, list[QualityObservation]]:
    try:
        family_id, repaired = catalog.resolve_family(value)
    except (KeyError, ValueError) as exc:
        raise DataContractError(str(exc).strip("'\"")) from exc
    observations = []
    if repaired:
        observations.append(
            QualityObservation(
                campo=f"{field_path}.familia_id",
                accion="reparar",
                detalle=f"{value!r} se interpreto como {family_id!r}",
            )
        )
    return family_id, observations


def clean_forecast(request: ForecastRequest, catalog: Catalog) -> CleanForecast:
    store_id, observations = _resolve_store(catalog, request.tienda_id)
    forecast_date, date_notes = _resolve_date(catalog, request.fecha)
    observations.extend(date_notes)

    families: list[CleanFamilyRequest] = []
    seen: dict[str, int] = {}
    for index, family in enumerate(request.familias):
        path = f"familias[{index}]"
        family_id, family_notes = _resolve_family(catalog, family.familia_id, path)
        observations.extend(family_notes)
        promotion, repaired, missing = normalize_non_negative_int(
            family.en_promocion,
            field_name=f"{path}.en_promocion",
            missing_allowed=True,
        )
        if missing:
            observations.append(
                QualityObservation(
                    campo=f"{path}.en_promocion",
                    accion="advertir",
                    detalle="en_promocion ausente; se asumio 0",
                )
            )
        elif repaired:
            observations.append(
                QualityObservation(
                    campo=f"{path}.en_promocion",
                    accion="reparar",
                    detalle=f"{family.en_promocion!r} se interpreto como {promotion}",
                )
            )
        if family_id in seen:
            if seen[family_id] != promotion:
                raise DataContractError(
                    f"familia duplicada con promociones contradictorias: {family_id}"
                )
            observations.append(
                QualityObservation(
                    campo=path,
                    accion="reparar",
                    detalle=f"entrada duplicada identica de {family_id!r} eliminada",
                )
            )
            continue
        seen[family_id] = promotion
        families.append(CleanFamilyRequest(family_id, promotion))

    missing_families = sorted(set(catalog.families) - set(seen))
    if missing_families:
        observations.append(
            QualityObservation(
                campo="familias",
                accion="advertir",
                detalle=(
                    "La solicitud omitio familias del catalogo; se respondio solo por "
                    "las familias presentes: " + ", ".join(missing_families)
                ),
            )
        )

    return CleanForecast(
        tienda_id=store_id,
        fecha=forecast_date,
        familias=tuple(families),
        quality=Quality.from_observations(observations),
    )


def clean_observation(request: ObservationRequest, catalog: Catalog) -> CleanObservation:
    store_id, observations = _resolve_store(catalog, request.tienda_id)
    observation_date, date_notes = _resolve_date(catalog, request.fecha)
    observations.extend(date_notes)

    sales: list[CleanFamilyObservation] = []
    seen: dict[str, int] = {}
    for index, item in enumerate(request.ventas):
        path = f"ventas[{index}]"
        family_id, family_notes = _resolve_family(catalog, item.familia_id, path)
        observations.extend(family_notes)
        amount, repaired, _ = normalize_non_negative_int(
            item.ventas,
            field_name=f"{path}.ventas",
        )
        if repaired:
            observations.append(
                QualityObservation(
                    campo=f"{path}.ventas",
                    accion="reparar",
                    detalle=f"{item.ventas!r} se interpreto como {amount}",
                )
            )
        if family_id in seen:
            if seen[family_id] != amount:
                raise DataContractError(
                    f"familia duplicada con ventas contradictorias: {family_id}"
                )
            observations.append(
                QualityObservation(
                    campo=path,
                    accion="reparar",
                    detalle=f"entrada duplicada identica de {family_id!r} eliminada",
                )
            )
            continue
        seen[family_id] = amount
        sales.append(CleanFamilyObservation(family_id, amount))

    return CleanObservation(
        tienda_id=store_id,
        fecha=observation_date,
        ventas=tuple(sales),
        quality=Quality.from_observations(observations),
    )
