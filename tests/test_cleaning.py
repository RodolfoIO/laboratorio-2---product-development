from __future__ import annotations

from datetime import date

import pytest

from laboratorio_2.cleaning import clean_forecast, clean_observation
from laboratorio_2.contracts import ForecastRequest, ObservationRequest
from laboratorio_2.errors import DataContractError


def test_forecast_repairs_numeric_text_family_accents_and_date(catalog) -> None:
    request = ForecastRequest.model_validate(
        {
            "tienda_id": "005",
            "fecha": "17/11/2025",
            "familias": [
                {"familia_id": "LÁCTEOS", "en_promocion": "8"},
                {"familia_id": "abarrotes", "en_promocion": 0},
            ],
        }
    )
    clean = clean_forecast(request, catalog)
    assert clean.tienda_id == 5
    assert clean.fecha == date(2025, 11, 17)
    assert clean.familias[0].familia_id == "lacteos"
    assert clean.familias[0].en_promocion == 8
    assert clean.quality.estado == "corregido"
    assert {note.campo for note in clean.quality.observaciones} >= {
        "tienda_id",
        "fecha",
        "familias[0].familia_id",
        "familias[0].en_promocion",
    }


def test_store_prefix_timestamp_and_missing_families_policy(catalog) -> None:
    request = ForecastRequest.model_validate(
        {
            "tienda_id": "T05",
            "fecha": "2025-11-17T00:00:00",
            "familias": [{"familia_id": "abarrotes", "en_promocion": 0}],
        }
    )
    clean = clean_forecast(request, catalog)
    assert clean.tienda_id == 5
    assert clean.quality.estado == "advertencia"
    assert any(note.campo == "familias" for note in clean.quality.observaciones)


def test_missing_promotion_warns_and_assumes_zero(catalog) -> None:
    request = ForecastRequest.model_validate(
        {
            "tienda_id": 5,
            "fecha": "2025-11-17",
            "familias": [{"familia_id": "abarrotes"}],
        }
    )
    clean = clean_forecast(request, catalog)
    assert clean.familias[0].en_promocion == 0
    assert clean.quality.estado == "advertencia"
    assert clean.quality.observaciones[0].accion == "advertir"


def test_identical_duplicate_is_repaired(catalog) -> None:
    request = ForecastRequest.model_validate(
        {
            "tienda_id": 5,
            "fecha": "2025-11-17",
            "familias": [
                {"familia_id": "abarrotes", "en_promocion": 2},
                {"familia_id": "ABARROTES", "en_promocion": 2},
            ],
        }
    )
    clean = clean_forecast(request, catalog)
    assert len(clean.familias) == 1
    assert any("duplicada" in note.detalle for note in clean.quality.observaciones)


def test_contradictory_duplicate_is_rejected(catalog) -> None:
    request = ForecastRequest.model_validate(
        {
            "tienda_id": 5,
            "fecha": "2025-11-17",
            "familias": [
                {"familia_id": "abarrotes", "en_promocion": 2},
                {"familia_id": "abarrotes", "en_promocion": 3},
            ],
        }
    )
    with pytest.raises(DataContractError, match="contradictorias"):
        clean_forecast(request, catalog)


def test_ambiguous_or_unknown_values_are_rejected(catalog) -> None:
    ambiguous = ForecastRequest.model_validate(
        {
            "tienda_id": 5,
            "fecha": "03/04/2025",
            "familias": [{"familia_id": "abarrotes", "en_promocion": 0}],
        }
    )
    with pytest.raises(DataContractError, match="ambigua"):
        clean_forecast(ambiguous, catalog)

    unknown = ForecastRequest.model_validate(
        {
            "tienda_id": 999,
            "fecha": "2025-04-03",
            "familias": [{"familia_id": "abarrotes", "en_promocion": 0}],
        }
    )
    with pytest.raises(DataContractError, match="desconocida"):
        clean_forecast(unknown, catalog)


def test_observation_rejects_negative_and_repairs_numeric_text(catalog) -> None:
    clean = clean_observation(
        ObservationRequest.model_validate(
            {
                "tienda_id": "5",
                "fecha": "2025-11-17",
                "ventas": [{"familia_id": "Abarrotes", "ventas": "99"}],
            }
        ),
        catalog,
    )
    assert clean.ventas[0].ventas == 99
    assert clean.quality.estado == "corregido"

    with pytest.raises(DataContractError):
        clean_observation(
            ObservationRequest.model_validate(
                {
                    "tienda_id": 5,
                    "fecha": "2025-11-17",
                    "ventas": [{"familia_id": "abarrotes", "ventas": -1}],
                }
            ),
            catalog,
        )
