from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import pandas as pd

from laboratorio_2.features import (
    PredictionFamily,
    build_prediction_features,
    build_training_features,
)
from laboratorio_2.forecasting import ForecastArtifact, LightGBMForecaster, SeasonalBaseline
from laboratorio_2.metrics import ForecastMetrics, calculate_metrics, should_promote


def test_training_features_do_not_use_same_day_sales(
    catalog, sales_frame, holidays_frame
) -> None:
    features = build_training_features(sales_frame, catalog, holidays_frame)
    row = features[
        (features["tienda_id"] == 5)
        & (features["familia_id"] == "abarrotes")
        & (features["fecha"] == pd.Timestamp("2025-01-10"))
    ].iloc[0]
    previous = sales_frame[
        (sales_frame["tienda_id"] == 5)
        & (sales_frame["familia_id"] == "abarrotes")
        & (sales_frame["fecha"] == pd.Timestamp("2025-01-09").date())
    ].iloc[0]
    assert row["lag_1"] == previous["ventas"]
    assert row["is_holiday"] == 0


def test_prediction_features_ignore_future_observations(
    catalog, sales_frame, holidays_frame
) -> None:
    changed = sales_frame.copy()
    target = pd.Timestamp("2025-02-15").date()
    changed.loc[
        (changed["tienda_id"] == 5)
        & (changed["familia_id"] == "abarrotes")
        & (changed["fecha"] >= target),
        "ventas",
    ] = 999999
    features = build_prediction_features(
        changed,
        catalog,
        holidays_frame,
        store_id=5,
        forecast_date=target,
        families=[PredictionFamily("abarrotes", 1)],
    )
    assert features.iloc[0]["lag_1"] < 200
    assert features.iloc[0]["en_promocion"] == 1


def test_local_holiday_and_commercial_flags(
    catalog, sales_frame, holidays_frame
) -> None:
    features = build_prediction_features(
        sales_frame,
        catalog,
        holidays_frame,
        store_id=5,
        forecast_date=pd.Timestamp("2025-02-14").date(),
        families=[PredictionFamily("abarrotes", 0)],
    )
    assert features.iloc[0]["is_commercial"] == 1


def test_baseline_and_lightgbm_improve_or_remain_serializable(
    catalog, sales_frame, holidays_frame
) -> None:
    features = build_training_features(sales_frame, catalog, holidays_frame)
    cutoff = pd.Timestamp("2025-03-01")
    train = features[pd.to_datetime(features["fecha"]) < cutoff]
    validation = features[pd.to_datetime(features["fecha"]) >= cutoff]

    baseline = SeasonalBaseline().fit(train, catalog)
    baseline_metrics, _, _ = baseline.evaluate(validation, catalog)
    model = LightGBMForecaster().fit(train, catalog)
    candidate_metrics, expected, suggested = model.evaluate(validation, catalog)
    assert len(expected) == len(validation)
    assert (suggested >= 0).all()
    assert candidate_metrics.mae < baseline_metrics.mae

    artifact = ForecastArtifact(
        algorithm="lightgbm",
        baseline=baseline,
        ml_model=model,
        metrics=candidate_metrics.as_dict(),
        trained_through=max(pd.to_datetime(train["fecha"]).dt.date),
        data_fingerprint="abc",
        code_revision="test",
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "model.joblib"
        joblib.dump(artifact, path)
        restored = joblib.load(path)
    restored_expected, restored_suggested = restored.predict(validation.head(4), catalog)
    assert len(restored_expected) == 4
    assert len(restored_suggested) == 4


def test_business_quantile_uses_family_cost_ratio(catalog) -> None:
    assert catalog.costs["abarrotes"].critical_ratio == 2 / 3
    assert catalog.costs["lacteos"].critical_ratio == 1 / 4


def test_promotion_gate_requires_cost_and_technical_improvement() -> None:
    champion = ForecastMetrics(10, 12, 0.1, 100, 10)
    candidate = ForecastMetrics(10.1, 12, 0.1, 98, 9.8)
    baseline = ForecastMetrics(20, 20, 0.2, 200, 20)
    promoted, reason = should_promote(
        candidate,
        champion,
        baseline,
        cost_improvement=0.01,
        mae_tolerance=0.05,
    )
    assert promoted
    assert "Costo en validacion" in reason

    protected, _ = should_promote(
        ForecastMetrics(12, 12, 0.1, 90, 9),
        champion,
        baseline,
        cost_improvement=0.01,
        mae_tolerance=0.05,
    )
    assert not protected


def test_metric_cost_matches_documented_formula(catalog) -> None:
    frame = pd.DataFrame(
        {
            "familia_id": ["abarrotes", "lacteos"],
            "fecha": [pd.Timestamp("2025-01-01")] * 2,
        }
    )
    metrics = calculate_metrics(
        frame,
        actual=[10, 10],
        expected=[10, 10],
        suggested=[8, 12],
        catalog=catalog,
    )
    assert metrics.business_cost == 2 * 2 + 3 * 2
