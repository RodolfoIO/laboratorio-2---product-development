from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from laboratorio_2.catalogs import Catalog
from laboratorio_2.features import FEATURE_COLUMNS
from laboratorio_2.metrics import ForecastMetrics, calculate_metrics


@dataclass
class SeasonalBaseline:
    weekday_median: dict[tuple[int, str, int], float] = field(default_factory=dict)
    family_weekday_median: dict[tuple[str, int], float] = field(default_factory=dict)
    family_median: dict[str, float] = field(default_factory=dict)
    family_quantiles: dict[str, dict[float, float]] = field(default_factory=dict)
    global_median: float = 0.0

    def fit(self, frame: pd.DataFrame, catalog: Catalog) -> SeasonalBaseline:
        training = frame.dropna(subset=["ventas"]).copy()
        training["fecha"] = pd.to_datetime(training["fecha"])
        training["weekday"] = training["fecha"].dt.dayofweek
        for key, values in training.groupby(["tienda_id", "familia_id", "weekday"]):
            self.weekday_median[tuple(key)] = float(values["ventas"].median())
        for key, values in training.groupby(["familia_id", "weekday"]):
            self.family_weekday_median[tuple(key)] = float(values["ventas"].median())
        for key, values in training.groupby("familia_id"):
            self.family_median[str(key)] = float(values["ventas"].median())
        self.global_median = float(training["ventas"].median()) if len(training) else 0.0
        for family in catalog.families:
            ratio = catalog.costs[family].critical_ratio
            for key, values in training[training["familia_id"] == family].groupby(
                "tienda_id"
            ):
                self.family_quantiles.setdefault(str(key), {})[ratio] = float(
                    np.quantile(values["ventas"], ratio)
                )
        return self

    def predict_expected(self, frame: pd.DataFrame) -> np.ndarray:
        dates = pd.to_datetime(frame["fecha"])
        values: list[float] = []
        for index, row in frame.reset_index(drop=True).iterrows():
            key = (int(row["tienda_id"]), str(row["familia_id"]), int(dates.iloc[index].dayofweek))
            value = self.weekday_median.get(key)
            if value is None:
                value = self.family_weekday_median.get((key[1], key[2]))
            if value is None:
                value = self.family_median.get(key[1], self.global_median)
            values.append(max(float(value), 0.0))
        return np.asarray(values, dtype=float)

    def predict_suggested(self, frame: pd.DataFrame, catalog: Catalog) -> np.ndarray:
        values: list[float] = []
        for row in frame.to_dict("records"):
            ratio = catalog.costs[str(row["familia_id"])].critical_ratio
            store_quantiles = self.family_quantiles.get(str(row["tienda_id"]), {})
            value = store_quantiles.get(ratio)
            if value is None:
                value = self.family_median.get(str(row["familia_id"]), self.global_median)
            values.append(max(float(value), 0.0))
        return np.ceil(np.asarray(values)).astype(int)

    def evaluate(
        self, frame: pd.DataFrame, catalog: Catalog
    ) -> tuple[ForecastMetrics, np.ndarray, np.ndarray]:
        expected = self.predict_expected(frame)
        suggested = self.predict_suggested(frame, catalog)
        metrics = calculate_metrics(frame, frame["ventas"], expected, suggested, catalog)
        return metrics, expected, suggested


@dataclass
class LightGBMForecaster:
    point_model: Any = None
    quantile_models: dict[float, Any] = field(default_factory=dict)
    categorical_categories: dict[str, list] = field(default_factory=dict)

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = frame[FEATURE_COLUMNS].copy()
        for column, categories in self.categorical_categories.items():
            features[column] = pd.Categorical(features[column], categories=categories)
        return features

    def fit(self, frame: pd.DataFrame, catalog: Catalog) -> LightGBMForecaster:
        training = frame.dropna(subset=["ventas"]).copy()
        if training.empty:
            raise ValueError("No hay filas para entrenar LightGBM")
        self.categorical_categories = {
            "tienda_id": sorted(catalog.stores),
            "familia_id": sorted(catalog.families),
        }
        features = self._prepare(training)
        target = training["ventas"].astype(float)
        common = {
            "n_estimators": 250,
            "learning_rate": 0.04,
            "num_leaves": 24,
            "max_depth": 7,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 0.5,
            "random_state": 42,
            "n_jobs": 2,
            "verbosity": -1,
        }
        self.point_model = lgb.LGBMRegressor(objective="regression", **common)
        self.point_model.fit(
            features,
            target,
            categorical_feature=["tienda_id", "familia_id"],
        )
        ratios = sorted({cost.critical_ratio for cost in catalog.costs.values()})
        for ratio in ratios:
            model = lgb.LGBMRegressor(objective="quantile", alpha=ratio, **common)
            model.fit(
                features,
                target,
                categorical_feature=["tienda_id", "familia_id"],
            )
            self.quantile_models[ratio] = model
        return self

    def predict_expected(self, frame: pd.DataFrame) -> np.ndarray:
        if self.point_model is None:
            raise ValueError("LightGBM no esta entrenado")
        return np.maximum(self.point_model.predict(self._prepare(frame)), 0.0)

    def predict_suggested(self, frame: pd.DataFrame, catalog: Catalog) -> np.ndarray:
        result = np.zeros(len(frame), dtype=float)
        for ratio, model in self.quantile_models.items():
            indices = [
                index
                for index, family in enumerate(frame["familia_id"])
                if catalog.costs[str(family)].critical_ratio == ratio
            ]
            if not indices:
                continue
            subset = frame.iloc[indices]
            result[indices] = model.predict(self._prepare(subset))
        return np.ceil(np.maximum(result, 0)).astype(int)

    def evaluate(
        self, frame: pd.DataFrame, catalog: Catalog
    ) -> tuple[ForecastMetrics, np.ndarray, np.ndarray]:
        expected = self.predict_expected(frame)
        suggested = self.predict_suggested(frame, catalog)
        metrics = calculate_metrics(frame, frame["ventas"], expected, suggested, catalog)
        return metrics, expected, suggested


@dataclass
class ForecastArtifact:
    algorithm: str
    baseline: SeasonalBaseline
    ml_model: LightGBMForecaster
    metrics: dict[str, float]
    trained_through: date
    data_fingerprint: str
    code_revision: str
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def predict(
        self, frame: pd.DataFrame, catalog: Catalog
    ) -> tuple[np.ndarray, np.ndarray]:
        model = self.ml_model if self.algorithm == "lightgbm" else self.baseline
        expected = model.predict_expected(frame)
        suggested = model.predict_suggested(frame, catalog)
        return expected, suggested

    def evaluate(self, frame: pd.DataFrame, catalog: Catalog) -> ForecastMetrics:
        expected, suggested = self.predict(frame, catalog)
        return calculate_metrics(frame, frame["ventas"], expected, suggested, catalog)
