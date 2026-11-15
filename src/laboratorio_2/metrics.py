from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from laboratorio_2.catalogs import Catalog


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    wape: float
    business_cost: float
    daily_cost: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "wape": self.wape,
            "business_cost": self.business_cost,
            "daily_cost": self.daily_cost,
        }


def inventory_cost(
    actual: np.ndarray | pd.Series,
    suggested: np.ndarray | pd.Series,
    shortage_cost: np.ndarray | pd.Series,
    surplus_cost: np.ndarray | pd.Series,
) -> np.ndarray:
    actual_array = np.asarray(actual, dtype=float)
    suggested_array = np.asarray(suggested, dtype=float)
    shortage_array = np.asarray(shortage_cost, dtype=float)
    surplus_array = np.asarray(surplus_cost, dtype=float)
    return shortage_array * np.maximum(actual_array - suggested_array, 0) + (
        surplus_array * np.maximum(suggested_array - actual_array, 0)
    )


def calculate_metrics(
    frame: pd.DataFrame,
    actual: np.ndarray,
    expected: np.ndarray,
    suggested: np.ndarray,
    catalog: Catalog,
) -> ForecastMetrics:
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    suggested_array = np.asarray(suggested, dtype=float)
    errors = actual_array - expected_array
    mae = float(np.mean(np.abs(errors))) if len(errors) else 0.0
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else 0.0
    denominator = float(np.sum(np.abs(actual_array)))
    wape = float(np.sum(np.abs(errors)) / denominator) if denominator else 0.0
    shortage = [
        catalog.costs[str(family)].shortage
        for family in frame["familia_id"].to_numpy()
    ]
    surplus = [
        catalog.costs[str(family)].surplus for family in frame["familia_id"].to_numpy()
    ]
    costs = inventory_cost(actual_array, suggested_array, shortage, surplus)
    total = float(np.sum(costs)) if len(costs) else 0.0
    day_count = max(int(frame["fecha"].nunique()), 1)
    return ForecastMetrics(
        mae=mae,
        rmse=rmse,
        wape=wape,
        business_cost=total,
        daily_cost=total / day_count,
    )


def should_promote(
    candidate: ForecastMetrics,
    champion: ForecastMetrics | None,
    baseline: ForecastMetrics,
    *,
    cost_improvement: float,
    mae_tolerance: float,
) -> tuple[bool, str]:
    if champion is None:
        if candidate.business_cost < baseline.business_cost:
            reduction = 1 - (
                candidate.business_cost / baseline.business_cost
                if baseline.business_cost > 0
                else 0
            )
            return True, (
                "No existia campeon; candidato supero baseline con "
                f"costo {candidate.business_cost:.2f} vs {baseline.business_cost:.2f} "
                f"({reduction:.1%})"
            )
        return False, (
            "No existia campeon, pero el candidato no supero al baseline: "
            f"{candidate.business_cost:.2f} vs {baseline.business_cost:.2f}"
        )

    enough_improvement = candidate.business_cost <= champion.business_cost * (
        1 - cost_improvement
    )
    technical_guardrail = candidate.mae <= champion.mae * (1 + mae_tolerance)
    promoted = enough_improvement and technical_guardrail
    reason = (
        f"Costo en validacion {candidate.business_cost:.2f} vs "
        f"{champion.business_cost:.2f} del vigente; MAE {candidate.mae:.2f} vs "
        f"{champion.mae:.2f}. "
    )
    if promoted:
        reason += "Candidato promovido."
    elif not enough_improvement:
        reason += f"No alcanzo la mejora minima de costo de {cost_improvement:.1%}."
    else:
        reason += f"MAE excedio la tolerancia tecnica de {mae_tolerance:.1%}."
    return promoted, reason
