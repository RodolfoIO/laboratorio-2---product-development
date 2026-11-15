from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from laboratorio_2.catalogs import Catalog, normalize_key

FEATURE_COLUMNS = [
    "tienda_id",
    "familia_id",
    "en_promocion",
    "day_of_week",
    "day",
    "month",
    "day_of_year",
    "is_weekend",
    "is_holiday",
    "is_commercial",
    "is_payday",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_median_7",
    "rolling_mean_28",
    "rolling_median_28",
]


@dataclass(frozen=True)
class PredictionFamily:
    familia_id: str
    en_promocion: int


def _calendar_frame(dates: pd.Series) -> pd.DataFrame:
    parsed = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "day_of_week": parsed.dt.dayofweek.astype(int),
            "day": parsed.dt.day.astype(int),
            "month": parsed.dt.month.astype(int),
            "day_of_year": parsed.dt.dayofyear.astype(int),
            "is_weekend": (parsed.dt.dayofweek >= 5).astype(int),
        },
        index=dates.index,
    )


def _holiday_flags(
    frame: pd.DataFrame,
    holidays: pd.DataFrame,
    catalog: Catalog,
) -> pd.DataFrame:
    result = frame.copy()
    result["is_holiday"] = 0
    result["is_commercial"] = 0
    result["is_payday"] = 0
    if holidays.empty:
        return result

    holiday_rows = holidays.copy()
    holiday_rows["fecha"] = pd.to_datetime(holiday_rows["fecha"]).dt.date
    holiday_rows["ambito_key"] = holiday_rows["ambito"].map(normalize_key)
    holiday_rows["tipo_key"] = holiday_rows["tipo"].map(normalize_key)
    keyed: dict[tuple[date, str], tuple[int, int, int]] = {}
    for row in holiday_rows.itertuples(index=False):
        scope = row.ambito_key
        kind = row.tipo_key
        is_holiday = int(
            "feriado" in kind or "holiday" in kind or kind in {"festivo", "asuelto"}
        )
        is_commercial = int(
            "comercial" in kind
            or "promocion" in kind
            or kind in {"black_friday", "cyber_monday"}
        )
        is_payday = int("pago" in kind or "payday" in kind or kind == "quincena")
        key = (row.fecha, scope)
        previous = keyed.get(key, (0, 0, 0))
        keyed[key] = (
            max(previous[0], is_holiday),
            max(previous[1], is_commercial),
            max(previous[2], is_payday),
        )

    for index, row in result.iterrows():
        requested_date = row["fecha"]
        if not isinstance(requested_date, date):
            requested_date = pd.Timestamp(requested_date).date()
        store = catalog.stores[int(row["tienda_id"])]
        city_key = normalize_key(store.city)
        values = [(0, 0, 0)]
        values.append(keyed.get((requested_date, "nacional"), (0, 0, 0)))
        values.append(keyed.get((requested_date, city_key), (0, 0, 0)))
        result.at[index, "is_holiday"] = max(value[0] for value in values)
        result.at[index, "is_commercial"] = max(value[1] for value in values)
        result.at[index, "is_payday"] = max(value[2] for value in values)
    return result


def build_training_features(
    sales: pd.DataFrame,
    catalog: Catalog,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    if sales.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["fecha", "ventas"])
    frame = sales.copy()
    frame["fecha"] = pd.to_datetime(frame["fecha"])
    frame["en_promocion"] = frame["en_promocion"].fillna(0).astype(int)
    frame = frame.sort_values(["tienda_id", "familia_id", "fecha"]).reset_index(drop=True)
    grouped = frame.groupby(["tienda_id", "familia_id"], sort=False)["ventas"]
    previous = grouped.shift(1)
    frame["lag_1"] = previous
    frame["lag_7"] = grouped.shift(7)
    frame["lag_14"] = grouped.shift(14)
    frame["lag_28"] = grouped.shift(28)
    frame["rolling_mean_7"] = previous.groupby(
        [frame["tienda_id"], frame["familia_id"]]
    ).transform(lambda values: values.rolling(7, min_periods=1).mean())
    frame["rolling_median_7"] = previous.groupby(
        [frame["tienda_id"], frame["familia_id"]]
    ).transform(lambda values: values.rolling(7, min_periods=1).median())
    frame["rolling_mean_28"] = previous.groupby(
        [frame["tienda_id"], frame["familia_id"]]
    ).transform(lambda values: values.rolling(28, min_periods=1).mean())
    frame["rolling_median_28"] = previous.groupby(
        [frame["tienda_id"], frame["familia_id"]]
    ).transform(lambda values: values.rolling(28, min_periods=1).median())

    calendar = _calendar_frame(frame["fecha"])
    for column in calendar:
        frame[column] = calendar[column].to_numpy()
    holiday_frame = pd.DataFrame(
        {
            "tienda_id": frame["tienda_id"],
            "fecha": frame["fecha"].dt.date,
        }
    )
    flags = _holiday_flags(holiday_frame, holidays, catalog)
    for column in ("is_holiday", "is_commercial", "is_payday"):
        frame[column] = flags[column].to_numpy()
    return frame


def build_prediction_features(
    sales: pd.DataFrame,
    catalog: Catalog,
    holidays: pd.DataFrame,
    *,
    store_id: int,
    forecast_date: date,
    families: list[PredictionFamily],
) -> pd.DataFrame:
    history = sales[
        (sales["tienda_id"] == store_id) & (sales["fecha"] < forecast_date)
    ].copy()
    history["fecha"] = pd.to_datetime(history["fecha"])
    rows: list[dict] = []
    target_timestamp = pd.Timestamp(forecast_date)
    for family in families:
        family_history = history[history["familia_id"] == family.familia_id].sort_values(
            "fecha"
        )
        exact = family_history.set_index("fecha")["ventas"]
        values = family_history["ventas"]
        lag_7_date = target_timestamp - pd.Timedelta(days=7)
        lag_14_date = target_timestamp - pd.Timedelta(days=14)
        lag_28_date = target_timestamp - pd.Timedelta(days=28)
        rows.append(
            {
                "tienda_id": store_id,
                "familia_id": family.familia_id,
                "fecha": target_timestamp,
                "en_promocion": family.en_promocion,
                "lag_1": values.iloc[-1] if len(values) else np.nan,
                "lag_7": exact.get(lag_7_date, np.nan),
                "lag_14": exact.get(lag_14_date, np.nan),
                "lag_28": exact.get(lag_28_date, np.nan),
                "rolling_mean_7": (
                    values.tail(7).mean() if len(values) else np.nan
                ),
                "rolling_median_7": (
                    values.tail(7).median() if len(values) else np.nan
                ),
                "rolling_mean_28": (
                    values.tail(28).mean() if len(values) else np.nan
                ),
                "rolling_median_28": (
                    values.tail(28).median() if len(values) else np.nan
                ),
            }
        )
    frame = pd.DataFrame(rows)
    calendar = _calendar_frame(frame["fecha"])
    for column in calendar:
        frame[column] = calendar[column].to_numpy()
    flags = _holiday_flags(
        pd.DataFrame({"tienda_id": frame["tienda_id"], "fecha": frame["fecha"].dt.date}),
        holidays,
        catalog,
    )
    for column in ("is_holiday", "is_commercial", "is_payday"):
        frame[column] = flags[column].to_numpy()
    return frame
