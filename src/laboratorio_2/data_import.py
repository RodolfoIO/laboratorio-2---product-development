from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from laboratorio_2.catalogs import Catalog, Cost, Family, Store, normalize_key
from laboratorio_2.database import Database
from laboratorio_2.db_models import (
    CostRecord,
    FamilyRecord,
    HistoricalSaleRecord,
    HolidayRecord,
    StoreRecord,
)

REQUIRED_FILES = {
    "stores": "tiendas.csv",
    "families": "familias.csv",
    "history": "historico_ventas.csv",
    "holidays": "feriados.csv",
    "costs": "costos.csv",
}


@dataclass(frozen=True)
class DataBundle:
    stores: pd.DataFrame
    families: pd.DataFrame
    history: pd.DataFrame
    holidays: pd.DataFrame
    costs: pd.DataFrame
    fingerprint: str

    @property
    def catalog(self) -> Catalog:
        history_dates = pd.to_datetime(self.history["fecha"])
        holiday_dates = pd.to_datetime(self.holidays["fecha"])
        return Catalog(
            stores={
                int(row.tienda_id): Store(
                    int(row.tienda_id),
                    str(row.nombre),
                    str(row.ciudad),
                    str(row.formato),
                )
                for row in self.stores.itertuples(index=False)
            },
            families={
                str(row.familia_id): Family(str(row.familia_id), str(row.nombre))
                for row in self.families.itertuples(index=False)
            },
            costs={
                str(row.familia_id): Cost(
                    str(row.familia_id),
                    shortage=float(row.costo_faltante),
                    surplus=float(row.costo_sobrante),
                )
                for row in self.costs.itertuples(index=False)
            },
            min_date=history_dates.min().date(),
            max_date=max(history_dates.max().date(), holiday_dates.max().date()),
        )


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    normalized = {normalize_key(column): str(column) for column in frame.columns}
    for candidate in candidates:
        found = normalized.get(normalize_key(candidate))
        if found is not None:
            return found
    raise ValueError(f"Falta la columna {label}; alternativas: {', '.join(candidates)}")


def _rename_columns(
    frame: pd.DataFrame,
    mapping: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    rename = {
        _find_column(frame, candidates, target): target
        for target, candidates in mapping.items()
    }
    return frame.rename(columns=rename)


def _validate_non_negative_integer(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError(f"{label} debe contener enteros no negativos")
    return values.astype(int)


def load_published_data(data_dir: Path) -> DataBundle:
    paths = {key: data_dir / filename for key, filename in REQUIRED_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos publicados en data/raw: " + ", ".join(sorted(missing))
        )

    stores = pd.read_csv(paths["stores"], encoding="utf-8-sig")
    stores = _rename_columns(
        stores,
        {
            "tienda_id": ("tienda_id", "id", "store_id"),
            "nombre": ("nombre", "tienda", "name"),
            "ciudad": ("ciudad", "city"),
            "formato": ("formato", "store_format", "tipo"),
        },
    )
    if stores["tienda_id"].isna().any():
        raise ValueError("tiendas.csv contiene tienda_id vacio")
    stores["tienda_id"] = _validate_non_negative_integer(
        stores["tienda_id"], "tienda_id"
    )
    if stores["tienda_id"].duplicated().any():
        raise ValueError("tiendas.csv contiene tienda_id duplicados")
    for column in ("nombre", "ciudad", "formato"):
        stores[column] = stores[column].fillna("").astype(str).str.strip()
        if (stores[column] == "").any():
            raise ValueError(f"tiendas.csv contiene {column} vacio")

    families = pd.read_csv(paths["families"], encoding="utf-8-sig")
    families = _rename_columns(
        families,
        {
            "familia_id": ("familia_id", "id", "family_id"),
            "nombre": ("nombre", "familia", "name"),
        },
    )
    families["familia_id"] = families["familia_id"].map(normalize_key)
    families["nombre"] = families["nombre"].fillna("").astype(str).str.strip()
    if (families["familia_id"] == "").any() or (families["nombre"] == "").any():
        raise ValueError("familias.csv contiene identificadores o nombres vacios")
    if families["familia_id"].duplicated().any():
        raise ValueError("familias.csv contiene familia_id duplicados")

    history = pd.read_csv(paths["history"], encoding="utf-8-sig")
    history = _rename_columns(
        history,
        {
            "fecha": ("fecha", "date"),
            "tienda_id": ("tienda_id", "store_id"),
            "familia_id": ("familia_id", "family_id"),
            "ventas": ("ventas", "sales"),
            "en_promocion": ("en_promocion", "promotion_count", "promociones"),
        },
    )
    history["fecha"] = pd.to_datetime(history["fecha"], errors="coerce").dt.date
    if history["fecha"].isna().any():
        raise ValueError("historico_ventas.csv contiene fechas invalidas")
    history["tienda_id"] = _validate_non_negative_integer(
        history["tienda_id"], "tienda_id"
    )
    history["ventas"] = _validate_non_negative_integer(history["ventas"], "ventas")
    history["en_promocion"] = _validate_non_negative_integer(
        history["en_promocion"].fillna(0), "en_promocion"
    )
    history["familia_id"] = history["familia_id"].map(normalize_key)
    family_lookup = set(families["familia_id"])
    store_lookup = set(stores["tienda_id"])
    if not set(history["familia_id"]).issubset(family_lookup):
        raise ValueError("historico_ventas.csv contiene familias fuera del catalogo")
    if not set(history["tienda_id"]).issubset(store_lookup):
        raise ValueError("historico_ventas.csv contiene tiendas fuera del catalogo")
    duplicates = history.duplicated(["tienda_id", "familia_id", "fecha"])
    if duplicates.any():
        raise ValueError("historico_ventas.csv contiene claves duplicadas")

    holidays = pd.read_csv(paths["holidays"], encoding="utf-8-sig")
    holidays = _rename_columns(
        holidays,
        {
            "fecha": ("fecha", "date"),
            "ambito": ("ambito", "scope", "ciudad"),
            "tipo": ("tipo", "type", "categoria"),
        },
    )
    holidays["fecha"] = pd.to_datetime(holidays["fecha"], errors="coerce").dt.date
    if holidays["fecha"].isna().any():
        raise ValueError("feriados.csv contiene fechas invalidas")
    holidays["ambito"] = holidays["ambito"].fillna("").astype(str).str.strip()
    holidays["tipo"] = holidays["tipo"].fillna("").astype(str).map(normalize_key)
    if (holidays["ambito"] == "").any() or (holidays["tipo"] == "").any():
        raise ValueError("feriados.csv contiene ambito o tipo vacio")
    description_column = next(
        (
            column
            for column in holidays.columns
            if normalize_key(column) in {"descripcion", "nombre", "description"}
        ),
        None,
    )
    holidays["descripcion"] = (
        holidays[description_column].fillna("").astype(str)
        if description_column is not None
        else ""
    )
    holidays = holidays.drop_duplicates(["fecha", "ambito", "tipo"])

    costs = pd.read_csv(paths["costs"], encoding="utf-8-sig")
    costs = _rename_columns(
        costs,
        {
            "familia_id": ("familia_id", "family_id"),
            "costo_faltante": ("costo_faltante", "missing_cost", "cfaltante"),
            "costo_sobrante": ("costo_sobrante", "surplus_cost", "csobrante"),
        },
    )
    costs["familia_id"] = costs["familia_id"].map(normalize_key)
    for column in ("costo_faltante", "costo_sobrante"):
        costs[column] = pd.to_numeric(costs[column], errors="coerce")
        if costs[column].isna().any() or (costs[column] < 0).any():
            raise ValueError(f"costos.csv contiene {column} invalido")
    if (costs["costo_faltante"] + costs["costo_sobrante"] <= 0).any():
        raise ValueError("cada familia debe tener un costo total positivo")
    if costs["familia_id"].duplicated().any():
        raise ValueError("costos.csv contiene familia_id duplicados")
    if not set(costs["familia_id"]).issubset(family_lookup):
        raise ValueError("costos.csv contiene familias fuera del catalogo")

    digest = hashlib.sha256()
    for key in ("stores", "families", "history", "holidays", "costs"):
        digest.update(paths[key].read_bytes())

    return DataBundle(
        stores=stores,
        families=families,
        history=history,
        holidays=holidays,
        costs=costs,
        fingerprint=digest.hexdigest(),
    )


async def import_data(database: Database, bundle: DataBundle) -> Catalog:
    async with database.session_factory() as session, session.begin():
        if bundle.stores.size:
            store_insert = pg_insert(StoreRecord).values(
                bundle.stores.rename(
                    columns={
                        "tienda_id": "id",
                        "nombre": "name",
                        "ciudad": "city",
                        "formato": "store_format",
                    }
                )[["id", "name", "city", "store_format"]].to_dict("records")
            )
            await session.execute(
                store_insert.on_conflict_do_update(
                    index_elements=[StoreRecord.id],
                    set_={
                        "name": store_insert.excluded.name,
                        "city": store_insert.excluded.city,
                        "store_format": store_insert.excluded.store_format,
                    },
                )
            )
        if bundle.families.size:
            family_insert = pg_insert(FamilyRecord).values(
                bundle.families.rename(
                    columns={"familia_id": "id", "nombre": "name"}
                )[["id", "name"]].to_dict("records")
            )
            await session.execute(
                family_insert.on_conflict_do_update(
                    index_elements=[FamilyRecord.id],
                    set_={"name": family_insert.excluded.name},
                )
            )
        if bundle.costs.size:
            cost_insert = pg_insert(CostRecord).values(
                bundle.costs.rename(
                    columns={
                        "costo_faltante": "shortage_cost",
                        "costo_sobrante": "surplus_cost",
                    }
                )[["familia_id", "shortage_cost", "surplus_cost"]].to_dict("records")
            )
            await session.execute(
                cost_insert.on_conflict_do_update(
                    index_elements=[CostRecord.family_id],
                    set_={
                        "shortage_cost": cost_insert.excluded.shortage_cost,
                        "surplus_cost": cost_insert.excluded.surplus_cost,
                    },
                )
            )
        if bundle.holidays.size:
            holiday_rows = bundle.holidays.rename(
                columns={
                    "fecha": "date",
                    "ambito": "scope",
                    "tipo": "holiday_type",
                    "descripcion": "description",
                }
            ).to_dict("records")
            holiday_insert = pg_insert(HolidayRecord).values(holiday_rows)
            await session.execute(
                holiday_insert.on_conflict_do_update(
                    constraint="uq_holiday_date_scope_type",
                    set_={"description": holiday_insert.excluded.description},
                )
            )

        existing_history = await session.scalar(
            select(func.count()).select_from(HistoricalSaleRecord)
        )
        if not existing_history and bundle.history.size:
            history_rows = (
                bundle.history.rename(
                    columns={
                        "tienda_id": "store_id",
                        "familia_id": "family_id",
                        "ventas": "sales",
                        "en_promocion": "promotion_count",
                    }
                )
                .to_dict("records")
            )
            await session.execute(pg_insert(HistoricalSaleRecord), history_rows)

    return bundle.catalog
