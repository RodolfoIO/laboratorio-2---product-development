from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import uuid4

import pandas as pd
from sqlalchemy import func, select

from laboratorio_2.catalogs import Catalog, Cost, Family, Store
from laboratorio_2.cleaning import CleanForecast, CleanObservation
from laboratorio_2.contracts import ModelInfo, Quality
from laboratorio_2.database import Database
from laboratorio_2.db_models import (
    CorrectionRecord,
    CostRecord,
    FamilyRecord,
    ForecastRequestRecord,
    HistoricalSaleRecord,
    HolidayRecord,
    ObservationRecord,
    PredictionRecord,
    RetrainingRunRecord,
    StoreRecord,
)


class ProductRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def get_catalog(self) -> Catalog:
        async with self.database.session_factory() as session:
            stores = (await session.scalars(select(StoreRecord).order_by(StoreRecord.id))).all()
            families = (
                await session.scalars(select(FamilyRecord).order_by(FamilyRecord.id))
            ).all()
            costs = (await session.scalars(select(CostRecord))).all()
            min_history = await session.scalar(select(func.min(HistoricalSaleRecord.date)))
            max_calendar = await session.scalar(select(func.max(HolidayRecord.date)))
            max_history = await session.scalar(select(func.max(HistoricalSaleRecord.date)))
        max_date = max_calendar or max_history
        return Catalog(
            stores={
                row.id: Store(row.id, row.name, row.city, row.store_format) for row in stores
            },
            families={row.id: Family(row.id, row.name) for row in families},
            costs={
                row.family_id: Cost(
                    row.family_id,
                    shortage=row.shortage_cost,
                    surplus=row.surplus_cost,
                )
                for row in costs
            },
            min_date=min_history,
            max_date=max_date,
        )

    async def save_forecast(
        self,
        clean: CleanForecast,
        predictions: list[dict],
        model: ModelInfo,
        raw_payload: dict,
    ) -> str:
        request_id = str(uuid4())
        async with self.database.session_factory() as session, session.begin():
            session.add(
                ForecastRequestRecord(
                    id=request_id,
                    store_id=clean.tienda_id,
                    date=clean.fecha,
                    model_name=model.nombre,
                    model_alias=model.alias,
                    model_version=model.version,
                    payload=raw_payload,
                    quality=clean.quality.model_dump(mode="json"),
                )
            )
            for item, family in zip(predictions, clean.familias, strict=True):
                session.add(
                    PredictionRecord(
                        request_id=request_id,
                        store_id=clean.tienda_id,
                        family_id=item["familia_id"],
                        date=clean.fecha,
                        expected_sales=item["venta_esperada"],
                        suggested_quantity=item["cantidad_sugerida"],
                        promotion_count=family.en_promocion,
                    )
                )
            for note in clean.quality.observaciones:
                session.add(
                    CorrectionRecord(
                        event_id=request_id,
                        event_type="forecast",
                        field=note.campo,
                        action=note.accion,
                        detail=note.detalle,
                    )
                )
        return request_id

    async def save_observations(self, clean: CleanObservation) -> int:
        event_id = str(uuid4())
        changed = 0
        async with self.database.session_factory() as session, session.begin():
            for item in clean.ventas:
                existing = await session.scalar(
                    select(ObservationRecord)
                    .where(
                        ObservationRecord.store_id == clean.tienda_id,
                        ObservationRecord.family_id == item.familia_id,
                        ObservationRecord.date == clean.fecha,
                    )
                    .with_for_update()
                )
                if existing is None:
                    session.add(
                        ObservationRecord(
                            store_id=clean.tienda_id,
                            family_id=item.familia_id,
                            date=clean.fecha,
                            sales=item.ventas,
                        )
                    )
                    changed += 1
                elif existing.sales != item.ventas:
                    existing.sales = item.ventas
                    existing.updated_at = datetime.now(UTC)
                    changed += 1
            if changed:
                for note in clean.quality.observaciones:
                    session.add(
                        CorrectionRecord(
                            event_id=event_id,
                            event_type="observation",
                            field=note.campo,
                            action=note.accion,
                            detail=note.detalle,
                        )
                    )
        return changed

    async def load_sales(
        self,
        *,
        store_id: int | None = None,
        before: date | None = None,
        after: date | None = None,
    ) -> pd.DataFrame:
        async with self.database.session_factory() as session:
            historical_query = select(HistoricalSaleRecord)
            observation_query = select(ObservationRecord)
            if store_id is not None:
                historical_query = historical_query.where(
                    HistoricalSaleRecord.store_id == store_id
                )
                observation_query = observation_query.where(
                    ObservationRecord.store_id == store_id
                )
            if before is not None:
                historical_query = historical_query.where(
                    HistoricalSaleRecord.date < before
                )
                observation_query = observation_query.where(ObservationRecord.date < before)
            if after is not None:
                historical_query = historical_query.where(
                    HistoricalSaleRecord.date >= after
                )
                observation_query = observation_query.where(ObservationRecord.date >= after)
            historical = (await session.scalars(historical_query)).all()
            observations = (await session.scalars(observation_query)).all()

        rows: dict[tuple[int, str, date], dict] = {}
        for row in historical:
            rows[(row.store_id, row.family_id, row.date)] = {
                "tienda_id": row.store_id,
                "familia_id": row.family_id,
                "fecha": row.date,
                "ventas": row.sales,
                "en_promocion": row.promotion_count,
                "origen": "historico",
            }
        for row in observations:
            rows[(row.store_id, row.family_id, row.date)] = {
                "tienda_id": row.store_id,
                "familia_id": row.family_id,
                "fecha": row.date,
                "ventas": row.sales,
                "en_promocion": None,
                "origen": "observacion",
            }
        frame = pd.DataFrame(rows.values())
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "tienda_id",
                    "familia_id",
                    "fecha",
                    "ventas",
                    "en_promocion",
                    "origen",
                ]
            )
        frame["fecha"] = pd.to_datetime(frame["fecha"]).dt.date
        frame["en_promocion"] = frame["en_promocion"].astype("Int64")
        await self._fill_observation_promotions(frame)
        frame["en_promocion"] = frame["en_promocion"].fillna(0).astype(int)
        return frame.sort_values(["fecha", "tienda_id", "familia_id"]).reset_index(drop=True)

    async def _fill_observation_promotions(self, frame: pd.DataFrame) -> None:
        pending = frame[frame["en_promocion"].isna()]
        if pending.empty:
            return
        async with self.database.session_factory() as session:
            records = (
                await session.execute(
                    select(
                        ForecastRequestRecord.store_id,
                        ForecastRequestRecord.date,
                        PredictionRecord.family_id,
                        PredictionRecord.promotion_count,
                        ForecastRequestRecord.created_at,
                    )
                    .join(
                        PredictionRecord,
                        PredictionRecord.request_id == ForecastRequestRecord.id,
                    )
                    .order_by(ForecastRequestRecord.created_at)
                )
            ).all()
        promotions: dict[tuple[int, str, date], int] = {}
        for row in records:
            promotions[(row.store_id, row.family_id, row.date)] = row.promotion_count
        for index, row in frame.iterrows():
            if pd.isna(row["en_promocion"]):
                key = (int(row["tienda_id"]), str(row["familia_id"]), row["fecha"])
                frame.at[index, "en_promocion"] = promotions.get(key, 0)

    async def latest_observation_date(self) -> date | None:
        async with self.database.session_factory() as session:
            return await session.scalar(select(func.max(ObservationRecord.date)))

    async def load_holidays(self) -> pd.DataFrame:
        async with self.database.session_factory() as session:
            rows = (
                await session.scalars(select(HolidayRecord).order_by(HolidayRecord.date))
            ).all()
        return pd.DataFrame(
            [
                {
                    "fecha": row.date,
                    "ambito": row.scope,
                    "tipo": row.holiday_type,
                    "descripcion": row.description or "",
                }
                for row in rows
            ],
            columns=["fecha", "ambito", "tipo", "descripcion"],
        )

    async def create_retraining_run(self, run_id: str) -> None:
        async with self.database.session_factory() as session, session.begin():
            session.add(RetrainingRunRecord(id=run_id, state="en_curso"))

    async def update_retraining_run(
        self,
        run_id: str,
        *,
        state: str,
        promoted: bool | None = None,
        version: str | None = None,
        motive: str | None = None,
        error: str | None = None,
    ) -> None:
        async with self.database.session_factory() as session, session.begin():
            record = await session.get(RetrainingRunRecord, run_id, with_for_update=True)
            if record is None:
                record = RetrainingRunRecord(id=run_id, state=state)
                session.add(record)
            record.state = state
            record.promoted = promoted
            record.version = version
            record.motive = motive
            record.error = error
            record.updated_at = datetime.now(UTC)

    async def get_retraining_run(self, run_id: str) -> RetrainingRunRecord | None:
        async with self.database.session_factory() as session:
            return await session.get(RetrainingRunRecord, run_id)

    async def quality_for_existing_observation(
        self,
        clean: CleanObservation,
    ) -> Quality:
        return clean.quality
