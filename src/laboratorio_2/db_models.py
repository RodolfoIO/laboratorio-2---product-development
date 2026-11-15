from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class StoreRecord(Base):
    __tablename__ = "stores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    city: Mapped[str] = mapped_column(String(200), nullable=False)
    store_format: Mapped[str] = mapped_column(String(80), nullable=False)


class FamilyRecord(Base):
    __tablename__ = "families"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)


class CostRecord(Base):
    __tablename__ = "costs"

    family_id: Mapped[str] = mapped_column(
        String(120), ForeignKey("families.id", ondelete="CASCADE"), primary_key=True
    )
    shortage_cost: Mapped[float] = mapped_column(Float, nullable=False)
    surplus_cost: Mapped[float] = mapped_column(Float, nullable=False)


class HolidayRecord(Base):
    __tablename__ = "holidays"
    __table_args__ = (
        UniqueConstraint("date", "scope", "holiday_type", name="uq_holiday_date_scope_type"),
        Index("ix_holidays_date", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    scope: Mapped[str] = mapped_column(String(200), nullable=False)
    holiday_type: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(String(300))


class HistoricalSaleRecord(Base):
    __tablename__ = "historical_sales"
    __table_args__ = (
        UniqueConstraint(
            "store_id", "family_id", "date", name="uq_historical_store_family_date"
        ),
        Index("ix_historical_store_date", "store_id", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    store_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("stores.id", ondelete="CASCADE"), nullable=False
    )
    family_id: Mapped[str] = mapped_column(
        String(120), ForeignKey("families.id", ondelete="CASCADE"), nullable=False
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    sales: Mapped[int] = mapped_column(Integer, nullable=False)
    promotion_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ForecastRequestRecord(Base):
    __tablename__ = "forecast_requests"
    __table_args__ = (Index("ix_forecast_requests_store_date", "store_id", "date"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    store_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("stores.id", ondelete="RESTRICT"), nullable=False
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_alias: Mapped[str] = mapped_column(String(120), nullable=False)
    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    quality: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PredictionRecord(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "request_id", "family_id", name="uq_prediction_request_family"
        ),
        Index("ix_predictions_request", "request_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("forecast_requests.id", ondelete="CASCADE"),
        nullable=False,
    )
    store_id: Mapped[int] = mapped_column(Integer, nullable=False)
    family_id: Mapped[str] = mapped_column(String(120), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    expected_sales: Mapped[float] = mapped_column(Float, nullable=False)
    suggested_quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    promotion_count: Mapped[int] = mapped_column(Integer, nullable=False)


class ObservationRecord(Base):
    __tablename__ = "observations"
    __table_args__ = (
        UniqueConstraint(
            "store_id", "family_id", "date", name="uq_observation_store_family_date"
        ),
        Index("ix_observations_store_date", "store_id", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    store_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("stores.id", ondelete="RESTRICT"), nullable=False
    )
    family_id: Mapped[str] = mapped_column(
        String(120), ForeignKey("families.id", ondelete="RESTRICT"), nullable=False
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    sales: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class CorrectionRecord(Base):
    __tablename__ = "corrections"
    __table_args__ = (Index("ix_corrections_event", "event_type", "event_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(36), nullable=False)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    field: Mapped[str] = mapped_column(String(300), nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    detail: Mapped[str] = mapped_column(String(1000), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RetrainingRunRecord(Base):
    __tablename__ = "retraining_runs"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="en_curso")
    promoted: Mapped[bool | None] = mapped_column(Boolean)
    version: Mapped[str | None] = mapped_column(String(80))
    motive: Mapped[str | None] = mapped_column(String(2000))
    error: Mapped[str | None] = mapped_column(String(4000))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
