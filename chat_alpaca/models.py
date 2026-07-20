from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Portfolio(Base):
    __tablename__ = "portfolios"
    __table_args__ = (
        CheckConstraint(
            "account_type IN ('traditional_ira', 'roth_ira', 'taxable', 'unknown')",
            name="ck_portfolio_account_type",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    account_type: Mapped[str] = mapped_column(String(24), nullable=False, default="unknown")
    cash: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False, default=Decimal("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    holdings: Mapped[list[HoldingLot]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )
    transactions: Mapped[list[PortfolioTransaction]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )
    benchmark_components: Mapped[list[PortfolioBenchmarkComponent]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )


class HoldingLot(Base):
    __tablename__ = "holding_lots"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    shares: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    acquired_on: Mapped[date] = mapped_column(Date, nullable=False)
    cost_basis: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    portfolio: Mapped[Portfolio] = relationship(back_populates="holdings")


class LedgerEntry(Base):
    __tablename__ = "ledger_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(16))
    quantity: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    price: Mapped[Decimal | None] = mapped_column(Numeric(18, 6))
    cash_delta: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    note: Mapped[str] = mapped_column(String(240), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class DataMigration(Base):
    """A durable marker for a one-time, data-level migration."""

    __tablename__ = "data_migrations"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class PortfolioTransaction(Base):
    """An auditable portfolio event used to derive cash and open lots."""

    __tablename__ = "portfolio_transactions"
    __table_args__ = (
        UniqueConstraint(
            "portfolio_id", "fingerprint", name="uq_transaction_portfolio_fingerprint"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), index=True
    )
    transaction_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(16))
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    quantity: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    price: Mapped[Decimal | None] = mapped_column(Numeric(18, 6))
    fees: Mapped[Decimal | None] = mapped_column(Numeric(18, 4))
    cash_delta: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    source: Mapped[str] = mapped_column(String(24), nullable=False, default="manual")
    fingerprint: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    portfolio: Mapped[Portfolio] = relationship(back_populates="transactions")


class TransactionOverride(Base):
    """An immutable snapshot of a manually updated or deleted transaction."""

    __tablename__ = "transaction_overrides"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    transaction_id: Mapped[int] = mapped_column(nullable=False, index=True)
    operation: Mapped[str] = mapped_column(String(12), nullable=False)
    original_source: Mapped[str] = mapped_column(String(24), nullable=False)
    before_state: Mapped[str] = mapped_column(Text, nullable=False)
    after_state: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class OrderAllocation(Base):
    __tablename__ = "order_allocations"
    __table_args__ = (
        UniqueConstraint("alpaca_order_id", name="uq_order_alpaca_id"),
        UniqueConstraint("client_order_id", name="uq_order_client_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), index=True
    )
    alpaca_order_id: Mapped[str] = mapped_column(String(64), nullable=False)
    client_order_id: Mapped[str] = mapped_column(String(48), nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    order_type: Mapped[str] = mapped_column(String(16), nullable=False)
    requested_qty: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 6))
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    filled_qty: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), nullable=False, default=Decimal("0")
    )
    filled_avg_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 6))
    applied_qty: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), nullable=False, default=Decimal("0")
    )
    applied_notional: Mapped[Decimal] = mapped_column(
        Numeric(18, 4), nullable=False, default=Decimal("0")
    )
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class Instrument(Base):
    """A provider-neutral, canonical market instrument."""

    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(primary_key=True)
    canonical_symbol: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    asset_type: Mapped[str] = mapped_column(String(24), nullable=False)
    exchange: Mapped[str | None] = mapped_column(String(32))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    base_currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    aliases: Mapped[list[SymbolAlias]] = relationship(
        back_populates="instrument", cascade="all, delete-orphan"
    )


class SymbolAlias(Base):
    __tablename__ = "symbol_aliases"
    __table_args__ = (
        UniqueConstraint("alias", "effective_from", name="uq_symbol_alias_effective_from"),
        CheckConstraint(
            "effective_to IS NULL OR effective_from IS NULL OR effective_to >= effective_from",
            name="ck_symbol_alias_dates",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    alias: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    effective_from: Mapped[date | None] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)
    instrument: Mapped[Instrument] = relationship(back_populates="aliases")


class MarketDataset(Base):
    """Immutable provenance for one acquired or imported market-data payload."""

    __tablename__ = "market_datasets"
    __table_args__ = (
        UniqueConstraint("imported_file_hash", name="uq_market_dataset_file_hash"),
        CheckConstraint(
            "coverage_end IS NULL OR coverage_start IS NULL OR coverage_end >= coverage_start",
            name="ck_market_dataset_coverage_dates",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(48), nullable=False)
    feed: Mapped[str | None] = mapped_column(String(32))
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    adjustment_method: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    coverage_start: Mapped[date | None] = mapped_column(Date)
    coverage_end: Mapped[date | None] = mapped_column(Date)
    quality_status: Mapped[str] = mapped_column(String(24), nullable=False)
    validation_warnings: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    override_priority: Mapped[int] = mapped_column(nullable=False, default=0, index=True)
    imported_file_hash: Mapped[str | None] = mapped_column(String(64))
    request_metadata: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    bars: Mapped[list[DailyBar]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class DailyBar(Base):
    __tablename__ = "daily_bars"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id", "instrument_id", "bar_date", name="uq_daily_bar_dataset_instrument_date"
        ),
        CheckConstraint(
            "open > 0 AND high > 0 AND low > 0 AND close > 0", name="ck_daily_bar_prices"
        ),
        CheckConstraint("high >= open AND high >= close AND high >= low", name="ck_daily_bar_high"),
        CheckConstraint("low <= open AND low <= close AND low <= high", name="ck_daily_bar_low"),
        CheckConstraint("volume IS NULL OR volume >= 0", name="ck_daily_bar_volume"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    bar_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    open: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[Decimal | None] = mapped_column(Numeric(24, 8))
    trade_count: Mapped[int | None] = mapped_column()
    vwap: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("market_datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset: Mapped[MarketDataset] = relationship(back_populates="bars")
    instrument: Mapped[Instrument] = relationship()


class ProxyAssignment(Base):
    __tablename__ = "proxy_assignments"
    __table_args__ = (
        CheckConstraint(
            "(proxy_instrument_id IS NOT NULL AND proxy_series IS NULL) OR "
            "(proxy_instrument_id IS NULL AND proxy_series IS NOT NULL)",
            name="ck_proxy_assignment_target",
        ),
        CheckConstraint(
            "effective_to IS NULL OR effective_to >= effective_from",
            name="ck_proxy_assignment_dates",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_proxy_assignment_confidence",
        ),
        CheckConstraint(
            "assignment_source IN ('manual', 'automatic')",
            name="ck_proxy_assignment_source",
        ),
        UniqueConstraint(
            "target_instrument_id", "effective_from", name="uq_proxy_assignment_effective_from"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    target_instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    proxy_instrument_id: Mapped[int | None] = mapped_column(
        ForeignKey("instruments.id", ondelete="RESTRICT"), index=True
    )
    proxy_series: Mapped[str | None] = mapped_column(String(80))
    effective_from: Mapped[date] = mapped_column(Date, nullable=False)
    effective_to: Mapped[date | None] = mapped_column(Date)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    data_sufficiency_rationale: Mapped[str] = mapped_column(Text, nullable=False)
    assignment_source: Mapped[str] = mapped_column(String(12), nullable=False)
    target_instrument: Mapped[Instrument] = relationship(foreign_keys=[target_instrument_id])
    proxy_instrument: Mapped[Instrument | None] = relationship(foreign_keys=[proxy_instrument_id])


class PortfolioBenchmarkComponent(Base):
    """One component of an append-only, effective-dated portfolio benchmark blend."""

    __tablename__ = "portfolio_benchmark_components"
    __table_args__ = (
        CheckConstraint("weight > 0 AND weight <= 1", name="ck_benchmark_component_weight"),
        CheckConstraint(
            "rebalancing_frequency IN ('daily', 'monthly', 'quarterly', 'annual', 'none')",
            name="ck_benchmark_rebalancing_frequency",
        ),
        UniqueConstraint(
            "portfolio_id", "effective_from", "symbol", name="uq_benchmark_period_symbol"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    effective_from: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    weight: Mapped[Decimal] = mapped_column(Numeric(9, 8), nullable=False)
    rebalancing_frequency: Mapped[str] = mapped_column(
        String(16), nullable=False, default="monthly"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    portfolio: Mapped[Portfolio] = relationship(back_populates="benchmark_components")


class SecurityMetadata(Base):
    """Cached, provenance-bearing security classification snapshot."""

    __tablename__ = "security_metadata"
    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_security_metadata_confidence",
        ),
        UniqueConstraint(
            "instrument_id", "source", "retrieved_at", name="uq_security_metadata_snapshot"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    security_name: Mapped[str | None] = mapped_column(String(240))
    asset_type: Mapped[str | None] = mapped_column(String(24))
    sector: Mapped[str | None] = mapped_column(String(120))
    industry: Mapped[str | None] = mapped_column(String(160))
    source: Mapped[str] = mapped_column(String(48), nullable=False)
    effective_date: Mapped[date | None] = mapped_column(Date)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    quality_status: Mapped[str] = mapped_column(String(24), nullable=False)
    manual_override: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    instrument: Mapped[Instrument] = relationship()


class EtfSectorWeight(Base):
    """A sector allocation in a dated ETF look-through snapshot."""

    __tablename__ = "etf_sector_weights"
    __table_args__ = (
        CheckConstraint("weight >= 0 AND weight <= 1", name="ck_etf_sector_weight"),
        UniqueConstraint(
            "instrument_id",
            "effective_date",
            "source",
            "sector",
            name="uq_etf_sector_snapshot_sector",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sector: Mapped[str] = mapped_column(String(120), nullable=False)
    weight: Mapped[Decimal] = mapped_column(Numeric(9, 8), nullable=False)
    source: Mapped[str] = mapped_column(String(48), nullable=False)
    effective_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(24), nullable=False)
    instrument: Mapped[Instrument] = relationship()
