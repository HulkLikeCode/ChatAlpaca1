from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import Date, DateTime, ForeignKey, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
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


class PortfolioTransaction(Base):
    """An immutable portfolio event used to derive cash and open lots."""

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
