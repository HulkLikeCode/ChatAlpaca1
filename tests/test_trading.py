from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_alpaca.models import PortfolioTransaction
from chat_alpaca.portfolio_service import list_portfolios, seed_database
from chat_alpaca.trading import submit_allocated_order, sync_allocations


class FakeTradingClient:
    def __init__(self) -> None:
        self.order_id = uuid4()
        self.filled = False

    def submit_order(self, request: object) -> SimpleNamespace:
        return SimpleNamespace(
            id=self.order_id,
            status="new",
            filled_qty="0",
            filled_avg_price=None,
        )

    def get_order_by_id(self, order_id: str) -> SimpleNamespace:
        assert order_id == str(self.order_id)
        return SimpleNamespace(
            status="filled" if self.filled else "new",
            filled_qty="2" if self.filled else "0",
            filled_avg_price="25" if self.filled else None,
        )


class PartialFillTradingClient:
    def __init__(self) -> None:
        self.order_id = uuid4()
        self.fill = ("0", None, "new")

    def submit_order(self, request: object) -> SimpleNamespace:
        return SimpleNamespace(
            id=self.order_id,
            status="new",
            filled_qty="0",
            filled_avg_price=None,
        )

    def get_order_by_id(self, order_id: str) -> SimpleNamespace:
        assert order_id == str(self.order_id)
        quantity, average, status = self.fill
        return SimpleNamespace(status=status, filled_qty=quantity, filled_avg_price=average)


def test_order_fill_updates_only_assigned_portfolio(session: Session, monkeypatch: object) -> None:
    seed_database(session)
    portfolios = list_portfolios(session)
    target = portfolios[3]
    fake = FakeTradingClient()
    monkeypatch.setattr("chat_alpaca.trading.get_trading_client", lambda: fake)

    allocation = submit_allocated_order(session, target.id, "MSFT", "buy", 2, "market")
    session.flush()
    assert allocation.portfolio_id == target.id

    fake.filled = True
    assert sync_allocations(session) == 1
    session.flush()
    assert sync_allocations(session) == 0

    refreshed = list_portfolios(session)
    updated = next(item for item in refreshed if item.id == target.id)
    untouched = next(item for item in refreshed if item.id == portfolios[4].id)
    assert updated.cash == Decimal("-50.0000")
    assert [(lot.symbol, lot.shares) for lot in updated.holdings] == [
        ("MSFT", Decimal("2.00000000"))
    ]
    assert untouched.cash == Decimal("0.0000")
    assert untouched.holdings == []


def test_partial_fills_apply_incremental_quantity_notional_and_effective_price_once(
    session: Session, monkeypatch: object
) -> None:
    seed_database(session)
    target = list_portfolios(session)[3]
    fake = PartialFillTradingClient()
    monkeypatch.setattr("chat_alpaca.trading.get_trading_client", lambda: fake)
    allocation = submit_allocated_order(session, target.id, "MSFT", "buy", 2, "market")

    fake.fill = ("1", "10", "partially_filled")
    assert sync_allocations(session) == 1
    session.flush()
    fake.fill = ("2", "12", "filled")
    assert sync_allocations(session) == 1
    session.flush()
    assert sync_allocations(session) == 0
    session.flush()

    fills = list(
        session.scalars(
            select(PortfolioTransaction)
            .where(PortfolioTransaction.portfolio_id == target.id)
            .where(PortfolioTransaction.source == "alpaca")
            .order_by(PortfolioTransaction.id)
        )
    )
    assert [(row.quantity, row.price, row.cash_delta) for row in fills] == [
        (Decimal("1.00000000"), Decimal("10.0000"), Decimal("-10.0000")),
        (Decimal("1.00000000"), Decimal("14.0000"), Decimal("-14.0000")),
    ]
    assert allocation.applied_qty == Decimal("2.00000000")
    assert allocation.applied_notional == Decimal("24.0000")
    assert target.cash == Decimal("-24.0000")
    assert sum(lot.shares for lot in target.holdings) == Decimal("2.00000000")
