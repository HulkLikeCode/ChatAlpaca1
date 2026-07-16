from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy.orm import Session

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


def test_order_fill_updates_only_assigned_portfolio(session: Session, monkeypatch: object) -> None:
    seed_database(session)
    portfolios = list_portfolios(session)
    target = portfolios[2]
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
    untouched = next(item for item in refreshed if item.id == portfolios[3].id)
    assert updated.cash == Decimal("-50.0000")
    assert [(lot.symbol, lot.shares) for lot in updated.holdings] == [
        ("MSFT", Decimal("2.00000000"))
    ]
    assert untouched.cash == Decimal("0.0000")
    assert untouched.holdings == []
