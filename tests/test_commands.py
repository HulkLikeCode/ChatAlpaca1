from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from chat_alpaca.commands import (
    TransactionCommand,
    build_transaction_draft,
    calculated_trade_cash,
    validate_transaction_symbol,
)
from chat_alpaca.models import Portfolio
from chat_alpaca.portfolio_service import record_transaction


def test_buy_command_normalizes_and_calculates_ledger_values() -> None:
    draft = build_transaction_draft(
        TransactionCommand(
            "7/19/26",
            "buy",
            symbol=" aapl ",
            description="  New lot  ",
            quantity=2,
            price=10,
            fees=1,
            cash_delta=999,
        )
    )

    assert draft.transaction_date == date(2026, 7, 19)
    assert draft.symbol == "AAPL"
    assert draft.description == "New lot"
    assert draft.quantity == Decimal("2.00000000")
    assert draft.cash_delta == Decimal("-21.00000000")
    assert calculated_trade_cash("buy", 2, 10, 1) == -21


def test_command_rejects_missing_or_invalid_trade_symbol() -> None:
    with pytest.raises(ValueError, match="symbol is required"):
        build_transaction_draft(TransactionCommand("7/19/26", "sell", quantity=1, price=10))
    with pytest.raises(ValueError, match="Invalid stock or ETF symbol"):
        validate_transaction_symbol("BAD SYMBOL")


def test_extracted_command_supports_transaction_workflow(session: Session) -> None:
    portfolio = Portfolio(name="Command workflow", cash=Decimal("100"))
    session.add(portfolio)
    session.flush()
    contribution = build_transaction_draft(
        TransactionCommand("7/18/26", "cash_adjustment", cash_delta=100)
    )
    draft = build_transaction_draft(TransactionCommand("7/19/26", "buy", "ABC", "Buy", 2, 10))

    record_transaction(session, portfolio.id, contribution)
    record_transaction(session, portfolio.id, draft)

    assert portfolio.cash == Decimal("80.0000")
    assert [(lot.symbol, lot.shares) for lot in portfolio.holdings] == [
        ("ABC", Decimal("2.00000000"))
    ]
