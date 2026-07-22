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


def test_sell_cash_reference_case_matches_ledger_identity() -> None:
    draft = build_transaction_draft(
        TransactionCommand("7/19/26", "sell", "AAPL", quantity=3, price=12.50, fees=1.25)
    )

    assert calculated_trade_cash("sell", 3, 12.50, 1.25) == pytest.approx(36.25)
    assert draft.cash_delta == Decimal("36.25000000")


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["quantity", "price", "fees"])
def test_trade_cash_rejects_negative_and_nonfinite_inputs(field: str, invalid: float) -> None:
    inputs = {"quantity": 1.0, "price": 10.0, "fees": 0.0}
    inputs[field] = invalid

    with pytest.raises(ValueError, match="finite nonnegative"):
        calculated_trade_cash("sell", **inputs)


@pytest.mark.parametrize(
    "command",
    [
        TransactionCommand("7/19/26", "buy", "AAA", quantity=float("nan"), price=10),
        TransactionCommand("7/19/26", "buy", "AAA", quantity=float("inf"), price=10),
        TransactionCommand("7/19/26", "buy", "AAA", quantity=1, price=float("nan")),
        TransactionCommand("7/19/26", "buy", "AAA", quantity=1, price=float("inf")),
        TransactionCommand("7/19/26", "buy", "AAA", quantity=1, price=-1),
        TransactionCommand("7/19/26", "buy", "AAA", quantity=1, price=10, fees=-1),
        TransactionCommand("7/19/26", "cash_adjustment", cash_delta=float("nan")),
        TransactionCommand("7/19/26", "cash_adjustment", cash_delta=float("inf")),
    ],
)
def test_transaction_draft_rejects_invalid_numeric_inputs(command: TransactionCommand) -> None:
    with pytest.raises(ValueError):
        build_transaction_draft(command)


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
