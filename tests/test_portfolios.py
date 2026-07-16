from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_alpaca.models import LedgerEntry
from chat_alpaca.portfolio_service import (
    list_portfolios,
    portfolio_cost,
    replace_holdings,
    seed_database,
    set_cash,
)


def test_seeded_portfolios_and_costs(session: Session) -> None:
    seed_database(session)
    portfolios = list_portfolios(session)

    assert [portfolio.name for portfolio in portfolios] == [
        "KCs Traditional IRA",
        "KCs Roth IRA",
        "Portfolio 3",
        "Portfolio 4",
        "Portfolio 5",
    ]
    assert portfolio_cost(portfolios[0]) == Decimal("118910.91052")
    assert portfolio_cost(portfolios[1]) == Decimal("95531.78934")


def test_cash_edit_is_persisted_in_ledger(session: Session) -> None:
    seed_database(session)
    portfolio = list_portfolios(session)[0]

    set_cash(session, portfolio.id, "1250.25", "Opening cash")
    session.flush()

    entry = session.scalar(select(LedgerEntry))
    assert portfolio.cash == Decimal("1250.2500")
    assert entry is not None
    assert entry.cash_delta == Decimal("1250.2500")
    assert entry.note == "Opening cash"


def test_replace_holdings_supports_short_positions(session: Session) -> None:
    seed_database(session)
    portfolio = list_portfolios(session)[2]

    replace_holdings(
        session,
        portfolio.id,
        [
            {
                "Symbol": "aapl",
                "Shares": -3,
                "Acquired": date(2026, 1, 5),
                "Cost / share": 200,
            }
        ],
    )
    session.flush()

    refreshed = list_portfolios(session)[2]
    assert refreshed.holdings[0].symbol == "AAPL"
    assert refreshed.holdings[0].shares == Decimal("-3.00000000")


def test_replace_holdings_enforces_25_symbol_limit(session: Session) -> None:
    seed_database(session)
    portfolio = list_portfolios(session)[2]
    rows = [
        {
            "Symbol": f"T{index}",
            "Shares": 1,
            "Acquired": date(2026, 1, 5),
            "Cost / share": 10,
        }
        for index in range(26)
    ]

    with pytest.raises(ValueError, match="at most 25"):
        replace_holdings(session, portfolio.id, rows)
