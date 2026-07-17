from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_alpaca.models import LedgerEntry, PortfolioTransaction
from chat_alpaca.portfolio_service import (
    MAX_PORTFOLIOS,
    TransactionDraft,
    create_portfolio,
    delete_portfolio,
    import_statement,
    list_portfolios,
    parse_statement_csv,
    portfolio_cost,
    rebuild_portfolio_from_csv,
    record_transaction,
    replace_holdings,
    seed_database,
    set_cash,
)

STATEMENT_PATH = Path(__file__).resolve().parent.parent / "KC and Papa.csv"


def test_seeded_portfolios_and_costs(session: Session) -> None:
    seed_database(session)
    portfolios = list_portfolios(session)

    assert [portfolio.name for portfolio in portfolios] == [
        "KCs Traditional IRA",
        "KCs Roth IRA",
        "KC and Papa",
        "Portfolio 4",
        "Portfolio 5",
    ]
    assert portfolio_cost(portfolios[0]) == Decimal("118910.91052")
    assert portfolio_cost(portfolios[1]) == Decimal("95531.78934")
    assert len(portfolios[2].holdings) == 37
    assert (
        session.scalar(
            select(PortfolioTransaction).where(
                PortfolioTransaction.portfolio_id == portfolios[2].id
            )
        )
        is not None
    )


def test_cash_edit_is_persisted_in_ledger(session: Session) -> None:
    seed_database(session)
    portfolio = list_portfolios(session)[0]

    set_cash(session, portfolio.id, "1250.25", "Opening cash")
    session.flush()

    entry = session.scalar(
        select(LedgerEntry)
        .where(LedgerEntry.portfolio_id == portfolio.id)
        .order_by(LedgerEntry.id.desc())
    )
    assert portfolio.cash == Decimal("1250.2500")
    assert entry is not None
    assert entry.cash_delta == Decimal("1250.2500")
    assert entry.note == "Opening cash"


def test_portfolios_can_be_added_up_to_the_limit_and_deleted(session: Session) -> None:
    seed_database(session)
    created = create_portfolio(session, "Another One!")
    created_id = created.id

    assert created.name == "Another One!"
    delete_portfolio(session, created_id)
    session.flush()
    assert all(item.id != created_id for item in list_portfolios(session))

    for index in range(MAX_PORTFOLIOS - len(list_portfolios(session))):
        create_portfolio(session, f"Portfolio {index + 6}")
    with pytest.raises(ValueError, match="maximum"):
        create_portfolio(session, "One Too Many")


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


def test_replace_holdings_has_no_symbol_limit(session: Session) -> None:
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

    replace_holdings(session, portfolio.id, rows)
    assert len(portfolio.holdings) == 26


def test_statement_import_is_idempotent_and_preserves_duplicate_rows(session: Session) -> None:
    seed_database(session)
    portfolio = list_portfolios(session)[3]
    parsed = parse_statement_csv(STATEMENT_PATH.read_bytes())

    assert parsed.errors == []
    assert len(parsed.transactions) == 75
    assert import_statement(session, portfolio.id, parsed) == (75, 0)
    assert import_statement(session, portfolio.id, parsed) == (0, 75)

    transfers = list(
        session.scalars(
            select(PortfolioTransaction).where(
                PortfolioTransaction.portfolio_id == portfolio.id,
                PortfolioTransaction.kind == "transfer",
            )
        )
    )
    assert len(transfers) == 3


def test_rebuild_replaces_portfolio_and_applies_fifo_sales(session: Session) -> None:
    seed_database(session)
    portfolio = list_portfolios(session)[3]
    record_transaction(
        session,
        portfolio.id,
        TransactionDraft(
            date(2026, 1, 1),
            "Buy",
            "buy",
            "TEST",
            "Old position",
            Decimal("1"),
            Decimal("10"),
            None,
            Decimal("-10"),
        ),
    )

    rebuilt = rebuild_portfolio_from_csv(session, portfolio.id, STATEMENT_PATH.read_bytes())
    session.flush()
    refreshed = list_portfolios(session)[3]

    assert rebuilt == 75
    assert all(lot.symbol != "TEST" for lot in refreshed.holdings)
    assert [(lot.symbol, lot.shares) for lot in refreshed.holdings if lot.symbol == "CRML"] == []
    assert sum(
        (lot.shares for lot in refreshed.holdings if lot.symbol == "AAPL"), Decimal("0")
    ) == Decimal("122.00000000")
