from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from chat_alpaca.models import (
    HoldingLot,
    LedgerEntry,
    Portfolio,
    PortfolioTransaction,
    TransactionOverride,
)
from chat_alpaca.portfolio_service import (
    MAX_PORTFOLIOS,
    TransactionDraft,
    create_portfolio,
    delete_portfolio,
    delete_transaction,
    dividend_totals,
    format_short_date,
    import_statement,
    list_portfolios,
    list_transactions,
    list_transactions_for_portfolios,
    parse_short_date,
    parse_statement_csv,
    portfolio_income_events,
    portfolio_income_summary,
    rebuild_portfolio_from_csv,
    record_transaction,
    replace_holdings,
    replay_integrity_diagnostic,
    replay_portfolio,
    seed_database,
    set_cash,
    update_transaction,
)

STATEMENT_FIXTURE = """Date,Action,Symbol,Description,Quantity,Price,Fees & Comm,Amount
1/2/2026,MoneyLink Transfer,,Opening cash,,,,1000.00
1/3/2026,Buy,AAPL,First lot,4,100.00,,-400.00
1/4/2026,Sell,AAPL,Partial sale,1,110.00,,110.00
"""


def test_seeded_portfolios_and_costs(session: Session) -> None:
    seed_database(session)
    portfolios = list_portfolios(session)

    assert [portfolio.name for portfolio in portfolios] == [
        "Portfolio 1",
        "Portfolio 2",
        "Portfolio 3",
        "Portfolio 4",
        "Portfolio 5",
    ]
    assert all(portfolio.cash == Decimal("0") for portfolio in portfolios)
    assert all(not portfolio.holdings for portfolio in portfolios)
    assert session.scalar(select(func.count()).select_from(PortfolioTransaction)) == 0


def test_non_private_seed_is_durable_and_idempotent(session: Session) -> None:
    seed_database(session)
    original_portfolios = session.scalar(select(func.count()).select_from(Portfolio))

    seed_database(session)
    session.flush()

    assert session.scalar(select(func.count()).select_from(Portfolio)) == original_portfolios == 5
    assert session.scalar(select(func.count()).select_from(PortfolioTransaction)) == 0


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


def test_transaction_updates_and_deletes_require_confirmation_and_replay(
    session: Session,
) -> None:
    seed_database(session)
    portfolio = list_portfolios(session)[3]
    buy = TransactionDraft(
        date(2026, 1, 1),
        "Buy",
        "buy",
        "AAPL",
        "First lot",
        Decimal("10"),
        Decimal("10"),
        None,
        Decimal("-100"),
    )
    sale = TransactionDraft(
        date(2026, 1, 2),
        "Sell",
        "sell",
        "AAPL",
        "Partial sale",
        Decimal("4"),
        Decimal("20"),
        None,
        Decimal("80"),
    )
    record_transaction(session, portfolio.id, buy, source="alpaca")
    record_transaction(session, portfolio.id, sale, source="csv")
    transactions = list_transactions(session, portfolio.id)
    buy_id = next(item.id for item in transactions if item.kind == "buy")
    sale_id = next(item.id for item in transactions if item.kind == "sell")
    revised_buy = TransactionDraft(
        date(2026, 1, 1),
        "Buy",
        "buy",
        "AAPL",
        "Corrected first lot",
        Decimal("8"),
        Decimal("10"),
        None,
        Decimal("-80"),
    )

    with pytest.raises(ValueError, match=f"UPDATE {buy_id}"):
        update_transaction(session, portfolio.id, buy_id, revised_buy)
    update_transaction(
        session,
        portfolio.id,
        buy_id,
        revised_buy,
        confirmation=f"UPDATE {buy_id}",
    )
    refreshed = list_portfolios(session)[3]
    updated_transaction = session.get(PortfolioTransaction, buy_id)
    assert updated_transaction is not None
    assert updated_transaction.source == "manual_override"
    assert refreshed.cash == Decimal("0.0000")
    assert [(lot.symbol, lot.shares) for lot in refreshed.holdings] == [
        ("AAPL", Decimal("4.00000000"))
    ]
    assert (
        len(
            list(
                session.scalars(select(LedgerEntry).where(LedgerEntry.portfolio_id == portfolio.id))
            )
        )
        == 2
    )
    update_audit = session.scalar(
        select(TransactionOverride).where(
            TransactionOverride.transaction_id == buy_id,
            TransactionOverride.operation == "update",
        )
    )
    assert update_audit is not None
    assert update_audit.original_source == "alpaca"
    assert json.loads(update_audit.before_state)["description"] == "First lot"
    assert json.loads(update_audit.after_state or "{}")["source"] == "manual_override"

    with pytest.raises(ValueError, match=f"DELETE {sale_id}"):
        delete_transaction(session, portfolio.id, sale_id)
    delete_transaction(session, portfolio.id, sale_id, confirmation=f"DELETE {sale_id}")
    refreshed = list_portfolios(session)[3]
    assert refreshed.cash == Decimal("-80.0000")
    assert [(lot.symbol, lot.shares) for lot in refreshed.holdings] == [
        ("AAPL", Decimal("8.00000000"))
    ]
    assert (
        len(
            list(
                session.scalars(select(LedgerEntry).where(LedgerEntry.portfolio_id == portfolio.id))
            )
        )
        == 1
    )
    delete_audit = session.scalar(
        select(TransactionOverride).where(
            TransactionOverride.transaction_id == sale_id,
            TransactionOverride.operation == "delete",
        )
    )
    assert delete_audit is not None
    assert delete_audit.original_source == "csv"
    assert json.loads(delete_audit.before_state)["description"] == "Partial sale"
    assert delete_audit.after_state is None


def test_transaction_listing_is_uncapped(session: Session) -> None:
    portfolio = create_portfolio(session, "Large ledger")
    session.add_all(
        [
            PortfolioTransaction(
                portfolio_id=portfolio.id,
                transaction_date=date(2026, 1, 1),
                kind="cash_adjustment",
                action="Cash Adjustment",
                description=f"Entry {index}",
                cash_delta=Decimal("1"),
                source="test",
            )
            for index in range(301)
        ]
    )
    session.flush()
    replay_portfolio(session, portfolio.id)

    assert len(list_transactions(session, portfolio.id)) == 301
    assert portfolio.cash == Decimal("301.0000")


def test_multi_portfolio_listing_and_dividend_period_totals(session: Session) -> None:
    first = create_portfolio(session, "First")
    second = create_portfolio(session, "Second")
    third = create_portfolio(session, "Not selected")
    entries = [
        (first.id, date(2025, 7, 17), "dividend", Decimal("99")),
        (first.id, date(2025, 7, 18), "dividend", Decimal("20")),
        (first.id, date(2026, 1, 1), "dividend", Decimal("10")),
        (second.id, date(2026, 6, 15), "dividend", Decimal("30")),
        (second.id, date(2026, 7, 1), "interest", Decimal("500")),
        (third.id, date(2026, 7, 1), "dividend", Decimal("1000")),
    ]
    for portfolio_id, transaction_date, kind, amount in entries:
        record_transaction(
            session,
            portfolio_id,
            TransactionDraft(
                transaction_date,
                kind.title(),
                kind,
                None,
                "Period test",
                None,
                None,
                None,
                amount,
            ),
        )

    selected = list_transactions_for_portfolios(session, [first.id, second.id, first.id])
    totals = dividend_totals(
        session,
        [first.id, second.id],
        date(2026, 6, 1),
        date(2026, 7, 17),
        as_of=date(2026, 7, 17),
    )

    assert len(selected) == 5
    assert totals.year_to_date == Decimal("40.0000")
    assert totals.trailing_365_days == Decimal("60.0000")
    assert totals.custom_range == Decimal("30.0000")


def test_portfolio_income_uses_master_end_and_includes_only_dividends_and_interest(
    session: Session,
) -> None:
    portfolio = create_portfolio(session, "Income portfolio")
    entries = [
        (date(2025, 7, 1), "dividend", "OLD", Decimal("100")),
        (date(2025, 7, 2), "interest", None, Decimal("5")),
        (date(2026, 1, 1), "dividend", "YTD", Decimal("10")),
        (date(2026, 6, 1), "dividend", "SOLD", Decimal("20")),
        (date(2026, 6, 15), "interest", None, Decimal("30")),
        (date(2026, 6, 20), "award", None, Decimal("500")),
        (date(2026, 7, 1), "dividend", "FUTURE", Decimal("1000")),
    ]
    for transaction_date, kind, symbol, amount in entries:
        record_transaction(
            session,
            portfolio.id,
            TransactionDraft(
                transaction_date,
                kind.title(),
                kind,
                symbol,
                "Income summary test",
                None,
                None,
                None,
                amount,
            ),
        )

    summary = portfolio_income_summary(session, [portfolio.id], date(2026, 6, 1), date(2026, 6, 30))
    events = portfolio_income_events(session, [portfolio.id], date(2026, 6, 1), date(2026, 6, 30))

    assert summary.selected_range == Decimal("50.0000")
    assert summary.year_to_date == Decimal("60.0000")
    assert summary.trailing_365_days == Decimal("165.0000")
    assert summary.normalized_quarterly_average == Decimal("152.1875")
    assert [(event.kind, event.symbol) for event in events] == [
        ("dividend", "SOLD"),
        ("interest", None),
    ]


def test_short_transaction_date_format_round_trip() -> None:
    assert format_short_date(date(2026, 7, 3)) == "7/3/26"
    assert parse_short_date("7/3/26") == date(2026, 7, 3)
    with pytest.raises(ValueError, match="M/D/YY"):
        parse_short_date("2026-07-03")


def test_future_transaction_date_is_rejected(session: Session) -> None:
    portfolio = create_portfolio(session, "No future transactions")
    future = date.today() + timedelta(days=1)

    with pytest.raises(ValueError, match="cannot be in the future"):
        record_transaction(
            session,
            portfolio.id,
            TransactionDraft(
                future,
                "Cash Adjustment",
                "cash_adjustment",
                None,
                "Future entry",
                None,
                None,
                None,
                Decimal("10"),
            ),
        )

    assert list_transactions(session, portfolio.id) == []


def test_statement_parser_rejects_future_transaction_date() -> None:
    future = date.today() + timedelta(days=1)
    content = (
        "Date,Action,Symbol,Description,Quantity,Price,Fees & Comm,Amount\n"
        f"{future:%m/%d/%Y},MoneyLink Transfer,,Future transfer,,,,100.00\n"
    )

    parsed = parse_statement_csv(content)

    assert parsed.transactions == []
    assert parsed.errors == ["Row 2: Transaction date cannot be in the future."]


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
        migration_or_test_only=True,
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

    replace_holdings(session, portfolio.id, rows, migration_or_test_only=True)
    assert len(portfolio.holdings) == 26


def test_replace_holdings_cannot_bypass_transaction_ledger(session: Session) -> None:
    portfolio = create_portfolio(session, "Guarded")

    with pytest.raises(PermissionError, match="opening-position transactions"):
        replace_holdings(session, portfolio.id, [])


def test_replay_integrity_diagnostic_identifies_mismatch_categories(session: Session) -> None:
    portfolio = create_portfolio(session, "Diagnostic")
    record_transaction(
        session,
        portfolio.id,
        TransactionDraft(
            date(2026, 1, 1),
            "Buy",
            "buy",
            "ABC",
            "Diagnostic buy",
            Decimal("2"),
            Decimal("10"),
            None,
            Decimal("-20"),
        ),
    )
    assert replay_integrity_diagnostic(session, portfolio.id).is_consistent

    portfolio.cash = Decimal("999")
    result = replay_integrity_diagnostic(session, portfolio.id)

    assert not result.is_consistent
    assert set(result.mismatches) == {"cash"}


def test_replay_integrity_uses_persisted_precision_and_ignores_ledger_row_order(
    session: Session,
) -> None:
    portfolio = create_portfolio(session, "Precision diagnostic")
    first = TransactionDraft(
        date(2026, 1, 2),
        "Buy",
        "buy",
        "ABC",
        "Fractional basis",
        Decimal("3"),
        Decimal("10"),
        None,
        Decimal("-10"),
    )
    second = TransactionDraft(
        date(2026, 1, 3),
        "Cash Adjustment",
        "cash_adjustment",
        None,
        "Unordered ledger",
        None,
        None,
        None,
        Decimal("10"),
    )
    record_transaction(session, portfolio.id, first)
    record_transaction(session, portfolio.id, second)
    transaction = next(
        item for item in list_transactions(session, portfolio.id) if item.kind == "cash_adjustment"
    )
    transaction.transaction_date = date(2026, 1, 1)
    session.flush()
    session.expire_all()

    result = replay_integrity_diagnostic(session, portfolio.id)

    assert result.is_consistent
    assert not result.mismatches


def test_sqlite_rejects_invalid_foreign_key_write(session: Session) -> None:
    session.add(
        HoldingLot(
            portfolio_id=999999,
            symbol="ABC",
            shares=Decimal("1"),
            acquired_on=date(2026, 1, 1),
            cost_basis=Decimal("10"),
        )
    )

    with pytest.raises(IntegrityError):
        session.flush()


def test_statement_import_is_idempotent_and_preserves_duplicate_rows(session: Session) -> None:
    seed_database(session)
    portfolio = list_portfolios(session)[3]
    content = STATEMENT_FIXTURE + (
        "1/5/2026,MoneyLink Transfer,,Repeated transfer,,,,50.00\n"
        "1/5/2026,MoneyLink Transfer,,Repeated transfer,,,,50.00\n"
    )
    parsed = parse_statement_csv(content)

    assert parsed.errors == []
    assert len(parsed.transactions) == 5
    assert import_statement(session, portfolio.id, parsed) == (5, 0)
    assert import_statement(session, portfolio.id, parsed) == (0, 5)

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

    rebuilt = rebuild_portfolio_from_csv(session, portfolio.id, STATEMENT_FIXTURE)
    session.flush()
    refreshed = list_portfolios(session)[3]

    assert rebuilt == 3
    assert all(lot.symbol != "TEST" for lot in refreshed.holdings)
    assert sum(
        (lot.shares for lot in refreshed.holdings if lot.symbol == "AAPL"), Decimal("0")
    ) == Decimal("3.00000000")
