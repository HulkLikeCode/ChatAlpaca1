from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd

from chat_alpaca.analytics import (
    consolidated_holdings,
    normalized_growth,
    portfolio_gain_loss,
    portfolio_series,
    summary_metrics,
    total_portfolio_value,
)
from chat_alpaca.models import HoldingLot, Portfolio, PortfolioTransaction


def test_buy_and_hold_series_includes_cash_and_acquisition_date() -> None:
    portfolio = Portfolio(id=1, name="Example", cash=Decimal("100"))
    portfolio.holdings = [
        HoldingLot(
            symbol="ABC",
            shares=Decimal("2"),
            acquired_on=date(2026, 1, 2),
            cost_basis=Decimal("10"),
        )
    ]
    index = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    closes = pd.DataFrame({"ABC": [9.0, 10.0, 12.0]}, index=index)

    values = portfolio_series(portfolio, closes)

    assert values.tolist() == [100.0, 120.0, 124.0]


def test_normalization_and_metrics() -> None:
    values = pd.Series(
        [50.0, 55.0, 60.0],
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        name="ABC",
    )

    normalized = normalized_growth(values)
    metrics = summary_metrics(normalized)

    assert normalized.tolist() == [100.0, 110.00000000000001, 120.0]
    assert round(metrics["Total return"], 6) == 0.2
    assert metrics["Max drawdown"] == 0.0


def test_transaction_aware_values_and_gain_loss_exclude_external_cash_flows() -> None:
    portfolio = Portfolio(id=1, name="Transaction portfolio", cash=Decimal("85"))
    portfolio.holdings = [
        HoldingLot(
            symbol="ABC",
            shares=Decimal("2"),
            acquired_on=date(2026, 1, 2),
            cost_basis=Decimal("10"),
        )
    ]
    portfolio.transactions = [
        PortfolioTransaction(
            id=1,
            transaction_date=date(2026, 1, 1),
            kind="transfer",
            action="Transfer",
            description="Contribution",
            cash_delta=Decimal("100"),
            source="test",
        ),
        PortfolioTransaction(
            id=2,
            transaction_date=date(2026, 1, 2),
            kind="buy",
            action="Buy",
            symbol="ABC",
            description="Buy shares",
            quantity=Decimal("2"),
            price=Decimal("10"),
            cash_delta=Decimal("-20"),
            source="test",
        ),
        PortfolioTransaction(
            id=3,
            transaction_date=date(2026, 1, 3),
            kind="dividend",
            action="Dividend",
            symbol="ABC",
            description="Dividend",
            cash_delta=Decimal("5"),
            source="test",
        ),
    ]
    closes = pd.DataFrame(
        {"ABC": [9.0, 10.0, 12.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )

    values = portfolio_series(portfolio, closes)
    gain_loss = portfolio_gain_loss(portfolio, closes, date(2026, 1, 2), date(2026, 1, 3))

    assert values.tolist() == [100.0, 100.0, 109.0]
    assert gain_loss.all_time == 9.0
    assert gain_loss.daily == 9.0
    assert gain_loss.custom == 9.0
    assert total_portfolio_value([portfolio], closes) == Decimal("109.0")


def test_consolidated_holdings_sum_symbols_and_preserve_lot_breakdown() -> None:
    first = Portfolio(id=1, name="First", cash=Decimal("0"))
    first.holdings = [
        HoldingLot(
            symbol="ABC",
            shares=Decimal("2"),
            acquired_on=date(2026, 1, 1),
            cost_basis=Decimal("10"),
        )
    ]
    second = Portfolio(id=2, name="Second", cash=Decimal("0"))
    second.holdings = [
        HoldingLot(
            symbol="ABC",
            shares=Decimal("3"),
            acquired_on=date(2026, 1, 2),
            cost_basis=Decimal("20"),
        )
    ]
    closes = pd.DataFrame(
        {"ABC": [12.0, 22.0, 25.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )

    summary, detail = consolidated_holdings(
        [first, second], closes, date(2026, 1, 2), date(2026, 1, 3)
    )

    row = summary.iloc[0]
    assert len(detail) == 2
    assert row["Symbol"] == "ABC"
    assert row["Portfolios"] == 2
    assert row["Shares"] == 5.0
    assert row["Average cost / share"] == 16.0
    assert row["Total cost basis"] == 80.0
    assert row["Market value"] == 125.0
    assert row["All-time gain/loss"] == 45.0
    assert row["Daily gain/loss"] == 15.0
    assert row["Custom gain/loss"] == 41.0
