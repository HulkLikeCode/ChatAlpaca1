from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from chat_alpaca.analytics import (
    IncompleteValuationError,
    consolidated_holdings,
    normalized_growth,
    performance_growth,
    portfolio_gain_loss,
    portfolio_series,
    portfolio_valuation,
    rebase_comparison_series,
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


def test_explicit_dividend_is_not_counted_twice_in_portfolio_history() -> None:
    portfolio = Portfolio(id=1, name="Dividend", cash=Decimal("1"))
    portfolio.transactions = [
        PortfolioTransaction(
            id=1,
            transaction_date=date(2026, 1, 1),
            kind="opening_position",
            action="Opening Position",
            symbol="ABC",
            quantity=Decimal("1"),
            price=Decimal("10"),
            cash_delta=Decimal("0"),
            source="test",
        ),
        PortfolioTransaction(
            id=2,
            transaction_date=date(2026, 1, 2),
            kind="dividend",
            action="Dividend",
            symbol="ABC",
            cash_delta=Decimal("1"),
            source="test",
        ),
    ]
    split_only_closes = pd.DataFrame(
        {"ABC": [10.0, 9.0]}, index=pd.to_datetime(["2026-01-01", "2026-01-02"])
    )

    assert portfolio_series(portfolio, split_only_closes).tolist() == [10.0, 10.0]


def _valuation_portfolio() -> Portfolio:
    portfolio = Portfolio(id=1, name="Coverage", cash=Decimal("5"))
    portfolio.holdings = [
        HoldingLot(
            symbol=symbol,
            shares=Decimal("1"),
            acquired_on=date(2026, 1, 1),
            cost_basis=Decimal("10"),
        )
        for symbol in ("ABC", "XYZ", "MISS")
    ]
    return portfolio


@pytest.mark.parametrize(
    ("columns", "missing"),
    [
        ({"ABC": [10.0], "XYZ": [20.0]}, ("MISS",)),
        ({"ABC": [10.0]}, ("MISS", "XYZ")),
        ({"ABC": [10.0], "XYZ": [20.0], "MISS": [30.0]}, ()),
    ],
)
def test_valuation_discloses_missing_symbols(
    columns: dict[str, list[float]], missing: tuple[str, ...]
) -> None:
    closes = pd.DataFrame(columns, index=pd.to_datetime(["2026-01-02"]))
    result = portfolio_valuation(_valuation_portfolio(), closes)

    assert result.missing_symbols == missing
    assert result.is_complete is (not missing)
    assert result.valued_market_value_percentage == (100.0 if not missing else None)
    if missing:
        with pytest.raises(IncompleteValuationError):
            total_portfolio_value([_valuation_portfolio()], closes)


def test_valuation_with_no_usable_prices_is_explicitly_incomplete() -> None:
    result = portfolio_valuation(
        _valuation_portfolio(), pd.DataFrame(columns=["ABC", "XYZ", "MISS"])
    )

    assert result.total_calculated_value == Decimal("5")
    assert result.market_value == Decimal("0")
    assert not result.is_complete
    assert result.missing_symbols == ("ABC", "MISS", "XYZ")
    assert result.valued_market_value_percentage == 0.0


def test_historical_series_marks_missing_held_symbol_incomplete() -> None:
    portfolio = _valuation_portfolio()
    closes = pd.DataFrame({"ABC": [10.0], "XYZ": [20.0]}, index=pd.to_datetime(["2026-01-02"]))

    assert pd.isna(portfolio_series(portfolio, closes).iloc[0])


def test_valuation_uses_common_as_of_and_reports_stale_symbols() -> None:
    portfolio = _valuation_portfolio()
    portfolio.holdings = portfolio.holdings[:2]
    closes = pd.DataFrame(
        {"ABC": [10.0, 11.0, 12.0], "XYZ": [19.0, 20.0, float("nan")]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )

    result = portfolio_valuation(portfolio, closes)

    assert result.total_calculated_value == Decimal("36.0")
    assert result.common_valuation_date == date(2026, 1, 2)
    assert result.stale_symbols == ("XYZ",)
    assert result.last_price_dates["XYZ"] == date(2026, 1, 2)
    assert result.is_complete


def test_custom_gain_loss_requires_a_prior_trading_close() -> None:
    portfolio = Portfolio(id=1, name="Baseline", cash=Decimal("100"))
    closes = pd.DataFrame({"ABC": [10.0]}, index=pd.to_datetime(["2026-01-05"]))

    result = portfolio_gain_loss(portfolio, closes, date(2026, 1, 5), date(2026, 1, 5))

    assert result.custom is None
    assert any("prior trading close" in warning for warning in result.warnings)


def test_custom_gain_loss_uses_earlier_close_for_weekend_start() -> None:
    portfolio = Portfolio(id=1, name="Weekend", cash=Decimal("100"))
    closes = pd.DataFrame({"ABC": [10.0, 11.0]}, index=pd.to_datetime(["2026-01-02", "2026-01-05"]))

    result = portfolio_gain_loss(portfolio, closes, date(2026, 1, 3), date(2026, 1, 5))

    assert result.custom == 0.0


def test_custom_gain_loss_handles_position_established_after_start() -> None:
    portfolio = Portfolio(id=1, name="New position", cash=Decimal("80"))
    portfolio.transactions = [
        PortfolioTransaction(
            id=1,
            transaction_date=date(2026, 1, 5),
            kind="buy",
            action="Buy",
            symbol="ABC",
            quantity=Decimal("2"),
            price=Decimal("10"),
            cash_delta=Decimal("-20"),
            source="test",
        )
    ]
    closes = pd.DataFrame(
        {"ABC": [9.0, 10.0, 12.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
    )

    result = portfolio_gain_loss(portfolio, closes, date(2026, 1, 3), date(2026, 1, 6))

    assert result.custom == 4.0


def test_all_comparison_series_rebase_at_selected_report_start() -> None:
    portfolio_growth = pd.Series(
        [125.0, 150.0], index=pd.to_datetime(["2026-01-02", "2026-01-03"]), name="Portfolio"
    )
    benchmark = pd.Series([50.0, 55.0], index=portfolio_growth.index, name="SPY")

    rebased = rebase_comparison_series([portfolio_growth, benchmark])

    assert [series.iloc[0] for series in rebased] == [100.0, 100.0]


def test_performance_growth_excludes_external_flows_and_opening_positions() -> None:
    portfolio = Portfolio(id=1, name="Performance portfolio", cash=Decimal("220"))
    portfolio.transactions = [
        PortfolioTransaction(
            id=1,
            transaction_date=date(2026, 1, 1),
            kind="transfer",
            action="Transfer",
            cash_delta=Decimal("100"),
            source="test",
        ),
        PortfolioTransaction(
            id=2,
            transaction_date=date(2026, 1, 2),
            kind="cash_adjustment",
            action="Cash Adjustment",
            cash_delta=Decimal("100"),
            source="test",
        ),
        PortfolioTransaction(
            id=3,
            transaction_date=date(2026, 1, 3),
            kind="opening_position",
            action="Opening Position",
            symbol="ABC",
            quantity=Decimal("2"),
            price=Decimal("10"),
            cash_delta=Decimal("0"),
            source="test",
        ),
    ]
    closes = pd.DataFrame(
        {"ABC": [10.0, 10.0, 10.0, 11.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
    )

    assert performance_growth(portfolio, closes).round(6).tolist() == [
        100.0,
        100.0,
        100.0,
        100.909091,
    ]


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
