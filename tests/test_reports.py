from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd

from chat_alpaca.models import HoldingLot, Portfolio, PortfolioTransaction
from chat_alpaca.reports import (
    assemble_combined_performance_report,
    assemble_comparison_report,
    assemble_portfolio_card_reports,
    comparison_acquisition_plan,
    historical_symbol_universe,
)


def _portfolio() -> Portfolio:
    portfolio = Portfolio(id=1, name="Example", cash=Decimal("0"))
    portfolio.holdings = [
        HoldingLot(
            symbol="ABC",
            shares=Decimal("1"),
            acquired_on=date(2026, 1, 2),
            cost_basis=Decimal("10"),
        )
    ]
    portfolio.transactions = [
        PortfolioTransaction(
            id=1,
            transaction_date=date(2026, 1, 2),
            kind="opening_position",
            action="Opening Position",
            symbol="OLD",
            quantity=Decimal("1"),
            price=Decimal("10"),
            cash_delta=Decimal("0"),
            source="test",
        )
    ]
    return portfolio


def test_universe_and_acquisition_plan_own_adjustment_and_baseline_policy() -> None:
    portfolio = _portfolio()

    assert historical_symbol_universe([portfolio], ("spy",)) == ("ABC", "OLD", "SPY")
    plan = comparison_acquisition_plan([portfolio], date(2026, 1, 10), date(2026, 1, 20), ("spy",))

    assert plan.portfolio.start == date(2026, 1, 3)
    assert plan.portfolio.price_policy == "portfolio_accounting"
    assert plan.portfolio.symbols == ("ABC", "OLD", "SPY")
    assert plan.benchmark.start == date(2026, 1, 10)
    assert plan.benchmark.price_policy == "benchmark_total_return"


def test_portfolio_card_annualizes_selected_range_dividends() -> None:
    portfolio = _portfolio()
    portfolio.transactions.append(
        PortfolioTransaction(
            id=2,
            transaction_date=date(2026, 2, 10),
            kind="dividend",
            action="Cash Dividend",
            symbol="ABC",
            cash_delta=Decimal("100"),
            source="test",
        )
    )

    report = assemble_portfolio_card_reports(
        [portfolio], pd.DataFrame(), date(2026, 1, 1), date(2026, 4, 10)
    )[0]

    assert report.expected_annual_dividends == Decimal("365.2425")


def test_combined_performance_report_returns_incomplete_coverage_warning() -> None:
    portfolio = _portfolio()
    closes = pd.DataFrame(
        {"OTHER": [10.0, 11.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )

    report = assemble_combined_performance_report(
        [portfolio], closes, date(2026, 1, 2), date(2026, 1, 3)
    )

    assert report.total_value is None
    assert report.coverage == "Complete valuations: 0 of 1 portfolios."
    assert any("Missing prices for held symbols: ABC" in warning for warning in report.warnings)


def test_comparison_report_assembles_series_metrics_and_missing_warning() -> None:
    portfolio = _portfolio()
    portfolio_closes = pd.DataFrame(
        {"ABC": [10.0, 11.0], "OLD": [10.0, 12.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )
    benchmark_closes = pd.DataFrame(
        {"SPY": [100.0, 101.0]},
        index=portfolio_closes.index,
    )

    report = assemble_comparison_report(
        [portfolio],
        portfolio_closes,
        benchmark_closes,
        date(2026, 1, 2),
        date(2026, 1, 3),
        ("SPY", "MISSING"),
    )

    assert [series.name for series in report.series] == [
        "Selected portfolios",
        "Example",
        "SPY",
    ]
    assert list(report.metrics["Series"]) == ["Selected portfolios", "Example", "SPY"]
    assert any("MISSING" in warning for warning in report.warnings)
    assert report.coverage == "Usable comparison series: 3 of 4 requested."
