from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

import chat_alpaca.analytics as analytics
import chat_alpaca.reconstruction as reconstruction
import chat_alpaca.reports as reports
from chat_alpaca.analytics import (
    consolidated_holdings,
    performance_growth,
    portfolio_gain_loss,
)
from chat_alpaca.models import HoldingLot, Portfolio, PortfolioTransaction
from chat_alpaca.reports import (
    assemble_combined_performance_report,
    assemble_comparison_report,
    assemble_portfolio_card_reports,
    build_portfolio_calculation_context,
    comparison_acquisition_plan,
    historical_symbol_universe,
    overlay_intraday_performance,
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


def _reconstruction_fixture(count: int) -> tuple[list[Portfolio], pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2025-01-02", periods=80)
    closes = pd.DataFrame(index=index)
    portfolios = []
    for number in range(1, count + 1):
        symbol = f"S{number:02d}"
        closes[symbol] = [80.0 + number + day * 0.1 for day in range(len(index))]
        portfolio = Portfolio(id=number, name=f"Portfolio {number}", cash=Decimal("1205"))
        portfolio.holdings = [
            HoldingLot(
                symbol=symbol,
                shares=Decimal("8"),
                acquired_on=index[1].date(),
                cost_basis=Decimal(str(80 + number)),
            )
        ]
        portfolio.transactions = [
            PortfolioTransaction(
                id=number * 10 + 1,
                transaction_date=index[0].date(),
                kind="transfer",
                action="Transfer",
                cash_delta=Decimal("2000"),
                source="test",
            ),
            PortfolioTransaction(
                id=number * 10 + 2,
                transaction_date=index[1].date(),
                kind="buy",
                action="Buy",
                symbol=symbol,
                quantity=Decimal("10"),
                price=Decimal(str(80 + number)),
                cash_delta=Decimal(str(-10 * (80 + number))),
                source="test",
            ),
            PortfolioTransaction(
                id=number * 10 + 3,
                transaction_date=index[30].date(),
                kind="sell",
                action="Sell",
                symbol=symbol,
                quantity=Decimal("2"),
                price=Decimal(str(84 + number)),
                cash_delta=Decimal(str(2 * (84 + number))),
                source="test",
            ),
            PortfolioTransaction(
                id=number * 10 + 4,
                transaction_date=index[45].date(),
                kind="dividend",
                action="Dividend",
                symbol=symbol,
                cash_delta=Decimal("5"),
                source="test",
            ),
            PortfolioTransaction(
                id=number * 10 + 5,
                transaction_date=index[60].date(),
                kind="interest",
                action="Interest",
                cash_delta=Decimal("2"),
                source="test",
            ),
        ]
        portfolios.append(portfolio)
    benchmark = pd.DataFrame(
        {"SPY": [400.0 + day * 0.2 for day in range(len(index))]},
        index=index,
    )
    return portfolios, closes, benchmark


def test_universe_and_acquisition_plan_own_adjustment_and_baseline_policy() -> None:
    portfolio = _portfolio()

    assert historical_symbol_universe([portfolio], ("spy",)) == ("ABC", "OLD", "SPY")
    plan = comparison_acquisition_plan([portfolio], date(2026, 1, 10), date(2026, 1, 20), ("spy",))

    assert plan.portfolio.start == date(2026, 1, 3)
    assert plan.portfolio.price_policy == "portfolio_accounting"
    assert plan.portfolio.symbols == ("ABC", "OLD", "SPY")
    assert plan.benchmark.start == date(2026, 1, 10)
    assert plan.benchmark.price_policy == "benchmark_total_return"


def test_portfolio_card_sums_selected_range_dividends() -> None:
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

    assert report.cumulative_dividends == Decimal("100")
    assert report.value_label == "Cost basis $10"


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
    assert report.total_label == "Selected Totals"
    assert report.coverage == "Complete valuations: 0 of 1 portfolios."
    assert any("Missing prices for held symbols: ABC" in warning for warning in report.warnings)
    assert report.rows[0].cash == 0


def test_selected_totals_and_cards_share_one_household_confirmed_date() -> None:
    first = Portfolio(id=1, name="First", cash=Decimal("0"))
    first.holdings = [
        HoldingLot(
            symbol="AAA",
            shares=Decimal("2"),
            acquired_on=date(2026, 1, 1),
            cost_basis=Decimal("80"),
        )
    ]
    second = Portfolio(id=2, name="Second", cash=Decimal("0"))
    second.holdings = [
        HoldingLot(
            symbol="BBB",
            shares=Decimal("3"),
            acquired_on=date(2026, 1, 1),
            cost_basis=Decimal("40"),
        )
    ]
    closes = pd.DataFrame(
        {"AAA": [100.0, 110.0], "BBB": [50.0, float("nan")]},
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )

    report = assemble_combined_performance_report(
        [first, second], closes, date(2026, 1, 2), date(2026, 1, 3)
    )
    cards = assemble_portfolio_card_reports(
        [first, second], closes, date(2026, 1, 2), date(2026, 1, 3)
    )

    assert report.total_value == Decimal("350.0")
    assert "2026-01-02" in report.coverage
    assert [card.value_label for card in cards] == ["$200", "$150"]


def test_intraday_overlay_updates_current_metrics_but_can_hold_custom_fixed() -> None:
    portfolio = _portfolio()
    closes = pd.DataFrame(
        {"ABC": [10.0, 12.0], "OLD": [10.0, 10.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )
    report = assemble_combined_performance_report(
        [portfolio], closes, date(2026, 1, 3), date(2026, 1, 3)
    )

    fixed = overlay_intraday_performance(
        report, {"Example": 5.0}, include_custom=False, indicative_total_value=17.0
    )
    live = overlay_intraday_performance(report, {"Example": 5.0}, include_custom=True)

    assert fixed.total_value == Decimal("17.00")
    assert fixed.total_label == "Selected Totals"
    assert fixed.daily == 5.0
    assert fixed.all_time == report.all_time + 5.0
    assert fixed.custom == report.custom
    assert live.custom == report.custom + 5.0


def test_intraday_overlay_preserves_confirmed_daily_value_when_quote_move_is_missing() -> None:
    portfolio = _portfolio()
    closes = pd.DataFrame(
        {"ABC": [10.0, 12.0], "OLD": [10.0, 10.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )
    report = assemble_combined_performance_report(
        [portfolio], closes, date(2026, 1, 3), date(2026, 1, 3)
    )

    partial = overlay_intraday_performance(report, {"Example": None}, include_custom=True)

    assert partial.daily == report.daily
    assert partial.all_time == report.all_time
    assert partial.custom == report.custom


def test_intraday_overlay_combines_fresh_and_confirmed_close_portfolio_rows() -> None:
    portfolio = _portfolio()
    second = _portfolio()
    second.id = 2
    second.name = "Fallback"
    second.transactions[0].id = 2
    closes = pd.DataFrame(
        {"ABC": [10.0, 12.0], "OLD": [10.0, 10.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )
    report = assemble_combined_performance_report(
        [portfolio, second], closes, date(2026, 1, 3), date(2026, 1, 3)
    )

    hybrid = overlay_intraday_performance(
        report, {"Example": 5.0, "Fallback": None}, include_custom=True
    )

    assert hybrid.rows[0].daily == 5.0
    assert hybrid.rows[1].daily == report.rows[1].daily
    assert hybrid.daily == hybrid.rows[0].daily + hybrid.rows[1].daily


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


@pytest.mark.parametrize("portfolio_count", [2, 8, 20])
def test_report_context_reconstructs_each_portfolio_once(portfolio_count: int, monkeypatch) -> None:
    portfolios, closes, benchmark = _reconstruction_fixture(portfolio_count)
    counts = {"typed": 0, "coverage": 0, "household": 0}
    per_portfolio: dict[int, int] = {}
    original_typed = reports.scoped_reconstruction
    original_coverage = analytics.reconstruct_from_coverage
    original_household = reports.household_valuation
    original_daily = reconstruction._daily_for_portfolio

    def count_typed(*args, **kwargs):
        counts["typed"] += 1
        return original_typed(*args, **kwargs)

    def count_coverage(*args, **kwargs):
        counts["coverage"] += 1
        return original_coverage(*args, **kwargs)

    def count_household(*args, **kwargs):
        counts["household"] += 1
        return original_household(*args, **kwargs)

    def count_daily(portfolio, *args, **kwargs):
        per_portfolio[portfolio.id] = per_portfolio.get(portfolio.id, 0) + 1
        return original_daily(portfolio, *args, **kwargs)

    monkeypatch.setattr(reports, "scoped_reconstruction", count_typed)
    monkeypatch.setattr(analytics, "reconstruct_from_coverage", count_coverage)
    monkeypatch.setattr(reports, "household_valuation", count_household)
    monkeypatch.setattr(reconstruction, "_daily_for_portfolio", count_daily)

    context = build_portfolio_calculation_context(portfolios, closes)
    assemble_combined_performance_report(
        portfolios,
        closes,
        closes.index[10].date(),
        closes.index[-1].date(),
        benchmark,
        calculation_context=context,
    )
    assemble_comparison_report(
        portfolios,
        closes,
        benchmark,
        closes.index[10].date(),
        closes.index[-1].date(),
        ("SPY",),
        context,
    )
    assemble_portfolio_card_reports(
        portfolios,
        closes,
        closes.index[10].date(),
        closes.index[-1].date(),
        context,
    )
    consolidated_holdings(
        portfolios,
        closes,
        closes.index[10].date(),
        closes.index[-1].date(),
        benchmark,
        household=context.household_valuation,
    )

    assert counts == {"typed": 1, "coverage": 1, "household": 1}
    assert per_portfolio == {portfolio.id: 1 for portfolio in portfolios}


def test_gain_loss_and_growth_share_reconstruction_without_mutation() -> None:
    portfolios, closes, _ = _reconstruction_fixture(2)
    closes_before = closes.copy(deep=True)
    context = build_portfolio_calculation_context(portfolios, closes)
    reconstruction_before = {
        portfolio_id: item.daily.portfolio_value.copy(deep=True)
        for portfolio_id, item in context.reconstruction.portfolios.items()
    }

    first_growth = performance_growth(portfolios[0], closes, context.reconstruction)
    gain_loss = portfolio_gain_loss(
        portfolios[0],
        closes,
        closes.index[10].date(),
        closes.index[-1].date(),
        context.reconstruction,
        context.household_valuation.valuations[0],
    )
    second_growth = performance_growth(portfolios[0], closes, context.reconstruction)

    assert gain_loss == portfolio_gain_loss(
        portfolios[0], closes, closes.index[10].date(), closes.index[-1].date()
    )
    pd.testing.assert_series_equal(first_growth, second_growth, check_exact=True)
    pd.testing.assert_frame_equal(closes, closes_before, check_exact=True)
    for portfolio_id, expected in reconstruction_before.items():
        pd.testing.assert_series_equal(
            context.reconstruction.portfolios[portfolio_id].daily.portfolio_value,
            expected,
            check_exact=True,
        )


def test_context_outputs_exactly_match_independent_report_assembly() -> None:
    portfolios, closes, benchmark = _reconstruction_fixture(2)
    start = closes.index[10].date()
    end = closes.index[-1].date()
    context = build_portfolio_calculation_context(portfolios, closes)

    shared = assemble_combined_performance_report(
        portfolios,
        closes,
        start,
        end,
        benchmark,
        calculation_context=context,
    )
    independent = assemble_combined_performance_report(portfolios, closes, start, end, benchmark)
    assert shared == independent

    shared_comparison = assemble_comparison_report(
        portfolios, closes, benchmark, start, end, ("SPY",), context
    )
    independent_comparison = assemble_comparison_report(
        portfolios, closes, benchmark, start, end, ("SPY",)
    )
    assert shared_comparison.warnings == independent_comparison.warnings
    assert shared_comparison.coverage == independent_comparison.coverage
    for shared_series, independent_series in zip(
        shared_comparison.series, independent_comparison.series, strict=True
    ):
        pd.testing.assert_series_equal(shared_series, independent_series, check_exact=True)
    pd.testing.assert_frame_equal(
        shared_comparison.metrics, independent_comparison.metrics, check_exact=True
    )


def test_context_rejects_different_portfolios_and_price_datasets() -> None:
    portfolios, closes, benchmark = _reconstruction_fixture(2)
    context = build_portfolio_calculation_context(portfolios, closes)
    start = closes.index[10].date()
    end = closes.index[-1].date()

    with pytest.raises(ValueError, match="different close-price dataset"):
        assemble_comparison_report(
            portfolios,
            closes.copy(),
            benchmark,
            start,
            end,
            ("SPY",),
            context,
        )
    with pytest.raises(ValueError, match="different portfolio selection"):
        assemble_combined_performance_report(
            portfolios[:1],
            closes,
            start,
            end,
            benchmark,
            calculation_context=context,
        )


def test_new_context_reconstructs_after_transactions_or_closes_change(monkeypatch) -> None:
    portfolios, closes, _ = _reconstruction_fixture(2)
    calls = 0
    original = reports.scoped_reconstruction

    def count_typed(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(reports, "scoped_reconstruction", count_typed)
    first = build_portfolio_calculation_context(portfolios, closes)
    portfolios[0].transactions[0].cash_delta += Decimal("1")
    second = build_portfolio_calculation_context(portfolios, closes)
    changed_closes = closes.copy()
    changed_closes.iloc[-1, 0] += 1.0
    third = build_portfolio_calculation_context(portfolios, changed_closes)

    assert calls == 3
    assert first.reconstruction is not second.reconstruction
    assert second.reconstruction is not third.reconstruction


def test_shared_reconstruction_supports_separate_custom_periods() -> None:
    portfolios, closes, benchmark = _reconstruction_fixture(2)
    context = build_portfolio_calculation_context(portfolios, closes)
    early_start = closes.index[10].date()
    late_start = closes.index[50].date()
    end = closes.index[-1].date()

    early = assemble_combined_performance_report(
        portfolios,
        closes,
        early_start,
        end,
        benchmark,
        calculation_context=context,
    )
    late = assemble_combined_performance_report(
        portfolios,
        closes,
        late_start,
        end,
        benchmark,
        calculation_context=context,
    )

    assert early.all_time == late.all_time
    assert early.daily == late.daily
    assert early.custom != late.custom
