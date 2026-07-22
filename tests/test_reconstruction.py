from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from chat_alpaca.historical_data import (
    BarValue,
    HistoricalCoverageResult,
    PriceAdjustment,
    ProviderDataset,
    SqlHistoricalDataRepository,
)
from chat_alpaca.models import Portfolio, PortfolioTransaction
from chat_alpaca.reconstruction import (
    PortfolioReconstructionService,
    ReconstructionRequest,
    SufficiencyStatus,
    calculate_xirr,
    reconstruct_from_coverage,
)


def _transaction(
    transaction_id: int,
    day: date,
    kind: str,
    cash: str,
    *,
    symbol: str | None = None,
    quantity: str | None = None,
    price: str | None = None,
    fees: str | None = None,
) -> PortfolioTransaction:
    return PortfolioTransaction(
        id=transaction_id,
        transaction_date=day,
        kind=kind,
        action=kind.replace("_", " ").title(),
        symbol=symbol,
        description="test",
        quantity=Decimal(quantity) if quantity else None,
        price=Decimal(price) if price else None,
        fees=Decimal(fees) if fees else None,
        cash_delta=Decimal(cash),
        source="test",
    )


def _coverage(
    prices: dict[str, list[float | None]],
    days: list[str],
    *,
    source: str | tuple[str, ...] = "test",
    adjustment: str = "split",
) -> HistoricalCoverageResult:
    frame = pd.DataFrame(prices, index=pd.to_datetime(days), dtype=float)
    return HistoricalCoverageResult(
        data=frame,
        source=source,
        feed="test",
        adjustment=adjustment,
        coverage_start=frame.index.min().date() if not frame.empty else None,
        coverage_end=frame.index.max().date() if not frame.empty else None,
        missing_symbols=tuple(symbol for symbol in frame if frame[symbol].dropna().empty),
        missing_date_ranges={},
        warnings=(),
        freshness={symbol: datetime.now(timezone.utc) for symbol in frame},
        usable=not frame.isna().any().any(),
        usability="test coverage",
    )


def _request(*portfolio_ids: int, start: date = date(2026, 1, 2), end: date = date(2026, 1, 9)):
    return ReconstructionRequest(portfolio_ids, start, end)


def test_reconstructs_flows_income_expenses_returns_and_rebasing() -> None:
    portfolio = Portfolio(id=1, name="Lifecycle", cash=Decimal("0"))
    portfolio.transactions = [
        _transaction(1, date(2026, 1, 2), "transfer", "100"),
        _transaction(2, date(2026, 1, 5), "buy", "-50", symbol="ABC", quantity="5", price="10"),
        _transaction(3, date(2026, 1, 6), "dividend", "5", symbol="ABC"),
        _transaction(4, date(2026, 1, 6), "interest", "2"),
        _transaction(5, date(2026, 1, 7), "fee", "-1"),
        _transaction(6, date(2026, 1, 7), "tax", "-2"),
        _transaction(7, date(2026, 1, 7), "award", "3"),
        _transaction(8, date(2026, 1, 8), "sell", "60", symbol="ABC", quantity="5", price="12"),
        _transaction(9, date(2026, 1, 9), "transfer", "-20"),
    ]
    coverage = _coverage(
        {"ABC": [10, 10, 11, 12, 12, 12]},
        ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"],
    )

    result = reconstruct_from_coverage([portfolio], _request(1), coverage)
    daily = result.portfolios[1].daily

    assert daily.portfolio_value.tolist() == [100, 100, 112, 117, 117, 97]
    assert daily.external_cash_flows.tolist() == [100, 0, 0, 3, 0, -20]
    assert daily.dividends.loc["2026-01-06"] == 5
    assert daily.interest.loc["2026-01-06"] == 2
    assert daily.fees.loc["2026-01-07"] == -1
    assert daily.taxes.loc["2026-01-07"] == -2
    assert daily.awards.loc["2026-01-07"] == 3
    assert daily.time_weighted_return.iloc[0] == 0
    assert daily.total_return.loc["2026-01-06"] == pytest.approx(0.12)
    assert daily.income_return.loc["2026-01-06"] == pytest.approx(0.07)
    assert daily.price_return.loc["2026-01-06"] == pytest.approx(0.05)
    assert daily.total_return.loc["2026-01-09"] == pytest.approx(0)
    assert result.gain_loss == pytest.approx(14)
    assert result.xirr is not None


def test_multiple_portfolios_remain_separate_and_combine() -> None:
    first = Portfolio(id=1, name="First", cash=Decimal("0"))
    first.transactions = [_transaction(1, date(2026, 1, 2), "transfer", "100")]
    second = Portfolio(id=2, name="Second", cash=Decimal("0"))
    second.transactions = [_transaction(2, date(2026, 1, 2), "transfer", "50")]

    result = reconstruct_from_coverage(
        [first, second], _request(1, 2), _coverage({}, ["2026-01-02", "2026-01-05"])
    )

    assert result.portfolios[1].daily.portfolio_value.iloc[-1] == 100
    assert result.portfolios[2].daily.portfolio_value.iloc[-1] == 50
    assert result.combined.portfolio_value.iloc[-1] == 150
    assert result.combined.positions.columns.names == ["portfolio", "symbol"]


def test_combined_returns_exclude_a_new_years_day_cash_adjustment() -> None:
    existing = Portfolio(id=1, name="Existing", cash=Decimal("0"))
    existing.transactions = [_transaction(1, date(2025, 12, 31), "transfer", "100")]
    new_year_contribution = Portfolio(id=2, name="New year", cash=Decimal("0"))
    new_year_contribution.transactions = [
        _transaction(2, date(2026, 1, 1), "cash_adjustment", "50")
    ]
    coverage = _coverage({}, ["2025-12-31", "2026-01-02"])
    request = ReconstructionRequest((1, 2), date(2025, 12, 31), date(2026, 1, 2))

    result = reconstruct_from_coverage([existing, new_year_contribution], request, coverage)

    assert result.combined.external_cash_flows.loc["2026-01-02"] == 50
    assert result.combined.total_return.loc["2026-01-02"] == 0
    assert result.combined.time_weighted_return.loc["2026-01-02"] == 0


def test_positions_opened_and_closed_mid_period_do_not_require_prices_while_flat() -> None:
    portfolio = Portfolio(id=1, name="Round trip", cash=Decimal("0"))
    portfolio.transactions = [
        _transaction(1, date(2026, 1, 2), "transfer", "100"),
        _transaction(2, date(2026, 1, 6), "buy", "-20", symbol="ABC", quantity="2", price="10"),
        _transaction(3, date(2026, 1, 8), "sell", "24", symbol="ABC", quantity="2", price="12"),
    ]
    coverage = _coverage(
        {"ABC": [None, None, 10, 11, 12, None]},
        ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"],
    )

    result = reconstruct_from_coverage([portfolio], _request(1), coverage)
    daily = result.portfolios[1].daily

    assert daily.positions["ABC"].tolist() == [0, 0, 2, 2, 0, 0]
    assert daily.portfolio_value.tolist() == [100, 100, 100, 102, 104, 104]
    assert result.missing_symbols == ()


def test_missing_held_price_is_unknown_not_zero_and_invalidates_common_date() -> None:
    portfolio = Portfolio(id=1, name="Missing", cash=Decimal("0"))
    portfolio.transactions = [
        _transaction(
            1, date(2026, 1, 2), "opening_position", "0", symbol="ABC", quantity="1", price="10"
        )
    ]
    coverage = _coverage({"ABC": [10, None]}, ["2026-01-02", "2026-01-05"])

    result = reconstruct_from_coverage(
        [portfolio], _request(1, start=date(2026, 1, 2), end=date(2026, 1, 5)), coverage
    )

    assert pd.isna(result.combined.portfolio_value.iloc[-1])
    assert result.common_as_of_date == date(2026, 1, 2)
    assert result.common_as_of_value == 10
    assert result.data_coverage.missing_observations == 1
    assert not result.suitable_for_forecasting


def test_common_as_of_uses_oldest_held_symbol_close_and_reports_staleness() -> None:
    portfolio = Portfolio(id=1, name="As of", cash=Decimal("0"))
    portfolio.transactions = [
        _transaction(
            1, date(2026, 1, 2), "opening_position", "0", symbol="ABC", quantity="1", price="10"
        ),
        _transaction(
            2, date(2026, 1, 2), "opening_position", "0", symbol="XYZ", quantity="1", price="20"
        ),
    ]
    coverage = _coverage({"ABC": [10, 11], "XYZ": [20, None]}, ["2026-01-02", "2026-01-05"])

    result = reconstruct_from_coverage(
        [portfolio], _request(1, start=date(2026, 1, 2), end=date(2026, 1, 5)), coverage
    )

    assert result.common_as_of_date == date(2026, 1, 2)
    assert result.common_as_of_value == 30
    assert result.common_as_of_portfolio_values == {1: 30}
    assert result.stale_symbols == ("XYZ",)
    assert any("Mixed price freshness" in warning for warning in result.warnings)


def test_mixed_sources_and_proxy_use_reduce_transparent_sufficiency() -> None:
    portfolio = Portfolio(id=1, name="Quality", cash=Decimal("0"))
    portfolio.transactions = [_transaction(1, date(2026, 1, 2), "transfer", "100")]
    request = ReconstructionRequest(
        (1,), date(2026, 1, 2), date(2026, 1, 5), proxy_symbols={"NEW": "SPY"}
    )

    result = reconstruct_from_coverage(
        [portfolio], request, _coverage({}, ["2026-01-02", "2026-01-05"], source=("alpaca", "csv"))
    )

    assert result.data_coverage.mixed_sources
    assert result.data_coverage.proxy_use == {"NEW": "SPY"}
    assert result.data_coverage.score_components["proxy_use"] == 5
    assert result.data_coverage.status in {
        SufficiencyStatus.LIMITED,
        SufficiencyStatus.INSUFFICIENT,
    }
    assert any("not used to value" in warning for warning in result.warnings)


def test_benchmark_total_return_is_rebased_to_report_start() -> None:
    portfolio = Portfolio(id=1, name="Benchmark", cash=Decimal("0"))
    portfolio.transactions = [_transaction(1, date(2026, 1, 2), "transfer", "100")]
    request = ReconstructionRequest((1,), date(2026, 1, 2), date(2026, 1, 5), ("SPY",))
    benchmark = _coverage({"SPY": [50, 55]}, ["2026-01-02", "2026-01-05"], adjustment="all")

    result = reconstruct_from_coverage(
        [portfolio], request, _coverage({}, ["2026-01-02", "2026-01-05"]), {"SPY": benchmark}
    )

    assert result.benchmarks["SPY"].benchmark_growth.tolist() == [100, pytest.approx(110)]
    assert result.benchmarks["SPY"].relative_return.iloc[0] == 0
    assert result.benchmarks["SPY"].relative_return.iloc[-1] == pytest.approx(-0.1)


def test_xirr_known_example() -> None:
    result = calculate_xirr([(date(2020, 1, 1), -1000), (date(2021, 1, 1), 1100)])

    assert result == pytest.approx(0.0997, abs=0.0002)


def test_reconstruction_is_reproducible() -> None:
    portfolio = Portfolio(id=1, name="Repeat", cash=Decimal("0"))
    portfolio.transactions = [
        _transaction(2, date(2026, 1, 5), "buy", "-10", symbol="ABC", quantity="1", price="10"),
        _transaction(1, date(2026, 1, 2), "transfer", "100"),
    ]
    coverage = _coverage({"ABC": [9, 10]}, ["2026-01-02", "2026-01-05"])

    first = reconstruct_from_coverage(
        [portfolio], _request(1, start=date(2026, 1, 2), end=date(2026, 1, 5)), coverage
    )
    second = reconstruct_from_coverage(
        [portfolio], _request(1, start=date(2026, 1, 2), end=date(2026, 1, 5)), coverage
    )

    pd.testing.assert_series_equal(first.combined.portfolio_value, second.combined.portfolio_value)
    pd.testing.assert_frame_equal(first.combined.positions, second.combined.positions)
    assert first.assumptions == second.assumptions
    assert first.data_coverage == second.data_coverage


def _bar(symbol: str, day: date, close: str) -> BarValue:
    value = Decimal(close)
    return BarValue(symbol, day, value, value, value, value)


def test_service_reads_canonical_ledger_and_repository_with_benchmark(session) -> None:
    portfolio = Portfolio(name="Stored", cash=Decimal("100"))
    session.add(portfolio)
    session.flush()
    session.add_all(
        [
            _transaction(1, date(2026, 1, 2), "transfer", "100"),
            _transaction(2, date(2026, 1, 5), "buy", "-10", symbol="ABC", quantity="1", price="10"),
        ]
    )
    for transaction in session.new:
        if isinstance(transaction, PortfolioTransaction):
            transaction.portfolio_id = portfolio.id
    repository = SqlHistoricalDataRepository(session)
    now = datetime.now(timezone.utc)
    repository.persist(
        ProviderDataset(
            "test",
            "primary",
            "iex",
            "1Day",
            PriceAdjustment.SPLIT,
            now,
            (_bar("ABC", date(2026, 1, 2), "9"), _bar("ABC", date(2026, 1, 5), "10")),
        )
    )
    repository.persist(
        ProviderDataset(
            "test",
            "benchmark",
            "iex",
            "1Day",
            PriceAdjustment.TOTAL_RETURN,
            now,
            (_bar("SPY", date(2026, 1, 2), "100"), _bar("SPY", date(2026, 1, 5), "101")),
        )
    )
    service = PortfolioReconstructionService(session, repository)

    result = service.reconstruct(
        ReconstructionRequest((portfolio.id,), date(2026, 1, 2), date(2026, 1, 5), ("SPY",))
    )

    assert result.portfolios[portfolio.id].daily.portfolio_value.tolist() == [100, 100]
    assert result.benchmarks["SPY"].coverage.adjustment == "all"
