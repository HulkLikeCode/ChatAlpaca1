from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from typing import Literal

import pandas as pd

from chat_alpaca.analytics import (
    HouseholdValuationResult,
    alpha_beta_from_levels,
    combined_performance_growth,
    household_valuation,
    performance_growth,
    portfolio_gain_loss,
    rebase_comparison_series,
    scoped_reconstruction,
    summary_metrics,
)
from chat_alpaca.market_data import get_benchmark_daily_closes, get_daily_closes
from chat_alpaca.models import Portfolio
from chat_alpaca.portfolio_service import money, portfolio_cost
from chat_alpaca.reconstruction import ReconstructionResult

BASELINE_LOOKBACK_DAYS = 7


@dataclass(frozen=True)
class HistoricalDataRequest:
    symbols: tuple[str, ...]
    start: date
    end: date
    price_policy: Literal["portfolio_accounting", "benchmark_total_return"]


@dataclass(frozen=True)
class ComparisonAcquisitionPlan:
    portfolio: HistoricalDataRequest
    benchmark: HistoricalDataRequest


@dataclass(frozen=True)
class PerformanceRow:
    portfolio: str
    cash: float
    all_time: float | None
    daily: float | None
    custom: float | None
    alpha: float | None
    beta: float | None
    alpha_beta_observations: int


@dataclass(frozen=True)
class PortfolioCardReport:
    name: str
    value_label: str
    cumulative_dividends: Decimal
    cash: Decimal
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CombinedPerformanceReport:
    total_value: Decimal | None
    total_label: str
    all_time: float | None
    daily: float | None
    custom: float | None
    alpha: float | None
    beta: float | None
    alpha_beta_observations: int
    rows: tuple[PerformanceRow, ...]
    warnings: tuple[str, ...]
    coverage: str


@dataclass(frozen=True)
class ComparisonReport:
    series: tuple[pd.Series, ...]
    metrics: pd.DataFrame
    warnings: tuple[str, ...]
    coverage: str


@dataclass(frozen=True)
class PortfolioCalculationContext:
    """Deterministic results shared only within one report or active-tab render."""

    portfolio_ids: tuple[int, ...]
    closes_identity: int
    reconstruction: ReconstructionResult | None
    household_valuation: HouseholdValuationResult | None


def build_portfolio_calculation_context(
    portfolios: list[Portfolio], closes: pd.DataFrame
) -> PortfolioCalculationContext:
    """Compute shared financial results without retaining them beyond the caller's operation."""
    return PortfolioCalculationContext(
        portfolio_ids=tuple(portfolio.id for portfolio in portfolios),
        closes_identity=id(closes),
        reconstruction=(
            scoped_reconstruction(portfolios, closes)
            if not closes.empty and any(portfolio.transactions for portfolio in portfolios)
            else None
        ),
        household_valuation=(household_valuation(portfolios, closes) if not closes.empty else None),
    )


def _validated_context(
    portfolios: list[Portfolio],
    closes: pd.DataFrame,
    context: PortfolioCalculationContext | None,
) -> PortfolioCalculationContext:
    if context is None:
        return build_portfolio_calculation_context(portfolios, closes)
    if context.portfolio_ids != tuple(portfolio.id for portfolio in portfolios):
        raise ValueError("The calculation context belongs to a different portfolio selection.")
    if context.closes_identity != id(closes):
        raise ValueError("The calculation context belongs to a different close-price dataset.")
    return context


def historical_symbol_universe(
    portfolios: list[Portfolio], extra_symbols: tuple[str, ...] = ()
) -> tuple[str, ...]:
    """Build the complete historical universe from holdings, ledger activity, and extras."""
    symbols = {lot.symbol for portfolio in portfolios for lot in portfolio.holdings if lot.symbol}
    symbols.update(
        transaction.symbol
        for portfolio in portfolios
        for transaction in portfolio.transactions
        if transaction.symbol
    )
    symbols.update(symbol.strip().upper() for symbol in extra_symbols if symbol.strip())
    return tuple(sorted(symbols))


def portfolio_acquisition_request(
    portfolios: list[Portfolio], report_start: date, report_end: date
) -> HistoricalDataRequest:
    """Apply the accounting baseline window and split-only portfolio policy."""
    if report_start > report_end:
        raise ValueError("Report start must be on or before report end.")
    return HistoricalDataRequest(
        historical_symbol_universe(portfolios),
        report_start - timedelta(days=BASELINE_LOOKBACK_DAYS),
        report_end,
        "portfolio_accounting",
    )


def comparison_acquisition_plan(
    portfolios: list[Portfolio],
    report_start: date,
    report_end: date,
    benchmark_symbols: tuple[str, ...],
) -> ComparisonAcquisitionPlan:
    normalized_benchmarks = tuple(
        dict.fromkeys(symbol.strip().upper() for symbol in benchmark_symbols if symbol.strip())
    )
    portfolio_request = portfolio_acquisition_request(portfolios, report_start, report_end)
    return ComparisonAcquisitionPlan(
        HistoricalDataRequest(
            historical_symbol_universe(portfolios, normalized_benchmarks),
            portfolio_request.start,
            report_end,
            "portfolio_accounting",
        ),
        HistoricalDataRequest(
            normalized_benchmarks,
            report_start,
            report_end,
            "benchmark_total_return",
        ),
    )


def acquire_historical_data(request: HistoricalDataRequest) -> pd.DataFrame:
    """Execute the request's explicit adjustment policy outside the UI layer."""
    if not request.symbols:
        return pd.DataFrame()
    if request.price_policy == "benchmark_total_return":
        return get_benchmark_daily_closes(list(request.symbols), request.start, request.end)
    return get_daily_closes(list(request.symbols), request.start, request.end)


def _summed(values: list[float | None]) -> float | None:
    return sum(values) if values and all(value is not None for value in values) else None


def assemble_portfolio_card_reports(
    portfolios: list[Portfolio],
    closes: pd.DataFrame,
    selected_start: date,
    selected_end: date,
    calculation_context: PortfolioCalculationContext | None = None,
) -> tuple[PortfolioCardReport, ...]:
    if selected_start > selected_end:
        raise ValueError("The portfolio card start date must be on or before the end date.")
    household = (
        _validated_context(portfolios, closes, calculation_context).household_valuation
        if not closes.empty
        else None
    )
    reports = []
    for index, portfolio in enumerate(portfolios):
        if closes.empty:
            value_label = f"Cost basis ${float(portfolio_cost(portfolio)):,.0f}"
            warnings: tuple[str, ...] = ()
        else:
            valuation = household.valuations[index]
            value_label = (
                f"${float(valuation.total_calculated_value):,.0f}"
                if valuation.is_complete
                else "Incomplete valuation"
            )
            warnings = valuation.warnings
        reports.append(
            PortfolioCardReport(
                portfolio.name,
                value_label,
                money(
                    sum(
                        (
                            transaction.cash_delta
                            for transaction in portfolio.transactions
                            if transaction.kind == "dividend"
                            and selected_start <= transaction.transaction_date <= selected_end
                        ),
                        Decimal("0"),
                    )
                ),
                Decimal(portfolio.cash),
                warnings,
            )
        )
    return tuple(reports)


def assemble_combined_performance_report(
    portfolios: list[Portfolio],
    closes: pd.DataFrame,
    custom_start: date,
    custom_end: date,
    benchmark_closes: pd.DataFrame | None = None,
    benchmark_symbol: str = "SPY",
    calculation_context: PortfolioCalculationContext | None = None,
) -> CombinedPerformanceReport:
    """Assemble portfolio value and gain/loss without presentation-layer accounting."""
    if closes.empty:
        return CombinedPerformanceReport(
            total_value=sum(
                (portfolio_cost(portfolio) for portfolio in portfolios), start=money(0)
            ),
            total_label="Selected Totals",
            all_time=None,
            daily=None,
            custom=None,
            alpha=None,
            beta=None,
            alpha_beta_observations=0,
            rows=tuple(
                PerformanceRow(
                    portfolio.name, float(portfolio.cash), None, None, None, None, None, 0
                )
                for portfolio in portfolios
            ),
            warnings=("Cost basis plus cash is shown; gain/loss requires market data.",),
            coverage="Market-price coverage unavailable.",
        )

    context = _validated_context(portfolios, closes, calculation_context)
    household = context.household_valuation
    if household is None:
        raise ValueError("A non-empty close-price dataset requires a household valuation.")
    valuations = household.valuations
    total_value = household.total_calculated_value
    gain_loss = [
        portfolio_gain_loss(
            portfolio,
            closes,
            custom_start,
            custom_end,
            context.reconstruction,
            valuations[index],
        )
        for index, portfolio in enumerate(portfolios)
    ]
    benchmark_levels = (
        benchmark_closes[benchmark_symbol].dropna()
        if benchmark_closes is not None and benchmark_symbol in benchmark_closes
        else pd.Series(dtype=float)
    )
    portfolio_risk = [
        alpha_beta_from_levels(
            performance_growth(portfolio, closes, context.reconstruction),
            benchmark_levels,
            custom_start,
            custom_end,
        )
        if not benchmark_levels.empty
        else None
        for portfolio in portfolios
    ]
    rows = tuple(
        PerformanceRow(
            portfolio.name,
            float(portfolio.cash),
            metrics.all_time,
            metrics.daily,
            metrics.custom,
            risk.alpha if risk else None,
            risk.beta if risk else None,
            risk.observations if risk else 0,
        )
        for portfolio, metrics, risk in zip(portfolios, gain_loss, portfolio_risk, strict=True)
    )
    combined_risk = (
        alpha_beta_from_levels(
            combined_performance_growth(portfolios, closes, context.reconstruction),
            benchmark_levels,
            custom_start,
            custom_end,
        )
        if not benchmark_levels.empty
        else None
    )
    warnings = sorted(
        {
            *closes.attrs.get("warnings", ()),
            *(warning for valuation in valuations for warning in valuation.warnings),
            *(warning for metrics in gain_loss for warning in metrics.warnings),
        }
    )
    complete_count = sum(valuation.is_complete for valuation in valuations)
    return CombinedPerformanceReport(
        total_value=total_value,
        total_label="Selected Totals",
        all_time=_summed([row.all_time for row in rows]),
        daily=_summed([row.daily for row in rows]),
        custom=_summed([row.custom for row in rows]),
        alpha=combined_risk.alpha if combined_risk else None,
        beta=combined_risk.beta if combined_risk else None,
        alpha_beta_observations=combined_risk.observations if combined_risk else 0,
        rows=rows,
        warnings=tuple(warnings),
        coverage=(
            f"Confirmed household date: {household.common_valuation_date}; complete valuations: "
            f"{complete_count} of {len(valuations)} portfolios."
            if household.common_valuation_date is not None
            else f"Complete valuations: {complete_count} of {len(valuations)} portfolios."
        ),
    )


def overlay_intraday_performance(
    report: CombinedPerformanceReport,
    portfolio_changes: dict[str, float | None],
    *,
    include_custom: bool,
    indicative_total_value: float | None = None,
) -> CombinedPerformanceReport:
    """Overlay complete indicative quote moves on close-based performance metrics."""
    rows = []
    for row in report.rows:
        change = portfolio_changes.get(row.portfolio)
        rows.append(
            replace(
                row,
                all_time=(
                    row.all_time + change
                    if row.all_time is not None and change is not None
                    else row.all_time
                ),
                daily=change if change is not None else row.daily,
                custom=(
                    row.custom + change
                    if include_custom and row.custom is not None and change is not None
                    else row.custom
                ),
            )
        )
    updated_rows = tuple(rows)
    return replace(
        report,
        total_value=(
            money(indicative_total_value)
            if indicative_total_value is not None
            else report.total_value
        ),
        total_label="Selected Totals",
        all_time=_summed([row.all_time for row in updated_rows]),
        daily=_summed([row.daily for row in updated_rows]),
        custom=_summed([row.custom for row in updated_rows]),
        rows=updated_rows,
    )


def assemble_comparison_report(
    portfolios: list[Portfolio],
    portfolio_closes: pd.DataFrame,
    benchmark_closes: pd.DataFrame,
    report_start: date,
    report_end: date,
    benchmark_symbols: tuple[str, ...],
    calculation_context: PortfolioCalculationContext | None = None,
) -> ComparisonReport:
    """Assemble normalized comparison series, statistics, warnings, and coverage."""
    context = _validated_context(portfolios, portfolio_closes, calculation_context)
    portfolio_series = [
        combined_performance_growth(portfolios, portfolio_closes, context.reconstruction),
        *(
            performance_growth(portfolio, portfolio_closes, context.reconstruction)
            for portfolio in portfolios
        ),
    ]
    candidates = [
        item[(item.index.date >= report_start) & (item.index.date <= report_end)]
        for item in portfolio_series
    ]
    missing_benchmarks: list[str] = []
    for symbol in benchmark_symbols:
        if symbol not in benchmark_closes or benchmark_closes[symbol].dropna().empty:
            missing_benchmarks.append(symbol)
            continue
        values = benchmark_closes.loc[
            (benchmark_closes.index.date >= report_start)
            & (benchmark_closes.index.date <= report_end),
            symbol,
        ].copy()
        values.name = symbol
        candidates.append(values)

    normalized = tuple(item for item in rebase_comparison_series(candidates) if not item.empty)
    rows = []
    for item in normalized:
        calculated = summary_metrics(item)
        metrics = {
            key: value * 100 if value is not None else None for key, value in calculated.items()
        }
        rows.append({"Series": item.name, **metrics})
    warnings = {
        *portfolio_closes.attrs.get("warnings", ()),
        *benchmark_closes.attrs.get("warnings", ()),
    }
    if missing_benchmarks:
        warnings.add("Comparison data is incomplete for: " + ", ".join(missing_benchmarks) + ".")
    unavailable = [
        item.name
        for item in normalized
        if all(value is None for value in summary_metrics(item).values())
    ]
    if unavailable:
        warnings.add(
            "Comparison metrics are unavailable for series with fewer than two valid "
            "observations: " + ", ".join(unavailable) + "."
        )
    requested_count = len(portfolios) + 1 + len(benchmark_symbols)
    return ComparisonReport(
        normalized,
        pd.DataFrame(rows),
        tuple(sorted(warnings)),
        f"Usable comparison series: {len(normalized)} of {requested_count} requested.",
    )
