from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from math import isfinite

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from chat_alpaca.historical_data import (
    BENCHMARK_TOTAL_RETURN_ADJUSTMENT,
    PORTFOLIO_ACCOUNTING_ADJUSTMENT,
    HistoricalCoverageResult,
    HistoricalDataRepository,
    HistoricalRequest,
)
from chat_alpaca.market_calendar import market_session_index
from chat_alpaca.models import Instrument, Portfolio, PortfolioTransaction, ProxyAssignment

EXTERNAL_FLOW_KINDS = {"transfer", "cash_adjustment", "award"}
POSITION_KINDS = {"buy", "sell", "opening_position", "award"}
INCOME_KINDS = {"dividend", "interest"}


class SufficiencyStatus(str, Enum):
    GOOD = "good"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class ReconstructionRequest:
    portfolio_ids: tuple[int, ...]
    start: date
    end: date
    benchmark_symbols: tuple[str, ...] = ()
    proxy_symbols: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = tuple(dict.fromkeys(int(item) for item in self.portfolio_ids))
        if not ids:
            raise ValueError("At least one portfolio is required.")
        if self.start > self.end:
            raise ValueError("Reconstruction start date must be on or before the end date.")
        object.__setattr__(self, "portfolio_ids", ids)
        object.__setattr__(
            self,
            "benchmark_symbols",
            tuple(dict.fromkeys(symbol.strip().upper() for symbol in self.benchmark_symbols)),
        )
        object.__setattr__(
            self,
            "proxy_symbols",
            {
                target.strip().upper(): proxy.strip().upper()
                for target, proxy in self.proxy_symbols.items()
            },
        )


@dataclass(frozen=True)
class DataCoverage:
    status: SufficiencyStatus
    score: int
    score_components: Mapping[str, int]
    history_days: int
    expected_observations: int
    available_observations: int
    missing_observations: int
    stale_observations: int
    sources: tuple[str, ...]
    mixed_sources: bool
    proxy_use: Mapping[str, str]
    adjustment: str
    adjustment_quality: str
    valid_start_baseline: bool
    common_date_complete: bool
    suitable_for_forecasting: bool
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class DailyReconstruction:
    portfolio_value: pd.Series
    cash: pd.Series
    positions: pd.DataFrame
    external_cash_flows: pd.Series
    dividends: pd.Series
    interest: pd.Series
    fees: pd.Series
    taxes: pd.Series
    awards: pd.Series
    price_return: pd.Series
    income_return: pd.Series
    total_return: pd.Series
    time_weighted_return: pd.Series
    gain_loss: pd.Series
    calculation_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortfolioReconstruction:
    portfolio_id: int
    portfolio_name: str
    daily: DailyReconstruction
    money_weighted_return: float | None
    xirr: float | None
    gain_loss: float | None


@dataclass(frozen=True)
class BenchmarkRelativeSeries:
    symbol: str
    benchmark_growth: pd.Series
    relative_return: pd.Series
    coverage: HistoricalCoverageResult


@dataclass(frozen=True)
class ReconstructionResult:
    request: ReconstructionRequest
    portfolios: Mapping[int, PortfolioReconstruction]
    combined: DailyReconstruction
    money_weighted_return: float | None
    xirr: float | None
    gain_loss: float | None
    benchmarks: Mapping[str, BenchmarkRelativeSeries]
    data_coverage: DataCoverage
    common_as_of_date: date | None
    common_as_of_value: float | None
    common_as_of_portfolio_values: Mapping[int, float | None]
    stale_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    suitable_for_forecasting: bool


def _transactions(portfolio: Portfolio) -> list[PortfolioTransaction]:
    return sorted(
        (item for item in portfolio.transactions),
        key=lambda item: (item.transaction_date, item.id or 0),
    )


def _symbols(portfolios: Sequence[Portfolio], end: date) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                transaction.symbol
                for portfolio in portfolios
                for transaction in _transactions(portfolio)
                if transaction.symbol
                and transaction.kind in POSITION_KINDS
                and transaction.transaction_date <= end
            }
        )
    )


def _report_index(start: date, end: date, prices: pd.DataFrame) -> pd.DatetimeIndex:
    requested = market_session_index(start, end)
    confirmed = pd.DatetimeIndex(prices.index).normalize()
    return requested.union(
        confirmed[(confirmed.date >= start) & (confirmed.date <= end)]
    ).sort_values()


def _empty_series(index: pd.DatetimeIndex, name: str) -> pd.Series:
    return pd.Series(0.0, index=index, name=name, dtype=float)


def _event_series(
    transactions: Sequence[PortfolioTransaction],
    index: pd.DatetimeIndex,
    kinds: set[str],
    name: str,
) -> pd.Series:
    result = _empty_series(index, name)
    for transaction in transactions:
        timestamp = _event_timestamp(transaction.transaction_date, result.index)
        if transaction.kind in kinds and timestamp is not None:
            result.loc[timestamp] += float(transaction.cash_delta)
    return result


def _event_timestamp(event_date: date, index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Post an event on its trading date or the next available calculation date."""
    eligible = index[index >= pd.Timestamp(event_date)]
    return eligible[0] if not eligible.empty else None


def _opening_asset_flow(transaction: PortfolioTransaction) -> float:
    if (
        transaction.kind == "opening_position"
        and transaction.quantity is not None
        and transaction.price is not None
    ):
        return float(transaction.quantity * transaction.price)
    return 0.0


def _award_asset_flow(transaction: PortfolioTransaction) -> float | None:
    if transaction.kind != "award" or transaction.quantity is None:
        return 0.0
    if transaction.price is None or transaction.price <= 0:
        return None
    return float(transaction.quantity * transaction.price)


def _daily_for_portfolio(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    start: date,
    end: date,
) -> DailyReconstruction:
    index = _report_index(start, end, prices)
    prior_dates = pd.DatetimeIndex(prices.index[prices.index.date < start])
    calculation_index = (
        index.insert(0, prior_dates.max()).drop_duplicates().sort_values()
        if not prior_dates.empty
        else index
    )
    transactions = _transactions(portfolio)
    symbols = sorted(
        {item.symbol for item in transactions if item.symbol and item.kind in POSITION_KINDS}
    )
    positions = pd.DataFrame(0.0, index=calculation_index, columns=symbols)
    cash = _empty_series(calculation_index, "cash")

    running_cash = 0.0
    running_positions = {symbol: 0.0 for symbol in symbols}
    cursor = 0
    for timestamp in calculation_index:
        while (
            cursor < len(transactions) and transactions[cursor].transaction_date <= timestamp.date()
        ):
            transaction = transactions[cursor]
            running_cash += float(transaction.cash_delta)
            if transaction.kind in POSITION_KINDS and transaction.quantity is not None:
                direction = -1.0 if transaction.kind == "sell" else 1.0
                running_positions[transaction.symbol] += direction * float(transaction.quantity)
            cursor += 1
        cash.loc[timestamp] = running_cash
        for symbol, quantity in running_positions.items():
            positions.loc[timestamp, symbol] = quantity

    values = cash.copy().rename("portfolio_value")
    for timestamp in calculation_index:
        market_value = 0.0
        complete = True
        for symbol in symbols:
            quantity = positions.loc[timestamp, symbol]
            if abs(quantity) < 1e-12:
                continue
            if (
                symbol not in prices
                or timestamp not in prices.index
                or pd.isna(prices.loc[timestamp, symbol])
            ):
                complete = False
                break
            market_value += quantity * float(prices.loc[timestamp, symbol])
        values.loc[timestamp] = cash.loc[timestamp] + market_value if complete else np.nan

    external = _event_series(
        transactions, calculation_index, EXTERNAL_FLOW_KINDS, "external_cash_flows"
    )
    for transaction in transactions:
        timestamp = _event_timestamp(transaction.transaction_date, external.index)
        if timestamp in external.index:
            external.loc[timestamp] += _opening_asset_flow(transaction)
            award_flow = _award_asset_flow(transaction)
            if award_flow is not None:
                external.loc[timestamp] += award_flow
    dividends = _event_series(transactions, calculation_index, {"dividend"}, "dividends")
    interest = _event_series(transactions, calculation_index, {"interest"}, "interest")
    fees = _event_series(transactions, calculation_index, {"fee"}, "fees")
    for transaction in transactions:
        timestamp = _event_timestamp(transaction.transaction_date, fees.index)
        if transaction.kind != "fee" and transaction.fees and timestamp in fees.index:
            fees.loc[timestamp] -= abs(float(transaction.fees))
    taxes = _event_series(transactions, calculation_index, {"tax"}, "taxes")
    awards = _event_series(transactions, calculation_index, {"award"}, "awards")
    missing_award_values = [
        transaction
        for transaction in transactions
        if transaction.kind == "award"
        and transaction.quantity is not None
        and _award_asset_flow(transaction) is None
    ]
    unavailable_award_dates = {
        _event_timestamp(transaction.transaction_date, calculation_index)
        for transaction in missing_award_values
    }

    total_return = pd.Series(np.nan, index=calculation_index, name="total_return", dtype=float)
    income_return = pd.Series(np.nan, index=calculation_index, name="income_return", dtype=float)
    price_return = pd.Series(np.nan, index=calculation_index, name="price_return", dtype=float)
    gain_loss = pd.Series(np.nan, index=calculation_index, name="gain_loss", dtype=float)
    previous_value: float | None = None
    for timestamp, value in values.items():
        if pd.isna(value):
            previous_value = None
            continue
        current = float(value)
        external_flow = float(external.loc[timestamp])
        if previous_value is not None:
            income = float(dividends.loc[timestamp] + interest.loc[timestamp])
            non_price = float(
                dividends.loc[timestamp]
                + interest.loc[timestamp]
                + fees.loc[timestamp]
                + taxes.loc[timestamp]
            )
            gain = current - previous_value - external_flow
            gain_loss.loc[timestamp] = gain
            if previous_value != 0:
                total = gain / previous_value
                income_component = income / previous_value
                total_return.loc[timestamp] = total
                income_return.loc[timestamp] = income_component
                price_return.loc[timestamp] = total - non_price / previous_value
        elif external_flow:
            gain_loss.loc[timestamp] = current - external_flow
        previous_value = current

    twr = (1.0 + total_return.fillna(0.0)).cumprod() - 1.0
    if missing_award_values:
        for timestamp in unavailable_award_dates:
            if timestamp is not None:
                gain_loss.loc[timestamp] = np.nan
                total_return.loc[timestamp] = np.nan
                income_return.loc[timestamp] = np.nan
                price_return.loc[timestamp] = np.nan
        twr.loc[:] = np.nan
    twr.name = "time_weighted_return"
    selected = calculation_index[calculation_index.date >= start]
    return DailyReconstruction(
        portfolio_value=values.loc[selected],
        cash=cash.loc[selected],
        positions=positions.loc[selected],
        external_cash_flows=external.loc[selected],
        dividends=dividends.loc[selected],
        interest=interest.loc[selected],
        fees=fees.loc[selected],
        taxes=taxes.loc[selected],
        awards=awards.loc[selected],
        price_return=price_return.loc[selected],
        income_return=income_return.loc[selected],
        total_return=total_return.loc[selected],
        time_weighted_return=twr.loc[selected],
        gain_loss=gain_loss.loc[selected],
        calculation_warnings=(
            (
                "Gain and return outputs are unavailable because a quantity award lacks an "
                "explicitly recorded fair value; no market price was inferred."
            ),
        )
        if missing_award_values
        else (),
    )


def _xnpv(rate: float, flows: Sequence[tuple[date, float]]) -> float:
    origin = flows[0][0]
    return sum(amount / (1.0 + rate) ** ((day - origin).days / 365.0) for day, amount in flows)


def calculate_xirr(flows: Sequence[tuple[date, float]]) -> float | None:
    """Return annualized XIRR using deterministic bisection, or None when undefined."""
    cleaned = [(day, float(amount)) for day, amount in flows if amount and isfinite(float(amount))]
    if (
        not cleaned
        or not any(amount < 0 for _, amount in cleaned)
        or not any(amount > 0 for _, amount in cleaned)
    ):
        return None
    cleaned.sort(key=lambda item: item[0])
    low, high = -0.999999, 10.0
    low_value, high_value = _xnpv(low, cleaned), _xnpv(high, cleaned)
    while low_value * high_value > 0 and high < 1_000_000:
        high *= 10
        high_value = _xnpv(high, cleaned)
    if low_value * high_value > 0:
        return None
    for _ in range(200):
        midpoint = (low + high) / 2
        value = _xnpv(midpoint, cleaned)
        if abs(value) < 1e-10:
            return midpoint
        if low_value * value <= 0:
            high = midpoint
        else:
            low, low_value = midpoint, value
    return (low + high) / 2


def _money_weighted_return(daily: DailyReconstruction, start: date, end: date) -> float | None:
    valid = daily.portfolio_value.dropna()
    if valid.empty:
        return None
    first_date = valid.index[0].date()
    last_date = valid.index[-1].date()
    flows: list[tuple[date, float]] = [(first_date, -float(valid.iloc[0]))]
    for timestamp, amount in daily.external_cash_flows.items():
        if timestamp.date() > first_date and amount:
            flows.append((timestamp.date(), -float(amount)))
    flows.append((last_date, float(valid.iloc[-1])))
    if first_date == last_date or first_date > end or last_date < start:
        return None
    return calculate_xirr(flows)


def _combine_daily(
    items: Sequence[DailyReconstruction], portfolio_ids: Sequence[int] | None = None
) -> DailyReconstruction:
    def summed(attribute: str, *, complete: bool = False) -> pd.Series:
        frame = pd.concat([getattr(item, attribute) for item in items], axis=1)
        return frame.sum(axis=1, min_count=len(items) if complete else 1).rename(attribute)

    positions = pd.concat(
        [item.positions for item in items],
        axis=1,
        keys=portfolio_ids or range(len(items)),
        names=["portfolio", "symbol"],
    )
    value = summed("portfolio_value", complete=True)
    external = summed("external_cash_flows")
    dividends = summed("dividends")
    interest = summed("interest")
    fees = summed("fees")
    taxes = summed("taxes")
    awards = summed("awards")
    gain = value.diff() - external
    total = gain / value.shift(1).replace(0, np.nan)
    income = (dividends + interest) / value.shift(1).replace(0, np.nan)
    non_price = dividends + interest + fees + taxes
    price = total - non_price / value.shift(1).replace(0, np.nan)
    twr = (1.0 + total.fillna(0.0)).cumprod() - 1.0
    calculation_warnings = tuple(
        dict.fromkeys(warning for item in items for warning in item.calculation_warnings)
    )
    if calculation_warnings:
        twr.loc[:] = np.nan
    return DailyReconstruction(
        portfolio_value=value,
        cash=summed("cash"),
        positions=positions,
        external_cash_flows=external,
        dividends=dividends,
        interest=interest,
        fees=fees,
        taxes=taxes,
        awards=awards,
        price_return=price.rename("price_return"),
        income_return=income.rename("income_return"),
        total_return=total.rename("total_return"),
        time_weighted_return=twr.rename("time_weighted_return"),
        gain_loss=gain.rename("gain_loss"),
        calculation_warnings=calculation_warnings,
    )


def _latest_price_status(
    prices: pd.DataFrame, positions: pd.DataFrame, end: date
) -> tuple[date | None, tuple[str, ...], tuple[str, ...]]:
    if positions.empty:
        return end, (), ()
    end_positions = positions.iloc[-1]
    if isinstance(end_positions.index, pd.MultiIndex):
        active = {
            symbol
            for (_, symbol), quantity in end_positions.items()
            if abs(float(quantity)) > 1e-12
        }
    else:
        active = {
            symbol for symbol, quantity in end_positions.items() if abs(float(quantity)) > 1e-12
        }
    last_dates: dict[str, date] = {}
    for symbol in active:
        if symbol in prices:
            eligible = prices.loc[prices.index.date <= end, symbol].dropna()
            if not eligible.empty:
                last_dates[symbol] = eligible.index[-1].date()
    missing = tuple(sorted(active - set(last_dates)))
    common = min(last_dates.values()) if last_dates and not missing else None
    freshest = max(last_dates.values()) if last_dates else None
    stale = tuple(
        sorted(
            symbol for symbol, last_date in last_dates.items() if freshest and last_date < freshest
        )
    )
    return common, stale, missing


def _as_of_value(portfolio: Portfolio, prices: pd.DataFrame, as_of: date) -> float | None:
    cash = 0.0
    positions: dict[str, float] = {}
    for transaction in _transactions(portfolio):
        if transaction.transaction_date > as_of:
            break
        cash += float(transaction.cash_delta)
        if transaction.kind in POSITION_KINDS and transaction.symbol and transaction.quantity:
            direction = -1.0 if transaction.kind == "sell" else 1.0
            positions[transaction.symbol] = positions.get(transaction.symbol, 0.0) + (
                direction * float(transaction.quantity)
            )
    value = cash
    for symbol, quantity in positions.items():
        if abs(quantity) < 1e-12:
            continue
        if symbol not in prices:
            return None
        confirmed = prices.loc[prices.index.date <= as_of, symbol].dropna()
        if confirmed.empty:
            return None
        value += quantity * float(confirmed.iloc[-1])
    return value


def _coverage_score(
    coverage: HistoricalCoverageResult,
    prices: pd.DataFrame,
    proxies: Mapping[str, str],
    baseline_valid: bool,
    common_complete: bool,
    stale_symbols: Sequence[str],
    start: date,
    end: date,
) -> DataCoverage:
    requested_index = pd.bdate_range(start, end)
    requested_prices = prices.reindex(requested_index)
    expected = int(len(requested_index) * prices.shape[1])
    available = int(requested_prices.notna().sum().sum())
    missing = max(expected - available, 0)
    history_days = (
        (coverage.coverage_end - coverage.coverage_start).days + 1
        if coverage.coverage_start and coverage.coverage_end
        else 0
    )
    history_score = min(20, round(20 * history_days / 365))
    observation_score = 25 if expected == 0 else round(25 * available / expected)
    stale_score = 15 if not stale_symbols else max(0, 15 - 5 * len(stale_symbols))
    proxy_score = 10 if not proxies else max(0, 10 - 5 * len(proxies))
    adjustment_score = 15 if coverage.adjustment == PORTFOLIO_ACCOUNTING_ADJUSTMENT.value else 0
    common_score = 15 if common_complete and baseline_valid else 0
    components = {
        "history_length": history_score,
        "observations": observation_score,
        "freshness": stale_score,
        "proxy_use": proxy_score,
        "adjustment_quality": adjustment_score,
        "common_date_completeness": common_score,
    }
    score = sum(components.values())
    status = (
        SufficiencyStatus.GOOD
        if score >= 80
        else SufficiencyStatus.LIMITED
        if score >= 50
        else SufficiencyStatus.INSUFFICIENT
    )
    sources = (coverage.source,) if isinstance(coverage.source, str) else tuple(coverage.source)
    suitable = (
        status is SufficiencyStatus.GOOD
        and history_days >= 365
        and not proxies
        and not stale_symbols
        and baseline_valid
        and common_complete
        and missing == 0
    )
    rationale = (
        f"{history_days} calendar days of confirmed history are available.",
        f"{available} of {expected} requested symbol-date observations are available.",
        "Forecasting requires at least one year, complete common-date valuation, and no proxy use.",
    )
    return DataCoverage(
        status=status,
        score=score,
        score_components=components,
        history_days=history_days,
        expected_observations=expected,
        available_observations=available,
        missing_observations=missing,
        stale_observations=len(stale_symbols),
        sources=sources,
        mixed_sources=len(sources) > 1,
        proxy_use=dict(proxies),
        adjustment=coverage.adjustment,
        adjustment_quality=(
            "confirmed split-adjusted, non-dividend-adjusted closes"
            if coverage.adjustment == PORTFOLIO_ACCOUNTING_ADJUSTMENT.value
            else "unsuitable adjustment for portfolio accounting"
        ),
        valid_start_baseline=baseline_valid,
        common_date_complete=common_complete,
        suitable_for_forecasting=suitable,
        rationale=rationale,
    )


def reconstruct_from_coverage(
    portfolios: Sequence[Portfolio],
    request: ReconstructionRequest,
    coverage: HistoricalCoverageResult,
    benchmark_coverages: Mapping[str, HistoricalCoverageResult] | None = None,
) -> ReconstructionResult:
    """Pure reconstruction entry point for callers that already resolved confirmed closes."""
    portfolios_by_id = {portfolio.id: portfolio for portfolio in portfolios}
    missing_ids = [
        portfolio_id
        for portfolio_id in request.portfolio_ids
        if portfolio_id not in portfolios_by_id
    ]
    if missing_ids:
        raise ValueError(f"Unknown portfolio IDs: {', '.join(map(str, missing_ids))}.")
    portfolios = [portfolios_by_id[portfolio_id] for portfolio_id in request.portfolio_ids]
    prices = coverage.data.copy()
    prices.index = pd.to_datetime(prices.index).normalize()
    per_portfolio: dict[int, PortfolioReconstruction] = {}
    for portfolio in portfolios:
        daily = _daily_for_portfolio(portfolio, prices, request.start, request.end)
        xirr = _money_weighted_return(daily, request.start, request.end)
        valid_gain = daily.gain_loss.dropna()
        per_portfolio[portfolio.id] = PortfolioReconstruction(
            portfolio_id=portfolio.id,
            portfolio_name=portfolio.name,
            daily=daily,
            money_weighted_return=xirr,
            xirr=xirr,
            gain_loss=(
                float(valid_gain.sum())
                if not daily.calculation_warnings and not valid_gain.empty
                else None
            ),
        )
    combined = _combine_daily([item.daily for item in per_portfolio.values()], tuple(per_portfolio))
    combined_xirr = _money_weighted_return(combined, request.start, request.end)
    common, stale, missing = _latest_price_status(prices, combined.positions, request.end)
    common_complete = common is not None and not missing
    common_portfolio_values = (
        {portfolio.id: _as_of_value(portfolio, prices, common) for portfolio in portfolios}
        if common is not None
        else {portfolio.id: None for portfolio in portfolios}
    )
    common_value = (
        sum(float(value) for value in common_portfolio_values.values() if value is not None)
        if common_portfolio_values
        and all(value is not None for value in common_portfolio_values.values())
        else None
    )
    baseline_positions: dict[str, float] = {}
    for portfolio in portfolios:
        for transaction in _transactions(portfolio):
            if (
                transaction.transaction_date >= request.start
                or transaction.kind not in POSITION_KINDS
            ):
                continue
            direction = -1.0 if transaction.kind == "sell" else 1.0
            baseline_positions[transaction.symbol] = baseline_positions.get(
                transaction.symbol, 0.0
            ) + (direction * float(transaction.quantity or 0))
    baseline_symbols = {
        symbol for symbol, quantity in baseline_positions.items() if abs(quantity) > 1e-12
    }
    baseline_valid = all(
        symbol in prices
        and not prices.loc[prices.index.date < request.start, symbol].dropna().empty
        for symbol in baseline_symbols
    )
    data_coverage = _coverage_score(
        coverage,
        prices,
        request.proxy_symbols,
        baseline_valid,
        common_complete,
        stale,
        request.start,
        request.end,
    )
    warnings = list(coverage.warnings)
    warnings.extend(
        warning for item in per_portfolio.values() for warning in item.daily.calculation_warnings
    )
    if missing:
        warnings.append("Missing confirmed prices for held symbols: " + ", ".join(missing) + ".")
    if stale:
        warnings.append("Mixed price freshness affects held symbols: " + ", ".join(stale) + ".")
    if request.proxy_symbols:
        warnings.append(
            "Proxies are disclosed for data sufficiency only and are not used to value holdings."
        )
    first_valid = combined.portfolio_value.dropna()
    if not baseline_valid or first_valid.empty or first_valid.index[0].date() > request.start:
        warnings.append("A valid confirmed start baseline is unavailable.")

    benchmarks: dict[str, BenchmarkRelativeSeries] = {}
    for symbol, benchmark_coverage in (benchmark_coverages or {}).items():
        benchmark = benchmark_coverage.data.get(symbol, pd.Series(dtype=float)).dropna()
        aligned = benchmark.loc[
            (benchmark.index.date >= request.start) & (benchmark.index.date <= request.end)
        ]
        benchmark_growth = aligned / aligned.iloc[0] * 100.0 if not aligned.empty else aligned
        portfolio_growth = combined.time_weighted_return.reindex(benchmark_growth.index)
        benchmarks[symbol] = BenchmarkRelativeSeries(
            symbol=symbol,
            benchmark_growth=benchmark_growth.rename(symbol),
            relative_return=(portfolio_growth - (benchmark_growth / 100.0 - 1.0)).rename(symbol),
            coverage=benchmark_coverage,
        )

    gain = combined.gain_loss.dropna()
    assumptions = (
        "Canonical dated transactions are the accounting source of truth.",
        "Portfolio valuation uses confirmed split-adjusted, non-dividend-adjusted closes only.",
        "Explicit dividends and interest remain ledger cash and are not embedded in valuation prices.",
        "Transactions are applied before the close on their transaction date.",
        "No missing price is forward-filled or valued at zero.",
        "Daily TWR removes external transfers, cash adjustments, awards, and contributed opening assets.",
        "XIRR annualizes dated investor cash flows and the terminal confirmed portfolio value.",
    )
    return ReconstructionResult(
        request=request,
        portfolios=per_portfolio,
        combined=combined,
        money_weighted_return=combined_xirr,
        xirr=combined_xirr,
        gain_loss=(
            float(gain.sum()) if not combined.calculation_warnings and not gain.empty else None
        ),
        benchmarks=benchmarks,
        data_coverage=data_coverage,
        common_as_of_date=common,
        common_as_of_value=common_value,
        common_as_of_portfolio_values=common_portfolio_values,
        stale_symbols=stale,
        missing_symbols=missing,
        assumptions=assumptions,
        warnings=tuple(dict.fromkeys(warnings)),
        suitable_for_forecasting=data_coverage.suitable_for_forecasting,
    )


class PortfolioReconstructionService:
    """Reconstruct typed portfolio history from the ledger and durable price repository."""

    def __init__(self, session: Session, repository: HistoricalDataRepository) -> None:
        self.session = session
        self.repository = repository

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        portfolios = list(
            self.session.scalars(
                select(Portfolio)
                .where(Portfolio.id.in_(request.portfolio_ids))
                .options(selectinload(Portfolio.transactions))
                .order_by(Portfolio.id)
            )
        )
        found = {portfolio.id for portfolio in portfolios}
        missing_ids = sorted(set(request.portfolio_ids) - found)
        if missing_ids:
            raise ValueError(f"Unknown portfolio IDs: {', '.join(map(str, missing_ids))}.")
        symbols = _symbols(portfolios, request.end)
        earliest_transaction = min(
            (
                transaction.transaction_date
                for portfolio in portfolios
                for transaction in _transactions(portfolio)
                if transaction.transaction_date <= request.end
            ),
            default=request.start,
        )
        history_start = min(earliest_transaction, request.start - timedelta(days=14))
        if symbols:
            coverage = self.repository.coverage(
                HistoricalRequest(
                    symbols, history_start, request.end, PORTFOLIO_ACCOUNTING_ADJUSTMENT
                )
            )
        else:
            index = pd.bdate_range(history_start, request.end)
            coverage = HistoricalCoverageResult(
                data=pd.DataFrame(index=index),
                source="none",
                feed=None,
                adjustment=PORTFOLIO_ACCOUNTING_ADJUSTMENT.value,
                coverage_start=history_start,
                coverage_end=request.end,
                missing_symbols=(),
                missing_date_ranges={},
                warnings=(),
                freshness={},
                usable=True,
                usability="complete cash-only reconstruction",
            )
        benchmark_coverages = {
            symbol: self.repository.coverage(
                HistoricalRequest(
                    (symbol,),
                    request.start,
                    request.end,
                    BENCHMARK_TOTAL_RETURN_ADJUSTMENT,
                )
            )
            for symbol in request.benchmark_symbols
        }
        recorded_proxies = self._recorded_proxies(symbols, request.start, request.end)
        merged_request = ReconstructionRequest(
            request.portfolio_ids,
            request.start,
            request.end,
            request.benchmark_symbols,
            {**recorded_proxies, **request.proxy_symbols},
        )
        return reconstruct_from_coverage(portfolios, merged_request, coverage, benchmark_coverages)

    def _recorded_proxies(self, symbols: Sequence[str], start: date, end: date) -> dict[str, str]:
        if not symbols:
            return {}
        rows = self.session.scalars(
            select(ProxyAssignment)
            .join(ProxyAssignment.target_instrument)
            .where(
                ProxyAssignment.target_instrument.has(Instrument.canonical_symbol.in_(symbols)),
                ProxyAssignment.effective_from <= end,
                (ProxyAssignment.effective_to.is_(None) | (ProxyAssignment.effective_to >= start)),
            )
        )
        result: dict[str, str] = {}
        for assignment in rows:
            target = assignment.target_instrument.canonical_symbol
            result[target] = (
                assignment.proxy_instrument.canonical_symbol
                if assignment.proxy_instrument is not None
                else str(assignment.proxy_series)
            )
        return result
