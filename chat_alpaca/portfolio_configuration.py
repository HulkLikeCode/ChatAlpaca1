from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_alpaca.models import Portfolio, PortfolioBenchmarkComponent

ACCOUNT_TYPES = ("traditional_ira", "roth_ira", "taxable", "unknown")
ACCOUNT_TYPE_LABELS = {
    "traditional_ira": "Traditional IRA",
    "roth_ira": "Roth IRA",
    "taxable": "Taxable",
    "unknown": "Unknown",
}
REBALANCING_FREQUENCIES = ("daily", "monthly", "quarterly", "annual", "none")
BENCHMARK_WEIGHT_TOLERANCE = Decimal("0.0001")  # percentage points
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,31}$")


@dataclass(frozen=True)
class BenchmarkConfiguration:
    portfolio_id: int
    effective_from: date
    weights: Mapping[str, Decimal]
    rebalancing_frequency: str


@dataclass(frozen=True)
class PortfolioBenchmarkSeries:
    portfolio_id: int
    growth: pd.Series
    configurations: tuple[BenchmarkConfiguration, ...]
    assumption: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class HouseholdBenchmarkSeries:
    """Portfolio-specific references; intentionally not collapsed into one household blend."""

    by_portfolio: Mapping[int, PortfolioBenchmarkSeries]
    assumption: str


def set_account_type(session: Session, portfolio_id: int, account_type: str) -> Portfolio:
    normalized = account_type.strip().lower()
    if normalized not in ACCOUNT_TYPES:
        raise ValueError(f"Account type must be one of: {', '.join(ACCOUNT_TYPES)}.")
    portfolio = session.get(Portfolio, portfolio_id)
    if portfolio is None:
        raise ValueError(f"Unknown portfolio ID: {portfolio_id}.")
    portfolio.account_type = normalized
    session.flush()
    return portfolio


def _percentage(value: object) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("Benchmark weights must be numeric percentages.") from exc
    if not parsed.is_finite() or parsed <= 0 or parsed > 100:
        raise ValueError("Each benchmark weight must be greater than 0% and at most 100%.")
    return parsed


def validate_benchmark_weights(weights: Mapping[str, object]) -> dict[str, Decimal]:
    normalized: dict[str, Decimal] = {}
    for raw_symbol, raw_weight in weights.items():
        symbol = raw_symbol.strip().upper()
        if not _SYMBOL.fullmatch(symbol):
            raise ValueError(f"Invalid benchmark symbol: {raw_symbol!r}.")
        if symbol in normalized:
            raise ValueError(f"Duplicate benchmark symbol: {symbol}.")
        normalized[symbol] = _percentage(raw_weight)
    if not normalized:
        raise ValueError("A benchmark blend requires at least one component.")
    total = sum(normalized.values(), Decimal("0"))
    if abs(total - Decimal("100")) > BENCHMARK_WEIGHT_TOLERANCE:
        raise ValueError(f"Benchmark weights must sum to 100%; received {total}%.")
    return normalized


def parse_benchmark_components(value: str) -> dict[str, Decimal]:
    """Parse an owner-facing ``SYMBOL:percentage`` comma-separated blend."""
    parsed: dict[str, Decimal] = {}
    for component in value.split(","):
        if not component.strip():
            continue
        symbol, separator, weight = component.partition(":")
        if not separator:
            raise ValueError("Use SYMBOL:weight entries separated by commas.")
        canonical = symbol.strip().upper()
        if canonical in parsed:
            raise ValueError(f"Duplicate benchmark symbol: {canonical}.")
        try:
            parsed[canonical] = Decimal(weight.strip())
        except InvalidOperation as exc:
            raise ValueError(f"Invalid benchmark weight for {canonical}.") from exc
    return validate_benchmark_weights(parsed)


def save_benchmark_configuration(
    session: Session,
    portfolio_id: int,
    effective_from: date,
    weights: Mapping[str, object],
    *,
    rebalancing_frequency: str = "monthly",
) -> BenchmarkConfiguration:
    if session.get(Portfolio, portfolio_id) is None:
        raise ValueError(f"Unknown portfolio ID: {portfolio_id}.")
    frequency = rebalancing_frequency.strip().lower()
    if frequency not in REBALANCING_FREQUENCIES:
        raise ValueError(
            "Rebalancing frequency must be one of: " + ", ".join(REBALANCING_FREQUENCIES) + "."
        )
    existing = session.scalar(
        select(PortfolioBenchmarkComponent.id).where(
            PortfolioBenchmarkComponent.portfolio_id == portfolio_id,
            PortfolioBenchmarkComponent.effective_from == effective_from,
        )
    )
    if existing is not None:
        raise ValueError(
            "A benchmark configuration already exists on this effective date; "
            "add a later effective date so prior history is not rewritten."
        )
    percentages = validate_benchmark_weights(weights)
    for symbol, percentage in percentages.items():
        session.add(
            PortfolioBenchmarkComponent(
                portfolio_id=portfolio_id,
                effective_from=effective_from,
                symbol=symbol,
                weight=percentage / Decimal("100"),
                rebalancing_frequency=frequency,
            )
        )
    session.flush()
    return BenchmarkConfiguration(
        portfolio_id,
        effective_from,
        {symbol: value / Decimal("100") for symbol, value in percentages.items()},
        frequency,
    )


def benchmark_configurations(
    session: Session, portfolio_id: int
) -> tuple[BenchmarkConfiguration, ...]:
    rows = list(
        session.scalars(
            select(PortfolioBenchmarkComponent)
            .where(PortfolioBenchmarkComponent.portfolio_id == portfolio_id)
            .order_by(
                PortfolioBenchmarkComponent.effective_from,
                PortfolioBenchmarkComponent.symbol,
            )
        )
    )
    grouped: dict[date, list[PortfolioBenchmarkComponent]] = {}
    for row in rows:
        grouped.setdefault(row.effective_from, []).append(row)
    return tuple(
        BenchmarkConfiguration(
            portfolio_id,
            effective,
            {row.symbol: Decimal(row.weight) for row in components},
            components[0].rebalancing_frequency,
        )
        for effective, components in grouped.items()
    )


def _rebalance_due(previous: pd.Timestamp, current: pd.Timestamp, frequency: str) -> bool:
    if frequency == "daily":
        return True
    if frequency == "monthly":
        return (previous.year, previous.month) != (current.year, current.month)
    if frequency == "quarterly":
        return (previous.year, (previous.month - 1) // 3) != (
            current.year,
            (current.month - 1) // 3,
        )
    if frequency == "annual":
        return previous.year != current.year
    return False


def reconstruct_benchmark_series(
    portfolio_id: int,
    configurations: Sequence[BenchmarkConfiguration],
    total_return_closes: pd.DataFrame,
    start: date,
    end: date,
) -> PortfolioBenchmarkSeries:
    configs = tuple(sorted(configurations, key=lambda item: item.effective_from))
    assumption = (
        "Component total returns are combined at target weights and rebalanced at each "
        "configuration's stated frequency; a new effective date changes only that date forward."
    )
    if start > end:
        raise ValueError("Benchmark start date must be on or before the end date.")
    frame = total_return_closes.sort_index().loc[
        lambda item: (item.index.date >= start) & (item.index.date <= end)
    ]
    growth = pd.Series(index=frame.index, dtype=float, name=f"Portfolio {portfolio_id} benchmark")
    warnings: list[str] = []
    if not configs or frame.empty:
        return PortfolioBenchmarkSeries(portfolio_id, growth, configs, assumption, ())

    returns = total_return_closes.sort_index().pct_change(fill_method=None)
    nav = 100.0
    sleeves: dict[str, float] = {}
    active: BenchmarkConfiguration | None = None
    previous_date: pd.Timestamp | None = None
    initialized = False
    for timestamp in frame.index:
        eligible = [item for item in configs if item.effective_from <= timestamp.date()]
        configuration = eligible[-1] if eligible else None
        if configuration is None:
            continue
        symbols = tuple(configuration.weights)
        missing_columns = [symbol for symbol in symbols if symbol not in returns.columns]
        daily = returns.loc[timestamp, list(symbols)] if not missing_columns else None
        if missing_columns or daily is None or daily.isna().any():
            warnings.append(
                f"Incomplete benchmark component returns on {timestamp.date().isoformat()}."
            )
            if not initialized and all(symbol in frame.columns for symbol in symbols):
                available = frame.loc[timestamp, list(symbols)]
                if not available.isna().any():
                    sleeves = {
                        symbol: nav * float(weight)
                        for symbol, weight in configuration.weights.items()
                    }
                    active = configuration
                    initialized = True
                    growth.loc[timestamp] = nav
            previous_date = timestamp
            continue
        changed = active is None or active.effective_from != configuration.effective_from
        scheduled = (
            active is not None
            and previous_date is not None
            and _rebalance_due(previous_date, timestamp, active.rebalancing_frequency)
        )
        if changed or scheduled:
            sleeves = {
                symbol: nav * float(weight) for symbol, weight in configuration.weights.items()
            }
            active = configuration
        sleeves = {symbol: sleeves[symbol] * (1.0 + float(daily[symbol])) for symbol in symbols}
        nav = sum(sleeves.values())
        growth.loc[timestamp] = nav
        initialized = True
        previous_date = timestamp
    return PortfolioBenchmarkSeries(
        portfolio_id, growth, configs, assumption, tuple(dict.fromkeys(warnings))
    )


def household_benchmark_series(
    configurations: Mapping[int, Sequence[BenchmarkConfiguration]],
    total_return_closes: pd.DataFrame,
    start: date,
    end: date,
) -> HouseholdBenchmarkSeries:
    by_portfolio = {
        portfolio_id: reconstruct_benchmark_series(
            portfolio_id, portfolio_configs, total_return_closes, start, end
        )
        for portfolio_id, portfolio_configs in configurations.items()
    }
    return HouseholdBenchmarkSeries(
        by_portfolio,
        "Household reporting preserves each portfolio's benchmark; it does not substitute a "
        "single household benchmark.",
    )
