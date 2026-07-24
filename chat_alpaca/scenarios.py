from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from numbers import Real
from typing import Protocol

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_alpaca.models import (
    ForecastRun,
    ForecastRunDataset,
    ModelValidation,
    Portfolio,
    PortfolioTransaction,
)
from chat_alpaca.presentation import assumption_comparisons, format_assumption_value

DETERMINISTIC_MODEL_TYPE = "deterministic_scenario"
DETERMINISTIC_MODEL_VERSION = "1.1.0"
VALIDATION_STATUSES = {"unvalidated", "in_review", "validated", "rejected"}

SHOCK_DISCLOSURE = (
    "Shock impacts are first-order exposure estimates; they do not model cross-security "
    "correlation, liquidity, or market impact."
)
DIVIDEND_DISCLOSURE = (
    "The dividend scenario holds positive trailing-365-day dividend income constant for every "
    "forecast year, applies no dividend growth, and does not model reinvestment or opportunity "
    "cost."
)


class ScenarioType(str, Enum):
    BROAD_MARKET_DECLINE = "broad_market_decline"
    HOLDING_DECLINE = "holding_decline"
    SECTOR_DECLINE = "sector_decline"
    DIVIDEND_REDUCTION = "dividend_reduction"
    CONTRIBUTION_INTERRUPTION = "contribution_interruption"
    INFLATION_INCREASE = "inflation_increase"
    LOW_RETURN_PERIOD = "low_return_period"
    LOST_DECADE = "lost_decade"
    RETIREMENT_DATE_DECLINE = "retirement_date_decline"
    HISTORICAL_REPLAY = "historical_replay"


@dataclass(frozen=True)
class DatasetReference:
    dataset_id: int
    purpose: str = "scenario_input"


@dataclass(frozen=True)
class ScenarioAssumptions:
    scenario_type: ScenarioType | str
    market_decline: float = -0.20
    holding_symbol: str | None = None
    holding_decline: float = -0.30
    sector: str | None = None
    sector_decline: float = -0.25
    dividend_reduction: float = 0.50
    contribution_amount: float = 0.0
    interruption_months: int = 12
    inflation: float = 0.025
    inflation_increase: float = 0.02
    spending: float = 0.0
    expected_return: float = 0.07
    low_return: float = 0.01
    horizon_years: int = 10
    retirement_date: date | None = None
    historical_start: date | None = None
    historical_end: date | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_type", ScenarioType(self.scenario_type))
        if self.horizon_years < 1 or self.horizon_years > 40:
            raise ValueError("Scenario horizon must be between 1 and 40 years.")
        if self.contribution_amount < 0 or self.spending < 0:
            raise ValueError("Contributions and spending cannot be negative.")
        if self.interruption_months < 0:
            raise ValueError("Contribution interruption cannot be negative.")
        for name in ("market_decline", "holding_decline", "sector_decline"):
            value = getattr(self, name)
            if value < -1 or value > 0:
                raise ValueError(f"{name.replace('_', ' ').title()} must be between -100% and 0%.")
        if not 0 <= self.dividend_reduction <= 1:
            raise ValueError("Dividend reduction must be between 0% and 100%.")
        if self.inflation <= -1 or self.inflation + self.inflation_increase <= -1:
            raise ValueError("Inflation assumptions must be greater than -100%.")


@dataclass(frozen=True)
class ScenarioResult:
    scenario_type: str
    model_type: str
    model_version: str
    baseline_value: float
    scenario_value: float
    total_household_impact: float
    impact_by_portfolio: Mapping[str, float]
    impact_by_holding: Mapping[str, float]
    impact_by_sector: Mapping[str, float]
    account_type_effects: Mapping[str, float]
    largest_loss_contributors: tuple[tuple[str, float], ...]
    assumptions: Mapping[str, object]
    coverage: Mapping[str, object]
    warnings: tuple[str, ...]
    comparison_with_baseline: Mapping[str, float]
    scenario_bands: Mapping[str, float] | None = None

    def summary(self) -> dict[str, object]:
        return {
            "baseline_value": self.baseline_value,
            "scenario_value": self.scenario_value,
            "total_household_impact": self.total_household_impact,
            "impact_by_portfolio": dict(self.impact_by_portfolio),
            "impact_by_holding": dict(self.impact_by_holding),
            "impact_by_sector": dict(self.impact_by_sector),
            "account_type_effects": dict(self.account_type_effects),
            "largest_loss_contributors": list(self.largest_loss_contributors),
            "comparison_with_baseline": dict(self.comparison_with_baseline),
            "warnings": list(self.warnings),
        }


def scenario_explanation(assumptions: ScenarioAssumptions) -> str:
    """Describe the selected deterministic branch directly from structured inputs."""
    current = asdict(assumptions)
    defaults = asdict(ScenarioAssumptions(assumptions.scenario_type))
    comparisons = assumption_comparisons(current, defaults)
    changed = [
        (
            f"{item.assumption} {format_assumption_value(item.default_value, item.unit)} → "
            f"{format_assumption_value(item.value, item.unit)}"
        )
        for item in comparisons
        if item.value != item.default_value
    ]
    unchanged = [item.assumption for item in comparisons if item.value == item.default_value]
    scenario_type = ScenarioType(assumptions.scenario_type)
    calculation = {
        ScenarioType.BROAD_MARKET_DECLINE: (
            "Each holding value receives the broad-market shock; cash is unchanged."
        ),
        ScenarioType.HOLDING_DECLINE: (
            f"Only {assumptions.holding_symbol or 'the selected holding'} receives the "
            "holding shock; cash and other holdings are unchanged."
        ),
        ScenarioType.SECTOR_DECLINE: (
            f"The {assumptions.sector or 'selected'} sector shock is weighted through each "
            "holding's disclosed sector exposure."
        ),
        ScenarioType.DIVIDEND_REDUCTION: (
            "Positive trailing-365-day ledger dividends are reduced for each portfolio and "
            "extended across the scenario horizon."
        ),
        ScenarioType.CONTRIBUTION_INTERRUPTION: (
            "Monthly compounding omits contributions during the interruption, then restores "
            "the configured contribution."
        ),
        ScenarioType.INFLATION_INCREASE: (
            "The same nominal compounded value is deflated by baseline inflation and by "
            "baseline plus additional inflation for comparison."
        ),
        ScenarioType.LOW_RETURN_PERIOD: (
            "Starting value, contributions, and spending compound at the selected low return."
        ),
        ScenarioType.LOST_DECADE: (
            "Starting value, contributions, and spending compound at the selected low return "
            "for the fixed ten-year lost-decade horizon."
        ),
        ScenarioType.RETIREMENT_DATE_DECLINE: (
            "Monthly compounding applies the market shock at the resolved retirement month."
        ),
        ScenarioType.HISTORICAL_REPLAY: (
            "Each holding uses its jointly complete historical endpoint return with no "
            "forward-filling."
        ),
    }[scenario_type]
    return (
        f"Adjusted inputs: {', '.join(changed) if changed else 'none'}. "
        f"{calculation} Unchanged defaults: {', '.join(unchanged) if unchanged else 'none'}. "
        "Impacts are allocated to household, portfolio, holding, sector, and account-type "
        "outputs using the selected ledger scope and disclosed classifications."
    )


class ModelValidator(Protocol):
    """Interface shared by deterministic and future stochastic validators."""

    model_type: str
    model_version: str

    def evidence(self) -> Sequence[str]: ...

    def limitations(self) -> Sequence[str]: ...


def _valid_price(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        return None
    price = float(value)
    return price if math.isfinite(price) and price > 0 else None


def _date_label(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _resolved_prices(
    closes: pd.DataFrame | Mapping[str, float], required_symbols: Sequence[str]
) -> tuple[dict[str, float], dict[str, object]]:
    required = tuple(sorted(set(required_symbols)))
    if isinstance(closes, pd.DataFrame):
        if not required:
            return {}, {
                "common_valuation_date": None,
                "price_source_dates": {},
                "price_observation_counts": {},
                "jointly_complete_price_observations": 0,
            }
        normalized = closes.copy()
        normalized.columns = [str(symbol).upper() for symbol in normalized.columns]
        if normalized.columns.duplicated().any():
            raise ValueError("Scenario price columns must identify each symbol only once.")
        missing = [symbol for symbol in required if symbol not in normalized]
        if missing:
            raise ValueError(
                "Scenario refused because current market data is missing for: "
                + ", ".join(missing)
                + "."
            )
        try:
            normalized.index = pd.to_datetime(normalized.index)
        except (TypeError, ValueError) as exc:
            raise ValueError("Scenario DataFrame prices require a date-like index.") from exc
        if normalized.index.hasnans:
            raise ValueError("Scenario DataFrame prices require a date for every observation.")
        normalized = normalized.sort_index()
        usable = normalized.loc[:, list(required)].map(_valid_price)
        latest_dates: dict[str, pd.Timestamp] = {}
        for symbol in required:
            available = usable[symbol].dropna()
            if available.empty:
                raise ValueError(
                    f"Scenario refused because current market data is missing for: {symbol}."
                )
            latest_dates[symbol] = pd.Timestamp(available.index[-1])
        common_date = min(latest_dates.values())
        resolved: dict[str, float] = {}
        source_dates: dict[str, str] = {}
        observation_counts: dict[str, int] = {}
        through_common = usable.loc[usable.index <= common_date]
        for symbol in required:
            available = through_common[symbol].dropna()
            if available.empty:
                raise ValueError(
                    f"Scenario refused because current market data is missing for: {symbol}."
                )
            resolved[symbol] = float(available.iloc[-1])
            source_dates[symbol] = _date_label(available.index[-1])
            observation_counts[symbol] = int(len(available))
        complete = int(through_common.dropna(how="any").shape[0])
        return resolved, {
            "common_valuation_date": _date_label(common_date),
            "price_source_dates": source_dates,
            "price_observation_counts": observation_counts,
            "jointly_complete_price_observations": complete,
        }

    resolved = {}
    for raw_symbol, raw_value in closes.items():
        symbol = str(raw_symbol).upper()
        price = _valid_price(raw_value)
        if price is None:
            raise ValueError(
                f"Scenario price for {symbol} must be numeric, finite, positive, and not boolean."
            )
        resolved[symbol] = price
    missing = [symbol for symbol in required if symbol not in resolved]
    if missing:
        raise ValueError(
            "Scenario refused because current market data is missing for: "
            + ", ".join(missing)
            + "."
        )
    return resolved, {
        "common_valuation_date": None,
        "price_source_dates": {},
        "price_observation_counts": {},
        "jointly_complete_price_observations": None,
    }


def _sector_weights(
    symbol: str, sectors: Mapping[str, str | Mapping[str, float]]
) -> dict[str, float]:
    classification = sectors.get(symbol, "Unclassified")
    if isinstance(classification, str):
        return {classification: 1.0}
    weights = {name: float(weight) for name, weight in classification.items() if float(weight) > 0}
    total = sum(weights.values())
    if total > 1.000001:
        weights = {name: weight / 100 for name, weight in weights.items()}
        total = sum(weights.values())
    if total > 1.000001:
        raise ValueError(f"Sector weights for {symbol} exceed 100%.")
    if total < 0.999999:
        weights["Unclassified"] = weights.get("Unclassified", 0) + (1 - total)
    return weights


def _compound(
    starting_value: float,
    annual_return: float,
    monthly_contribution: float,
    annual_spending: float,
    years: int,
) -> float:
    value = starting_value
    monthly_return = (1 + annual_return) ** (1 / 12) - 1
    for _ in range(years * 12):
        value = value * (1 + monthly_return) + monthly_contribution - annual_spending / 12
    return value


def run_deterministic_scenario(
    portfolios: Sequence[Portfolio],
    closes: pd.DataFrame | Mapping[str, float],
    assumptions: ScenarioAssumptions,
    *,
    sectors: Mapping[str, str | Mapping[str, float]] | None = None,
    historical_prices: pd.DataFrame | None = None,
    dataset_references: Sequence[DatasetReference] = (),
    proxy_use: Mapping[str, str] | None = None,
    as_of: date | None = None,
) -> ScenarioResult:
    """Run one deterministic scenario without randomness or raw path generation."""
    if not portfolios:
        raise ValueError("A deterministic scenario requires at least one portfolio.")
    required_symbols = sorted(
        {
            lot.symbol.upper()
            for portfolio in portfolios
            for lot in portfolio.holdings
            if float(lot.shares) != 0
        }
    )
    prices, price_coverage = _resolved_prices(closes, required_symbols)
    sectors = sectors or {}
    proxy_use = proxy_use or {}
    holdings: list[dict[str, object]] = []
    missing: set[str] = set()
    cash_by_portfolio: dict[str, float] = {}
    for portfolio in portfolios:
        cash_by_portfolio[portfolio.name] = float(portfolio.cash)
        for lot in portfolio.holdings:
            if float(lot.shares) == 0:
                continue
            symbol = lot.symbol.upper()
            if symbol not in prices:
                missing.add(symbol)
                continue
            holdings.append(
                {
                    "portfolio": portfolio.name,
                    "account_type": portfolio.account_type,
                    "symbol": symbol,
                    "value": float(lot.shares) * prices[symbol],
                    "sectors": _sector_weights(symbol, sectors),
                }
            )
    if missing:
        raise ValueError(
            "Scenario refused because current market data is missing for: "
            + ", ".join(sorted(missing))
            + "."
        )
    starting_value = sum(cash_by_portfolio.values()) + sum(float(row["value"]) for row in holdings)
    if starting_value <= 0:
        raise ValueError("A deterministic scenario requires a positive household value.")

    scenario_type = ScenarioType(assumptions.scenario_type)
    years = 10 if scenario_type == ScenarioType.LOST_DECADE else assumptions.horizon_years
    baseline = _compound(
        starting_value,
        assumptions.expected_return,
        assumptions.contribution_amount,
        assumptions.spending,
        years,
    )
    impacts = [0.0 for _ in holdings]
    direct_portfolio_impacts: dict[str, float] | None = None
    warnings: list[str] = []
    if price_coverage["common_valuation_date"] is not None:
        warnings.append(f"Scenario valuation date: {price_coverage['common_valuation_date']}.")
    scenario_value = baseline

    if scenario_type == ScenarioType.BROAD_MARKET_DECLINE:
        warnings.append(SHOCK_DISCLOSURE)
        impacts = [float(row["value"]) * assumptions.market_decline for row in holdings]
        baseline = starting_value
        scenario_value = baseline + sum(impacts)
    elif scenario_type == ScenarioType.HOLDING_DECLINE:
        warnings.append(SHOCK_DISCLOSURE)
        target = (assumptions.holding_symbol or "").strip().upper()
        if not target:
            raise ValueError("A holding-specific decline requires a holding symbol.")
        if target not in {str(row["symbol"]) for row in holdings}:
            raise ValueError(f"Holding {target} is not in the selected portfolio scope.")
        impacts = [
            float(row["value"]) * assumptions.holding_decline if row["symbol"] == target else 0.0
            for row in holdings
        ]
        baseline = starting_value
        scenario_value = baseline + sum(impacts)
    elif scenario_type == ScenarioType.SECTOR_DECLINE:
        warnings.append(SHOCK_DISCLOSURE)
        target_sector = (assumptions.sector or "").strip()
        if not target_sector:
            raise ValueError("A sector decline requires a sector.")
        covered = sum(
            float(row["value"]) * dict(row["sectors"]).get(target_sector, 0) for row in holdings
        )
        if covered == 0:
            raise ValueError(f"No selected holding has disclosed {target_sector} exposure.")
        impacts = [
            float(row["value"])
            * dict(row["sectors"]).get(target_sector, 0)
            * assumptions.sector_decline
            for row in holdings
        ]
        baseline = starting_value
        scenario_value = baseline + sum(impacts)
    elif scenario_type == ScenarioType.DIVIDEND_REDUCTION:
        warnings.append(DIVIDEND_DISCLOSURE)
        dividend_start = (as_of or date.today()) - timedelta(days=365)
        dividend_by_portfolio: dict[str, float] = {}
        for portfolio in portfolios:
            dividend_by_portfolio[portfolio.name] = sum(
                float(transaction.cash_delta)
                for transaction in portfolio.transactions
                if transaction.kind == "dividend"
                and transaction.cash_delta > 0
                and transaction.transaction_date >= dividend_start
            )
        direct_portfolio_impacts = {
            name: -income * assumptions.dividend_reduction * years
            for name, income in dividend_by_portfolio.items()
        }
        dividend_loss = sum(direct_portfolio_impacts.values())
        if dividend_loss == 0:
            warnings.append(
                "No positive trailing-365-day dividend transactions were available in the "
                "ledger scope."
            )
        scenario_value = baseline + dividend_loss
        invested_by_portfolio = {
            portfolio.name: sum(
                float(row["value"]) for row in holdings if row["portfolio"] == portfolio.name
            )
            for portfolio in portfolios
        }
        impacts = [
            direct_portfolio_impacts[str(row["portfolio"])]
            * float(row["value"])
            / invested_by_portfolio[str(row["portfolio"])]
            if invested_by_portfolio[str(row["portfolio"])]
            else 0.0
            for row in holdings
        ]
    elif scenario_type == ScenarioType.CONTRIBUTION_INTERRUPTION:
        scenario_value = starting_value
        monthly_return = (1 + assumptions.expected_return) ** (1 / 12) - 1
        for month in range(years * 12):
            contribution = (
                0.0 if month < assumptions.interruption_months else assumptions.contribution_amount
            )
            scenario_value = (
                scenario_value * (1 + monthly_return) + contribution - assumptions.spending / 12
            )
    elif scenario_type == ScenarioType.INFLATION_INCREASE:
        baseline = baseline / (1 + assumptions.inflation) ** years
        scenario_value = (
            scenario_value / (1 + assumptions.inflation + assumptions.inflation_increase) ** years
        )
    elif scenario_type in {ScenarioType.LOW_RETURN_PERIOD, ScenarioType.LOST_DECADE}:
        scenario_value = _compound(
            starting_value,
            assumptions.low_return,
            assumptions.contribution_amount,
            assumptions.spending,
            years,
        )
    elif scenario_type == ScenarioType.RETIREMENT_DATE_DECLINE:
        retirement_date = assumptions.retirement_date
        if retirement_date is None:
            raise ValueError("A retirement-date decline requires a retirement date.")
        today = as_of or date.today()
        months_until = (
            (retirement_date.year - today.year) * 12 + retirement_date.month - today.month
        )
        if months_until < 0 or months_until > years * 12:
            raise ValueError("Retirement date must fall within the scenario horizon.")
        scenario_value = starting_value
        monthly_return = (1 + assumptions.expected_return) ** (1 / 12) - 1
        if months_until == 0:
            scenario_value *= 1 + assumptions.market_decline
        for month in range(1, years * 12 + 1):
            scenario_value = (
                scenario_value * (1 + monthly_return)
                + assumptions.contribution_amount
                - assumptions.spending / 12
            )
            if month == months_until:
                scenario_value *= 1 + assumptions.market_decline
    else:
        if historical_prices is None or historical_prices.empty:
            raise ValueError("Historical replay requires an available prior-period price dataset.")
        if not dataset_references:
            raise ValueError("Historical replay requires persisted market-dataset references.")
        window = historical_prices.copy()
        if assumptions.historical_start:
            window = window.loc[window.index >= pd.Timestamp(assumptions.historical_start)]
        if assumptions.historical_end:
            window = window.loc[window.index <= pd.Timestamp(assumptions.historical_end)]
        required = sorted({str(row["symbol"]) for row in holdings})
        unavailable = [symbol for symbol in required if symbol not in window]
        if unavailable:
            raise ValueError(
                "Historical replay refused because prior-period data is missing for: "
                + ", ".join(unavailable)
                + "."
            )
        joint = window.loc[:, required].map(_valid_price).dropna(how="any")
        if len(joint) < 2:
            raise ValueError(
                "Historical replay requires at least two jointly complete observations for every "
                "required symbol."
            )
        replay_start = _date_label(joint.index[0])
        replay_end = _date_label(joint.index[-1])
        jointly_complete_observations = int(len(joint))
        warnings.extend(
            [
                f"Historical replay endpoints: {replay_start} through {replay_end}.",
                f"Historical replay observations: {jointly_complete_observations}.",
            ]
        )
        returns = {
            symbol: float(joint[symbol].iloc[-1] / joint[symbol].iloc[0] - 1) for symbol in required
        }
        impacts = [float(row["value"]) * returns[str(row["symbol"])] for row in holdings]
        baseline = starting_value
        scenario_value = baseline + sum(impacts)

    total_impact = scenario_value - baseline
    if scenario_type not in {
        ScenarioType.BROAD_MARKET_DECLINE,
        ScenarioType.HOLDING_DECLINE,
        ScenarioType.SECTOR_DECLINE,
        ScenarioType.DIVIDEND_REDUCTION,
        ScenarioType.HISTORICAL_REPLAY,
    }:
        invested = sum(float(row["value"]) for row in holdings)
        impacts = [
            total_impact * float(row["value"]) / invested if invested else 0.0 for row in holdings
        ]

    by_portfolio = {portfolio.name: 0.0 for portfolio in portfolios}
    by_holding: dict[str, float] = {}
    by_sector: dict[str, float] = {}
    by_account = {portfolio.account_type: 0.0 for portfolio in portfolios}
    for row, impact in zip(holdings, impacts, strict=True):
        portfolio_name = str(row["portfolio"])
        symbol = str(row["symbol"])
        account_type = str(row["account_type"])
        by_portfolio[portfolio_name] += impact
        by_holding[symbol] = by_holding.get(symbol, 0.0) + impact
        by_account[account_type] += impact
        for sector_name, weight in dict(row["sectors"]).items():
            by_sector[sector_name] = by_sector.get(sector_name, 0.0) + impact * weight
    if direct_portfolio_impacts is not None:
        by_portfolio = direct_portfolio_impacts
        by_account = {portfolio.account_type: 0.0 for portfolio in portfolios}
        for portfolio in portfolios:
            by_account[portfolio.account_type] += direct_portfolio_impacts[portfolio.name]
    allocated = sum(impacts)
    unallocated = total_impact - allocated
    if abs(unallocated) > 0.005:
        portfolio_values = {
            portfolio.name: float(portfolio.cash)
            + sum(float(row["value"]) for row in holdings if row["portfolio"] == portfolio.name)
            for portfolio in portfolios
        }
        if direct_portfolio_impacts is None:
            for portfolio in portfolios:
                share = portfolio_values[portfolio.name] / starting_value
                by_portfolio[portfolio.name] += unallocated * share
                by_account[portfolio.account_type] += unallocated * share
        by_holding["Cash flows / purchasing power"] = unallocated
        by_sector["Cash flows / purchasing power"] = unallocated

    proxy_warnings = [
        f"{symbol} uses proxy {proxy}." for symbol, proxy in sorted(proxy_use.items())
    ]
    warnings.extend(proxy_warnings)
    coverage = {
        "priced_holdings": len(holdings),
        "required_symbols": sorted({str(row["symbol"]) for row in holdings}),
        "missing_symbols": [],
        "dataset_ids": [reference.dataset_id for reference in dataset_references],
        "proxy_symbols": sorted(proxy_use),
        "proxy_use": dict(sorted(proxy_use.items())),
        **price_coverage,
        "replay_start_date": replay_start
        if scenario_type == ScenarioType.HISTORICAL_REPLAY
        else None,
        "replay_end_date": replay_end if scenario_type == ScenarioType.HISTORICAL_REPLAY else None,
        "jointly_complete_replay_observations": jointly_complete_observations
        if scenario_type == ScenarioType.HISTORICAL_REPLAY
        else None,
    }
    assumption_payload = asdict(assumptions)
    assumption_payload["scenario_type"] = scenario_type.value
    assumption_payload = {
        key: value.isoformat() if isinstance(value, date) else value
        for key, value in assumption_payload.items()
    }
    largest = tuple(sorted(by_holding.items(), key=lambda item: item[1])[:5])
    return ScenarioResult(
        scenario_type.value,
        DETERMINISTIC_MODEL_TYPE,
        DETERMINISTIC_MODEL_VERSION,
        baseline,
        scenario_value,
        total_impact,
        by_portfolio,
        by_holding,
        by_sector,
        by_account,
        largest,
        assumption_payload,
        coverage,
        tuple(warnings),
        {
            "absolute_impact": total_impact,
            "percentage_impact": total_impact / baseline if baseline else 0.0,
        },
        {"baseline": baseline, "scenario": scenario_value},
    )


def sensitivity_grid(
    portfolios: Sequence[Portfolio],
    closes: pd.DataFrame | Mapping[str, float],
    assumptions: ScenarioAssumptions,
    variations: Mapping[str, Sequence[object]],
    **scenario_inputs: object,
) -> pd.DataFrame:
    """Evaluate a Cartesian grid of selected assumptions with stable row ordering."""
    allowed = {
        "market_decline",
        "inflation",
        "contribution_amount",
        "spending",
        "retirement_date",
        "expected_return",
        "low_return",
    }
    unknown = set(variations) - allowed
    if unknown:
        raise ValueError("Unsupported sensitivity assumptions: " + ", ".join(sorted(unknown)))
    rows: list[dict[str, object]] = [{}]
    for name, values in variations.items():
        rows = [{**row, name: value} for row in rows for value in values]
    outputs = []
    for values in rows:
        replacements = dict(values)
        if "expected_return" in replacements and assumptions.scenario_type in {
            ScenarioType.LOW_RETURN_PERIOD,
            ScenarioType.LOST_DECADE,
        }:
            replacements["low_return"] = replacements.pop("expected_return")
        if "market_decline" in replacements:
            if assumptions.scenario_type == ScenarioType.HOLDING_DECLINE:
                replacements["holding_decline"] = replacements["market_decline"]
            elif assumptions.scenario_type == ScenarioType.SECTOR_DECLINE:
                replacements["sector_decline"] = replacements["market_decline"]
            elif assumptions.scenario_type == ScenarioType.DIVIDEND_REDUCTION:
                replacements["dividend_reduction"] = abs(replacements["market_decline"])
        result = run_deterministic_scenario(
            portfolios, closes, replace(assumptions, **replacements), **scenario_inputs
        )
        outputs.append(
            {
                **values,
                "baseline_value": result.baseline_value,
                "scenario_value": result.scenario_value,
                "household_impact": result.total_household_impact,
                "impact_percent": result.comparison_with_baseline["percentage_impact"],
            }
        )
    return pd.DataFrame(outputs)


def ledger_state_hash(session: Session, portfolio_ids: Sequence[int]) -> str:
    """Hash canonical ledger rows for the exact ordered portfolio scope."""
    scope = sorted(set(portfolio_ids))
    rows = session.scalars(
        select(PortfolioTransaction)
        .where(PortfolioTransaction.portfolio_id.in_(scope))
        .order_by(
            PortfolioTransaction.portfolio_id,
            PortfolioTransaction.transaction_date,
            PortfolioTransaction.id,
        )
    )
    payload = {
        "portfolio_ids": scope,
        "transactions": [
            {
                "id": row.id,
                "portfolio_id": row.portfolio_id,
                "date": row.transaction_date.isoformat(),
                "kind": row.kind,
                "symbol": row.symbol,
                "quantity": str(row.quantity) if row.quantity is not None else None,
                "price": str(row.price) if row.price is not None else None,
                "fees": str(row.fees) if row.fees is not None else None,
                "cash_delta": str(row.cash_delta),
                "source": row.source,
                "fingerprint": row.fingerprint,
            }
            for row in rows
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def save_scenario_run(
    session: Session,
    portfolios: Sequence[Portfolio],
    result: ScenarioResult,
    *,
    dataset_references: Sequence[DatasetReference] = (),
    ledger_hash: str | None = None,
    status: str = "completed",
) -> ForecastRun:
    """Persist reproducibility inputs and summary outputs, never raw path data."""
    if status not in {"pending", "completed", "failed"}:
        raise ValueError("Unknown forecast run status.")
    ids = [portfolio.id for portfolio in portfolios]
    validation = session.scalar(
        select(ModelValidation).where(
            ModelValidation.model_type == result.model_type,
            ModelValidation.model_version == result.model_version,
        )
    )
    run = ForecastRun(
        model_type=result.model_type,
        model_version=result.model_version,
        portfolio_scope=json.dumps(
            [{"id": portfolio.id, "name": portfolio.name} for portfolio in portfolios],
            sort_keys=True,
        ),
        ledger_state_hash=ledger_hash or ledger_state_hash(session, ids),
        assumptions=json.dumps(dict(result.assumptions), sort_keys=True),
        data_coverage=json.dumps(dict(result.coverage), sort_keys=True),
        proxy_use=json.dumps(result.coverage.get("proxy_use", {}), sort_keys=True),
        status=status,
        validation_status=validation.status if validation else "unvalidated",
        summary_outputs=json.dumps(result.summary(), sort_keys=True),
        scenario_bands=(
            json.dumps(dict(result.scenario_bands), sort_keys=True)
            if result.scenario_bands is not None
            else None
        ),
    )
    session.add(run)
    session.flush()
    for reference in dataset_references:
        session.add(
            ForecastRunDataset(
                forecast_run_id=run.id,
                dataset_id=reference.dataset_id,
                purpose=reference.purpose,
            )
        )
    session.flush()
    return run


def record_validation_evidence(
    session: Session,
    model_type: str,
    model_version: str,
    *,
    automated_tests_passed: bool,
    evidence: Sequence[str] = (),
    limitations: Sequence[str] = (),
) -> ModelValidation:
    """Record evidence without treating test success as model validation."""
    row = session.scalar(
        select(ModelValidation).where(
            ModelValidation.model_type == model_type,
            ModelValidation.model_version == model_version,
        )
    )
    if row is None:
        row = ModelValidation(
            model_type=model_type, model_version=model_version, status="unvalidated"
        )
        session.add(row)
    row.automated_tests_passed = automated_tests_passed
    row.evidence = json.dumps(list(evidence), sort_keys=True)
    row.limitations = json.dumps(list(limitations), sort_keys=True)
    if row.status in {None, "unvalidated"} and (evidence or automated_tests_passed):
        row.status = "in_review"
    session.flush()
    return row


def set_validation_status(
    session: Session,
    model_type: str,
    model_version: str,
    status: str,
    *,
    reviewer: str,
) -> ModelValidation:
    """Apply an explicit human governance decision to a model version."""
    if status not in VALIDATION_STATUSES:
        raise ValueError("Unknown validation status.")
    if status == "validated" and not reviewer.strip():
        raise ValueError("Validated status requires an identified reviewer.")
    row = session.scalar(
        select(ModelValidation).where(
            ModelValidation.model_type == model_type,
            ModelValidation.model_version == model_version,
        )
    )
    if row is None:
        row = ModelValidation(
            model_type=model_type, model_version=model_version, status="unvalidated"
        )
        session.add(row)
    row.status = status
    row.reviewer = reviewer.strip() or None
    row.reviewed_at = datetime.now(timezone.utc)
    session.flush()
    return row
