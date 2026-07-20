from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_alpaca.forecasting import simulate_portfolio_projection
from chat_alpaca.models import HypotheticalScenario, Portfolio
from chat_alpaca.scenarios import ledger_state_hash

HYPOTHETICAL_MODEL_VERSION = "1.0.0"


class HypotheticalActionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    ADD_CASH = "add_cash"
    REMOVE_CASH = "remove_cash"
    REASSIGN = "reassign"


@dataclass(frozen=True)
class BaselineLot:
    portfolio_id: int
    portfolio_name: str
    symbol: str
    shares: float
    unit_cost_basis: float
    acquired_ordinal: int = 0

    def __post_init__(self) -> None:
        if self.shares < 0 or self.unit_cost_basis < 0:
            raise ValueError("Hypothetical baselines support nonnegative long lots only.")


@dataclass(frozen=True)
class PortfolioBaseline:
    portfolio_id: int
    portfolio_name: str
    cash: float
    lots: tuple[BaselineLot, ...]


@dataclass(frozen=True)
class ProposedAction:
    action: HypotheticalActionType | str
    portfolio_id: int
    symbol: str | None = None
    quantity: float | None = None
    price: float | None = None
    amount: float | None = None
    destination_portfolio_id: int | None = None
    fees: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", HypotheticalActionType(self.action))
        symbol = self.symbol.strip().upper() if self.symbol else None
        object.__setattr__(self, "symbol", symbol)
        if self.fees < 0:
            raise ValueError("Hypothetical fees cannot be negative.")
        if self.action in {HypotheticalActionType.BUY, HypotheticalActionType.SELL}:
            if not symbol or not self.quantity or self.quantity <= 0:
                raise ValueError("A buy or sell requires a symbol and positive quantity.")
            if self.price is None or self.price <= 0:
                raise ValueError("A buy or sell requires a positive hypothetical price.")
        elif self.action in {
            HypotheticalActionType.ADD_CASH,
            HypotheticalActionType.REMOVE_CASH,
        }:
            if self.amount is None or self.amount <= 0:
                raise ValueError("A cash action requires a positive amount.")
        elif self.action == HypotheticalActionType.REASSIGN:
            if not symbol or not self.quantity or self.quantity <= 0:
                raise ValueError("An assignment change requires a symbol and positive quantity.")
            if self.destination_portfolio_id is None:
                raise ValueError("An assignment change requires a destination portfolio.")
            if self.destination_portfolio_id == self.portfolio_id:
                raise ValueError("Assignment source and destination must differ.")


@dataclass(frozen=True)
class RetirementAnalysisAssumptions:
    horizon_years: int
    annual_spending: float
    simulations: int = 5_000
    seed: int = 20260720

    def __post_init__(self) -> None:
        if not 1 <= self.horizon_years <= 40:
            raise ValueError("Retirement horizon must be between 1 and 40 years.")
        if self.annual_spending < 0 or self.simulations < 1:
            raise ValueError("Retirement spending cannot be negative and simulations must run.")


@dataclass(frozen=True)
class HypotheticalAssumptions:
    expected_returns: Mapping[str, float]
    sectors: Mapping[str, str | Mapping[str, float]]
    benchmark_weights: Mapping[str, float]
    stress_shocks: Mapping[str, float]
    forecast_horizon_years: int = 10
    forecast_target: float | None = None
    retirement: RetirementAnalysisAssumptions | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.forecast_horizon_years <= 10:
            raise ValueError("Forecast horizon must be between 1 and 10 years.")
        if self.forecast_target is not None and self.forecast_target <= 0:
            raise ValueError("Forecast target must be positive.")
        if any(value <= -1 for value in self.expected_returns.values()):
            raise ValueError("Expected returns must be greater than -100%.")
        if any(value < -1 or value > 1 for value in self.stress_shocks.values()):
            raise ValueError("Stress shocks must be between -100% and 100%.")


@dataclass(frozen=True)
class AnalysisSnapshot:
    total_value: float
    cash: float
    market_value: float
    cost_basis: float
    portfolio_values: Mapping[str, float]
    portfolio_cash: Mapping[str, float]
    holding_weights: Mapping[str, float]
    assignment_weights: Mapping[str, float]
    sector_exposure: Mapping[str, float]
    benchmark_relative_exposure: Mapping[str, float]
    concentration: Mapping[str, float]
    effective_number_of_holdings: float
    volatility: float | None
    beta: float | None
    risk_contribution: Mapping[str, float]
    drawdown_exposure: float | None
    expected_return: float
    forecast_target_probability: float | None
    downside_percentiles: Mapping[str, float]
    deterministic_stress_losses: Mapping[str, float]
    retirement_success_probability: float | None


@dataclass(frozen=True)
class HypotheticalResult:
    model_version: str
    market_data_as_of: datetime
    before: AnalysisSnapshot
    after: AnalysisSnapshot
    changes: Mapping[str, float]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SavedScenario:
    id: int
    name: str
    creator: str
    created_at: datetime
    portfolio_scope: tuple[Mapping[str, object], ...]
    baseline_ledger_hash: str
    market_data_as_of: datetime
    assumptions: Mapping[str, object]
    proposed_trades: tuple[Mapping[str, object], ...]
    results: Mapping[str, object]
    stale_baseline: bool


@dataclass(frozen=True)
class OrderTicketTransferDraft:
    """Reviewed copy data only; this type has no brokerage or persistence behavior."""

    portfolio_id: int
    symbol: str
    side: str
    quantity: float
    reviewed_market_price: float
    reviewed_price_as_of: datetime
    source_scenario_id: int | None


@dataclass
class _Lot:
    portfolio_id: int
    portfolio_name: str
    symbol: str
    shares: float
    unit_basis: float
    order: int


def baseline_from_portfolios(portfolios: Sequence[Portfolio]) -> tuple[PortfolioBaseline, ...]:
    """Copy ledger-derived state into immutable analysis inputs."""
    return tuple(
        PortfolioBaseline(
            portfolio.id,
            portfolio.name,
            float(portfolio.cash),
            tuple(
                BaselineLot(
                    portfolio.id,
                    portfolio.name,
                    lot.symbol.upper(),
                    float(lot.shares),
                    float(lot.cost_basis),
                    lot.acquired_on.toordinal(),
                )
                for lot in portfolio.holdings
            ),
        )
        for portfolio in portfolios
    )


def _sector_weights(
    symbol: str, sectors: Mapping[str, str | Mapping[str, float]]
) -> dict[str, float]:
    raw = sectors.get(symbol, "Unclassified")
    if isinstance(raw, str):
        return {raw: 1.0}
    weights = {str(key): float(value) for key, value in raw.items() if float(value) > 0}
    total = sum(weights.values())
    if total > 1.000001:
        weights = {key: value / 100 for key, value in weights.items()}
        total = sum(weights.values())
    if total > 1.000001:
        raise ValueError(f"Sector weights for {symbol} exceed 100%.")
    if total < 0.999999:
        weights["Unclassified"] = weights.get("Unclassified", 0.0) + 1 - total
    return weights


def _apply_actions(
    baselines: Sequence[PortfolioBaseline], actions: Sequence[ProposedAction]
) -> tuple[dict[int, float], list[_Lot], dict[str, float]]:
    names = {item.portfolio_id: item.portfolio_name for item in baselines}
    cash = {item.portfolio_id: item.cash for item in baselines}
    lots = [
        _Lot(
            lot.portfolio_id,
            lot.portfolio_name,
            lot.symbol,
            lot.shares,
            lot.unit_cost_basis,
            lot.acquired_ordinal,
        )
        for baseline in baselines
        for lot in baseline.lots
    ]
    trade_prices: dict[str, float] = {}

    def remove_fifo(portfolio_id: int, symbol: str, quantity: float) -> list[tuple[float, float]]:
        remaining = quantity
        removed: list[tuple[float, float]] = []
        for lot in sorted(lots, key=lambda item: item.order):
            if lot.portfolio_id != portfolio_id or lot.symbol != symbol or lot.shares <= 0:
                continue
            used = min(lot.shares, remaining)
            lot.shares -= used
            remaining -= used
            removed.append((used, lot.unit_basis))
            if remaining <= 1e-10:
                break
        if remaining > 1e-8:
            raise ValueError(f"The scenario tries to move or sell more {symbol} than is assigned.")
        return removed

    for index, action in enumerate(actions, start=1):
        if action.portfolio_id not in cash:
            raise ValueError(f"Action {index} references a portfolio outside the scenario scope.")
        if action.action == HypotheticalActionType.ADD_CASH:
            cash[action.portfolio_id] += float(action.amount)
        elif action.action == HypotheticalActionType.REMOVE_CASH:
            cash[action.portfolio_id] -= float(action.amount)
        elif action.action == HypotheticalActionType.BUY:
            notional = float(action.quantity) * float(action.price)
            cash[action.portfolio_id] -= notional + action.fees
            lots.append(
                _Lot(
                    action.portfolio_id,
                    names[action.portfolio_id],
                    str(action.symbol),
                    float(action.quantity),
                    (notional + action.fees) / float(action.quantity),
                    10**9 + index,
                )
            )
            trade_prices[str(action.symbol)] = float(action.price)
        elif action.action == HypotheticalActionType.SELL:
            remove_fifo(action.portfolio_id, str(action.symbol), float(action.quantity))
            cash[action.portfolio_id] += float(action.quantity) * float(action.price) - action.fees
            trade_prices[str(action.symbol)] = float(action.price)
        else:
            destination = int(action.destination_portfolio_id)
            if destination not in cash:
                raise ValueError(
                    f"Action {index} destination is outside the scenario portfolio scope."
                )
            moved = remove_fifo(action.portfolio_id, str(action.symbol), float(action.quantity))
            for moved_shares, unit_basis in moved:
                lots.append(
                    _Lot(
                        destination,
                        names[destination],
                        str(action.symbol),
                        moved_shares,
                        unit_basis,
                        10**9 + index,
                    )
                )
    return cash, [lot for lot in lots if lot.shares > 1e-10], trade_prices


def _retirement_probability(
    value: float,
    expected_return: float,
    volatility: float,
    assumptions: RetirementAnalysisAssumptions | None,
) -> float | None:
    if assumptions is None:
        return None
    months = assumptions.horizon_years * 12
    rng = np.random.default_rng(assumptions.seed)
    drift = (math.log1p(expected_return) - 0.5 * volatility**2) / 12
    diffusion = volatility / math.sqrt(12)
    values = np.full(assumptions.simulations, value, dtype=float)
    spending = assumptions.annual_spending / 12
    for _ in range(months):
        active = values > 0
        values[active] *= np.exp(drift + diffusion * rng.standard_normal(int(active.sum())))
        values = np.maximum(values - spending, 0)
    return float(np.mean(values > 0))


def _snapshot(
    cash: Mapping[int, float],
    names: Mapping[int, str],
    lots: Sequence[_Lot],
    prices: Mapping[str, float],
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    assumptions: HypotheticalAssumptions,
) -> AnalysisSnapshot:
    by_assignment: dict[tuple[int, str], float] = {}
    basis = 0.0
    for lot in lots:
        if lot.symbol not in prices:
            raise ValueError(f"A current hypothetical-analysis price is required for {lot.symbol}.")
        value = lot.shares * prices[lot.symbol]
        by_assignment[(lot.portfolio_id, lot.symbol)] = (
            by_assignment.get((lot.portfolio_id, lot.symbol), 0.0) + value
        )
        basis += lot.shares * lot.unit_basis
    symbol_values: dict[str, float] = {}
    portfolio_values = {names[key]: float(value) for key, value in cash.items()}
    for (portfolio_id, symbol), value in by_assignment.items():
        symbol_values[symbol] = symbol_values.get(symbol, 0.0) + value
        portfolio_values[names[portfolio_id]] += value
    total_cash = sum(cash.values())
    market_value = sum(symbol_values.values())
    total_value = total_cash + market_value
    if total_value <= 0:
        raise ValueError("A hypothetical scenario must retain a positive total value.")
    holding_weights = {symbol: value / total_value for symbol, value in symbol_values.items()}
    assignment_weights = {
        f"{names[portfolio_id]} · {symbol}": value / total_value
        for (portfolio_id, symbol), value in by_assignment.items()
    }
    sectors: dict[str, float] = {}
    for symbol, value in symbol_values.items():
        for sector, weight in _sector_weights(symbol, assumptions.sectors).items():
            sectors[sector] = sectors.get(sector, 0.0) + value * weight / total_value
    benchmark = {
        str(key).upper(): float(value) for key, value in assumptions.benchmark_weights.items()
    }
    benchmark_relative = {
        symbol: holding_weights.get(symbol, 0.0) - benchmark.get(symbol, 0.0)
        for symbol in sorted(set(holding_weights) | set(benchmark))
    }
    invested_weights = (
        {symbol: value / market_value for symbol, value in symbol_values.items()}
        if market_value > 0
        else {}
    )
    hhi = sum(weight**2 for weight in invested_weights.values())
    concentration = {
        "largest_holding_weight": max(holding_weights.values(), default=0.0),
        "top_five_weight": sum(sorted(holding_weights.values(), reverse=True)[:5]),
        "herfindahl_index": hhi,
    }

    symbols = tuple(sorted(symbol_values))
    volatility: float | None = None
    beta: float | None = None
    drawdown: float | None = None
    risk_contribution: dict[str, float] = {}
    if symbols and all(symbol in returns for symbol in symbols):
        aligned = returns[list(symbols)].replace([np.inf, -np.inf], np.nan)
        if benchmark_returns is not None:
            aligned = aligned.join(benchmark_returns.rename("__benchmark__"), how="inner")
        aligned = aligned.dropna()
        if len(aligned) >= 2:
            weights = np.array([holding_weights[symbol] for symbol in symbols], dtype=float)
            asset_returns = aligned[list(symbols)]
            covariance = asset_returns.cov().to_numpy(dtype=float) * 252
            variance = float(weights @ covariance @ weights)
            volatility = math.sqrt(max(variance, 0.0))
            if variance > 0:
                components = weights * (covariance @ weights) / variance
                risk_contribution = dict(zip(symbols, components.astype(float), strict=True))
            portfolio_returns = asset_returns.to_numpy(dtype=float) @ weights
            wealth = np.cumprod(1 + portfolio_returns)
            peaks = np.maximum.accumulate(wealth)
            drawdown = float(np.min(wealth / peaks - 1))
            if benchmark_returns is not None:
                benchmark_values = aligned["__benchmark__"].to_numpy(dtype=float)
                benchmark_variance = float(np.var(benchmark_values, ddof=1))
                if benchmark_variance > 0:
                    beta = float(
                        np.cov(portfolio_returns, benchmark_values, ddof=1)[0, 1]
                        / benchmark_variance
                    )

    expected_return = sum(
        holding_weights.get(symbol, 0.0) * float(assumptions.expected_returns.get(symbol, 0.0))
        for symbol in symbol_values
    )
    forecast_volatility = volatility or 0.0
    projection = simulate_portfolio_projection(
        total_value,
        expected_return,
        forecast_volatility,
        0,
        assumptions.forecast_horizon_years,
        assumptions.forecast_target,
        simulations=5_000,
        seed=20260720,
    )
    downside = {
        "P5": float(projection.annual_percentiles.iloc[-1]["P5"]),
        "P25": float(projection.annual_percentiles.iloc[-1]["P25"]),
    }
    stress_losses: dict[str, float] = {}
    for stress_name, shock in assumptions.stress_shocks.items():
        if stress_name.strip().lower() in {"all", "broad market", "broad_market"}:
            stress_losses[stress_name] = market_value * float(shock)
        elif stress_name.upper() in symbol_values:
            stress_losses[stress_name] = symbol_values[stress_name.upper()] * float(shock)
        else:
            exposed = sectors.get(stress_name, 0.0) * total_value
            stress_losses[stress_name] = exposed * float(shock)
    retirement_probability = _retirement_probability(
        total_value,
        expected_return,
        forecast_volatility,
        assumptions.retirement,
    )
    return AnalysisSnapshot(
        total_value,
        total_cash,
        market_value,
        basis,
        portfolio_values,
        {names[key]: float(value) for key, value in cash.items()},
        holding_weights,
        assignment_weights,
        sectors,
        benchmark_relative,
        concentration,
        1 / hhi if hhi > 0 else 0.0,
        volatility,
        beta,
        risk_contribution,
        drawdown,
        expected_return,
        projection.target_probability,
        downside,
        stress_losses,
        retirement_probability,
    )


def analyze_hypothetical_scenario(
    baselines: Sequence[PortfolioBaseline],
    actions: Sequence[ProposedAction],
    prices: Mapping[str, float],
    returns: pd.DataFrame,
    assumptions: HypotheticalAssumptions,
    *,
    market_data_as_of: datetime,
    benchmark_returns: pd.Series | None = None,
) -> HypotheticalResult:
    """Analyze copied state only; canonical ORM entities are never mutated."""
    if not baselines:
        raise ValueError("Select at least one portfolio for hypothetical analysis.")
    if not actions:
        raise ValueError("Add at least one proposed action.")
    normalized_prices = {str(key).upper(): float(value) for key, value in prices.items()}
    if any(value <= 0 for value in normalized_prices.values()):
        raise ValueError("Hypothetical-analysis prices must be positive.")
    names = {item.portfolio_id: item.portfolio_name for item in baselines}
    baseline_cash = {item.portfolio_id: item.cash for item in baselines}
    baseline_lots = [
        _Lot(
            lot.portfolio_id,
            lot.portfolio_name,
            lot.symbol,
            lot.shares,
            lot.unit_cost_basis,
            lot.acquired_ordinal,
        )
        for item in baselines
        for lot in item.lots
    ]
    after_cash, after_lots, action_prices = _apply_actions(baselines, actions)
    after_prices = {**action_prices, **normalized_prices}
    before = _snapshot(
        baseline_cash,
        names,
        baseline_lots,
        normalized_prices,
        returns,
        benchmark_returns,
        assumptions,
    )
    after = _snapshot(
        after_cash,
        names,
        after_lots,
        after_prices,
        returns,
        benchmark_returns,
        assumptions,
    )
    numeric_fields = (
        "total_value",
        "cash",
        "market_value",
        "cost_basis",
        "effective_number_of_holdings",
        "expected_return",
    )
    changes = {field: getattr(after, field) - getattr(before, field) for field in numeric_fields}
    for field in ("volatility", "beta", "drawdown_exposure", "forecast_target_probability"):
        before_value = getattr(before, field)
        after_value = getattr(after, field)
        if before_value is not None and after_value is not None:
            changes[field] = after_value - before_value
    warnings = [
        "Analysis only: this scenario does not create transactions, lots, ledger rows, "
        "Alpaca allocations, or orders.",
        "Expected return and probability outputs are assumption-driven and not guarantees.",
    ]
    if after.cash < 0:
        warnings.append("The proposed actions produce negative cash.")
    if before.volatility is None or after.volatility is None:
        warnings.append(
            "One or more historical risk measures are unavailable because the scenario "
            "does not have at least two complete, aligned return observations for every holding."
        )
    return HypotheticalResult(
        HYPOTHETICAL_MODEL_VERSION,
        market_data_as_of,
        before,
        after,
        changes,
        tuple(warnings),
    )


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def save_hypothetical_scenario(
    session: Session,
    *,
    name: str,
    creator: str,
    portfolios: Sequence[Portfolio],
    market_data_as_of: datetime,
    assumptions: HypotheticalAssumptions,
    actions: Sequence[ProposedAction],
    result: HypotheticalResult,
) -> HypotheticalScenario:
    clean_name = name.strip()
    clean_creator = creator.strip()
    if not clean_name or not clean_creator:
        raise ValueError("A scenario name and creator are required.")
    existing = session.scalar(
        select(HypotheticalScenario.id).where(
            HypotheticalScenario.creator == clean_creator,
            HypotheticalScenario.name == clean_name,
        )
    )
    if existing is not None:
        raise ValueError("This creator already has a saved scenario with that name.")
    row = HypotheticalScenario(
        name=clean_name,
        creator=clean_creator,
        portfolio_scope=json.dumps(
            [{"id": item.id, "name": item.name} for item in portfolios], sort_keys=True
        ),
        baseline_ledger_hash=ledger_state_hash(session, [item.id for item in portfolios]),
        market_data_as_of=market_data_as_of,
        assumptions=json.dumps(asdict(assumptions), sort_keys=True, default=_json_default),
        proposed_trades=json.dumps(
            [asdict(action) for action in actions], sort_keys=True, default=_json_default
        ),
        results=json.dumps(asdict(result), sort_keys=True, default=_json_default),
    )
    session.add(row)
    session.flush()
    return row


def load_hypothetical_scenarios(
    session: Session, *, creator: str | None = None
) -> tuple[SavedScenario, ...]:
    statement = select(HypotheticalScenario).order_by(HypotheticalScenario.created_at.desc())
    if creator is not None:
        statement = statement.where(HypotheticalScenario.creator == creator)
    rows = session.scalars(statement)
    saved: list[SavedScenario] = []
    for row in rows:
        scope = tuple(json.loads(row.portfolio_scope))
        ids = [int(item["id"]) for item in scope]
        current_hash = ledger_state_hash(session, ids)
        saved.append(
            SavedScenario(
                row.id,
                row.name,
                row.creator,
                row.created_at,
                scope,
                row.baseline_ledger_hash,
                row.market_data_as_of,
                json.loads(row.assumptions),
                tuple(json.loads(row.proposed_trades)),
                json.loads(row.results),
                current_hash != row.baseline_ledger_hash,
            )
        )
    return tuple(saved)


def prepare_order_ticket_transfer(
    action: ProposedAction,
    *,
    owner_review_confirmed: bool,
    current_ledger_hash: str,
    scenario_ledger_hash: str,
    reviewed_market_price: float | None,
    reviewed_price_as_of: datetime | None,
    now: datetime | None = None,
    maximum_price_age: timedelta = timedelta(minutes=15),
    source_scenario_id: int | None = None,
) -> OrderTicketTransferDraft:
    """Enforce the explicit, fresh-review boundary without submitting an order."""
    if action.action not in {HypotheticalActionType.BUY, HypotheticalActionType.SELL}:
        raise ValueError("Only a proposed buy or sell can be copied to an order ticket.")
    if not owner_review_confirmed:
        raise PermissionError("Fresh owner review and confirmation are required.")
    if current_ledger_hash != scenario_ledger_hash:
        raise ValueError("The scenario baseline is stale; rerun it before preparing an order.")
    if reviewed_market_price is None or reviewed_market_price <= 0 or reviewed_price_as_of is None:
        raise ValueError(
            "A freshly reviewed market price is required; scenario prices are ignored."
        )
    effective_now = now or datetime.now(timezone.utc)
    price_time = reviewed_price_as_of
    if price_time.tzinfo is None:
        price_time = price_time.replace(tzinfo=timezone.utc)
    if effective_now - price_time > maximum_price_age or price_time > effective_now:
        raise ValueError("The reviewed market price is stale or future-dated.")
    return OrderTicketTransferDraft(
        action.portfolio_id,
        str(action.symbol),
        action.action.value,
        float(action.quantity),
        float(reviewed_market_price),
        reviewed_price_as_of,
        source_scenario_id,
    )
