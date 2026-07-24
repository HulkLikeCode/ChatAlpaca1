from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from numbers import Integral, Real

import numpy as np
import pandas as pd

from chat_alpaca.analytics import household_valuation
from chat_alpaca.bootstrap_forecasting import (
    BOOTSTRAP_MODEL_TYPE as BOOTSTRAP_MODEL_TYPE,
)
from chat_alpaca.bootstrap_forecasting import (
    BOOTSTRAP_MODEL_VERSION as BOOTSTRAP_MODEL_VERSION,
)
from chat_alpaca.bootstrap_forecasting import (
    BacktestCriteria as BacktestCriteria,
)
from chat_alpaca.bootstrap_forecasting import (
    BacktestResult as BacktestResult,
)
from chat_alpaca.bootstrap_forecasting import (
    BootstrapAssumptions as BootstrapAssumptions,
)
from chat_alpaca.bootstrap_forecasting import (
    BootstrapRequest as BootstrapRequest,
)
from chat_alpaca.bootstrap_forecasting import (
    BootstrapResult as BootstrapResult,
)
from chat_alpaca.bootstrap_forecasting import (
    build_bootstrap_request as build_bootstrap_request,
)
from chat_alpaca.bootstrap_forecasting import (
    monthly_returns_from_prices as monthly_returns_from_prices,
)
from chat_alpaca.bootstrap_forecasting import (
    rolling_origin_backtest as rolling_origin_backtest,
)
from chat_alpaca.bootstrap_forecasting import (
    run_block_bootstrap as run_block_bootstrap,
)
from chat_alpaca.bootstrap_forecasting import (
    sample_block_indices as sample_block_indices,
)
from chat_alpaca.bootstrap_forecasting import (
    save_bootstrap_run as save_bootstrap_run,
)
from chat_alpaca.models import Portfolio
from chat_alpaca.parametric_forecasting import (
    PARAMETRIC_MODEL_TYPE as PARAMETRIC_MODEL_TYPE,
)
from chat_alpaca.parametric_forecasting import (
    PARAMETRIC_MODEL_VERSION as PARAMETRIC_MODEL_VERSION,
)
from chat_alpaca.parametric_forecasting import (
    CapitalMarketAssumption as CapitalMarketAssumption,
)
from chat_alpaca.parametric_forecasting import (
    ParameterEstimate as ParameterEstimate,
)
from chat_alpaca.parametric_forecasting import (
    ParametricAssumptions as ParametricAssumptions,
)
from chat_alpaca.parametric_forecasting import (
    ParametricRequest as ParametricRequest,
)
from chat_alpaca.parametric_forecasting import (
    ParametricResult as ParametricResult,
)
from chat_alpaca.parametric_forecasting import (
    build_parametric_request as build_parametric_request,
)
from chat_alpaca.parametric_forecasting import (
    calibration_comparison_table as calibration_comparison_table,
)
from chat_alpaca.parametric_forecasting import (
    estimate_parameters as estimate_parameters,
)
from chat_alpaca.parametric_forecasting import (
    import_external_assumptions as import_external_assumptions,
)
from chat_alpaca.parametric_forecasting import (
    model_comparison_table as model_comparison_table,
)
from chat_alpaca.parametric_forecasting import (
    normal_vs_fat_tail_comparison as normal_vs_fat_tail_comparison,
)
from chat_alpaca.parametric_forecasting import (
    parametric_sensitivity as parametric_sensitivity,
)
from chat_alpaca.parametric_forecasting import (
    rolling_parametric_backtest as rolling_parametric_backtest,
)
from chat_alpaca.parametric_forecasting import (
    run_parametric_forecast as run_parametric_forecast,
)
from chat_alpaca.parametric_forecasting import (
    save_parametric_run as save_parametric_run,
)
from chat_alpaca.parametric_forecasting import (
    validate_correlation_matrix as validate_correlation_matrix,
)
from chat_alpaca.parametric_forecasting import (
    validate_covariance_matrix as validate_covariance_matrix,
)
from chat_alpaca.portfolio_service import money, portfolio_cost

PERCENTILES = (5, 25, 50, 75, 95)
LEGACY_FORECAST_MODEL_TYPE = "legacy_projection"
LEGACY_FORECAST_MODEL_VERSION = "1.0.0"
LEGACY_FORECAST_DEFAULT_SIMULATIONS = 10_000
LEGACY_FORECAST_DEFAULT_SEED = 20260719


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number and cannot be Boolean.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer and cannot be Boolean.")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be at least one.")
    return normalized


@dataclass(frozen=True)
class ProjectionResult:
    """Percentile outcomes from an assumption-driven portfolio projection."""

    monthly_percentiles: pd.DataFrame
    annual_percentiles: pd.DataFrame
    target_probability: float | None
    contract: LegacyForecastContract


@dataclass(frozen=True)
class ForecastAssumptions:
    annual_return: float
    annual_volatility: float
    monthly_contribution: float
    horizon_years: int
    target_value: float | None = None

    def __post_init__(self) -> None:
        _validate_forecast_inputs(
            current_value=1.0,
            annual_return=self.annual_return,
            annual_volatility=self.annual_volatility,
            monthly_contribution=self.monthly_contribution,
            horizon_years=self.horizon_years,
            target_value=self.target_value,
            simulations=1,
            seed=0,
        )


@dataclass(frozen=True)
class LegacyForecastContract:
    model_type: str
    model_version: str
    seed: int
    simulation_count: int
    assumptions: ForecastAssumptions
    source_valuation_date: date | None
    source_valuation_methodology: str
    result_generated_at: datetime


@dataclass(frozen=True)
class ForecastRequest:
    current_value: float
    assumptions: ForecastAssumptions
    valuation_basis: str
    warnings: tuple[str, ...] = ()
    coverage: str = ""
    source_valuation_date: date | None = None
    seed: int = LEGACY_FORECAST_DEFAULT_SEED
    simulation_count: int = LEGACY_FORECAST_DEFAULT_SIMULATIONS
    portfolio_ids: tuple[int, ...] = ()
    portfolio_names: tuple[str, ...] = ()


def build_forecast_request(
    portfolios: list[Portfolio],
    closes: pd.DataFrame,
    assumptions: ForecastAssumptions,
) -> ForecastRequest:
    """Construct a reproducible request with an explicit starting-value policy."""
    if not portfolios:
        raise ValueError("A forecast requires at least one portfolio.")
    if closes.empty:
        current_value = float(
            sum((portfolio_cost(portfolio) for portfolio in portfolios), start=money(0))
        )
        warnings = (
            "Market data is unavailable; this scenario explicitly uses cost basis plus cash "
            "as its starting value.",
        )
        valuation_basis = "cost_basis_plus_cash"
        coverage = "Market-price coverage unavailable; cost basis fallback disclosed."
        source_valuation_date = None
    else:
        household = household_valuation(portfolios, closes)
        valuations = household.valuations
        incomplete = [
            portfolio.name
            for portfolio, valuation in zip(portfolios, valuations, strict=True)
            if not valuation.is_complete
        ]
        if incomplete:
            raise ValueError(
                "A projection is unavailable until every held symbol has a usable price. "
                "Incomplete portfolios: " + ", ".join(incomplete) + "."
            )
        current_value = float(household.total_calculated_value)
        warnings = tuple(
            sorted(
                {
                    *closes.attrs.get("warnings", ()),
                    *(warning for valuation in valuations for warning in valuation.warnings),
                }
            )
        )
        valuation_basis = "confirmed_market_value"
        coverage = (
            f"Common confirmed household valuation date: {household.common_valuation_date}; "
            f"complete valuations: {len(valuations)} of {len(valuations)} portfolios."
        )
        source_valuation_date = household.common_valuation_date
    if current_value <= 0:
        raise ValueError("A projection requires a selected portfolio with a positive value.")
    return ForecastRequest(
        current_value=current_value,
        assumptions=assumptions,
        valuation_basis=valuation_basis,
        warnings=warnings,
        coverage=coverage,
        source_valuation_date=source_valuation_date,
        portfolio_ids=tuple(portfolio.id for portfolio in portfolios),
        portfolio_names=tuple(portfolio.name for portfolio in portfolios),
    )


def run_forecast(request: ForecastRequest) -> ProjectionResult:
    assumptions = request.assumptions
    return simulate_portfolio_projection(
        current_value=request.current_value,
        annual_return=assumptions.annual_return,
        annual_volatility=assumptions.annual_volatility,
        monthly_contribution=assumptions.monthly_contribution,
        horizon_years=assumptions.horizon_years,
        target_value=assumptions.target_value,
        simulations=request.simulation_count,
        seed=request.seed,
        source_valuation_date=request.source_valuation_date,
        source_valuation_methodology=request.valuation_basis,
    )


def projection_calendar_dates(contract: LegacyForecastContract) -> pd.DatetimeIndex:
    """Return timezone-free calendar labels for every modeled monthly step.

    Month zero is the explicit confirmed valuation date when one exists. Otherwise, the UTC
    generation date is the disclosed fallback. Each later modeled month is labeled at the end of
    the corresponding following calendar month; labels do not participate in the simulation.
    """
    generated_at = contract.result_generated_at
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    fallback_date = generated_at.astimezone(timezone.utc).date()
    starting_date = contract.source_valuation_date or fallback_date
    starting_timestamp = pd.Timestamp(starting_date)
    starting_period = starting_timestamp.to_period("M")
    months = contract.assumptions.horizon_years * 12
    month_ends = [
        (starting_period + month).to_timestamp(how="end").normalize()
        for month in range(1, months + 1)
    ]
    return pd.DatetimeIndex([starting_timestamp, *month_ends], name="Month date")


def _validate_forecast_inputs(
    *,
    current_value: object,
    annual_return: object,
    annual_volatility: object,
    monthly_contribution: object,
    horizon_years: object,
    target_value: object | None,
    simulations: object,
    seed: object,
) -> None:
    current = _finite_real("Current portfolio value", current_value)
    annual = _finite_real("Annual return", annual_return)
    volatility = _finite_real("Annual volatility", annual_volatility)
    contribution = _finite_real("Monthly contribution", monthly_contribution)
    horizon = _positive_integer("Forecast horizon", horizon_years)
    simulation_count = _positive_integer("Simulation count", simulations)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("Forecast seed must be a nonnegative integer and cannot be Boolean.")
    target = _finite_real("Target value", target_value) if target_value is not None else None
    if current <= 0:
        raise ValueError("Current portfolio value must be greater than zero.")
    if annual <= -1:
        raise ValueError("Annual return must be greater than -100%.")
    if volatility < 0:
        raise ValueError("Annual volatility cannot be negative.")
    if contribution < 0:
        raise ValueError("Monthly contribution cannot be negative.")
    if horizon > 10:
        raise ValueError("Forecast horizon must be between 1 and 10 years.")
    if simulation_count < 1:
        raise ValueError("At least one simulation is required.")
    if target is not None and target <= 0:
        raise ValueError("Target value must be greater than zero.")


def simulate_portfolio_projection(
    current_value: float,
    annual_return: float,
    annual_volatility: float,
    monthly_contribution: float,
    horizon_years: int,
    target_value: float | None = None,
    simulations: int = LEGACY_FORECAST_DEFAULT_SIMULATIONS,
    seed: int = LEGACY_FORECAST_DEFAULT_SEED,
    source_valuation_date: date | None = None,
    source_valuation_methodology: str = "direct_inputs",
) -> ProjectionResult:
    """Project monthly portfolio values with a lognormal return model.

    ``annual_return`` is an arithmetic expected annual return. Contributions are
    added at each month-end in today's dollars. Results are planning scenarios,
    not forecasts inferred from the portfolio's short transaction history.
    """
    _validate_forecast_inputs(
        current_value=current_value,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        monthly_contribution=monthly_contribution,
        horizon_years=horizon_years,
        target_value=target_value,
        simulations=simulations,
        seed=seed,
    )

    months = horizon_years * 12
    monthly_step = 1 / 12
    # Calibrate the lognormal process so its expected annual gross return is
    # 1 + annual_return, while volatility controls the spread of outcomes.
    drift = (np.log1p(annual_return) - 0.5 * annual_volatility**2) * monthly_step
    diffusion = annual_volatility * np.sqrt(monthly_step)
    shocks = np.random.default_rng(seed).standard_normal((simulations, months))
    gross_returns = np.exp(drift + diffusion * shocks)

    paths = np.empty((simulations, months + 1), dtype=float)
    paths[:, 0] = current_value
    for month in range(1, months + 1):
        paths[:, month] = paths[:, month - 1] * gross_returns[:, month - 1]
        paths[:, month] += monthly_contribution
    if not np.isfinite(paths).all():
        raise ValueError("Forecast inputs produced nonfinite simulated values.")

    month_index = pd.RangeIndex(0, months + 1, name="Month")
    monthly_percentiles = pd.DataFrame(
        np.percentile(paths, PERCENTILES, axis=0).T,
        columns=[f"P{percentile}" for percentile in PERCENTILES],
        index=month_index,
    )
    annual_percentiles = monthly_percentiles.iloc[::12].copy()
    annual_percentiles.index = pd.Index(range(horizon_years + 1), name="Year")
    target_probability = (
        float(np.mean(paths[:, -1] >= target_value)) if target_value is not None else None
    )
    assumptions = ForecastAssumptions(
        annual_return,
        annual_volatility,
        monthly_contribution,
        horizon_years,
        target_value,
    )
    contract = LegacyForecastContract(
        LEGACY_FORECAST_MODEL_TYPE,
        LEGACY_FORECAST_MODEL_VERSION,
        int(seed),
        int(simulations),
        assumptions,
        source_valuation_date,
        source_valuation_methodology,
        datetime.now(timezone.utc),
    )
    return ProjectionResult(monthly_percentiles, annual_percentiles, target_probability, contract)
