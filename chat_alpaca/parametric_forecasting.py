from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_alpaca.bootstrap_forecasting import (
    BOOTSTRAP_PERCENTILES,
    REBALANCE_FREQUENCIES,
    BacktestCriteria,
    BacktestResult,
    BootstrapAssumptions,
    BootstrapRequest,
    BootstrapResult,
    _distribution,
    _normalized_values,
    _prepare_returns,
    monthly_returns_from_prices,
)
from chat_alpaca.historical_data import HistoricalCoverageResult
from chat_alpaca.models import ForecastRun, ForecastRunDataset, ModelValidation, Portfolio
from chat_alpaca.reconstruction import ReconstructionResult, SufficiencyStatus
from chat_alpaca.scenarios import DatasetReference, ledger_state_hash

PARAMETRIC_MODEL_TYPE = "correlated_parametric_monte_carlo"
PARAMETRIC_MODEL_VERSION = "1.0.0"
PARAMETER_COLUMNS = ("symbol", "annual_return", "annual_volatility", "source")


@dataclass(frozen=True)
class CapitalMarketAssumption:
    annual_return: float | None = None
    annual_volatility: float | None = None
    source: str = "owner entry"
    publication: str | None = None
    as_of_date: str | None = None

    def __post_init__(self) -> None:
        if self.annual_return is not None and self.annual_return <= -1:
            raise ValueError("Annual return assumptions must be greater than -100%.")
        if self.annual_volatility is not None and self.annual_volatility < 0:
            raise ValueError("Annual volatility assumptions cannot be negative.")
        if not self.source.strip():
            raise ValueError("Every external or override assumption requires a source.")


@dataclass(frozen=True)
class ParametricAssumptions:
    horizon_years: int
    distribution: Literal["normal", "student_t"] = "normal"
    degrees_of_freedom: float | None = None
    simulations: int = 10_000
    seed: int = 20260720
    monthly_contribution: float = 0.0
    annual_inflation: float = 0.025
    annual_fee: float = 0.0
    rebalancing: Literal["monthly", "quarterly", "annual", "never"] = "monthly"
    target_value: float | None = None
    minimum_history_months: int = 24
    return_shrinkage: float = 0.50
    covariance_shrinkage: float = 0.20
    historical_weight: float = 1.0
    external_weight: float = 1.0
    override_weight: float = 1.0
    parameter_uncertainty: bool = True
    expected_return_shift: float = 0.0
    volatility_multiplier: float = 1.0
    correlation_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not 1 <= self.horizon_years <= 10:
            raise ValueError("Forecast horizon must be between 1 and 10 years.")
        if self.distribution not in {"normal", "student_t"}:
            raise ValueError("Distribution must be normal or student_t.")
        if self.distribution == "student_t":
            if self.degrees_of_freedom is None or self.degrees_of_freedom <= 2:
                raise ValueError("Student's t degrees of freedom must be greater than 2.")
        elif self.degrees_of_freedom is not None:
            raise ValueError("Degrees of freedom apply only to the Student's t distribution.")
        if self.simulations < 1:
            raise ValueError("At least one simulation is required.")
        if self.monthly_contribution < 0:
            raise ValueError("Monthly contribution cannot be negative.")
        if self.annual_inflation <= -1:
            raise ValueError("Annual inflation must be greater than -100%.")
        if not 0 <= self.annual_fee < 1:
            raise ValueError("Annual fee must be between 0% and 100%.")
        if self.rebalancing not in REBALANCE_FREQUENCIES:
            raise ValueError("Rebalancing must be monthly, quarterly, annual, or never.")
        if self.target_value is not None and self.target_value <= 0:
            raise ValueError("Target value must be greater than zero.")
        if self.minimum_history_months < 2:
            raise ValueError("At least two months of history are required.")
        for name in ("return_shrinkage", "covariance_shrinkage"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name.replace('_', ' ').title()} must be between 0 and 1.")
        if min(self.historical_weight, self.external_weight, self.override_weight) < 0:
            raise ValueError("Parameter source weights cannot be negative.")
        if self.volatility_multiplier <= 0:
            raise ValueError("Volatility multiplier must be positive.")
        if not 0 <= self.correlation_multiplier <= 1:
            raise ValueError("Correlation multiplier must be between 0 and 1.")


@dataclass(frozen=True)
class ParametricRequest:
    holding_values: Mapping[str, float]
    monthly_returns: pd.DataFrame
    assumptions: ParametricAssumptions
    cash: float = 0.0
    benchmark_returns: pd.Series | None = None
    proxies: Mapping[str, str] | None = None
    external_assumptions: Mapping[str, CapitalMarketAssumption | Mapping[str, object]] | None = None
    user_overrides: Mapping[str, CapitalMarketAssumption | Mapping[str, object]] | None = None
    external_correlation: pd.DataFrame | np.ndarray | None = None
    correlation_override: pd.DataFrame | np.ndarray | None = None
    external_correlation_source: str | None = None
    correlation_override_source: str | None = None
    dataset_ids: tuple[int, ...] = ()
    source_coverage: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ParameterEstimate:
    symbols: tuple[str, ...]
    annual_returns: np.ndarray
    annual_volatilities: np.ndarray
    annual_covariance: np.ndarray
    correlation: np.ndarray
    effective_history_months: int
    parameter_sources: Mapping[str, object]
    shrinkage_method: str
    covariance_method: str

    def disclosure(self) -> dict[str, object]:
        return {
            "symbols": list(self.symbols),
            "annual_returns": dict(zip(self.symbols, self.annual_returns.tolist(), strict=True)),
            "annual_volatilities": dict(
                zip(self.symbols, self.annual_volatilities.tolist(), strict=True)
            ),
            "annual_covariance": self.annual_covariance.tolist(),
            "correlation": self.correlation.tolist(),
            "effective_history_months": self.effective_history_months,
            "parameter_sources": dict(self.parameter_sources),
            "shrinkage_method": self.shrinkage_method,
            "covariance_method": self.covariance_method,
        }


@dataclass(frozen=True)
class ParametricResult:
    model_type: str
    model_version: str
    monthly_percentiles: pd.DataFrame
    annual_percentiles: pd.DataFrame
    terminal_values: np.ndarray
    terminal_distribution: Mapping[str, object]
    target_probability: float | None
    downside_percentiles: Mapping[str, float]
    probability_nominal_loss: float
    probability_real_loss: float
    probability_beating_benchmark: float | None
    downside_contribution_by_holding: Mapping[str, float]
    assumptions: Mapping[str, object]
    data_coverage: Mapping[str, object]
    proxies: Mapping[str, str]
    parameter_estimates: Mapping[str, object]
    parameter_sources: Mapping[str, object]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

    def summary(self) -> dict[str, object]:
        return {
            "terminal_distribution": dict(self.terminal_distribution),
            "target_probability": self.target_probability,
            "downside_percentiles": dict(self.downside_percentiles),
            "probability_nominal_loss": self.probability_nominal_loss,
            "probability_real_loss": self.probability_real_loss,
            "probability_beating_benchmark": self.probability_beating_benchmark,
            "downside_contribution_by_holding": dict(self.downside_contribution_by_holding),
            "parameter_estimates": dict(self.parameter_estimates),
            "parameter_sources": dict(self.parameter_sources),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


def validate_correlation_matrix(
    matrix: pd.DataFrame | np.ndarray,
    symbols: Sequence[str] | None = None,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Validate shape, labels, symmetry, unit diagonal, range, and PSD status."""
    if isinstance(matrix, pd.DataFrame):
        if symbols is not None:
            expected = [str(symbol).upper() for symbol in symbols]
            rows = [str(value).upper() for value in matrix.index]
            columns = [str(value).upper() for value in matrix.columns]
            if rows != expected or columns != expected:
                raise ValueError("Correlation matrix labels and order must match forecast symbols.")
        values = matrix.to_numpy(dtype=float)
    else:
        values = np.asarray(matrix, dtype=float)
    expected_size = len(symbols) if symbols is not None else None
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    if expected_size is not None and values.shape != (expected_size, expected_size):
        raise ValueError("Correlation matrix dimensions must match forecast symbols.")
    if not np.isfinite(values).all():
        raise ValueError("Correlation matrix must contain only finite values.")
    if not np.allclose(values, values.T, atol=tolerance, rtol=0):
        raise ValueError("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(values), 1.0, atol=tolerance, rtol=0):
        raise ValueError("Correlation matrix diagonal must contain ones.")
    if np.any(values < -1 - tolerance) or np.any(values > 1 + tolerance):
        raise ValueError("Correlation values must be between -1 and 1.")
    minimum_eigenvalue = float(np.linalg.eigvalsh(values).min())
    if minimum_eigenvalue < -tolerance:
        raise ValueError("Correlation matrix must be positive semidefinite.")
    return (values + values.T) / 2


def validate_covariance_matrix(
    matrix: pd.DataFrame | np.ndarray,
    symbols: Sequence[str] | None = None,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Validate a finite, symmetric, positive-semidefinite covariance matrix."""
    if isinstance(matrix, pd.DataFrame):
        if symbols is not None:
            expected = [str(symbol).upper() for symbol in symbols]
            rows = [str(value).upper() for value in matrix.index]
            columns = [str(value).upper() for value in matrix.columns]
            if rows != expected or columns != expected:
                raise ValueError("Covariance matrix labels and order must match forecast symbols.")
        values = matrix.to_numpy(dtype=float)
    else:
        values = np.asarray(matrix, dtype=float)
    expected_size = len(symbols) if symbols is not None else None
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if expected_size is not None and values.shape != (expected_size, expected_size):
        raise ValueError("Covariance matrix dimensions must match forecast symbols.")
    if not np.isfinite(values).all():
        raise ValueError("Covariance matrix must contain only finite values.")
    if not np.allclose(values, values.T, atol=tolerance, rtol=0):
        raise ValueError("Covariance matrix must be symmetric.")
    if np.any(np.diag(values) < -tolerance):
        raise ValueError("Covariance matrix variances cannot be negative.")
    if float(np.linalg.eigvalsh(values).min()) < -tolerance:
        raise ValueError("Covariance matrix must be positive semidefinite.")
    return (values + values.T) / 2


def _coerce_assumption(
    value: CapitalMarketAssumption | Mapping[str, object], *, default_source: str
) -> CapitalMarketAssumption:
    if isinstance(value, CapitalMarketAssumption):
        return value
    return CapitalMarketAssumption(
        annual_return=(
            float(value["annual_return"]) if value.get("annual_return") not in {None, ""} else None
        ),
        annual_volatility=(
            float(value["annual_volatility"])
            if value.get("annual_volatility") not in {None, ""}
            else None
        ),
        source=str(value.get("source") or default_source),
        publication=str(value["publication"]) if value.get("publication") else None,
        as_of_date=str(value["as_of_date"]) if value.get("as_of_date") else None,
    )


def import_external_assumptions(
    source: str | Path | TextIO,
    *,
    source_name: str = "owner-imported capital-market assumptions",
) -> dict[str, CapitalMarketAssumption]:
    """Import optional owner-provided CSV assumptions without requiring a data vendor."""
    if hasattr(source, "read"):
        content = source.read()
    elif isinstance(source, Path):
        content = source.read_text(encoding="utf-8")
    else:
        candidate = str(source)
        if "\n" in candidate or "," in candidate:
            content = candidate
        else:
            content = Path(candidate).read_text(encoding="utf-8")
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    missing = set(PARAMETER_COLUMNS[:3]) - set(reader.fieldnames or ())
    if missing:
        raise ValueError(
            "External assumption CSV is missing columns: " + ", ".join(sorted(missing))
        )
    imported: dict[str, CapitalMarketAssumption] = {}
    for line, row in enumerate(reader, start=2):
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            raise ValueError(f"External assumption CSV row {line} has no symbol.")
        if symbol in imported:
            raise ValueError(f"External assumption CSV contains duplicate symbol {symbol}.")
        try:
            imported[symbol] = CapitalMarketAssumption(
                annual_return=(
                    float(row["annual_return"]) if str(row["annual_return"]).strip() else None
                ),
                annual_volatility=(
                    float(row["annual_volatility"])
                    if str(row["annual_volatility"]).strip()
                    else None
                ),
                source=str(row.get("source") or source_name),
                publication=str(row.get("publication") or "") or None,
                as_of_date=str(row.get("as_of_date") or "") or None,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid external assumption CSV row {line}: {exc}") from exc
    if not imported:
        raise ValueError("External assumption CSV contains no data rows.")
    return imported


def _blend_value(
    historical: float,
    external: float | None,
    override: float | None,
    assumptions: ParametricAssumptions,
) -> tuple[float, dict[str, float]]:
    if override is not None and assumptions.override_weight > 0:
        return float(override), {"user_override": 1.0}
    candidates = [("historical", historical, assumptions.historical_weight)]
    if external is not None:
        candidates.append(("external", external, assumptions.external_weight))
    total_weight = sum(weight for _, _, weight in candidates)
    if total_weight <= 0:
        raise ValueError("At least one available parameter source must have positive weight.")
    normalized = {name: weight / total_weight for name, _, weight in candidates if weight > 0}
    value = sum(item * weight for _, item, weight in candidates) / total_weight
    return float(value), normalized


def estimate_parameters(
    request: ParametricRequest,
) -> tuple[ParameterEstimate, pd.DataFrame, dict[str, str], list[str]]:
    """Estimate shrunk, blended annual parameters from aligned monthly returns."""
    symbols, _ = _normalized_values(request.holding_values)
    bootstrap_request = BootstrapRequest(
        request.holding_values,
        request.monthly_returns,
        BootstrapAssumptions(
            horizon_years=request.assumptions.horizon_years,
            minimum_history_months=request.assumptions.minimum_history_months,
        ),
        cash=request.cash,
        benchmark_returns=request.benchmark_returns,
        proxies=request.proxies,
    )
    returns, _, proxies, warnings = _prepare_returns(bootstrap_request, symbols)
    log_returns = np.log1p(returns.clip(lower=-0.999999))
    historical_geometric = np.expm1(log_returns.mean().to_numpy(dtype=float) * 12)
    shrinkage_target = float(np.median(historical_geometric))
    alpha = request.assumptions.return_shrinkage
    shrunk_returns = (1 - alpha) * historical_geometric + alpha * shrinkage_target

    monthly_covariance = returns.cov().to_numpy(dtype=float)
    diagonal = np.diag(np.diag(monthly_covariance))
    covariance_alpha = request.assumptions.covariance_shrinkage
    shrunk_covariance = (1 - covariance_alpha) * monthly_covariance + covariance_alpha * diagonal
    historical_volatility = np.sqrt(np.maximum(np.diag(shrunk_covariance) * 12, 0))

    external = {
        str(symbol).upper(): _coerce_assumption(value, default_source="external owner entry")
        for symbol, value in (request.external_assumptions or {}).items()
    }
    overrides = {
        str(symbol).upper(): _coerce_assumption(value, default_source="user override")
        for symbol, value in (request.user_overrides or {}).items()
    }
    annual_returns: list[float] = []
    annual_volatilities: list[float] = []
    disclosures: dict[str, object] = {}
    for index, symbol in enumerate(symbols):
        ext = external.get(symbol)
        override = overrides.get(symbol)
        annual_return, return_weights = _blend_value(
            float(shrunk_returns[index]),
            ext.annual_return if ext else None,
            override.annual_return if override else None,
            request.assumptions,
        )
        annual_volatility, volatility_weights = _blend_value(
            float(historical_volatility[index]),
            ext.annual_volatility if ext else None,
            override.annual_volatility if override else None,
            request.assumptions,
        )
        final_annual_return = annual_return + request.assumptions.expected_return_shift
        if final_annual_return <= -1:
            raise ValueError(
                f"Final annual return for {symbol} must be greater than -100% after shifts."
            )
        annual_returns.append(final_annual_return)
        annual_volatilities.append(annual_volatility * request.assumptions.volatility_multiplier)
        disclosures[symbol] = {
            "historical_method": "cross-sectional median shrinkage",
            "return_weights": return_weights,
            "volatility_weights": volatility_weights,
            "external": asdict(ext) if ext else None,
            "user_override": asdict(override) if override else None,
        }

    standard_deviations = np.sqrt(np.maximum(np.diag(shrunk_covariance), 0))
    historical_correlation = np.divide(
        shrunk_covariance,
        np.outer(standard_deviations, standard_deviations),
        out=np.eye(len(symbols)),
        where=np.outer(standard_deviations, standard_deviations) > 0,
    )
    np.fill_diagonal(historical_correlation, 1.0)
    correlation_source = "historical covariance with diagonal shrinkage"
    correlation = historical_correlation
    if request.external_correlation is not None:
        external_correlation = validate_correlation_matrix(request.external_correlation, symbols)
        correlation = (correlation + external_correlation) / 2
        external_source = (
            request.external_correlation_source
            or getattr(request.external_correlation, "attrs", {}).get("source")
            or "external owner input"
        )
        correlation_source = f"equal blend of shrunk historical and {external_source}"
    if request.correlation_override is not None:
        correlation = validate_correlation_matrix(request.correlation_override, symbols)
        correlation_source = request.correlation_override_source or "user override"
    multiplier = request.assumptions.correlation_multiplier
    correlation = np.eye(len(symbols)) + multiplier * (correlation - np.eye(len(symbols)))
    correlation = validate_correlation_matrix(correlation, symbols)
    annual_volatility_array = np.asarray(annual_volatilities)
    annual_covariance = validate_covariance_matrix(
        correlation * np.outer(annual_volatility_array, annual_volatility_array), symbols
    )
    disclosures["correlation"] = {"source": correlation_source, "multiplier": multiplier}
    estimate = ParameterEstimate(
        symbols,
        np.asarray(annual_returns),
        annual_volatility_array,
        annual_covariance,
        correlation,
        len(returns),
        disclosures,
        f"cross-sectional median shrinkage ({alpha:.3f})",
        f"fixed diagonal covariance shrinkage ({covariance_alpha:.3f})",
    )
    return estimate, returns, proxies, warnings


def build_parametric_request(
    reconstruction: ReconstructionResult,
    market_coverage: HistoricalCoverageResult,
    assumptions: ParametricAssumptions,
    *,
    proxies: Mapping[str, str] | None = None,
    benchmark_returns: pd.Series | None = None,
    external_assumptions: Mapping[str, CapitalMarketAssumption | Mapping[str, object]]
    | None = None,
    user_overrides: Mapping[str, CapitalMarketAssumption | Mapping[str, object]] | None = None,
    external_correlation: pd.DataFrame | np.ndarray | None = None,
    correlation_override: pd.DataFrame | np.ndarray | None = None,
    external_correlation_source: str | None = None,
    correlation_override_source: str | None = None,
) -> ParametricRequest:
    """Build a parametric request from the shared ledger reconstruction boundary."""
    as_of = reconstruction.common_as_of_date
    if as_of is None or reconstruction.common_as_of_value is None:
        raise ValueError("A complete common reconstruction date is required for forecasting.")
    positions = reconstruction.combined.positions.loc[pd.Timestamp(as_of)]
    prices = market_coverage.data.copy()
    prices.index = pd.to_datetime(prices.index).normalize()
    holding_values: dict[str, float] = {}
    for symbol, quantity in positions.items():
        if abs(float(quantity)) <= 1e-12:
            continue
        available = prices.loc[prices.index <= pd.Timestamp(as_of), symbol].dropna()
        if available.empty:
            raise ValueError(f"Missing confirmed current price for {symbol}.")
        holding_values[str(symbol).upper()] = float(quantity) * float(available.iloc[-1])
    return ParametricRequest(
        holding_values=holding_values,
        cash=float(reconstruction.combined.cash.loc[pd.Timestamp(as_of)]),
        monthly_returns=monthly_returns_from_prices(market_coverage.data),
        assumptions=assumptions,
        benchmark_returns=benchmark_returns,
        proxies=proxies,
        external_assumptions=external_assumptions,
        user_overrides=user_overrides,
        external_correlation=external_correlation,
        correlation_override=correlation_override,
        external_correlation_source=external_correlation_source,
        correlation_override_source=correlation_override_source,
        dataset_ids=tuple(market_coverage.data.attrs.get("dataset_ids", ())),
        source_coverage={
            "reconstruction_status": reconstruction.data_coverage.status.value,
            "reconstruction_score": reconstruction.data_coverage.score,
            "adjustment": market_coverage.adjustment,
            "sources": reconstruction.data_coverage.sources,
        },
    )


def _draw_returns(
    estimate: ParameterEstimate, assumptions: ParametricAssumptions, months: int
) -> np.ndarray:
    monthly_mean = np.expm1(np.log1p(estimate.annual_returns) / 12)
    monthly_covariance = estimate.annual_covariance / 12
    rng = np.random.default_rng(assumptions.seed)
    if assumptions.parameter_uncertainty:
        mean_uncertainty = monthly_covariance / max(estimate.effective_history_months, 1)
        simulation_means = rng.multivariate_normal(
            monthly_mean,
            mean_uncertainty,
            size=assumptions.simulations,
            check_valid="raise",
        )
    else:
        simulation_means = np.broadcast_to(
            monthly_mean, (assumptions.simulations, len(monthly_mean))
        )
    standard = rng.multivariate_normal(
        np.zeros(len(monthly_mean)),
        monthly_covariance,
        size=(assumptions.simulations, months),
        check_valid="raise",
    )
    if assumptions.distribution == "student_t":
        degrees = float(assumptions.degrees_of_freedom)
        scales = np.sqrt(
            rng.chisquare(degrees, size=(assumptions.simulations, months, 1)) / degrees
        )
        standard = standard / scales * np.sqrt((degrees - 2) / degrees)
    return np.maximum(standard + simulation_means[:, None, :], -0.999999)


def run_parametric_forecast(request: ParametricRequest) -> ParametricResult:
    assumptions = request.assumptions
    symbols, starting_holdings = _normalized_values(request.holding_values)
    if request.cash < 0:
        raise ValueError("Cash cannot be negative.")
    estimate, returns, proxies, warnings = estimate_parameters(request)
    months = assumptions.horizon_years * 12
    sampled = _draw_returns(estimate, assumptions, months)
    target_weights = starting_holdings / starting_holdings.sum()
    holdings = np.broadcast_to(starting_holdings, (assumptions.simulations, len(symbols))).copy()
    total_start = float(starting_holdings.sum() + request.cash)
    cash = np.full(assumptions.simulations, float(request.cash))
    paths = np.empty((assumptions.simulations, months + 1), dtype=float)
    paths[:, 0] = total_start
    holding_return_pnl = np.zeros_like(holdings)
    monthly_fee = 1 - (1 - assumptions.annual_fee) ** (1 / 12)
    rebalance_every = REBALANCE_FREQUENCIES[assumptions.rebalancing]
    for month in range(months):
        pnl = holdings * sampled[:, month, :]
        holding_return_pnl += pnl
        holdings += pnl
        invested = holdings.sum(axis=1)
        fee = invested * monthly_fee
        holdings -= fee[:, None] * np.divide(
            holdings,
            invested[:, None],
            out=np.zeros_like(holdings),
            where=invested[:, None] != 0,
        )
        holdings += assumptions.monthly_contribution * target_weights
        if rebalance_every is not None and (month + 1) % rebalance_every == 0:
            holdings = holdings.sum(axis=1)[:, None] * target_weights
        paths[:, month + 1] = holdings.sum(axis=1) + cash

    columns = [f"P{percentile}" for percentile in BOOTSTRAP_PERCENTILES]
    monthly_percentiles = pd.DataFrame(
        np.percentile(paths, BOOTSTRAP_PERCENTILES, axis=0).T,
        columns=columns,
        index=pd.RangeIndex(months + 1, name="Month"),
    )
    annual_percentiles = monthly_percentiles.iloc[::12].copy()
    annual_percentiles.index = pd.Index(range(assumptions.horizon_years + 1), name="Year")
    terminal = paths[:, -1].copy()
    total_contributed = total_start + assumptions.monthly_contribution * months
    monthly_inflation = (1 + assumptions.annual_inflation) ** (1 / 12) - 1
    real_capital = total_start + sum(
        assumptions.monthly_contribution / (1 + monthly_inflation) ** month
        for month in range(1, months + 1)
    )
    real_terminal = terminal / (1 + monthly_inflation) ** months
    downside = terminal <= np.percentile(terminal, 25)
    contributions = holding_return_pnl[downside].mean(axis=0)
    downside_by_holding = dict(
        sorted(zip(symbols, contributions.tolist(), strict=True), key=lambda item: item[1])
    )
    benchmark_probability = None
    if request.benchmark_returns is not None:
        benchmark = request.benchmark_returns.reindex(returns.index).dropna()
        benchmark_annual = float(np.expm1(np.log1p(benchmark.clip(lower=-0.999999)).mean() * 12))
        benchmark_volatility = float(benchmark.std(ddof=1) * np.sqrt(12))
        benchmark_rng = np.random.default_rng(assumptions.seed + 1)
        benchmark_monthly_mean = np.expm1(np.log1p(benchmark_annual) / 12)
        benchmark_values = np.full(assumptions.simulations, total_start)
        for _ in range(months):
            benchmark_shock = benchmark_rng.normal(
                benchmark_monthly_mean,
                benchmark_volatility / np.sqrt(12),
                assumptions.simulations,
            )
            benchmark_values *= 1 + np.maximum(benchmark_shock, -0.999999)
            benchmark_values += assumptions.monthly_contribution
        benchmark_probability = float(np.mean(terminal > benchmark_values))
    status = SufficiencyStatus.LIMITED if proxies else SufficiencyStatus.GOOD
    coverage = {
        "status": status.value,
        "history_months": len(returns),
        "data_start": returns.index.min().date().isoformat(),
        "data_end": returns.index.max().date().isoformat(),
        "dataset_ids": list(request.dataset_ids),
        "symbols": list(symbols),
        "proxy_use": dict(proxies),
        "source_coverage": dict(request.source_coverage or {}),
    }
    limitations = (
        "Parametric distributions and stable correlations may not represent future regimes.",
        "Normal returns understate fat tails; Student's t is a sensitivity, not a guarantee.",
        "Simple returns are floored at -99.9999% because invested value cannot fall below zero.",
        "Taxes, withdrawals, and full retirement-account tax treatment are not modeled.",
        "Parameter-source weights express assumptions and do not establish forecast truth.",
    )
    disclosure = estimate.disclosure()
    return ParametricResult(
        PARAMETRIC_MODEL_TYPE,
        PARAMETRIC_MODEL_VERSION,
        monthly_percentiles,
        annual_percentiles,
        terminal,
        _distribution(terminal),
        (
            float(np.mean(terminal >= assumptions.target_value))
            if assumptions.target_value is not None
            else None
        ),
        {f"P{p}": float(np.percentile(terminal, p)) for p in (5, 10, 25)},
        float(np.mean(terminal < total_contributed)),
        float(np.mean(real_terminal < real_capital)),
        benchmark_probability,
        downside_by_holding,
        asdict(assumptions),
        coverage,
        dict(proxies),
        disclosure,
        disclosure["parameter_sources"],
        tuple(warnings),
        limitations,
    )


def parametric_sensitivity(
    request: ParametricRequest,
    *,
    return_shifts: Sequence[float] = (-0.02, 0.0, 0.02),
    volatility_multipliers: Sequence[float] = (0.8, 1.0, 1.2),
    correlation_multipliers: Sequence[float] = (0.5, 1.0),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for kind, values in (
        ("expected_return", return_shifts),
        ("volatility", volatility_multipliers),
        ("correlation", correlation_multipliers),
    ):
        for value in values:
            changes = {
                "expected_return_shift": request.assumptions.expected_return_shift,
                "volatility_multiplier": request.assumptions.volatility_multiplier,
                "correlation_multiplier": request.assumptions.correlation_multiplier,
            }
            changes[
                {
                    "expected_return": "expected_return_shift",
                    "volatility": "volatility_multiplier",
                    "correlation": "correlation_multiplier",
                }[kind]
            ] = float(value)
            result = run_parametric_forecast(
                replace(request, assumptions=replace(request.assumptions, **changes))
            )
            rows.append(
                {
                    "parameter": kind,
                    "value": float(value),
                    "p5_terminal": result.downside_percentiles["P5"],
                    "median_terminal": float(np.percentile(result.terminal_values, 50)),
                    "probability_nominal_loss": result.probability_nominal_loss,
                    "target_probability": result.target_probability,
                }
            )
    return pd.DataFrame(rows)


def normal_vs_fat_tail_comparison(
    request: ParametricRequest, *, degrees_of_freedom: float = 5
) -> pd.DataFrame:
    rows = []
    for distribution, degrees in (("normal", None), ("student_t", degrees_of_freedom)):
        result = run_parametric_forecast(
            replace(
                request,
                assumptions=replace(
                    request.assumptions,
                    distribution=distribution,
                    degrees_of_freedom=degrees,
                ),
            )
        )
        rows.append(
            {
                "distribution": distribution,
                "degrees_of_freedom": degrees,
                "p1_terminal": float(np.percentile(result.terminal_values, 1)),
                "p5_terminal": result.downside_percentiles["P5"],
                "p10_terminal": result.downside_percentiles["P10"],
                "probability_nominal_loss": result.probability_nominal_loss,
                "probability_real_loss": result.probability_real_loss,
            }
        )
    return pd.DataFrame(rows)


def model_comparison_table(
    bootstrap: BootstrapResult, parametric: ParametricResult
) -> pd.DataFrame:
    """Compare like-for-like core outputs without ranking either model."""
    rows = []
    for result in (bootstrap, parametric):
        rows.append(
            {
                "model": result.model_type,
                "model_version": result.model_version,
                "p5_terminal": result.downside_percentiles["P5"],
                "median_terminal": float(np.percentile(result.terminal_values, 50)),
                "p95_terminal": float(np.percentile(result.terminal_values, 95)),
                "target_probability": result.target_probability,
                "probability_nominal_loss": result.probability_nominal_loss,
                "probability_real_loss": result.probability_real_loss,
                "probability_beating_benchmark": result.probability_beating_benchmark,
                "data_status": result.data_coverage.get("status"),
            }
        )
    return pd.DataFrame(rows)


def calibration_comparison_table(
    bootstrap: BacktestResult, parametric: BacktestResult
) -> pd.DataFrame:
    """Compare calibration evidence without assigning a universally superior model."""
    rows = []
    for model, result in (
        ("historical_block_bootstrap", bootstrap),
        (PARAMETRIC_MODEL_TYPE, parametric),
    ):
        rows.append(
            {
                "model": model,
                "forecast_interval_coverage": result.forecast_interval_coverage,
                "median_forecast_bias": result.median_forecast_bias,
                "downside_band_performance": result.downside_band_performance,
                "valid_windows": result.valid_windows,
                "invalid_windows": result.invalid_windows,
                "insufficient_windows": result.insufficient_windows,
                "criteria_met": result.criteria_met,
                "validation_status": result.validation_status,
            }
        )
    return pd.DataFrame(rows)


def rolling_parametric_backtest(
    request: ParametricRequest, *, criteria: BacktestCriteria = BacktestCriteria()
) -> BacktestResult:
    symbols, values = _normalized_values(request.holding_values)
    estimate, frame, _, _ = estimate_parameters(request)
    del estimate
    horizon = request.assumptions.horizon_years * 12
    minimum = request.assumptions.minimum_history_months
    weights = values / values.sum()
    rows: list[dict[str, object]] = []
    invalid = 0
    insufficient = 0
    last_origin = len(frame) - horizon
    if last_origin <= minimum:
        insufficient = max(last_origin, 0) + 1
    else:
        for origin in range(minimum, last_origin + 1):
            training = frame.iloc[:origin]
            future = frame.iloc[origin : origin + horizon]
            if len(training.dropna()) < minimum or len(future.dropna()) < horizon:
                insufficient += 1
                continue
            try:
                forecast = run_parametric_forecast(
                    replace(
                        request,
                        monthly_returns=training,
                        benchmark_returns=None,
                        assumptions=replace(
                            request.assumptions, seed=request.assumptions.seed + origin
                        ),
                    )
                )
                actual = float(values.sum() + request.cash)
                for _, observed in future.iterrows():
                    actual *= 1 + float(np.dot(observed[list(symbols)], weights))
                    actual += request.assumptions.monthly_contribution
                p5 = forecast.downside_percentiles["P5"]
                p50 = float(np.percentile(forecast.terminal_values, 50))
                p95 = float(np.percentile(forecast.terminal_values, 95))
                tolerance = max(abs(actual), abs(p5), abs(p95), 1.0) * 1e-12
                rows.append(
                    {
                        "origin": frame.index[origin - 1],
                        "actual": actual,
                        "p5": p5,
                        "p50": p50,
                        "p95": p95,
                        "interval_covered": p5 - tolerance <= actual <= p95 + tolerance,
                        "downside_covered": actual >= p5 - tolerance,
                        "median_bias": (p50 - actual) / actual if actual else np.nan,
                    }
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                invalid += 1
    results = pd.DataFrame(rows)
    valid = len(results)
    interval = float(results.interval_covered.mean()) if valid else None
    bias = float(results.median_bias.median()) if valid else None
    downside = float(results.downside_covered.mean()) if valid else None
    met = bool(
        valid >= criteria.minimum_valid_windows
        and interval is not None
        and criteria.interval_coverage_min <= interval <= criteria.interval_coverage_max
        and bias is not None
        and abs(bias) <= criteria.maximum_absolute_median_bias
        and downside is not None
        and downside >= criteria.minimum_downside_coverage
    )
    return BacktestResult(
        interval,
        bias,
        downside,
        valid,
        invalid,
        insufficient,
        met,
        "eligible_for_review" if met else "unvalidated",
        results,
        criteria,
    )


def save_parametric_run(
    session: Session,
    portfolios: Sequence[Portfolio],
    result: ParametricResult,
    *,
    dataset_references: Sequence[DatasetReference] = (),
    backtest: BacktestResult | None = None,
    ledger_hash: str | None = None,
) -> ForecastRun:
    validation = session.scalar(
        select(ModelValidation).where(
            ModelValidation.model_type == result.model_type,
            ModelValidation.model_version == result.model_version,
        )
    )
    summary = result.summary()
    if backtest is not None:
        summary["backtest"] = backtest.summary()
    run = ForecastRun(
        model_type=result.model_type,
        model_version=result.model_version,
        portfolio_scope=json.dumps(
            [{"id": portfolio.id, "name": portfolio.name} for portfolio in portfolios],
            sort_keys=True,
        ),
        ledger_state_hash=ledger_hash or ledger_state_hash(session, [p.id for p in portfolios]),
        assumptions=json.dumps(dict(result.assumptions), sort_keys=True),
        data_coverage=json.dumps(dict(result.data_coverage), sort_keys=True),
        proxy_use=json.dumps(dict(result.proxies), sort_keys=True),
        status="completed",
        validation_status=validation.status if validation else "unvalidated",
        summary_outputs=json.dumps(summary, sort_keys=True),
        scenario_bands=json.dumps(
            {
                "monthly": result.monthly_percentiles.to_dict(orient="index"),
                "annual": result.annual_percentiles.to_dict(orient="index"),
            },
            sort_keys=True,
        ),
    )
    session.add(run)
    session.flush()
    references = dataset_references or tuple(
        DatasetReference(dataset_id, "parametric_estimation")
        for dataset_id in result.data_coverage.get("dataset_ids", [])
    )
    for reference in references:
        session.add(
            ForecastRunDataset(
                forecast_run_id=run.id,
                dataset_id=reference.dataset_id,
                purpose=reference.purpose,
            )
        )
    session.flush()
    return run
