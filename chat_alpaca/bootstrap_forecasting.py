from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_alpaca.historical_data import HistoricalCoverageResult
from chat_alpaca.models import ForecastRun, ForecastRunDataset, ModelValidation, Portfolio
from chat_alpaca.reconstruction import ReconstructionResult, SufficiencyStatus
from chat_alpaca.scenarios import DatasetReference, ledger_state_hash

BOOTSTRAP_MODEL_TYPE = "historical_block_bootstrap"
BOOTSTRAP_MODEL_VERSION = "1.0.0"
BOOTSTRAP_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
REBALANCE_FREQUENCIES = {"monthly": 1, "quarterly": 3, "annual": 12, "never": None}


@dataclass(frozen=True)
class BootstrapAssumptions:
    horizon_years: int
    block_length: Literal[3, 6, 12] = 6
    simulations: int = 10_000
    seed: int = 20260719
    monthly_contribution: float = 0.0
    annual_inflation: float = 0.025
    annual_fee: float = 0.0
    rebalancing: Literal["monthly", "quarterly", "annual", "never"] = "monthly"
    sampling_level: Literal["holding", "portfolio"] = "holding"
    target_value: float | None = None
    minimum_history_months: int = 24

    def __post_init__(self) -> None:
        if not 1 <= self.horizon_years <= 10:
            raise ValueError("Forecast horizon must be between 1 and 10 years.")
        if self.block_length not in {3, 6, 12}:
            raise ValueError("Block length must be 3, 6, or 12 months.")
        if self.simulations < 1:
            raise ValueError("At least one simulation is required.")
        if self.monthly_contribution < 0:
            raise ValueError("Monthly contribution cannot be negative.")
        if self.annual_inflation <= -1:
            raise ValueError("Annual inflation must be greater than -100%.")
        if self.annual_fee < 0 or self.annual_fee >= 1:
            raise ValueError("Annual fee must be between 0% and 100%.")
        if self.rebalancing not in REBALANCE_FREQUENCIES:
            raise ValueError("Rebalancing must be monthly, quarterly, annual, or never.")
        if self.sampling_level not in {"holding", "portfolio"}:
            raise ValueError("Sampling level must be holding or portfolio.")
        if self.target_value is not None and self.target_value <= 0:
            raise ValueError("Target value must be greater than zero.")
        if self.minimum_history_months < self.block_length:
            raise ValueError("Minimum history cannot be shorter than the selected block.")


@dataclass(frozen=True)
class BootstrapRequest:
    holding_values: Mapping[str, float]
    monthly_returns: pd.DataFrame
    assumptions: BootstrapAssumptions
    cash: float = 0.0
    benchmark_returns: pd.Series | None = None
    proxies: Mapping[str, str] | None = None
    dataset_ids: tuple[int, ...] = ()
    source_coverage: Mapping[str, object] | None = None


@dataclass(frozen=True)
class BootstrapResult:
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
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class BacktestCriteria:
    minimum_valid_windows: int = 12
    interval_coverage_min: float = 0.80
    interval_coverage_max: float = 1.0
    maximum_absolute_median_bias: float = 0.10
    minimum_downside_coverage: float = 0.90


@dataclass(frozen=True)
class BacktestResult:
    forecast_interval_coverage: float | None
    median_forecast_bias: float | None
    downside_band_performance: float | None
    valid_windows: int
    invalid_windows: int
    insufficient_windows: int
    criteria_met: bool
    validation_status: str
    window_results: pd.DataFrame
    criteria: BacktestCriteria

    def summary(self) -> dict[str, object]:
        return {
            "forecast_interval_coverage": self.forecast_interval_coverage,
            "median_forecast_bias": self.median_forecast_bias,
            "downside_band_performance": self.downside_band_performance,
            "valid_windows": self.valid_windows,
            "invalid_windows": self.invalid_windows,
            "insufficient_windows": self.insufficient_windows,
            "criteria_met": self.criteria_met,
            "validation_status": self.validation_status,
            "criteria": asdict(self.criteria),
        }


def monthly_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert confirmed daily closes to month-end simple returns without filling gaps."""
    if prices.empty:
        return pd.DataFrame(columns=prices.columns, dtype=float)
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    monthly = frame.resample("ME").last()
    return monthly.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).iloc[1:]


def sample_block_indices(
    history_length: int,
    horizon_months: int,
    block_length: int,
    simulations: int,
    seed: int,
) -> np.ndarray:
    """Sample circular contiguous blocks; columns within each block remain adjacent."""
    if history_length < block_length:
        raise ValueError("History is shorter than the selected block length.")
    if horizon_months < 1 or simulations < 1:
        raise ValueError("Horizon and simulation count must be positive.")
    rng = np.random.default_rng(seed)
    blocks = (horizon_months + block_length - 1) // block_length
    starts = rng.integers(0, history_length, size=(simulations, blocks))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets) % history_length
    return indices.reshape(simulations, blocks * block_length)[:, :horizon_months]


def _normalized_values(values: Mapping[str, float]) -> tuple[tuple[str, ...], np.ndarray]:
    normalized = {str(symbol).strip().upper(): float(value) for symbol, value in values.items()}
    if not normalized or any(value < 0 for value in normalized.values()):
        raise ValueError("Holding values must include nonnegative values for at least one symbol.")
    symbols = tuple(sorted(symbol for symbol, value in normalized.items() if value > 0))
    amounts = np.array([normalized[symbol] for symbol in symbols], dtype=float)
    if not symbols or amounts.sum() <= 0:
        raise ValueError("At least one holding must have a positive value.")
    return symbols, amounts


def _prepare_returns(
    request: BootstrapRequest, symbols: Sequence[str]
) -> tuple[pd.DataFrame, pd.Series | None, dict[str, str], list[str]]:
    frame = request.monthly_returns.copy()
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index().replace([np.inf, -np.inf], np.nan)
    configured_proxies = {
        str(symbol).strip().upper(): str(proxy).strip().upper()
        for symbol, proxy in (request.proxies or {}).items()
    }
    warnings: list[str] = []
    used_proxies: dict[str, str] = {}
    prepared: dict[str, pd.Series] = {}
    minimum = request.assumptions.minimum_history_months
    for symbol in symbols:
        direct = frame[symbol] if symbol in frame else pd.Series(dtype=float)
        if int(direct.notna().sum()) >= minimum:
            prepared[symbol] = direct
            continue
        proxy = configured_proxies.get(symbol)
        if not proxy:
            raise ValueError(
                f"Insufficient monthly history for {symbol}; an explicit proxy is required."
            )
        if proxy not in frame or int(frame[proxy].notna().sum()) < minimum:
            raise ValueError(f"Proxy {proxy} for {symbol} also has insufficient monthly history.")
        prepared[symbol] = frame[proxy]
        used_proxies[symbol] = proxy
        warnings.append(f"{symbol} uses explicit return-history proxy {proxy}.")
    aligned = pd.DataFrame(prepared).dropna(how="any")
    if len(aligned) < minimum:
        raise ValueError(
            "Insufficient overlapping monthly history after aligning holdings and proxies."
        )
    benchmark = None
    if request.benchmark_returns is not None:
        benchmark = request.benchmark_returns.copy()
        benchmark.index = pd.to_datetime(benchmark.index)
        benchmark = benchmark.replace([np.inf, -np.inf], np.nan).rename("__BENCHMARK__")
        joint = aligned.join(benchmark, how="inner").dropna()
        if len(joint) < minimum:
            raise ValueError("Insufficient overlapping portfolio and benchmark history.")
        aligned = joint[list(symbols)]
        benchmark = joint["__BENCHMARK__"]
    return aligned, benchmark, used_proxies, warnings


def _distribution(values: np.ndarray) -> dict[str, object]:
    counts, edges = np.histogram(values, bins=min(20, max(5, round(np.sqrt(len(values))))))
    return {
        "count": int(len(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=0)),
        "histogram_counts": counts.astype(int).tolist(),
        "histogram_edges": edges.astype(float).tolist(),
    }


def run_block_bootstrap(request: BootstrapRequest) -> BootstrapResult:
    """Bootstrap observed joint monthly returns; no expected return is imposed."""
    assumptions = request.assumptions
    symbols, starting_holdings = _normalized_values(request.holding_values)
    if request.cash < 0:
        raise ValueError("Cash cannot be negative.")
    returns, benchmark, proxies, warnings = _prepare_returns(request, symbols)
    months = assumptions.horizon_years * 12
    indices = sample_block_indices(
        len(returns), months, assumptions.block_length, assumptions.simulations, assumptions.seed
    )
    sampled = returns.to_numpy(dtype=float)[indices]
    target_weights = starting_holdings / starting_holdings.sum()
    holdings = np.broadcast_to(starting_holdings, (assumptions.simulations, len(symbols))).copy()
    total_start = float(starting_holdings.sum() + request.cash)
    if total_start <= 0:
        raise ValueError("The forecast requires a positive starting value.")
    cash = np.full(assumptions.simulations, float(request.cash))
    paths = np.empty((assumptions.simulations, months + 1), dtype=float)
    paths[:, 0] = total_start
    holding_return_pnl = np.zeros_like(holdings)
    monthly_fee = 1 - (1 - assumptions.annual_fee) ** (1 / 12)
    rebalance_every = REBALANCE_FREQUENCIES[assumptions.rebalancing]

    if assumptions.sampling_level == "portfolio":
        portfolio_sampled = np.einsum("smh,h->sm", sampled, target_weights)
        sampled = np.repeat(portfolio_sampled[:, :, None], len(symbols), axis=2)

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
    real_capital = total_start
    for month in range(1, months + 1):
        real_capital += assumptions.monthly_contribution / (1 + monthly_inflation) ** month
    real_terminal = terminal / (1 + monthly_inflation) ** months

    benchmark_probability = None
    if benchmark is not None:
        benchmark_sampled = benchmark.to_numpy(dtype=float)[indices]
        benchmark_values = np.full(assumptions.simulations, total_start)
        for month in range(months):
            benchmark_values *= 1 + benchmark_sampled[:, month]
            benchmark_values += assumptions.monthly_contribution
        benchmark_probability = float(np.mean(terminal > benchmark_values))

    downside_cutoff = np.percentile(terminal, 25)
    downside = terminal <= downside_cutoff
    contributions = holding_return_pnl[downside].mean(axis=0)
    downside_by_holding = dict(
        sorted(zip(symbols, contributions, strict=True), key=lambda item: item[1])
    )
    downside_percentiles = {
        f"P{percentile}": float(np.percentile(terminal, percentile)) for percentile in (5, 10, 25)
    }
    effective_proxies = dict(proxies)
    status = SufficiencyStatus.LIMITED if effective_proxies else SufficiencyStatus.GOOD
    coverage = {
        "status": status.value,
        "history_months": len(returns),
        "data_start": returns.index.min().date().isoformat(),
        "data_end": returns.index.max().date().isoformat(),
        "dataset_ids": list(request.dataset_ids),
        "symbols": list(symbols),
        "proxy_use": effective_proxies,
        "source_coverage": dict(request.source_coverage or {}),
    }
    limitations = (
        "Observed history is sampled without an imposed expected-return assumption.",
        "Circular monthly block bootstrap samples jointly aligned contiguous 3-, 6-, or "
        "12-month blocks and wraps from the end of observed history to its beginning; this "
        "retained mechanic preserves within-block order and cross-asset row dependence.",
        "Historical blocks may not represent future regimes or unprecedented events.",
        "Taxes, withdrawals, and full retirement-account tax treatment are not modeled.",
        "Nominal and real loss compare terminal value with contributed capital; real values use the configured constant inflation rate.",
        "Downside holding contributions are average cumulative return P&L in terminal outcomes at or below the 25th percentile.",
    )
    return BootstrapResult(
        BOOTSTRAP_MODEL_TYPE,
        BOOTSTRAP_MODEL_VERSION,
        monthly_percentiles,
        annual_percentiles,
        terminal,
        _distribution(terminal),
        (
            float(np.mean(terminal >= assumptions.target_value))
            if assumptions.target_value is not None
            else None
        ),
        downside_percentiles,
        float(np.mean(terminal < total_contributed)),
        float(np.mean(real_terminal < real_capital)),
        benchmark_probability,
        downside_by_holding,
        asdict(assumptions),
        coverage,
        effective_proxies,
        tuple(warnings),
        limitations,
    )


def build_bootstrap_request(
    reconstruction: ReconstructionResult,
    market_coverage: HistoricalCoverageResult,
    assumptions: BootstrapAssumptions,
    *,
    proxies: Mapping[str, str] | None = None,
    benchmark_returns: pd.Series | None = None,
) -> BootstrapRequest:
    """Create a bootstrap request from shared reconstruction and market-data results."""
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
    cash = float(reconstruction.combined.cash.loc[pd.Timestamp(as_of)])
    return BootstrapRequest(
        holding_values=holding_values,
        cash=cash,
        monthly_returns=monthly_returns_from_prices(market_coverage.data),
        benchmark_returns=benchmark_returns,
        proxies=proxies,
        dataset_ids=tuple(market_coverage.data.attrs.get("dataset_ids", ())),
        source_coverage={
            "reconstruction_status": reconstruction.data_coverage.status.value,
            "reconstruction_score": reconstruction.data_coverage.score,
            "adjustment": market_coverage.adjustment,
            "sources": reconstruction.data_coverage.sources,
        },
        assumptions=assumptions,
    )


def rolling_origin_backtest(
    request: BootstrapRequest,
    *,
    criteria: BacktestCriteria = BacktestCriteria(),
) -> BacktestResult:
    """Run expanding-window forecasts against later realized portfolio outcomes."""
    symbols, values = _normalized_values(request.holding_values)
    frame, _, _, _ = _prepare_returns(request, symbols)
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
                forecast = run_block_bootstrap(
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
            except (ValueError, FloatingPointError):
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


def save_bootstrap_run(
    session: Session,
    portfolios: Sequence[Portfolio],
    result: BootstrapResult,
    *,
    dataset_references: Sequence[DatasetReference] = (),
    backtest: BacktestResult | None = None,
    ledger_hash: str | None = None,
) -> ForecastRun:
    """Persist reproducibility inputs and summaries, excluding terminal values and paths."""
    validation = session.scalar(
        select(ModelValidation).where(
            ModelValidation.model_type == result.model_type,
            ModelValidation.model_version == result.model_version,
        )
    )
    summary = result.summary()
    if backtest is not None:
        summary["backtest"] = backtest.summary()
    coverage = dict(result.data_coverage)
    run = ForecastRun(
        model_type=result.model_type,
        model_version=result.model_version,
        portfolio_scope=json.dumps(
            [{"id": portfolio.id, "name": portfolio.name} for portfolio in portfolios],
            sort_keys=True,
        ),
        ledger_state_hash=ledger_hash or ledger_state_hash(session, [p.id for p in portfolios]),
        assumptions=json.dumps(dict(result.assumptions), sort_keys=True),
        data_coverage=json.dumps(coverage, sort_keys=True),
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
        DatasetReference(dataset_id, "bootstrap_returns")
        for dataset_id in coverage.get("dataset_ids", [])
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
