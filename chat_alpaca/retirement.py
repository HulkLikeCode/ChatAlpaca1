from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from numbers import Integral, Real
from typing import Literal

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_alpaca.bootstrap_forecasting import (
    BOOTSTRAP_PERCENTILES,
    REBALANCE_FREQUENCIES,
    BootstrapAssumptions,
    BootstrapRequest,
    _distribution,
    _normalized_values,
    _prepare_returns,
    monthly_returns_from_prices,
    sample_block_indices,
)
from chat_alpaca.historical_data import HistoricalCoverageResult
from chat_alpaca.models import ForecastRun, ForecastRunDataset, ModelValidation, Portfolio
from chat_alpaca.parametric_forecasting import (
    CapitalMarketAssumption,
    ParametricAssumptions,
    ParametricRequest,
    _draw_returns,
    estimate_parameters,
)
from chat_alpaca.reconstruction import ReconstructionResult
from chat_alpaca.rmd_tables import RMD_TABLE_VERSION, lifetime_divisor
from chat_alpaca.scenarios import DatasetReference, ledger_state_hash

RETIREMENT_MODEL_TYPE = "long_horizon_retirement"
RETIREMENT_MODEL_VERSION = "1.3.0"
ACCOUNT_TYPES = ("traditional_ira", "roth_ira", "taxable", "unknown")
UNKNOWN_ACCOUNT_ERROR = (
    "Retirement withdrawal and tax calculations require every in-scope account to be classified "
    "as taxable, Traditional IRA, or Roth IRA."
)
SPENDING_EVENT_TIMING_DISCLOSURE = (
    "One-time spending events are assigned to the nearest monthly model step; intra-month timing "
    "is not modeled."
)
RETAINED_CASH_DISCLOSURE = (
    "Retained net outside-income and RMD surplus remains zero-return household cash until the "
    "next configured rebalance, but remains available to fund subsequent spending before account "
    "withdrawals."
)


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number and cannot be Boolean.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _integer(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer and cannot be Boolean.")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return normalized


@dataclass(frozen=True)
class RetirementProfile:
    current_age: float
    planning_horizon_years: int
    fixed_real_annual_spending: float
    planned_retirement_age: float | None = None
    planned_retirement_date: date | None = None
    as_of_date: date = date(2026, 7, 20)
    annual_inflation: float = 0.025
    contribution_amount: float = 0.0
    contribution_frequency: Literal["monthly", "annual"] = "monthly"
    target_estate_value: float | None = None
    owner_date_of_birth: date | None = None
    spouse_date_of_birth: date | None = None
    spouse_is_sole_beneficiary: bool = False

    def __post_init__(self) -> None:
        current_age = _finite_number("Current age", self.current_age)
        horizon = _integer("Retirement planning horizon", self.planning_horizon_years)
        spending = _finite_number("Fixed real annual spending", self.fixed_real_annual_spending)
        inflation = _finite_number("Annual inflation", self.annual_inflation)
        contribution = _finite_number("Contribution amount", self.contribution_amount)
        if current_age < 0:
            raise ValueError("Current age cannot be negative.")
        if not 20 <= horizon <= 40:
            raise ValueError("Retirement planning horizon must be between 20 and 40 years.")
        if (self.planned_retirement_age is None) == (self.planned_retirement_date is None):
            raise ValueError("Provide either a planned retirement age or date, but not both.")
        if self.planned_retirement_age is not None:
            retirement_age = _finite_number("Planned retirement age", self.planned_retirement_age)
            if retirement_age < current_age:
                raise ValueError("Planned retirement age cannot precede current age.")
        if (
            self.planned_retirement_date is not None
            and self.planned_retirement_date < self.as_of_date
        ):
            raise ValueError("Planned retirement date cannot precede the as-of date.")
        if spending < 0 or contribution < 0:
            raise ValueError("Spending and contributions cannot be negative.")
        if inflation <= -1:
            raise ValueError("Inflation must be greater than -100%.")
        if self.contribution_frequency not in {"monthly", "annual"}:
            raise ValueError("Contribution frequency must be monthly or annual.")
        if self.target_estate_value is not None:
            target = _finite_number("Target estate value", self.target_estate_value)
            if target < 0:
                raise ValueError("Target estate value cannot be negative.")
        if self.spouse_is_sole_beneficiary and self.spouse_date_of_birth is None:
            raise ValueError("A sole-beneficiary spouse requires the spouse's date of birth.")

    @property
    def retirement_month(self) -> int:
        if self.planned_retirement_age is not None:
            months = round((self.planned_retirement_age - self.current_age) * 12)
        else:
            assert self.planned_retirement_date is not None
            months = (self.planned_retirement_date.year - self.as_of_date.year) * 12
            months += self.planned_retirement_date.month - self.as_of_date.month
        if not 0 <= months <= self.planning_horizon_years * 12:
            raise ValueError("Retirement must fall within the planning horizon.")
        return months


@dataclass(frozen=True)
class RetirementAccount:
    name: str
    account_type: Literal["traditional_ira", "roth_ira", "taxable", "unknown"]
    balance: float
    taxable_cost_basis: float | None = None
    prior_december_31_balance: float | None = None
    allocation: Mapping[str, float] | None = None
    is_inherited: bool = False

    def __post_init__(self) -> None:
        if self.account_type not in ACCOUNT_TYPES:
            raise ValueError(f"Unsupported account type: {self.account_type}.")
        balance = _finite_number("Account balance", self.balance)
        if balance < 0:
            raise ValueError("Account balances cannot be negative.")
        if self.taxable_cost_basis is not None:
            if self.account_type != "taxable":
                raise ValueError("Taxable cost basis applies only to taxable accounts.")
            if _finite_number("Taxable cost basis", self.taxable_cost_basis) < 0:
                raise ValueError("Taxable cost basis cannot be negative.")
        if self.prior_december_31_balance is not None:
            prior_balance = _finite_number(
                "Prior December 31 balance", self.prior_december_31_balance
            )
            if prior_balance < 0:
                raise ValueError("Prior December 31 balance cannot be negative.")
        if self.allocation is not None:
            weights = [
                _finite_number(f"Account allocation weight for {symbol}", value)
                for symbol, value in self.allocation.items()
            ]
            if not weights or any(value < 0 for value in weights):
                raise ValueError("Account allocation weights must be nonnegative and nonempty.")


@dataclass(frozen=True)
class OutsideIncome:
    name: str
    kind: Literal["social_security", "pension", "other"]
    annual_real_amount: float
    start_age: float | None = None
    start_date: date | None = None
    end_age: float | None = None
    annual_cola: float | None = None
    taxable_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"social_security", "pension", "other"}:
            raise ValueError("Outside income kind must be social_security, pension, or other.")
        amount = _finite_number("Outside income annual real amount", self.annual_real_amount)
        if amount < 0:
            raise ValueError("Outside income cannot be negative.")
        if (self.start_age is None) == (self.start_date is None):
            raise ValueError("Provide either an outside-income start age or date, but not both.")
        if self.start_age is not None:
            start_age = _finite_number("Outside-income start age", self.start_age)
            if start_age < 0:
                raise ValueError("Outside-income start age cannot be negative.")
        if self.end_age is not None:
            end_age = _finite_number("Outside-income end age", self.end_age)
        else:
            end_age = None
        if self.end_age is not None and self.start_age is not None and end_age < self.start_age:
            raise ValueError("Outside-income end age cannot precede its start age.")
        if self.annual_cola is not None:
            cola = _finite_number("Outside-income COLA", self.annual_cola)
            if cola <= -1:
                raise ValueError("Outside-income COLA must be greater than -100%.")
        if self.taxable_fraction is not None:
            taxable_fraction = _finite_number(
                "Outside-income taxable fraction", self.taxable_fraction
            )
            if not 0 <= taxable_fraction <= 1:
                raise ValueError("Outside-income taxable fraction must be between zero and one.")


@dataclass(frozen=True)
class SpendingEvent:
    name: str
    real_amount: float
    age: float

    def __post_init__(self) -> None:
        amount = _finite_number("One-time spending amount", self.real_amount)
        age = _finite_number("One-time spending age", self.age)
        if amount < 0 or age < 0:
            raise ValueError("One-time spending amount and age cannot be negative.")

    def resolved_model_month(self, profile: RetirementProfile) -> int:
        month = round((self.age - profile.current_age) * 12)
        if not 0 <= month < profile.planning_horizon_years * 12:
            raise ValueError(
                f"One-time spending event {self.name} must fall within the planning horizon."
            )
        return month


@dataclass(frozen=True)
class RetirementTaxAssumptions:
    ordinary_income_rate: float = 0.22
    capital_gains_rate: float = 0.15
    dividend_tax_rate: float = 0.15
    qualified_dividend_yield: float = 0.0
    social_security_taxable_fraction: float = 0.85
    taxable_realization_fraction: float = 0.50
    unknown_account_withdrawal_rate: float = 0.22

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            rate = _finite_number(name.replace("_", " ").title(), value)
            if not 0 <= rate <= 1:
                raise ValueError(f"{name.replace('_', ' ').title()} must be between zero and one.")


@dataclass(frozen=True)
class RetirementAssumptions:
    engine: Literal["historical_block_bootstrap", "parametric"] = "historical_block_bootstrap"
    simulations: int = 10_000
    seed: int = 20260720
    block_length: Literal[3, 6, 12] = 6
    minimum_history_months: int = 24
    annual_fee: float = 0.0
    rebalancing: Literal["monthly", "quarterly", "annual", "never"] = "annual"
    withdrawal_order: tuple[str, ...] = (
        "taxable",
        "traditional_ira",
        "roth_ira",
    )
    distribution: Literal["normal", "student_t"] = "normal"
    degrees_of_freedom: float | None = None
    parameter_uncertainty: bool = True
    expected_return_shift: float = 0.0
    volatility_multiplier: float = 1.0
    correlation_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.engine not in {"historical_block_bootstrap", "parametric"}:
            raise ValueError("Retirement engine must be historical_block_bootstrap or parametric.")
        _integer("Retirement simulation count", self.simulations)
        _integer("Retirement seed", self.seed, minimum=0)
        block_length = _integer("Retirement block length", self.block_length)
        minimum_history = _integer("Retirement minimum history", self.minimum_history_months)
        annual_fee = _finite_number("Retirement annual fee", self.annual_fee)
        expected_shift = _finite_number(
            "Retirement expected-return shift", self.expected_return_shift
        )
        volatility_multiplier = _finite_number(
            "Retirement volatility multiplier", self.volatility_multiplier
        )
        correlation_multiplier = _finite_number(
            "Retirement correlation multiplier", self.correlation_multiplier
        )
        if block_length not in {3, 6, 12}:
            raise ValueError("Block length must be 3, 6, or 12 months.")
        if minimum_history < block_length:
            raise ValueError("Minimum history cannot be shorter than the selected block.")
        if not 0 <= annual_fee < 1:
            raise ValueError("Annual fee must be between zero and one.")
        if self.rebalancing not in REBALANCE_FREQUENCIES:
            raise ValueError("Unsupported rebalancing frequency.")
        if not self.withdrawal_order or len(set(self.withdrawal_order)) != len(
            self.withdrawal_order
        ):
            raise ValueError("Withdrawal order must contain unique account types.")
        if any(item not in ACCOUNT_TYPES for item in self.withdrawal_order):
            raise ValueError("Withdrawal order contains an unsupported account type.")
        if self.distribution == "student_t":
            if (
                self.degrees_of_freedom is None
                or _finite_number("Student's t degrees of freedom", self.degrees_of_freedom) <= 2
            ):
                raise ValueError("Student's t degrees of freedom must be greater than two.")
        elif self.degrees_of_freedom is not None:
            raise ValueError("Degrees of freedom apply only to Student's t returns.")
        if expected_shift <= -1:
            raise ValueError("Expected-return shift must be greater than -100%.")
        if volatility_multiplier <= 0 or not 0 <= correlation_multiplier <= 1:
            raise ValueError("Volatility must be positive and correlation multiplier 0–1.")


@dataclass(frozen=True)
class RetirementRequest:
    profile: RetirementProfile
    accounts: tuple[RetirementAccount, ...]
    holding_values: Mapping[str, float]
    monthly_returns: pd.DataFrame
    assumptions: RetirementAssumptions = RetirementAssumptions()
    tax_assumptions: RetirementTaxAssumptions = RetirementTaxAssumptions()
    outside_income: tuple[OutsideIncome, ...] = ()
    spending_events: tuple[SpendingEvent, ...] = ()
    contribution_allocation: Mapping[str, float] | None = None
    proxies: Mapping[str, str] | None = None
    external_assumptions: Mapping[str, CapitalMarketAssumption | Mapping[str, object]] | None = None
    user_overrides: Mapping[str, CapitalMarketAssumption | Mapping[str, object]] | None = None
    dataset_ids: tuple[int, ...] = ()
    source_coverage: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        for symbol, value in self.holding_values.items():
            _finite_number(f"Retirement holding value for {symbol}", value)
        if self.contribution_allocation is not None:
            for account_type, value in self.contribution_allocation.items():
                weight = _finite_number(f"Contribution allocation weight for {account_type}", value)
                if weight < 0:
                    raise ValueError("Contribution allocation weights cannot be negative.")
        for event in self.spending_events:
            event.resolved_model_month(self.profile)


@dataclass(frozen=True)
class RetirementResult:
    model_type: str
    model_version: str
    nominal_monthly_percentiles: pd.DataFrame
    real_monthly_percentiles: pd.DataFrame
    nominal_annual_percentiles: pd.DataFrame
    real_annual_percentiles: pd.DataFrame
    terminal_values: np.ndarray
    real_terminal_values: np.ndarray
    scenario_shortfalls: np.ndarray
    scenario_depletion_ages: np.ndarray
    probability_funding_full_horizon: float
    probability_depletion: float
    depletion_age_distribution: Mapping[str, float | None]
    retirement_date_value_distribution: Mapping[str, object]
    lifetime_taxes_estimate: Mapping[str, float]
    withdrawals_by_account_type: Mapping[str, Mapping[str, float]]
    outside_income_contribution: Mapping[str, float]
    retained_household_cash: Mapping[str, float]
    rmd_withdrawals: Mapping[str, float]
    cash_flow_reconciliation: Mapping[str, float]
    shortfall_reconciliation: Mapping[str, float]
    shortfall_distribution: Mapping[str, object]
    target_estate_probability: float | None
    sequence_risk_diagnostics: Mapping[str, object]
    worst_decile_scenarios: pd.DataFrame
    assumptions: Mapping[str, object]
    data_coverage: Mapping[str, object]
    proxies: Mapping[str, str]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

    def summary(self) -> dict[str, object]:
        return {
            "probability_funding_full_horizon": self.probability_funding_full_horizon,
            "probability_depletion": self.probability_depletion,
            "depletion_age_distribution": dict(self.depletion_age_distribution),
            "retirement_date_value_distribution": dict(self.retirement_date_value_distribution),
            "lifetime_taxes_estimate": dict(self.lifetime_taxes_estimate),
            "withdrawals_by_account_type": {
                key: dict(value) for key, value in self.withdrawals_by_account_type.items()
            },
            "outside_income_contribution": dict(self.outside_income_contribution),
            "retained_household_cash": dict(self.retained_household_cash),
            "rmd_withdrawals": dict(self.rmd_withdrawals),
            "cash_flow_reconciliation": dict(self.cash_flow_reconciliation),
            "shortfall_reconciliation": dict(self.shortfall_reconciliation),
            "shortfall_distribution": dict(self.shortfall_distribution),
            "target_estate_probability": self.target_estate_probability,
            "sequence_risk_diagnostics": dict(self.sequence_risk_diagnostics),
            "worst_decile_scenarios": self.worst_decile_scenarios.to_dict(orient="records"),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class HistoricalReplayResult:
    valid_windows: int
    insufficient_windows: int
    probability_funding_full_horizon: float | None
    probability_depletion: float | None
    window_results: pd.DataFrame
    validation_status: str = "unvalidated"

    def summary(self) -> dict[str, object]:
        return {
            "valid_windows": self.valid_windows,
            "insufficient_windows": self.insufficient_windows,
            "probability_funding_full_horizon": self.probability_funding_full_horizon,
            "probability_depletion": self.probability_depletion,
            "validation_status": self.validation_status,
        }


def _return_paths(
    request: RetirementRequest,
) -> tuple[np.ndarray, tuple[str, ...], pd.DataFrame, dict[str, str], list[str]]:
    symbols, _ = _normalized_values(request.holding_values)
    common = BootstrapRequest(
        request.holding_values,
        request.monthly_returns,
        BootstrapAssumptions(
            1,
            block_length=request.assumptions.block_length,
            minimum_history_months=request.assumptions.minimum_history_months,
        ),
        proxies=request.proxies,
    )
    frame, _, proxies, warnings = _prepare_returns(common, symbols)
    months = request.profile.planning_horizon_years * 12
    if request.assumptions.engine == "historical_block_bootstrap":
        indices = sample_block_indices(
            len(frame),
            months,
            request.assumptions.block_length,
            request.assumptions.simulations,
            request.assumptions.seed,
        )
        sampled = frame.to_numpy(dtype=float)[indices]
        if request.assumptions.expected_return_shift:
            if request.assumptions.expected_return_shift <= -1:
                raise ValueError("Expected-return shift must be greater than -100%.")
            monthly_shift = np.expm1(np.log1p(request.assumptions.expected_return_shift) / 12)
            sampled = np.maximum(sampled + monthly_shift, -0.999999)
        return sampled, symbols, frame, proxies, warnings
    parametric = ParametricRequest(
        request.holding_values,
        request.monthly_returns,
        ParametricAssumptions(
            1,
            distribution=request.assumptions.distribution,
            degrees_of_freedom=request.assumptions.degrees_of_freedom,
            simulations=request.assumptions.simulations,
            seed=request.assumptions.seed,
            minimum_history_months=request.assumptions.minimum_history_months,
            parameter_uncertainty=request.assumptions.parameter_uncertainty,
            expected_return_shift=request.assumptions.expected_return_shift,
            volatility_multiplier=request.assumptions.volatility_multiplier,
            correlation_multiplier=request.assumptions.correlation_multiplier,
        ),
        proxies=request.proxies,
        external_assumptions=request.external_assumptions,
        user_overrides=request.user_overrides,
    )
    estimate, frame, proxies, warnings = estimate_parameters(parametric)
    return (
        _draw_returns(estimate, parametric.assumptions, months),
        symbols,
        frame,
        proxies,
        warnings,
    )


def _percentile_frame(paths: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        np.percentile(paths, BOOTSTRAP_PERCENTILES, axis=0).T,
        columns=[f"P{value}" for value in BOOTSTRAP_PERCENTILES],
        index=pd.RangeIndex(paths.shape[1], name="Month"),
    )


def _distribution_percentiles(values: np.ndarray) -> dict[str, float]:
    return {f"P{p}": float(np.percentile(values, p)) for p in (5, 10, 25, 50, 75, 90, 95)}


def _contribution_weights(request: RetirementRequest) -> np.ndarray:
    accounts = request.accounts
    configured = dict(request.contribution_allocation or {})
    if configured:
        if any(key not in ACCOUNT_TYPES for key in configured) or any(
            value < 0 for value in configured.values()
        ):
            raise ValueError("Contribution allocation contains an invalid type or weight.")
        total = sum(configured.values())
        if total <= 0:
            raise ValueError("Contribution allocation must have a positive total weight.")
        type_weights = {key: value / total for key, value in configured.items()}
    else:
        balances = {kind: 0.0 for kind in ACCOUNT_TYPES}
        for account in accounts:
            balances[account.account_type] += account.balance
        total = sum(balances.values())
        type_weights = {kind: value / total for kind, value in balances.items()} if total else {}
    weights = np.zeros(len(accounts))
    for kind, type_weight in type_weights.items():
        indices = [index for index, account in enumerate(accounts) if account.account_type == kind]
        if not indices and type_weight > 0:
            raise ValueError(f"Contribution allocation references missing account type {kind}.")
        denominator = sum(accounts[index].balance for index in indices)
        for index in indices:
            weights[index] = (
                type_weight * accounts[index].balance / denominator
                if denominator > 0
                else type_weight / len(indices)
            )
    return weights


def _income_start_month(profile: RetirementProfile, income: OutsideIncome) -> int:
    if income.start_age is not None:
        return round((income.start_age - profile.current_age) * 12)
    assert income.start_date is not None
    months = (income.start_date.year - profile.as_of_date.year) * 12
    return months + income.start_date.month - profile.as_of_date.month


def applicable_rmd_start_age(date_of_birth: date) -> float:
    """Return the owner RMD start age selected by date of birth under current federal law."""
    if date_of_birth < date(1949, 7, 1):
        return 70.5
    if date_of_birth < date(1951, 1, 1):
        return 72
    if date_of_birth < date(1960, 1, 1):
        return 73
    return 75


def first_rmd_year(date_of_birth: date) -> int:
    start_age = applicable_rmd_start_age(date_of_birth)
    if start_age != 70.5:
        return date_of_birth.year + int(start_age)
    month_index = date_of_birth.month - 1 + 6
    return date_of_birth.year + 70 + month_index // 12


def _age_in_year(date_of_birth: date, year: int) -> int:
    return year - date_of_birth.year


def owner_rmd_divisor(profile: RetirementProfile, year: int) -> tuple[float, str]:
    if profile.owner_date_of_birth is None:
        raise ValueError("Owner date of birth is required for an RMD divisor.")
    owner_age = _age_in_year(profile.owner_date_of_birth, year)
    spouse_age = None
    if profile.spouse_is_sole_beneficiary and profile.spouse_date_of_birth is not None:
        candidate = _age_in_year(profile.spouse_date_of_birth, year)
        if owner_age - candidate > 10:
            spouse_age = candidate
    return lifetime_divisor(owner_age, spouse_age)


def _account_allocation_weights(
    accounts: Sequence[RetirementAccount], symbols: Sequence[str], fallback: np.ndarray
) -> np.ndarray:
    weights = np.empty((len(accounts), len(symbols)))
    symbol_indices = {symbol: index for index, symbol in enumerate(symbols)}
    for account_index, account in enumerate(accounts):
        if account.allocation is None:
            weights[account_index] = fallback
            continue
        unknown = set(account.allocation) - set(symbols)
        if unknown:
            raise ValueError(
                f"Account {account.name} allocation references unavailable symbols: "
                + ", ".join(sorted(unknown))
                + "."
            )
        total = sum(account.allocation.values())
        if total <= 0:
            raise ValueError(f"Account {account.name} allocation must have a positive total.")
        weights[account_index] = 0
        for symbol, value in account.allocation.items():
            weights[account_index, symbol_indices[symbol]] = value / total
    return weights


def aggregate_taxable_basis(portfolio: Portfolio, cash_balance: float | None = None) -> float:
    """Return remaining FIFO security basis plus taxable cash, dollar for dollar."""
    security_basis = sum(float(lot.shares) * float(lot.cost_basis) for lot in portfolio.holdings)
    return security_basis + (float(portfolio.cash) if cash_balance is None else float(cash_balance))


def _require_classified_accounts(accounts: Sequence[RetirementAccount]) -> None:
    if any(account.account_type == "unknown" for account in accounts):
        raise ValueError(UNKNOWN_ACCOUNT_ERROR)


def _simulate(
    request: RetirementRequest, sampled: np.ndarray, symbols: Sequence[str]
) -> RetirementResult:
    profile = request.profile
    assumptions = request.assumptions
    taxes = request.tax_assumptions
    _require_classified_accounts(request.accounts)
    if not request.accounts or sum(account.balance for account in request.accounts) <= 0:
        raise ValueError("At least one retirement account must have a positive balance.")
    months = profile.planning_horizon_years * 12
    if sampled.ndim != 3 or sampled.shape[1:] != (months, len(symbols)):
        raise ValueError("Sampled returns do not match the retirement horizon and holdings.")
    if not np.isfinite(sampled).all():
        raise ValueError("Retirement return inputs must be finite.")
    simulations = sampled.shape[0]
    accounts = request.accounts
    balances = np.broadcast_to(
        np.array([account.balance for account in accounts], dtype=float),
        (simulations, len(accounts)),
    ).copy()
    taxable_bases = np.broadcast_to(
        np.array(
            [
                account.taxable_cost_basis
                if account.account_type == "taxable" and account.taxable_cost_basis is not None
                else np.nan
                for account in accounts
            ],
            dtype=float,
        ),
        (simulations, len(accounts)),
    ).copy()
    contribution_weights = _contribution_weights(request)
    holding_amounts = np.array([request.holding_values[symbol] for symbol in symbols], dtype=float)
    asset_weights = holding_amounts / holding_amounts.sum()
    account_asset_weights = _account_allocation_weights(accounts, symbols, asset_weights)
    account_assets = balances[:, :, None] * account_asset_weights[None, :, :]
    paths = np.empty((simulations, months + 1))
    paths[:, 0] = balances.sum(axis=1)
    beginning_household_assets = balances.sum(axis=1).copy()
    total_shortfall = np.zeros(simulations)
    lifetime_taxes = np.zeros(simulations)
    outside_used = np.zeros(simulations)
    retained_cash = np.zeros(simulations)
    total_rmd_withdrawals = np.zeros(simulations)
    total_additional_withdrawals = np.zeros(simulations)
    total_investment_return = np.zeros(simulations)
    total_fees = np.zeros(simulations)
    total_dividend_tax_drag = np.zeros(simulations)
    total_contributions = np.zeros(simulations)
    total_gross_outside_income = np.zeros(simulations)
    total_outside_income_tax = np.zeros(simulations)
    total_rmd_tax = np.zeros(simulations)
    total_additional_withdrawal_tax = np.zeros(simulations)
    total_recurring_spending_required = np.zeros(simulations)
    total_one_time_spending_required = np.zeros(simulations)
    total_recurring_spending_funded = np.zeros(simulations)
    total_one_time_spending_funded = np.zeros(simulations)
    withdrawals = {kind: np.zeros(simulations) for kind in ACCOUNT_TYPES}
    first_depletion_month = np.full(simulations, np.nan)
    retirement_values: np.ndarray | None = None
    retirement_returns = np.zeros((simulations, min(60, max(months - profile.retirement_month, 0))))
    monthly_inflation = (1 + profile.annual_inflation) ** (1 / 12) - 1
    monthly_fee = 1 - (1 - assumptions.annual_fee) ** (1 / 12)
    monthly_dividend_yield = (1 + taxes.qualified_dividend_yield) ** (1 / 12) - 1
    rebalance_every = REBALANCE_FREQUENCIES[assumptions.rebalancing]
    order = list(assumptions.withdrawal_order) + [
        kind for kind in ACCOUNT_TYPES if kind not in assumptions.withdrawal_order
    ]
    prior_december_balances = np.broadcast_to(
        np.array(
            [
                account.prior_december_31_balance
                if account.prior_december_31_balance is not None
                else account.balance
                for account in accounts
            ],
            dtype=float,
        ),
        (simulations, len(accounts)),
    ).copy()
    traditional_indices = [
        index
        for index, account in enumerate(accounts)
        if account.account_type == "traditional_ira" and not account.is_inherited
    ]
    if (
        profile.owner_date_of_birth is not None
        and profile.as_of_date.year >= first_rmd_year(profile.owner_date_of_birth)
        and any(accounts[index].prior_december_31_balance is None for index in traditional_indices)
    ):
        raise ValueError(
            "Each owner Traditional IRA requires its explicit prior December 31 balance when "
            "the projection begins in an RMD year."
        )

    for month in range(months):
        month_end = pd.Timestamp(profile.as_of_date) + pd.offsets.MonthEnd(month + 1)
        surplus_this_month = np.zeros(simulations)
        if month == profile.retirement_month:
            retirement_values = balances.sum(axis=1) + retained_cash
        prior_total = balances.sum(axis=1)
        account_assets *= 1 + sampled[:, month, None, :]
        balances = account_assets.sum(axis=2)
        investment_return = balances.sum(axis=1) - prior_total
        total_investment_return += investment_return
        portfolio_return = np.divide(
            investment_return,
            prior_total,
            out=np.zeros(simulations),
            where=prior_total > 0,
        )
        if (
            profile.retirement_month
            <= month
            < profile.retirement_month + retirement_returns.shape[1]
        ):
            retirement_returns[:, month - profile.retirement_month] = portfolio_return
        fee_amount = account_assets.sum(axis=(1, 2)) * monthly_fee
        total_fees += fee_amount
        account_assets *= 1 - monthly_fee
        balances = account_assets.sum(axis=2)
        taxable_indices = [
            i for i, account in enumerate(accounts) if account.account_type == "taxable"
        ]
        if taxable_indices and monthly_dividend_yield:
            dividend_tax = (
                balances[:, taxable_indices].sum(axis=1)
                * monthly_dividend_yield
                * taxes.dividend_tax_rate
            )
            lifetime_taxes += dividend_tax
            total_dividend_tax_drag += dividend_tax
            taxable_total = balances[:, taxable_indices].sum(axis=1)
            for index in taxable_indices:
                reduction = dividend_tax * np.divide(
                    balances[:, index],
                    taxable_total,
                    out=np.zeros(simulations),
                    where=taxable_total > 0,
                )
                prior_balance = balances[:, index].copy()
                balances[:, index] -= reduction
                account_assets[:, index, :] *= np.divide(
                    balances[:, index],
                    prior_balance,
                    out=np.zeros(simulations),
                    where=prior_balance > 0,
                )[:, None]

        rmd_net = np.zeros(simulations)
        if (
            month_end.month == 12
            and profile.owner_date_of_birth is not None
            and month_end.year >= first_rmd_year(profile.owner_date_of_birth)
            and traditional_indices
        ):
            divisor, _ = owner_rmd_divisor(profile, month_end.year)
            required = sum(
                prior_december_balances[:, index] / divisor for index in traditional_indices
            )
            remaining_rmd = required.copy()
            for index in traditional_indices:
                gross = np.minimum(balances[:, index], remaining_rmd)
                prior_balance = balances[:, index].copy()
                balances[:, index] -= gross
                account_assets[:, index, :] *= np.divide(
                    balances[:, index],
                    prior_balance,
                    out=np.zeros(simulations),
                    where=prior_balance > 0,
                )[:, None]
                remaining_rmd = np.maximum(remaining_rmd - gross, 0)
                tax = gross * taxes.ordinary_income_rate
                lifetime_taxes += tax
                total_rmd_tax += tax
                withdrawals["traditional_ira"] += gross
                total_rmd_withdrawals += gross
                rmd_net += gross - tax

        if month < profile.retirement_month:
            contribution = (
                profile.contribution_amount
                if profile.contribution_frequency == "monthly"
                else profile.contribution_amount / 12
            )
            balances += contribution * contribution_weights
            total_contributions += contribution
            account_assets += (
                contribution
                * contribution_weights[None, :, None]
                * account_asset_weights[None, :, :]
            )
            for index, account in enumerate(accounts):
                if (
                    account.account_type == "taxable"
                    and not np.isnan(taxable_bases[:, index]).all()
                ):
                    taxable_bases[:, index] += contribution * contribution_weights[index]
            surplus_this_month = rmd_net
        else:
            inflation_factor = (1 + monthly_inflation) ** month
            recurring_spending = profile.fixed_real_annual_spending / 12 * inflation_factor
            one_time_spending = 0.0
            age = profile.current_age + month / 12
            for event in request.spending_events:
                event_month = event.resolved_model_month(profile)
                if event_month == month:
                    one_time_spending += event.real_amount * inflation_factor
            spending = recurring_spending + one_time_spending
            total_recurring_spending_required += recurring_spending
            total_one_time_spending_required += one_time_spending
            outside_net = np.zeros(simulations)
            for income in request.outside_income:
                start_month = _income_start_month(profile, income)
                if month < start_month or (income.end_age is not None and age >= income.end_age):
                    continue
                cola = (
                    profile.annual_inflation if income.annual_cola is None else income.annual_cola
                )
                gross = (
                    income.annual_real_amount
                    / 12
                    * (1 + cola) ** ((month - max(start_month, 0)) / 12)
                )
                fraction = (
                    income.taxable_fraction
                    if income.taxable_fraction is not None
                    else taxes.social_security_taxable_fraction
                    if income.kind == "social_security"
                    else 1.0
                )
                income_tax = gross * fraction * taxes.ordinary_income_rate
                lifetime_taxes += income_tax
                total_gross_outside_income += gross
                total_outside_income_tax += income_tax
                outside_net += gross - income_tax
            current_cash = outside_net + rmd_net
            outside_used += np.minimum(outside_net, spending)
            surplus_this_month = np.maximum(current_cash - spending, 0)
            remaining = np.maximum(spending - current_cash, 0)
            retained_used = np.minimum(retained_cash, remaining)
            retained_cash -= retained_used
            remaining = np.where(remaining - retained_used > 1e-8, remaining - retained_used, 0)
            for kind in order:
                for index, account in enumerate(accounts):
                    if account.account_type != kind:
                        continue
                    if kind == "roth_ira":
                        effective_rate = np.zeros(simulations)
                    elif kind == "traditional_ira":
                        effective_rate = np.full(simulations, taxes.ordinary_income_rate)
                    elif kind == "taxable":
                        assumed = np.full(simulations, taxes.taxable_realization_fraction)
                        has_basis = ~np.isnan(taxable_bases[:, index])
                        embedded_gain = np.divide(
                            np.maximum(balances[:, index] - taxable_bases[:, index], 0),
                            balances[:, index],
                            out=np.zeros(simulations),
                            where=balances[:, index] > 0,
                        )
                        gain_fraction = np.where(has_basis, embedded_gain, assumed)
                        effective_rate = taxes.capital_gains_rate * gain_fraction
                    else:
                        raise ValueError(UNKNOWN_ACCOUNT_ERROR)
                    required_gross = remaining / np.maximum(1 - effective_rate, 1e-12)
                    gross = np.minimum(balances[:, index], required_gross)
                    tax = gross * effective_rate
                    net = gross - tax
                    prior_balance = balances[:, index].copy()
                    balances[:, index] -= gross
                    account_assets[:, index, :] *= np.divide(
                        balances[:, index],
                        prior_balance,
                        out=np.zeros(simulations),
                        where=prior_balance > 0,
                    )[:, None]
                    if kind == "taxable":
                        taxable_bases[:, index] *= 1 - np.divide(
                            gross,
                            prior_balance,
                            out=np.zeros(simulations),
                            where=prior_balance > 0,
                        )
                    remaining = np.where(remaining - net > 1e-8, remaining - net, 0)
                    lifetime_taxes += tax
                    total_additional_withdrawal_tax += tax
                    total_additional_withdrawals += gross
                    withdrawals[kind] += gross
            spending_funded = spending - remaining
            recurring_funded = np.minimum(spending_funded, recurring_spending)
            one_time_funded = np.maximum(spending_funded - recurring_funded, 0)
            total_recurring_spending_funded += recurring_funded
            total_one_time_spending_funded += one_time_funded
            depleted = (remaining > 1e-8) & np.isnan(first_depletion_month)
            first_depletion_month[depleted] = month
            total_shortfall += remaining

        balances = np.maximum(balances, 0)
        account_assets = np.maximum(account_assets, 0)
        if rebalance_every is not None and (month + 1) % rebalance_every == 0:
            taxable_indices = [
                index for index, account in enumerate(accounts) if account.account_type == "taxable"
            ]
            if taxable_indices:
                taxable_total = balances[:, taxable_indices].sum(axis=1)
                for index in taxable_indices:
                    share = np.divide(
                        balances[:, index],
                        taxable_total,
                        out=np.full(simulations, 1 / len(taxable_indices)),
                        where=taxable_total > 0,
                    )
                    invested = retained_cash * share
                    balances[:, index] += invested
                    account_assets[:, index, :] += (
                        invested[:, None] * account_asset_weights[index][None, :]
                    )
                    if not np.isnan(taxable_bases[:, index]).all():
                        taxable_bases[:, index] += invested
                retained_cash[:] = 0
            account_assets = balances[:, :, None] * account_asset_weights[None, :, :]
        retained_cash += surplus_this_month
        if month_end.month == 12:
            prior_december_balances = balances.copy()
        paths[:, month + 1] = balances.sum(axis=1) + retained_cash

    if retirement_values is None:
        retirement_values = paths[:, -1].copy()
    inflation_factors = (1 + monthly_inflation) ** np.arange(months + 1)
    real_paths = paths / inflation_factors[None, :]
    terminal = paths[:, -1].copy()
    real_terminal = real_paths[:, -1]
    total_spending_funded = total_recurring_spending_funded + total_one_time_spending_funded
    total_spending_required = total_recurring_spending_required + total_one_time_spending_required
    total_withdrawal_tax = total_rmd_tax + total_additional_withdrawal_tax
    reconciled_assets = (
        beginning_household_assets
        + total_investment_return
        - total_fees
        - total_dividend_tax_drag
        + total_contributions
        + total_gross_outside_income
        - total_outside_income_tax
        - total_spending_funded
        - total_withdrawal_tax
    )
    asset_residual = reconciled_assets - terminal
    shortfall_residual = total_spending_required - total_spending_funded - total_shortfall
    cash_flow_reconciliation = {
        "beginning_household_assets_mean": float(beginning_household_assets.mean()),
        "investment_return_mean": float(total_investment_return.mean()),
        "fees_mean": float(total_fees.mean()),
        "dividend_tax_drag_mean": float(total_dividend_tax_drag.mean()),
        "contributions_mean": float(total_contributions.mean()),
        "gross_outside_income_mean": float(total_gross_outside_income.mean()),
        "outside_income_tax_mean": float(total_outside_income_tax.mean()),
        "recurring_spending_funded_mean": float(total_recurring_spending_funded.mean()),
        "one_time_spending_funded_mean": float(total_one_time_spending_funded.mean()),
        "withdrawal_tax_mean": float(total_withdrawal_tax.mean()),
        "rmd_withdrawal_tax_mean": float(total_rmd_tax.mean()),
        "additional_withdrawal_tax_mean": float(total_additional_withdrawal_tax.mean()),
        "gross_rmd_withdrawals_mean": float(total_rmd_withdrawals.mean()),
        "gross_additional_withdrawals_mean": float(total_additional_withdrawals.mean()),
        "ending_invested_balances_mean": float(balances.sum(axis=1).mean()),
        "ending_retained_cash_mean": float(retained_cash.mean()),
        "maximum_absolute_residual": float(np.max(np.abs(asset_residual))),
    }
    shortfall_reconciliation = {
        "required_spending_obligations_mean": float(total_spending_required.mean()),
        "spending_obligations_funded_mean": float(total_spending_funded.mean()),
        "unpaid_shortfall_mean": float(total_shortfall.mean()),
        "maximum_absolute_residual": float(np.max(np.abs(shortfall_residual))),
    }
    funded = total_shortfall <= 1e-8
    depletion_ages = (
        profile.current_age + first_depletion_month[~np.isnan(first_depletion_month)] / 12
    )
    depletion_distribution: dict[str, float | None] = {
        f"P{p}": float(np.percentile(depletion_ages, p)) if len(depletion_ages) else None
        for p in (10, 25, 50, 75, 90)
    }
    early_compound = (
        np.prod(1 + retirement_returns, axis=1) - 1
        if retirement_returns.shape[1]
        else np.zeros(simulations)
    )
    correlation = (
        float(np.corrcoef(early_compound, real_terminal)[0, 1])
        if np.std(early_compound) > 0 and np.std(real_terminal) > 0
        else None
    )
    low_sequence = early_compound <= np.percentile(early_compound, 25)
    high_sequence = early_compound >= np.percentile(early_compound, 75)
    worst_indices = np.argsort(real_terminal)[: max(1, int(np.ceil(simulations * 0.10)))]
    worst = pd.DataFrame(
        {
            "scenario": worst_indices,
            "early_retirement_return": early_compound[worst_indices],
            "terminal_nominal": terminal[worst_indices],
            "terminal_real": real_terminal[worst_indices],
            "lifetime_taxes": lifetime_taxes[worst_indices],
            "total_shortfall": total_shortfall[worst_indices],
            "depletion_age": profile.current_age + first_depletion_month[worst_indices] / 12,
        }
    )
    nominal_monthly = _percentile_frame(paths)
    real_monthly = _percentile_frame(real_paths)
    nominal_annual = nominal_monthly.iloc[::12].copy()
    real_annual = real_monthly.iloc[::12].copy()
    nominal_annual.index = pd.Index(range(profile.planning_horizon_years + 1), name="Year")
    real_annual.index = nominal_annual.index
    assumptions_disclosure = {
        "profile": asdict(profile),
        "stochastic": asdict(assumptions),
        "tax": asdict(taxes),
        "accounts": [asdict(account) for account in accounts],
        "outside_income": [asdict(item) for item in request.outside_income],
        "spending_events": [
            {**asdict(item), "resolved_model_month": item.resolved_model_month(profile)}
            for item in request.spending_events
        ],
        "contribution_allocation": dict(request.contribution_allocation or {}),
    }
    limitations = (
        "This is a transparent planning tax estimate, not tax advice or a tax-return calculation.",
        "Tax rates are fixed user assumptions; brackets, deductions, state/local law, filing status, loss harvesting, and account-specific legal rules are not modeled.",
        "Owner Traditional IRA RMDs use date-of-birth-dependent starting ages, prior-December balances, December month-end timing, and versioned "
        + RMD_TABLE_VERSION
        + "; inherited-account rules are out of scope.",
        "Traditional IRAs are assumed to have no nondeductible basis, so RMDs and withdrawals are fully ordinary income at the configured rate.",
        "Roth IRA withdrawals are assumed qualified and tax free; qualification-date calculations are not modeled.",
        "Taxable withdrawals use aggregate FIFO security basis plus taxable cash when supplied; basis may exceed value and is reduced proportionally because detailed projected lots are not modeled.",
        "Social Security taxation uses the configured fixed taxable fraction rather than provisional-income rules.",
        RETAINED_CASH_DISCLOSURE,
        "Account-specific allocations are supported; accounts without one use the household allocation.",
        "Fixed real spending is inflation adjusted; no guardrail or percentage-spending strategy is modeled.",
        SPENDING_EVENT_TIMING_DISCLOSURE,
        "Return history and parametric assumptions may not represent future regimes or unprecedented events.",
    )
    return RetirementResult(
        RETIREMENT_MODEL_TYPE,
        RETIREMENT_MODEL_VERSION,
        nominal_monthly,
        real_monthly,
        nominal_annual,
        real_annual,
        terminal,
        real_terminal,
        total_shortfall.copy(),
        profile.current_age + first_depletion_month / 12,
        float(funded.mean()),
        float((~funded).mean()),
        depletion_distribution,
        {**_distribution(retirement_values), **_distribution_percentiles(retirement_values)},
        {**_distribution_percentiles(lifetime_taxes), "mean": float(lifetime_taxes.mean())},
        {
            kind: {**_distribution_percentiles(values), "mean": float(values.mean())}
            for kind, values in withdrawals.items()
        },
        {**_distribution_percentiles(outside_used), "mean": float(outside_used.mean())},
        {**_distribution_percentiles(retained_cash), "mean": float(retained_cash.mean())},
        {
            **_distribution_percentiles(total_rmd_withdrawals),
            "mean": float(total_rmd_withdrawals.mean()),
        },
        cash_flow_reconciliation,
        shortfall_reconciliation,
        {**_distribution(total_shortfall), **_distribution_percentiles(total_shortfall)},
        (
            float(np.mean(real_terminal >= profile.target_estate_value))
            if profile.target_estate_value is not None
            else None
        ),
        {
            "early_retirement_window_months": retirement_returns.shape[1],
            "early_return_terminal_real_correlation": correlation,
            "depletion_probability_low_early_return_quartile": float(
                (~funded[low_sequence]).mean()
            ),
            "depletion_probability_high_early_return_quartile": float(
                (~funded[high_sequence]).mean()
            ),
        },
        worst,
        assumptions_disclosure,
        {},
        {},
        (),
        limitations,
    )


def run_retirement_forecast(request: RetirementRequest) -> RetirementResult:
    """Run a 20–40 year account-aware accumulation and fixed-real-spending forecast."""
    _require_classified_accounts(request.accounts)
    sampled, symbols, history, proxies, warnings = _return_paths(request)
    result = _simulate(request, sampled, symbols)
    coverage = {
        "status": "limited" if proxies else "good",
        "history_months": len(history),
        "data_start": history.index.min().date().isoformat(),
        "data_end": history.index.max().date().isoformat(),
        "dataset_ids": list(request.dataset_ids),
        "symbols": list(symbols),
        "proxy_use": dict(proxies),
        "source_coverage": dict(request.source_coverage or {}),
    }
    return replace(result, data_coverage=coverage, proxies=dict(proxies), warnings=tuple(warnings))


def build_retirement_request(
    reconstruction: ReconstructionResult,
    market_coverage: HistoricalCoverageResult,
    portfolios: Sequence[Portfolio],
    profile: RetirementProfile,
    assumptions: RetirementAssumptions = RetirementAssumptions(),
    *,
    tax_assumptions: RetirementTaxAssumptions = RetirementTaxAssumptions(),
    outside_income: Sequence[OutsideIncome] = (),
    spending_events: Sequence[SpendingEvent] = (),
    contribution_allocation: Mapping[str, float] | None = None,
    account_allocations: Mapping[str, Mapping[str, float]] | None = None,
    prior_december_31_balances: Mapping[str, float] | None = None,
    proxies: Mapping[str, str] | None = None,
    external_assumptions: Mapping[str, CapitalMarketAssumption | Mapping[str, object]]
    | None = None,
    user_overrides: Mapping[str, CapitalMarketAssumption | Mapping[str, object]] | None = None,
) -> RetirementRequest:
    """Build a retirement request from ledger reconstruction and immutable market coverage."""
    as_of = reconstruction.common_as_of_date
    if as_of is None or reconstruction.common_as_of_value is None:
        raise ValueError(
            "A complete common reconstruction date is required for retirement planning."
        )
    portfolio_by_id = {portfolio.id: portfolio for portfolio in portfolios}
    accounts: list[RetirementAccount] = []
    for portfolio_id, value in reconstruction.common_as_of_portfolio_values.items():
        portfolio = portfolio_by_id.get(portfolio_id)
        if portfolio is None or value is None:
            continue
        taxable_basis = None
        if portfolio.account_type == "taxable":
            cash_series = reconstruction.portfolios[portfolio_id].daily.cash
            cash_value = (
                float(cash_series.loc[pd.Timestamp(as_of)])
                if pd.Timestamp(as_of) in cash_series.index
                else float(portfolio.cash)
            )
            taxable_basis = aggregate_taxable_basis(portfolio, cash_value)
        accounts.append(
            RetirementAccount(
                portfolio.name,
                portfolio.account_type,
                float(value),
                taxable_cost_basis=taxable_basis,
                prior_december_31_balance=(prior_december_31_balances or {}).get(portfolio.name),
                allocation=(account_allocations or {}).get(portfolio.name),
            )
        )
    positions = reconstruction.combined.positions.loc[pd.Timestamp(as_of)]
    prices = market_coverage.data.copy()
    prices.index = pd.to_datetime(prices.index).normalize()
    holding_values: dict[str, float] = {}
    quantities: dict[str, float] = {}
    for key, quantity in positions.items():
        symbol = key[-1] if isinstance(key, tuple) else key
        quantities[str(symbol)] = quantities.get(str(symbol), 0.0) + float(quantity)
    for symbol, quantity in quantities.items():
        if abs(float(quantity)) <= 1e-12:
            continue
        available = prices.loc[prices.index <= pd.Timestamp(as_of), symbol].dropna()
        if available.empty:
            raise ValueError(f"Missing confirmed current price for {symbol}.")
        holding_values[str(symbol).upper()] = float(quantity) * float(available.iloc[-1])
    return RetirementRequest(
        profile=profile,
        accounts=tuple(accounts),
        holding_values=holding_values,
        monthly_returns=monthly_returns_from_prices(market_coverage.data),
        assumptions=assumptions,
        tax_assumptions=tax_assumptions,
        outside_income=tuple(outside_income),
        spending_events=tuple(spending_events),
        contribution_allocation=contribution_allocation,
        proxies=proxies,
        external_assumptions=external_assumptions,
        user_overrides=user_overrides,
        dataset_ids=tuple(market_coverage.data.attrs.get("dataset_ids", ())),
        source_coverage={
            "reconstruction_status": reconstruction.data_coverage.status.value,
            "reconstruction_score": reconstruction.data_coverage.score,
            "adjustment": market_coverage.adjustment,
            "sources": reconstruction.data_coverage.sources,
        },
    )


def retirement_sensitivity(
    request: RetirementRequest, parameter: str, values: Sequence[object]
) -> pd.DataFrame:
    """Vary one documented planning assumption while holding the seed and other inputs fixed."""
    rows: list[dict[str, object]] = []
    supported = {
        "retirement_age",
        "spending",
        "inflation",
        "social_security_start_age",
        "contribution",
        "expected_return",
        "ordinary_income_tax",
        "capital_gains_tax",
        "dividend_tax",
        "withdrawal_order",
    }
    if parameter not in supported:
        raise ValueError(f"Unsupported retirement sensitivity parameter: {parameter}.")
    for value in values:
        candidate = request
        if parameter == "retirement_age":
            candidate = replace(
                request,
                profile=replace(
                    request.profile,
                    planned_retirement_age=float(value),
                    planned_retirement_date=None,
                ),
            )
        elif parameter == "spending":
            candidate = replace(
                request,
                profile=replace(request.profile, fixed_real_annual_spending=float(value)),
            )
        elif parameter == "inflation":
            candidate = replace(
                request, profile=replace(request.profile, annual_inflation=float(value))
            )
        elif parameter == "contribution":
            candidate = replace(
                request, profile=replace(request.profile, contribution_amount=float(value))
            )
        elif parameter == "expected_return":
            candidate = replace(
                request,
                assumptions=replace(request.assumptions, expected_return_shift=float(value)),
            )
        elif parameter == "withdrawal_order":
            candidate = replace(
                request, assumptions=replace(request.assumptions, withdrawal_order=tuple(value))
            )
        elif parameter == "social_security_start_age":
            streams = tuple(
                replace(item, start_age=float(value), start_date=None)
                if item.kind == "social_security"
                else item
                for item in request.outside_income
            )
            if not any(item.kind == "social_security" for item in streams):
                raise ValueError("Social Security sensitivity requires a Social Security stream.")
            candidate = replace(request, outside_income=streams)
        else:
            field = {
                "ordinary_income_tax": "ordinary_income_rate",
                "capital_gains_tax": "capital_gains_rate",
                "dividend_tax": "dividend_tax_rate",
            }[parameter]
            candidate = replace(
                request,
                tax_assumptions=replace(request.tax_assumptions, **{field: float(value)}),
            )
        result = run_retirement_forecast(candidate)
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "probability_funding_full_horizon": result.probability_funding_full_horizon,
                "probability_depletion": result.probability_depletion,
                "median_terminal_real": float(result.real_annual_percentiles.iloc[-1].P50),
                "mean_lifetime_taxes": result.lifetime_taxes_estimate["mean"],
                "target_estate_probability": result.target_estate_probability,
            }
        )
    return pd.DataFrame(rows)


def historical_sequence_replay(request: RetirementRequest) -> HistoricalReplayResult:
    """Replay every complete rolling historical sequence without random resampling."""
    _require_classified_accounts(request.accounts)
    symbols, _ = _normalized_values(request.holding_values)
    common = BootstrapRequest(
        request.holding_values,
        request.monthly_returns,
        BootstrapAssumptions(1, minimum_history_months=request.assumptions.minimum_history_months),
        proxies=request.proxies,
    )
    frame, _, _, _ = _prepare_returns(common, symbols)
    horizon = request.profile.planning_horizon_years * 12
    windows = len(frame) - horizon + 1
    if windows <= 0:
        return HistoricalReplayResult(0, 1, None, None, pd.DataFrame())
    sampled = np.stack(
        [frame.iloc[start : start + horizon].to_numpy(dtype=float) for start in range(windows)]
    )
    result = _simulate(request, sampled, symbols)
    rows = result.worst_decile_scenarios.iloc[0:0].copy()
    rows = pd.DataFrame(
        {
            "origin": frame.index[:windows],
            "terminal_nominal": result.terminal_values,
            "terminal_real": result.real_terminal_values,
            "total_shortfall": result.scenario_shortfalls,
            "depletion_age": result.scenario_depletion_ages,
            "funded_full_horizon": result.scenario_shortfalls <= 1e-8,
        }
    )
    # Re-run-free per-window depletion state is represented by the aggregate and terminal outcome;
    # historical replay is validation evidence, never self-validation.
    return HistoricalReplayResult(
        windows,
        0,
        result.probability_funding_full_horizon,
        result.probability_depletion,
        rows,
    )


rolling_retirement_backtest = historical_sequence_replay


def save_retirement_run(
    session: Session,
    portfolios: Sequence[Portfolio],
    result: RetirementResult,
    *,
    dataset_references: Sequence[DatasetReference] = (),
    replay: HistoricalReplayResult | None = None,
    ledger_hash: str | None = None,
) -> ForecastRun:
    """Persist reproducibility inputs and summaries; raw scenarios and paths are excluded."""
    validation = session.scalar(
        select(ModelValidation).where(
            ModelValidation.model_type == result.model_type,
            ModelValidation.model_version == result.model_version,
        )
    )
    summary = result.summary()
    if replay is not None:
        summary["historical_replay"] = replay.summary()
    run = ForecastRun(
        model_type=result.model_type,
        model_version=result.model_version,
        portfolio_scope=json.dumps(
            [{"id": portfolio.id, "name": portfolio.name} for portfolio in portfolios],
            sort_keys=True,
        ),
        ledger_state_hash=ledger_hash or ledger_state_hash(session, [p.id for p in portfolios]),
        assumptions=json.dumps(dict(result.assumptions), sort_keys=True, default=str),
        data_coverage=json.dumps(dict(result.data_coverage), sort_keys=True),
        proxy_use=json.dumps(dict(result.proxies), sort_keys=True),
        status="completed",
        validation_status=validation.status if validation else "unvalidated",
        summary_outputs=json.dumps(summary, sort_keys=True),
        scenario_bands=json.dumps(
            {
                "nominal_annual": result.nominal_annual_percentiles.to_dict(orient="index"),
                "real_annual": result.real_annual_percentiles.to_dict(orient="index"),
            },
            sort_keys=True,
        ),
    )
    session.add(run)
    session.flush()
    references = dataset_references or tuple(
        DatasetReference(dataset_id, "retirement_returns")
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
