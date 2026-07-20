from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
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
from chat_alpaca.scenarios import DatasetReference, ledger_state_hash

RETIREMENT_MODEL_TYPE = "long_horizon_retirement"
RETIREMENT_MODEL_VERSION = "1.0.0"
ACCOUNT_TYPES = ("traditional_ira", "roth_ira", "taxable", "unknown")


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

    def __post_init__(self) -> None:
        if self.current_age < 0:
            raise ValueError("Current age cannot be negative.")
        if not 20 <= self.planning_horizon_years <= 40:
            raise ValueError("Retirement planning horizon must be between 20 and 40 years.")
        if (self.planned_retirement_age is None) == (self.planned_retirement_date is None):
            raise ValueError("Provide either a planned retirement age or date, but not both.")
        if (
            self.planned_retirement_age is not None
            and self.planned_retirement_age < self.current_age
        ):
            raise ValueError("Planned retirement age cannot precede current age.")
        if (
            self.planned_retirement_date is not None
            and self.planned_retirement_date < self.as_of_date
        ):
            raise ValueError("Planned retirement date cannot precede the as-of date.")
        if self.fixed_real_annual_spending < 0 or self.contribution_amount < 0:
            raise ValueError("Spending and contributions cannot be negative.")
        if self.annual_inflation <= -1:
            raise ValueError("Inflation must be greater than -100%.")
        if self.contribution_frequency not in {"monthly", "annual"}:
            raise ValueError("Contribution frequency must be monthly or annual.")
        if self.target_estate_value is not None and self.target_estate_value < 0:
            raise ValueError("Target estate value cannot be negative.")

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

    def __post_init__(self) -> None:
        if self.account_type not in ACCOUNT_TYPES:
            raise ValueError(f"Unsupported account type: {self.account_type}.")
        if self.balance < 0:
            raise ValueError("Account balances cannot be negative.")
        if self.taxable_cost_basis is not None:
            if self.account_type != "taxable":
                raise ValueError("Taxable cost basis applies only to taxable accounts.")
            if not 0 <= self.taxable_cost_basis <= self.balance:
                raise ValueError("Taxable cost basis must be between zero and account balance.")


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
        if self.annual_real_amount < 0:
            raise ValueError("Outside income cannot be negative.")
        if (self.start_age is None) == (self.start_date is None):
            raise ValueError("Provide either an outside-income start age or date, but not both.")
        if self.start_age is not None and self.start_age < 0:
            raise ValueError("Outside-income start age cannot be negative.")
        if (
            self.end_age is not None
            and self.start_age is not None
            and self.end_age < self.start_age
        ):
            raise ValueError("Outside-income end age cannot precede its start age.")
        if self.annual_cola is not None and self.annual_cola <= -1:
            raise ValueError("Outside-income COLA must be greater than -100%.")
        if self.taxable_fraction is not None and not 0 <= self.taxable_fraction <= 1:
            raise ValueError("Outside-income taxable fraction must be between zero and one.")


@dataclass(frozen=True)
class SpendingEvent:
    name: str
    real_amount: float
    age: float

    def __post_init__(self) -> None:
        if self.real_amount < 0 or self.age < 0:
            raise ValueError("One-time spending amount and age cannot be negative.")


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
            if not 0 <= value <= 1:
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
        "unknown",
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
        if self.simulations < 1:
            raise ValueError("At least one simulation is required.")
        if self.block_length not in {3, 6, 12}:
            raise ValueError("Block length must be 3, 6, or 12 months.")
        if self.minimum_history_months < self.block_length:
            raise ValueError("Minimum history cannot be shorter than the selected block.")
        if not 0 <= self.annual_fee < 1:
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
            if self.degrees_of_freedom is None or self.degrees_of_freedom <= 2:
                raise ValueError("Student's t degrees of freedom must be greater than two.")
        elif self.degrees_of_freedom is not None:
            raise ValueError("Degrees of freedom apply only to Student's t returns.")
        if self.volatility_multiplier <= 0 or not 0 <= self.correlation_multiplier <= 1:
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


def _simulate(
    request: RetirementRequest, sampled: np.ndarray, symbols: Sequence[str]
) -> RetirementResult:
    profile = request.profile
    assumptions = request.assumptions
    taxes = request.tax_assumptions
    if not request.accounts or sum(account.balance for account in request.accounts) <= 0:
        raise ValueError("At least one retirement account must have a positive balance.")
    months = profile.planning_horizon_years * 12
    if sampled.ndim != 3 or sampled.shape[1:] != (months, len(symbols)):
        raise ValueError("Sampled returns do not match the retirement horizon and holdings.")
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
    _, holding_amounts = _normalized_values(request.holding_values)
    asset_weights = holding_amounts / holding_amounts.sum()
    asset_values = np.broadcast_to(asset_weights, (simulations, len(symbols))).copy()
    paths = np.empty((simulations, months + 1))
    paths[:, 0] = balances.sum(axis=1)
    total_shortfall = np.zeros(simulations)
    lifetime_taxes = np.zeros(simulations)
    outside_used = np.zeros(simulations)
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

    for month in range(months):
        if month == profile.retirement_month:
            retirement_values = balances.sum(axis=1).copy()
        before_assets = asset_values.sum(axis=1)
        asset_values *= 1 + sampled[:, month, :]
        portfolio_return = (
            np.divide(
                asset_values.sum(axis=1),
                before_assets,
                out=np.ones(simulations),
                where=before_assets > 0,
            )
            - 1
        )
        balances *= 1 + portfolio_return[:, None]
        if (
            profile.retirement_month
            <= month
            < profile.retirement_month + retirement_returns.shape[1]
        ):
            retirement_returns[:, month - profile.retirement_month] = portfolio_return
        balances -= balances * monthly_fee
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
            taxable_total = balances[:, taxable_indices].sum(axis=1)
            for index in taxable_indices:
                balances[:, index] -= dividend_tax * np.divide(
                    balances[:, index],
                    taxable_total,
                    out=np.zeros(simulations),
                    where=taxable_total > 0,
                )

        if month < profile.retirement_month:
            contribution = (
                profile.contribution_amount
                if profile.contribution_frequency == "monthly"
                else profile.contribution_amount
                if month % 12 == 0
                else 0.0
            )
            balances += contribution * contribution_weights
            for index, account in enumerate(accounts):
                if (
                    account.account_type == "taxable"
                    and not np.isnan(taxable_bases[:, index]).all()
                ):
                    taxable_bases[:, index] += contribution * contribution_weights[index]
        else:
            inflation_factor = (1 + monthly_inflation) ** month
            spending = profile.fixed_real_annual_spending / 12 * inflation_factor
            age = profile.current_age + month / 12
            for event in request.spending_events:
                event_month = round((event.age - profile.current_age) * 12)
                if event_month == month:
                    spending += event.real_amount * inflation_factor
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
                outside_net += gross - income_tax
            used = np.minimum(outside_net, spending)
            outside_used += used
            remaining = np.maximum(spending - outside_net, 0)
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
                        effective_rate = np.full(simulations, taxes.unknown_account_withdrawal_rate)
                    required_gross = remaining / np.maximum(1 - effective_rate, 1e-12)
                    gross = np.minimum(balances[:, index], required_gross)
                    tax = gross * effective_rate
                    net = gross - tax
                    prior_balance = balances[:, index].copy()
                    balances[:, index] -= gross
                    if kind == "taxable":
                        taxable_bases[:, index] *= 1 - np.divide(
                            gross,
                            prior_balance,
                            out=np.zeros(simulations),
                            where=prior_balance > 0,
                        )
                    remaining = np.maximum(remaining - net, 0)
                    lifetime_taxes += tax
                    withdrawals[kind] += gross
            depleted = (remaining > 1e-8) & np.isnan(first_depletion_month)
            first_depletion_month[depleted] = month
            total_shortfall += remaining

        balances = np.maximum(balances, 0)
        if rebalance_every is not None and (month + 1) % rebalance_every == 0:
            asset_values = asset_values.sum(axis=1)[:, None] * asset_weights
        paths[:, month + 1] = balances.sum(axis=1)

    if retirement_values is None:
        retirement_values = paths[:, -1].copy()
    inflation_factors = (1 + monthly_inflation) ** np.arange(months + 1)
    real_paths = paths / inflation_factors[None, :]
    terminal = paths[:, -1].copy()
    real_terminal = real_paths[:, -1]
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
        "spending_events": [asdict(item) for item in request.spending_events],
        "contribution_allocation": dict(request.contribution_allocation or {}),
    }
    limitations = (
        "This is a transparent planning tax estimate, not tax advice or a tax-return calculation.",
        "Tax rates are fixed user assumptions; brackets, deductions, RMDs, state/local law, filing status, loss harvesting, and account-specific legal rules are not modeled.",
        "Taxable withdrawals use aggregate embedded gain when cost basis is provided, otherwise the configured realization fraction; detailed tax lots are not projected.",
        "Social Security taxation uses a configurable fixed taxable fraction rather than provisional-income rules.",
        "Outside income offsets spending; unspent outside-income surplus is not automatically reinvested.",
        "Fixed real spending is inflation adjusted; no guardrail or percentage-spending strategy is modeled.",
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
        accounts.append(RetirementAccount(portfolio.name, portfolio.account_type, float(value)))
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
