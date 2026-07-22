from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chat_alpaca.analytics import portfolio_gain_loss, summary_metrics
from chat_alpaca.historical_data import HistoricalCoverageResult
from chat_alpaca.hypothetical import RetirementAnalysisAssumptions, _retirement_probability
from chat_alpaca.models import HoldingLot, Portfolio, PortfolioTransaction
from chat_alpaca.parametric_forecasting import (
    ParametricAssumptions,
    ParametricRequest,
    estimate_parameters,
)
from chat_alpaca.realtime import FreshnessStatus, QuoteRecord, build_portfolio_pulse
from chat_alpaca.reconstruction import ReconstructionRequest, reconstruct_from_coverage
from chat_alpaca.retirement import (
    RETAINED_CASH_DISCLOSURE,
    OutsideIncome,
    RetirementAccount,
    RetirementAssumptions,
    RetirementProfile,
    RetirementRequest,
    RetirementTaxAssumptions,
    SpendingEvent,
    _simulate,
    aggregate_taxable_basis,
    applicable_rmd_start_age,
    first_rmd_year,
    owner_rmd_divisor,
)


def _coverage(values: dict[str, list[float]], days: list[str]) -> HistoricalCoverageResult:
    frame = pd.DataFrame(values, index=pd.to_datetime(days))
    return HistoricalCoverageResult(
        data=frame,
        source="reference",
        feed="test",
        adjustment="split",
        coverage_start=frame.index.min().date(),
        coverage_end=frame.index.max().date(),
        missing_symbols=(),
        missing_date_ranges={},
        warnings=(),
        freshness={},
        usable=True,
        usability="reference case",
    )


def test_tax_002_google_sheets_formula_is_exactly_the_approved_validation_formula() -> None:
    with Path("Calculations Audit.csv").open(newline="") as handle:
        rows = {row["audit_id"]: row for row in csv.DictReader(handle)}

    assert rows["TAX-002"]["google_sheets_formula_template"] == (
        "'=TaxableBalance*(POWER(1+AnnualQualifiedDividendYield,1/12)-1)*DividendTaxRate"
    )


def _retirement_request(
    profile: RetirementProfile,
    accounts: tuple[RetirementAccount, ...],
    *,
    outside: tuple[OutsideIncome, ...] = (),
    rebalancing: str = "never",
) -> RetirementRequest:
    return RetirementRequest(
        profile,
        accounts,
        {"UP": 75, "DOWN": 25},
        pd.DataFrame(
            {"UP": np.zeros(24), "DOWN": np.zeros(24)},
            index=pd.date_range("2020-01-31", periods=24, freq="ME"),
        ),
        RetirementAssumptions(simulations=1, rebalancing=rebalancing),
        RetirementTaxAssumptions(
            ordinary_income_rate=0.2, capital_gains_rate=0, dividend_tax_rate=0
        ),
        outside,
    )


def test_quantity_award_without_recorded_fair_value_makes_performance_unavailable() -> None:
    portfolio = Portfolio(id=1, name="Award", cash=Decimal("0"))
    portfolio.transactions = [
        PortfolioTransaction(
            id=1,
            transaction_date=date(2026, 1, 2),
            kind="transfer",
            action="Transfer",
            cash_delta=Decimal("100"),
            source="test",
        ),
        PortfolioTransaction(
            id=2,
            transaction_date=date(2026, 1, 5),
            kind="award",
            action="Award",
            symbol="AAA",
            quantity=Decimal("1"),
            price=None,
            cash_delta=Decimal("0"),
            source="legacy",
        ),
    ]
    result = reconstruct_from_coverage(
        [portfolio],
        ReconstructionRequest((1,), date(2026, 1, 2), date(2026, 1, 6)),
        _coverage({"AAA": [10, 11, 12]}, ["2026-01-02", "2026-01-05", "2026-01-06"]),
    )

    assert result.portfolios[1].gain_loss is None
    assert result.portfolios[1].daily.time_weighted_return.isna().all()
    assert any("quantity award lacks" in warning for warning in result.warnings)


def test_cash_award_is_an_external_contribution_not_performance() -> None:
    portfolio = Portfolio(id=1, name="Cash award", cash=Decimal("0"))
    portfolio.transactions = [
        PortfolioTransaction(
            id=1,
            transaction_date=date(2026, 1, 2),
            kind="transfer",
            action="Transfer",
            cash_delta=Decimal("100"),
            source="test",
        ),
        PortfolioTransaction(
            id=2,
            transaction_date=date(2026, 1, 5),
            kind="award",
            action="Award",
            cash_delta=Decimal("5"),
            source="test",
        ),
    ]
    result = reconstruct_from_coverage(
        [portfolio],
        ReconstructionRequest((1,), date(2026, 1, 2), date(2026, 1, 5)),
        _coverage({}, ["2026-01-02", "2026-01-05"]),
    )

    assert result.combined.external_cash_flows.tolist() == [100, 5]
    assert result.gain_loss == 0


def test_all_time_gain_uses_complete_frame_when_custom_end_is_earlier() -> None:
    portfolio = Portfolio(id=1, name="All time", cash=Decimal("0"))
    portfolio.transactions = [
        PortfolioTransaction(
            id=1,
            transaction_date=date(2026, 1, 2),
            kind="opening_position",
            action="Opening",
            symbol="AAA",
            quantity=Decimal("1"),
            price=Decimal("10"),
            cash_delta=Decimal("0"),
            source="test",
        )
    ]
    closes = pd.DataFrame(
        {"AAA": [10, 11, 12]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
    )

    metrics = portfolio_gain_loss(portfolio, closes, date(2026, 1, 5), date(2026, 1, 5))

    assert metrics.all_time == 2
    assert metrics.custom == 1


def test_single_observation_comparison_is_unavailable() -> None:
    metrics = summary_metrics(pd.Series([100], index=pd.to_datetime(["2026-01-02"])))

    assert all(value is None for value in metrics.values())


def test_cash_only_monitor_reference_value() -> None:
    pulse = build_portfolio_pulse([SimpleNamespace(name="Cash", cash=12.34, holdings=[])], {})

    assert pulse.indicative_total_value == 12.34


def test_monitor_denominator_boundary_distinguishes_below_cent_from_exact_cent() -> None:
    portfolio = SimpleNamespace(
        name="Monitor", cash=0, holdings=[SimpleNamespace(symbol="AAA", shares=1)]
    )

    below = build_portfolio_pulse(
        [portfolio],
        {
            "AAA": QuoteRecord(
                "AAA",
                latest_trade=100.009,
                previous_close=100,
                status=FreshnessStatus.STREAMING,
            )
        },
    )
    exact = build_portfolio_pulse(
        [portfolio],
        {
            "AAA": QuoteRecord(
                "AAA",
                latest_trade=100.01,
                previous_close=100,
                status=FreshnessStatus.STREAMING,
            )
        },
    )

    assert below.holdings[0].contribution is None
    assert exact.holdings[0].contribution == pytest.approx(1)


@pytest.mark.parametrize(
    ("born", "age"),
    [
        (date(1949, 6, 30), 70.5),
        (date(1949, 7, 1), 72),
        (date(1959, 12, 31), 73),
        (date(1960, 1, 1), 75),
    ],
)
def test_rmd_start_age_is_date_of_birth_dependent(born: date, age: float) -> None:
    assert applicable_rmd_start_age(born) == age


def test_owner_born_in_1960_has_no_age_73_or_74_rmd_and_first_at_75_in_december() -> None:
    profile = RetirementProfile(
        current_age=74,
        planned_retirement_age=74,
        planning_horizon_years=20,
        fixed_real_annual_spending=0,
        as_of_date=date(2034, 1, 1),
        annual_inflation=0,
        owner_date_of_birth=date(1960, 1, 1),
    )
    account = RetirementAccount(
        "Traditional", "traditional_ira", 246_000, prior_december_31_balance=246_000
    )
    request = _retirement_request(profile, (account,))
    sampled = np.zeros((1, 240, 2))

    result = _simulate(request, sampled, ("UP", "DOWN"))

    assert first_rmd_year(date(1960, 1, 1)) == 2035
    assert result.nominal_monthly_percentiles.loc[12, "P50"] == 246_000
    assert result.nominal_monthly_percentiles.loc[23, "P50"] == 246_000
    assert result.nominal_monthly_percentiles.loc[24, "P50"] == pytest.approx(244_000)


def test_1951_through_1959_owner_first_rmd_is_age_73() -> None:
    assert first_rmd_year(date(1955, 8, 1)) == 2028


def test_legacy_half_age_rule_selects_the_calendar_year_age_70_and_a_half_is_reached() -> None:
    assert first_rmd_year(date(1948, 12, 31)) == 2019


def test_multiple_owner_iras_aggregate_rmd_while_roth_and_inherited_accounts_are_excluded() -> None:
    profile = RetirementProfile(
        current_age=74,
        planned_retirement_age=74,
        planning_horizon_years=20,
        fixed_real_annual_spending=0,
        as_of_date=date(2034, 1, 1),
        annual_inflation=0,
        owner_date_of_birth=date(1960, 1, 1),
    )
    owner_iras = _retirement_request(
        profile,
        (
            RetirementAccount(
                "Traditional A", "traditional_ira", 123_000, prior_december_31_balance=123_000
            ),
            RetirementAccount(
                "Traditional B", "traditional_ira", 123_000, prior_december_31_balance=123_000
            ),
        ),
    )
    excluded = _retirement_request(
        profile,
        (
            RetirementAccount("Roth", "roth_ira", 123_000),
            RetirementAccount(
                "Inherited",
                "traditional_ira",
                123_000,
                prior_december_31_balance=123_000,
                is_inherited=True,
            ),
        ),
    )
    sampled = np.zeros((1, 240, 2))

    owner_result = _simulate(owner_iras, sampled, ("UP", "DOWN"))
    excluded_result = _simulate(excluded, sampled, ("UP", "DOWN"))

    assert owner_result.nominal_monthly_percentiles.loc[24, "P50"] == pytest.approx(244_000)
    assert owner_result.rmd_withdrawals["mean"] > 0
    assert excluded_result.nominal_monthly_percentiles.loc[24, "P50"] == 246_000
    assert excluded_result.rmd_withdrawals["mean"] == 0


def test_uniform_and_younger_spouse_divisors_follow_table_selection_rules() -> None:
    uniform = RetirementProfile(
        75, 20, 0, planned_retirement_age=75, owner_date_of_birth=date(1960, 1, 1)
    )
    exception = RetirementProfile(
        75,
        20,
        0,
        planned_retirement_age=75,
        owner_date_of_birth=date(1960, 1, 1),
        spouse_date_of_birth=date(1975, 1, 1),
        spouse_is_sole_beneficiary=True,
    )
    not_sole = RetirementProfile(
        75,
        20,
        0,
        planned_retirement_age=75,
        owner_date_of_birth=date(1960, 1, 1),
        spouse_date_of_birth=date(1975, 1, 1),
    )

    assert owner_rmd_divisor(uniform, 2035) == (24.6, "Table III")
    assert owner_rmd_divisor(exception, 2035)[1] == "Table II"
    assert owner_rmd_divisor(exception, 2035)[0] > 24.6
    assert owner_rmd_divisor(not_sole, 2035) == (24.6, "Table III")


def test_excess_outside_income_is_retained_as_cash_until_rebalance() -> None:
    profile = RetirementProfile(65, 20, 0, planned_retirement_age=65, annual_inflation=0)
    request = _retirement_request(
        profile,
        (RetirementAccount("Roth", "roth_ira", 100_000),),
        outside=(OutsideIncome("Pension", "pension", 1_200, 65),),
    )

    result = _simulate(request, np.zeros((1, 240, 2)), ("UP", "DOWN"))

    assert result.terminal_values[0] == pytest.approx(119_200)
    assert result.retained_household_cash["mean"] == pytest.approx(19_200)
    assert result.outside_income_contribution["mean"] == 0


def test_retained_cash_funds_later_spending_before_any_account_withdrawal() -> None:
    profile = RetirementProfile(65, 20, 0, planned_retirement_age=65, annual_inflation=0)
    request = _retirement_request(
        profile,
        (RetirementAccount("Roth", "roth_ira", 100_000),),
        outside=(OutsideIncome("Pension", "pension", 1_200, 65, end_age=66, taxable_fraction=0),),
    )
    request = RetirementRequest(
        **{
            **request.__dict__,
            "spending_events": (SpendingEvent("Retained-cash spending", 1_200, 66),),
        }
    )

    result = _simulate(request, np.zeros((1, 240, 2)), ("UP", "DOWN"))

    assert result.nominal_monthly_percentiles.loc[12, "P50"] == pytest.approx(101_200)
    assert result.nominal_monthly_percentiles.loc[13, "P50"] == pytest.approx(100_000)
    assert result.withdrawals_by_account_type["roth_ira"]["mean"] == 0
    assert result.lifetime_taxes_estimate["mean"] == 0
    assert result.shortfall_distribution["P50"] == 0


def test_retained_cash_earns_zero_return_until_rebalance() -> None:
    profile = RetirementProfile(65, 20, 0, planned_retirement_age=65, annual_inflation=0)
    request = _retirement_request(
        profile,
        (RetirementAccount("Roth", "roth_ira", 100_000),),
        outside=(
            OutsideIncome(
                "One-month pension",
                "pension",
                1_200,
                65,
                end_age=65 + 1 / 12,
                taxable_fraction=0,
            ),
        ),
    )
    sampled = np.zeros((1, 240, 2))
    sampled[:, 1, :] = 1.0

    result = _simulate(request, sampled, ("UP", "DOWN"))

    # Only the invested account doubles in month two; the retained $100 does not.
    assert result.nominal_monthly_percentiles.loc[2, "P50"] == pytest.approx(200_100)
    assert RETAINED_CASH_DISCLOSURE in result.limitations


def test_rmd_net_cash_covers_spending_once_without_an_additional_withdrawal() -> None:
    profile = RetirementProfile(
        current_age=75,
        planned_retirement_age=75,
        planning_horizon_years=20,
        fixed_real_annual_spending=0,
        as_of_date=date(2035, 1, 1),
        annual_inflation=0,
        owner_date_of_birth=date(1960, 1, 1),
    )
    account = RetirementAccount(
        "Traditional",
        "traditional_ira",
        1_000,
        prior_december_31_balance=24_600,
    )
    request = _retirement_request(profile, (account,))
    request = RetirementRequest(
        **{
            **request.__dict__,
            "spending_events": (SpendingEvent("December spending", 800, 75 + 11 / 12),),
        }
    )

    result = _simulate(request, np.zeros((1, 240, 2)), ("UP", "DOWN"))

    assert result.rmd_withdrawals["mean"] == pytest.approx(1_000)
    assert result.withdrawals_by_account_type["traditional_ira"]["mean"] == pytest.approx(1_000)
    assert result.cash_flow_reconciliation["gross_additional_withdrawals_mean"] == pytest.approx(0)
    assert result.cash_flow_reconciliation["rmd_withdrawal_tax_mean"] == pytest.approx(200)
    assert result.cash_flow_reconciliation["additional_withdrawal_tax_mean"] == 0
    assert result.nominal_monthly_percentiles.loc[12, "P50"] == pytest.approx(0, abs=1e-8)
    assert result.retained_household_cash["mean"] == pytest.approx(0, abs=1e-8)
    assert result.shortfall_distribution["P50"] == 0
    assert result.lifetime_taxes_estimate["mean"] == pytest.approx(200)


def test_retirement_date_value_includes_spendable_retained_cash() -> None:
    profile = RetirementProfile(
        current_age=75,
        planned_retirement_age=76,
        planning_horizon_years=20,
        fixed_real_annual_spending=0,
        as_of_date=date(2035, 1, 1),
        annual_inflation=0,
        owner_date_of_birth=date(1960, 1, 1),
    )
    account = RetirementAccount(
        "Traditional",
        "traditional_ira",
        1_000,
        prior_december_31_balance=24_600,
    )

    result = _simulate(
        _retirement_request(profile, (account,)),
        np.zeros((1, 240, 2)),
        ("UP", "DOWN"),
    )

    assert result.retirement_date_value_distribution["P50"] == pytest.approx(800)
    assert result.terminal_values[0] == pytest.approx(800)


def test_household_asset_and_shortfall_reconciliations_are_independent() -> None:
    profile = RetirementProfile(65, 20, 1_200, planned_retirement_age=65, annual_inflation=0)
    request = _retirement_request(
        profile,
        (RetirementAccount("Traditional", "traditional_ira", 10_000),),
    )

    result = _simulate(request, np.zeros((1, 240, 2)), ("UP", "DOWN"))
    assets = result.cash_flow_reconciliation
    shortfall = result.shortfall_reconciliation

    reconciled_ending_assets = (
        assets["beginning_household_assets_mean"]
        + assets["investment_return_mean"]
        - assets["fees_mean"]
        - assets["dividend_tax_drag_mean"]
        + assets["contributions_mean"]
        + assets["gross_outside_income_mean"]
        - assets["outside_income_tax_mean"]
        - assets["recurring_spending_funded_mean"]
        - assets["one_time_spending_funded_mean"]
        - assets["withdrawal_tax_mean"]
    )
    ending_assets = assets["ending_invested_balances_mean"] + assets["ending_retained_cash_mean"]

    assert reconciled_ending_assets == pytest.approx(ending_assets, abs=0.01)
    assert assets["maximum_absolute_residual"] < 0.01
    assert shortfall["required_spending_obligations_mean"] - shortfall[
        "spending_obligations_funded_mean"
    ] == pytest.approx(shortfall["unpaid_shortfall_mean"], abs=0.01)
    assert shortfall["maximum_absolute_residual"] < 0.01


def test_fixed_social_security_taxable_fraction_is_applied() -> None:
    profile = RetirementProfile(65, 20, 1_200, planned_retirement_age=65, annual_inflation=0)
    request = _retirement_request(
        profile,
        (RetirementAccount("Roth", "roth_ira", 100_000),),
        outside=(OutsideIncome("Social Security", "social_security", 1_200, 65),),
    )

    result = _simulate(request, np.zeros((1, 240, 2)), ("UP", "DOWN"))

    assert result.lifetime_taxes_estimate["mean"] == pytest.approx(4_080)


def test_qualified_roth_assumption_produces_no_withdrawal_tax() -> None:
    profile = RetirementProfile(65, 20, 1_200, planned_retirement_age=65, annual_inflation=0)
    request = _retirement_request(profile, (RetirementAccount("Roth", "roth_ira", 100_000),))

    result = _simulate(request, np.zeros((1, 240, 2)), ("UP", "DOWN"))

    assert result.withdrawals_by_account_type["roth_ira"]["mean"] == pytest.approx(24_000)
    assert result.lifetime_taxes_estimate["mean"] == 0


def test_taxable_cash_is_dollar_for_dollar_basis_and_not_embedded_gain() -> None:
    portfolio = Portfolio(name="Taxable", cash=Decimal("40"), account_type="taxable")
    portfolio.holdings = [
        HoldingLot(
            symbol="AAA",
            shares=Decimal("2"),
            acquired_on=date(2020, 1, 1),
            cost_basis=Decimal("30"),
        )
    ]

    basis = aggregate_taxable_basis(portfolio)

    assert basis == 100
    assert 100 - basis == 0
    assert (
        RetirementAccount("Taxable", "taxable", 80, taxable_cost_basis=basis).taxable_cost_basis
        == 100
    )


def test_account_specific_allocations_produce_distinct_account_returns() -> None:
    profile = RetirementProfile(65, 20, 0, planned_retirement_age=65, annual_inflation=0)
    allocated = _retirement_request(
        profile,
        (
            RetirementAccount("Up", "roth_ira", 100, allocation={"UP": 1}),
            RetirementAccount("Down", "roth_ira", 100, allocation={"DOWN": 1}),
        ),
    )
    common = _retirement_request(
        profile,
        (RetirementAccount("One", "roth_ira", 100), RetirementAccount("Two", "roth_ira", 100)),
    )
    sampled = np.zeros((1, 240, 2))
    sampled[:, 0, :] = (0.10, -0.10)

    allocated_result = _simulate(allocated, sampled, ("UP", "DOWN"))
    common_result = _simulate(common, sampled, ("UP", "DOWN"))

    assert allocated_result.terminal_values[0] == pytest.approx(200)
    assert common_result.terminal_values[0] == pytest.approx(210)


def test_annual_contribution_is_twelve_equal_month_end_deposits() -> None:
    profile = RetirementProfile(
        45,
        20,
        0,
        planned_retirement_age=46,
        annual_inflation=0,
        contribution_amount=1_200,
        contribution_frequency="annual",
    )
    request = _retirement_request(profile, (RetirementAccount("Roth", "roth_ira", 100_000),))

    result = _simulate(request, np.zeros((1, 240, 2)), ("UP", "DOWN"))

    assert result.nominal_monthly_percentiles.loc[1, "P50"] == 100_100
    assert result.nominal_monthly_percentiles.loc[12, "P50"] == 101_200


def test_parametric_final_return_at_or_below_total_loss_is_rejected() -> None:
    returns = pd.DataFrame(
        {"AAA": np.zeros(24)}, index=pd.date_range("2020-01-31", periods=24, freq="ME")
    )
    request = ParametricRequest(
        {"AAA": 1}, returns, ParametricAssumptions(1, expected_return_shift=-1)
    )

    with pytest.raises(ValueError, match="greater than -100% after shifts"):
        estimate_parameters(request)


def test_parametric_methodology_names_are_exact() -> None:
    returns = pd.DataFrame(
        {"AAA": np.zeros(24)}, index=pd.date_range("2020-01-31", periods=24, freq="ME")
    )
    estimate, *_ = estimate_parameters(
        ParametricRequest({"AAA": 1}, returns, ParametricAssumptions(1))
    )

    assert estimate.shrinkage_method.startswith("cross-sectional median shrinkage")
    assert estimate.covariance_method.startswith("fixed diagonal covariance shrinkage")


def test_simplified_hypothetical_reports_depletion_not_survival() -> None:
    probability = _retirement_probability(
        0,
        0,
        0,
        RetirementAnalysisAssumptions(horizon_years=1, annual_spending=0, simulations=10),
    )

    assert probability == 1
