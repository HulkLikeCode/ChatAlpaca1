from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from chat_alpaca.models import MarketDataset, Portfolio
from chat_alpaca.retirement import (
    RETIREMENT_MODEL_TYPE,
    OutsideIncome,
    RetirementAccount,
    RetirementAssumptions,
    RetirementProfile,
    RetirementRequest,
    RetirementTaxAssumptions,
    SpendingEvent,
    historical_sequence_replay,
    retirement_sensitivity,
    run_retirement_forecast,
    save_retirement_run,
)


def _returns(months: int = 300, value: float | np.ndarray = 0.0) -> pd.DataFrame:
    values = np.full(months, value) if np.isscalar(value) else value
    return pd.DataFrame(
        {"AAA": values}, index=pd.date_range("2000-01-31", periods=months, freq="ME")
    )


def _request(
    *,
    current_age: float = 65,
    retirement_age: float = 65,
    horizon: int = 20,
    spending: float = 12_000,
    accounts: tuple[RetirementAccount, ...] | None = None,
    returns: pd.DataFrame | None = None,
    contribution: float = 0,
    inflation: float = 0,
    outside: tuple[OutsideIncome, ...] = (),
    events: tuple[SpendingEvent, ...] = (),
    taxes: RetirementTaxAssumptions | None = None,
    assumptions: RetirementAssumptions | None = None,
    target: float | None = None,
) -> RetirementRequest:
    return RetirementRequest(
        RetirementProfile(
            current_age=current_age,
            planned_retirement_age=retirement_age,
            planning_horizon_years=horizon,
            fixed_real_annual_spending=spending,
            annual_inflation=inflation,
            contribution_amount=contribution,
            target_estate_value=target,
        ),
        accounts or (RetirementAccount("Taxable", "taxable", 500_000),),
        {"AAA": 500_000},
        _returns() if returns is None else returns,
        assumptions or RetirementAssumptions(simulations=8, seed=9, minimum_history_months=24),
        taxes
        or RetirementTaxAssumptions(
            ordinary_income_rate=0,
            capital_gains_rate=0,
            dividend_tax_rate=0,
        ),
        outside,
        events,
    )


def test_accumulation_period_and_retirement_transition() -> None:
    result = run_retirement_forecast(
        _request(
            current_age=45,
            retirement_age=55,
            contribution=100,
            spending=12_000,
            accounts=(RetirementAccount("Roth", "roth_ira", 100_000),),
        )
    )

    assert result.nominal_monthly_percentiles.loc[119, "P50"] == pytest.approx(111_900)
    assert result.retirement_date_value_distribution["P50"] == pytest.approx(112_000)
    assert result.nominal_monthly_percentiles.loc[121, "P50"] == pytest.approx(111_000)


def test_fixed_real_spending_inflation_and_real_nominal_reporting() -> None:
    no_inflation = run_retirement_forecast(_request(inflation=0))
    inflation = run_retirement_forecast(_request(inflation=0.03))

    assert no_inflation.terminal_values == pytest.approx(np.full(8, 260_000))
    assert inflation.terminal_values.mean() < no_inflation.terminal_values.mean()
    assert inflation.real_monthly_percentiles.iloc[-1].P50 < inflation.terminal_values.mean()
    assert inflation.nominal_annual_percentiles.index.name == "Year"


def test_social_security_and_pension_reduce_portfolio_withdrawals() -> None:
    baseline = run_retirement_forecast(_request())
    income = run_retirement_forecast(
        _request(
            outside=(
                OutsideIncome("Social Security", "social_security", 6_000, 65),
                OutsideIncome("Pension", "pension", 3_000, 65),
            )
        )
    )

    assert income.outside_income_contribution["mean"] == pytest.approx(180_000)
    assert income.withdrawals_by_account_type["taxable"]["mean"] == pytest.approx(60_000)
    assert baseline.withdrawals_by_account_type["taxable"]["mean"] == pytest.approx(240_000)


def test_retirement_and_social_security_support_calendar_dates() -> None:
    request = RetirementRequest(
        RetirementProfile(
            current_age=55,
            planned_retirement_date=date(2036, 7, 1),
            planning_horizon_years=20,
            fixed_real_annual_spending=12_000,
            contribution_amount=100,
            annual_inflation=0,
        ),
        (RetirementAccount("Roth", "roth_ira", 500_000),),
        {"AAA": 500_000},
        _returns(),
        RetirementAssumptions(simulations=4),
        RetirementTaxAssumptions(ordinary_income_rate=0, capital_gains_rate=0, dividend_tax_rate=0),
        (OutsideIncome("Social Security", "social_security", 12_000, start_date=date(2036, 7, 1)),),
    )

    result = run_retirement_forecast(request)

    assert request.profile.retirement_month == 120
    assert result.retirement_date_value_distribution["P50"] == pytest.approx(512_000)
    assert result.terminal_values == pytest.approx(np.full(4, 512_000))


def test_traditional_roth_taxable_and_withdrawal_ordering() -> None:
    accounts = (
        RetirementAccount("Traditional", "traditional_ira", 100_000),
        RetirementAccount("Roth", "roth_ira", 100_000),
        RetirementAccount("Taxable", "taxable", 100_000),
        RetirementAccount("Unknown", "unknown", 100_000),
    )
    result = run_retirement_forecast(
        _request(
            spending=12_000,
            accounts=accounts,
            taxes=RetirementTaxAssumptions(
                ordinary_income_rate=0.20,
                capital_gains_rate=0.10,
                dividend_tax_rate=0,
                taxable_realization_fraction=0.50,
                unknown_account_withdrawal_rate=0.25,
            ),
            assumptions=RetirementAssumptions(
                simulations=4,
                withdrawal_order=("taxable", "traditional_ira", "unknown", "roth_ira"),
            ),
        )
    )

    assert result.withdrawals_by_account_type["taxable"]["mean"] == pytest.approx(100_000)
    assert result.withdrawals_by_account_type["traditional_ira"]["mean"] == pytest.approx(100_000)
    assert result.withdrawals_by_account_type["unknown"]["mean"] > 0
    assert result.withdrawals_by_account_type["roth_ira"]["mean"] == pytest.approx(0)
    assert result.lifetime_taxes_estimate["mean"] > 0


def test_roth_is_tax_free_and_traditional_withdrawals_are_tax_deferred_then_ordinary() -> None:
    roth = run_retirement_forecast(
        _request(accounts=(RetirementAccount("Roth", "roth_ira", 500_000),))
    )
    traditional = run_retirement_forecast(
        _request(
            accounts=(RetirementAccount("Traditional", "traditional_ira", 500_000),),
            taxes=RetirementTaxAssumptions(
                ordinary_income_rate=0.20, capital_gains_rate=0, dividend_tax_rate=0
            ),
        )
    )

    assert roth.lifetime_taxes_estimate["mean"] == 0
    assert roth.withdrawals_by_account_type["roth_ira"]["mean"] == pytest.approx(240_000)
    assert traditional.withdrawals_by_account_type["traditional_ira"]["mean"] == pytest.approx(
        300_000
    )
    assert traditional.lifetime_taxes_estimate["mean"] == pytest.approx(60_000)


def test_depletion_shortfall_and_depletion_age_distribution() -> None:
    result = run_retirement_forecast(
        _request(accounts=(RetirementAccount("Roth", "roth_ira", 12_000),))
    )

    assert result.probability_funding_full_horizon == 0
    assert result.probability_depletion == 1
    assert result.depletion_age_distribution["P50"] == pytest.approx(66)
    assert result.shortfall_distribution["P50"] == pytest.approx(228_000)


def test_one_time_spending_target_estate_and_worst_decile_outputs() -> None:
    result = run_retirement_forecast(
        _request(events=(SpendingEvent("Roof", 20_000, 70),), target=250_000)
    )

    assert result.terminal_values == pytest.approx(np.full(8, 240_000))
    assert result.target_estate_probability == 0
    assert {"early_retirement_return", "terminal_real", "total_shortfall"} <= set(
        result.worst_decile_scenarios
    )


def test_seed_reproducibility_and_sequence_risk_diagnostics() -> None:
    pattern = np.tile(np.array([-0.08, 0.01, 0.04, 0.02]), 75)
    request = _request(returns=_returns(value=pattern))

    first = run_retirement_forecast(request)
    second = run_retirement_forecast(request)

    assert np.array_equal(first.terminal_values, second.terminal_values)
    assert first.nominal_monthly_percentiles.equals(second.nominal_monthly_percentiles)
    assert "depletion_probability_low_early_return_quartile" in first.sequence_risk_diagnostics


def test_sensitivity_covers_required_planning_and_tax_dimensions() -> None:
    request = _request(
        assumptions=RetirementAssumptions(
            engine="parametric", simulations=30, seed=3, parameter_uncertainty=False
        ),
        outside=(OutsideIncome("Social Security", "social_security", 6_000, 67),),
    )
    spending = retirement_sensitivity(request, "spending", (8_000, 20_000))
    retirement = retirement_sensitivity(request, "retirement_age", (65, 67))
    social_security = retirement_sensitivity(request, "social_security_start_age", (65, 70))
    returns = retirement_sensitivity(request, "expected_return", (-0.02, 0.02))
    tax = retirement_sensitivity(request, "ordinary_income_tax", (0.1, 0.3))
    order = retirement_sensitivity(
        request,
        "withdrawal_order",
        (("taxable", "traditional_ira", "roth_ira"), ("roth_ira", "taxable")),
    )

    assert spending.iloc[0].median_terminal_real > spending.iloc[1].median_terminal_real
    assert set(retirement.parameter) == {"retirement_age"}
    assert set(social_security.parameter) == {"social_security_start_age"}
    assert returns.iloc[0].median_terminal_real < returns.iloc[1].median_terminal_real
    assert set(tax.parameter) == {"ordinary_income_tax"}
    assert set(order.parameter) == {"withdrawal_order"}


def test_historical_sequence_replay_and_rolling_backtest_contract() -> None:
    replay = historical_sequence_replay(_request(returns=_returns(245)))

    assert replay.valid_windows == 6
    assert replay.insufficient_windows == 0
    assert replay.probability_funding_full_horizon == 1
    assert replay.validation_status == "unvalidated"
    assert len(replay.window_results) == 6


def test_historical_replay_reports_insufficient_history() -> None:
    replay = historical_sequence_replay(_request(returns=_returns(100)))

    assert replay.valid_windows == 0
    assert replay.insufficient_windows == 1
    assert replay.probability_depletion is None


def test_persistence_excludes_raw_paths_and_scenarios(session: Session) -> None:
    portfolio = Portfolio(name="Phase 10", cash=Decimal("500000"), account_type="taxable")
    dataset = MarketDataset(
        provider="test",
        source="fixture",
        timeframe="1Day",
        adjustment_method="split",
        retrieved_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
        coverage_start=date(2000, 1, 1),
        coverage_end=date(2024, 12, 31),
        quality_status="validated",
    )
    session.add_all([portfolio, dataset])
    session.flush()
    request = _request()
    request = RetirementRequest(**{**request.__dict__, "dataset_ids": (dataset.id,)})
    result = run_retirement_forecast(request)
    replay = historical_sequence_replay(request)

    run = save_retirement_run(session, [portfolio], result, replay=replay)
    summary = json.loads(run.summary_outputs)
    assumptions = json.loads(run.assumptions)

    assert run.model_type == RETIREMENT_MODEL_TYPE
    assert assumptions["profile"]["planning_horizon_years"] == 20
    assert summary["historical_replay"]["validation_status"] == "unvalidated"
    assert "terminal_values" not in summary
    assert "scenario_shortfalls" not in summary
    assert [(item.dataset_id, item.purpose) for item in run.dataset_references] == [
        (dataset.id, "retirement_returns")
    ]


def test_tax_limitations_do_not_claim_tax_advisory_precision() -> None:
    result = run_retirement_forecast(_request())
    text = " ".join(result.limitations).lower()

    assert "not tax advice" in text
    assert "state/local" in text
    assert "rmd" in text
