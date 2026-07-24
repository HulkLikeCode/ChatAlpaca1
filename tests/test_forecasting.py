from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from chat_alpaca.forecasting import (
    LEGACY_FORECAST_MODEL_TYPE,
    LEGACY_FORECAST_MODEL_VERSION,
    ForecastAssumptions,
    build_forecast_request,
    projection_calendar_dates,
    run_forecast,
    simulate_portfolio_projection,
)
from chat_alpaca.models import HoldingLot, Portfolio


def test_projection_without_volatility_has_known_month_end_values() -> None:
    result = simulate_portfolio_projection(
        current_value=100.0,
        annual_return=0.12,
        annual_volatility=0.0,
        monthly_contribution=10.0,
        horizon_years=1,
        simulations=3,
    )

    expected = 238.4649790835
    assert result.annual_percentiles.loc[1].tolist() == pytest.approx([expected] * 5, abs=1e-9)
    assert result.contract.model_type == LEGACY_FORECAST_MODEL_TYPE
    assert result.contract.model_version == LEGACY_FORECAST_MODEL_VERSION
    assert result.contract.seed == 20260719
    assert result.contract.simulation_count == 3
    assert result.contract.assumptions == ForecastAssumptions(0.12, 0.0, 10.0, 1)
    assert result.contract.source_valuation_methodology == "direct_inputs"
    assert result.contract.result_generated_at.tzinfo is not None


def test_projection_is_seeded_and_reports_target_probability() -> None:
    arguments = dict(
        current_value=100_000.0,
        annual_return=0.07,
        annual_volatility=0.15,
        monthly_contribution=500.0,
        horizon_years=2,
        target_value=115_000.0,
        simulations=1_000,
        seed=7,
    )
    first = simulate_portfolio_projection(**arguments)
    second = simulate_portfolio_projection(**arguments)

    assert first.monthly_percentiles.equals(second.monthly_percentiles)
    assert 0 <= first.target_probability <= 1
    assert list(first.annual_percentiles.index) == [0, 1, 2]


def test_projection_calendar_dates_use_explicit_as_of_across_year_and_leap_day() -> None:
    result = simulate_portfolio_projection(
        current_value=100.0,
        annual_return=0.0,
        annual_volatility=0.0,
        monthly_contribution=0.0,
        horizon_years=1,
        simulations=1,
        source_valuation_date=date(2023, 12, 15),
    )
    altered_clock = replace(
        result.contract,
        result_generated_at=datetime(2040, 6, 1, 23, 30, tzinfo=timezone.utc),
    )

    dates = projection_calendar_dates(altered_clock)

    assert dates[0] == pd.Timestamp("2023-12-15")
    assert dates[1] == pd.Timestamp("2024-01-31")
    assert dates[2] == pd.Timestamp("2024-02-29")
    assert dates[12] == pd.Timestamp("2024-12-31")
    assert dates.equals(projection_calendar_dates(result.contract))


@pytest.mark.parametrize(
    ("starting_date", "first_step", "final_step", "annual_year"),
    [
        (date(2025, 10, 20), "2025-11-30", "2026-10-31", 2026),
        (date(2025, 3, 20), "2025-04-30", "2026-03-31", 2026),
    ],
)
def test_projection_calendar_dates_preserve_partial_calendar_years(
    starting_date: date, first_step: str, final_step: str, annual_year: int
) -> None:
    result = simulate_portfolio_projection(
        current_value=100.0,
        annual_return=0.07,
        annual_volatility=0.12,
        monthly_contribution=10.0,
        horizon_years=1,
        simulations=10,
        seed=7,
        source_valuation_date=starting_date,
    )
    monthly_before = result.monthly_percentiles.copy(deep=True)
    annual_before = result.annual_percentiles.copy(deep=True)

    dates = projection_calendar_dates(result.contract)

    assert len(dates) == 13
    assert dates[1] == pd.Timestamp(first_step)
    assert dates[-1] == pd.Timestamp(final_step)
    assert dates[12].year == annual_year
    pd.testing.assert_frame_equal(result.monthly_percentiles, monthly_before)
    pd.testing.assert_frame_equal(result.annual_percentiles, annual_before)


def test_projection_calendar_dates_fall_back_to_utc_generation_date() -> None:
    result = simulate_portfolio_projection(
        current_value=100.0,
        annual_return=0.0,
        annual_volatility=0.0,
        monthly_contribution=0.0,
        horizon_years=1,
        simulations=1,
    )
    contract = replace(
        result.contract,
        source_valuation_date=None,
        result_generated_at=datetime(2026, 12, 31, 23, 30, tzinfo=timezone.utc),
    )

    dates = projection_calendar_dates(contract)

    assert dates[0] == pd.Timestamp("2026-12-31")
    assert dates[1] == pd.Timestamp("2027-01-31")


@pytest.mark.parametrize(
    "arguments",
    [
        {"current_value": 0},
        {"annual_return": -1},
        {"annual_volatility": -0.1},
        {"monthly_contribution": -1},
        {"horizon_years": 11},
    ],
)
def test_projection_rejects_invalid_assumptions(arguments: dict[str, float]) -> None:
    values = {
        "current_value": 100.0,
        "annual_return": 0.07,
        "annual_volatility": 0.15,
        "monthly_contribution": 0.0,
        "horizon_years": 1,
    }
    values.update(arguments)

    with pytest.raises(ValueError):
        simulate_portfolio_projection(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_value", True),
        ("current_value", float("nan")),
        ("annual_return", float("inf")),
        ("annual_volatility", float("nan")),
        ("monthly_contribution", float("-inf")),
        ("horizon_years", True),
        ("target_value", float("nan")),
        ("simulations", True),
        ("simulations", float("inf")),
        ("seed", True),
    ],
)
def test_projection_rejects_boolean_and_nonfinite_inputs(field: str, value: object) -> None:
    values = {
        "current_value": 100.0,
        "annual_return": 0.07,
        "annual_volatility": 0.15,
        "monthly_contribution": 0.0,
        "horizon_years": 1,
        "target_value": 200.0,
        "simulations": 10,
        "seed": 7,
    }
    values[field] = value

    with pytest.raises(ValueError):
        simulate_portfolio_projection(**values)


def test_forecast_request_discloses_cost_basis_fallback() -> None:
    portfolio = Portfolio(id=1, name="Example", cash=Decimal("5"))
    portfolio.holdings = [
        HoldingLot(
            symbol="ABC",
            shares=Decimal("2"),
            acquired_on=date(2026, 1, 1),
            cost_basis=Decimal("10"),
        )
    ]
    assumptions = ForecastAssumptions(0.07, 0.12, 0, 10)

    request = build_forecast_request([portfolio], pd.DataFrame(), assumptions)

    assert request.current_value == 25
    assert request.valuation_basis == "cost_basis_plus_cash"
    assert any("explicitly uses cost basis" in warning for warning in request.warnings)


def test_forecast_request_does_not_silently_fill_incomplete_prices() -> None:
    portfolio = Portfolio(id=1, name="Example", cash=Decimal("5"))
    portfolio.holdings = [
        HoldingLot(
            symbol="ABC",
            shares=Decimal("2"),
            acquired_on=date(2026, 1, 1),
            cost_basis=Decimal("10"),
        )
    ]
    closes = pd.DataFrame({"OTHER": [10.0]}, index=pd.to_datetime(["2026-01-02"]))

    with pytest.raises(ValueError, match="Incomplete portfolios: Example"):
        build_forecast_request([portfolio], closes, ForecastAssumptions(0.07, 0.12, 0, 10))


def test_forecast_request_uses_common_confirmed_household_value() -> None:
    first = Portfolio(id=1, name="First", cash=Decimal("0"))
    first.holdings = [
        HoldingLot(
            symbol="AAA",
            shares=Decimal("2"),
            acquired_on=date(2026, 1, 1),
            cost_basis=Decimal("80"),
        )
    ]
    second = Portfolio(id=2, name="Second", cash=Decimal("0"))
    second.holdings = [
        HoldingLot(
            symbol="BBB",
            shares=Decimal("3"),
            acquired_on=date(2026, 1, 1),
            cost_basis=Decimal("40"),
        )
    ]
    closes = pd.DataFrame(
        {"AAA": [100.0, 110.0], "BBB": [50.0, float("nan")]},
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )

    request = build_forecast_request(
        [first, second], closes, ForecastAssumptions(0.07, 0.12, 0, 10)
    )

    assert request.current_value == 350.0
    assert request.portfolio_ids == (1, 2)
    assert request.portfolio_names == ("First", "Second")
    assert "2026-01-02" in request.coverage
    assert request.source_valuation_date == date(2026, 1, 2)

    result = run_forecast(request)
    assert result.contract.source_valuation_date == date(2026, 1, 2)
    assert result.contract.source_valuation_methodology == "confirmed_market_value"
