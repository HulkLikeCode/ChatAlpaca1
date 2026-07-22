from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from chat_alpaca.bootstrap_forecasting import (
    BacktestCriteria,
    BacktestResult,
    BootstrapAssumptions,
    BootstrapRequest,
    run_block_bootstrap,
)
from chat_alpaca.models import MarketDataset, Portfolio
from chat_alpaca.parametric_forecasting import (
    PARAMETRIC_MODEL_TYPE,
    CapitalMarketAssumption,
    ParametricAssumptions,
    ParametricRequest,
    calibration_comparison_table,
    estimate_parameters,
    import_external_assumptions,
    model_comparison_table,
    normal_vs_fat_tail_comparison,
    parametric_sensitivity,
    rolling_parametric_backtest,
    run_parametric_forecast,
    save_parametric_run,
    validate_correlation_matrix,
    validate_covariance_matrix,
)


def _returns(months: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(91)
    values = rng.multivariate_normal(
        [0.006, 0.004],
        [[0.0016, 0.0008], [0.0008, 0.0012]],
        size=months,
    )
    return pd.DataFrame(
        values,
        columns=["AAA", "BBB"],
        index=pd.date_range("2018-01-31", periods=months, freq="ME"),
    )


def _request(
    *,
    assumptions: ParametricAssumptions | None = None,
    frame: pd.DataFrame | None = None,
    **kwargs: object,
) -> ParametricRequest:
    return ParametricRequest(
        {"AAA": 60.0, "BBB": 40.0},
        _returns() if frame is None else frame,
        assumptions or ParametricAssumptions(1, simulations=500, seed=17),
        **kwargs,
    )


def test_parametric_seed_reproducibility() -> None:
    request = _request()

    first = run_parametric_forecast(request)
    second = run_parametric_forecast(request)

    assert np.array_equal(first.terminal_values, second.terminal_values)
    assert first.monthly_percentiles.equals(second.monthly_percentiles)
    assert first.assumptions["distribution"] == "normal"
    assert first.assumptions["seed"] == 17


def test_correlation_validation_and_invalid_matrix_handling() -> None:
    valid = validate_correlation_matrix(
        pd.DataFrame([[1.0, 0.4], [0.4, 1.0]], index=["AAA", "BBB"], columns=["AAA", "BBB"]),
        ("AAA", "BBB"),
    )
    assert np.linalg.eigvalsh(valid).min() >= 0

    with pytest.raises(ValueError, match="symmetric"):
        validate_correlation_matrix([[1.0, 0.2], [0.3, 1.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_correlation_matrix([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    with pytest.raises(ValueError, match="labels"):
        run_parametric_forecast(
            _request(
                correlation_override=pd.DataFrame(
                    [[1.0, 0.0], [0.0, 1.0]], columns=["BBB", "AAA"], index=["BBB", "AAA"]
                )
            )
        )


def test_covariance_validation_rejects_negative_variance_and_non_psd_input() -> None:
    assert validate_covariance_matrix([[0.04, 0.01], [0.01, 0.09]]).shape == (2, 2)
    with pytest.raises(ValueError, match="variances"):
        validate_covariance_matrix([[-0.01, 0.0], [0.0, 0.02]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_covariance_matrix([[0.01, 0.02], [0.02, 0.01]])


def test_student_t_has_fatter_extreme_downside_than_normal() -> None:
    request = _request(
        assumptions=ParametricAssumptions(
            1, simulations=30_000, seed=22, parameter_uncertainty=False
        )
    )

    comparison = normal_vs_fat_tail_comparison(request, degrees_of_freedom=3)
    normal = comparison.set_index("distribution").loc["normal"]
    student = comparison.set_index("distribution").loc["student_t"]

    assert student.degrees_of_freedom == 3
    assert student.p1_terminal < normal.p1_terminal


def test_parameter_blending_uses_shrunk_history_and_external_assumptions() -> None:
    external = {
        "AAA": CapitalMarketAssumption(
            annual_return=0.12, annual_volatility=0.30, source="CMA 2026"
        )
    }
    request = _request(external_assumptions=external)
    historical, *_ = estimate_parameters(_request())
    blended, *_ = estimate_parameters(request)

    assert (
        min(historical.annual_returns[0], 0.12)
        < blended.annual_returns[0]
        < max(historical.annual_returns[0], 0.12)
    )
    assert blended.parameter_sources["AAA"]["external"]["source"] == "CMA 2026"
    assert blended.shrinkage_method.startswith("cross-sectional median shrinkage")
    assert blended.covariance_method.startswith("fixed diagonal covariance shrinkage")


def test_correlation_multiplier_scales_only_off_diagonal_entries_exactly() -> None:
    override = pd.DataFrame(
        [[1.0, 0.8], [0.8, 1.0]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
    )
    estimate, *_ = estimate_parameters(
        _request(
            assumptions=ParametricAssumptions(
                1,
                simulations=10,
                seed=1,
                parameter_uncertainty=False,
                correlation_multiplier=0.25,
            ),
            correlation_override=override,
        )
    )

    assert estimate.correlation == pytest.approx(np.array([[1.0, 0.2], [0.2, 1.0]]))


def test_user_overrides_take_precedence_and_are_disclosed() -> None:
    override = {
        "AAA": CapitalMarketAssumption(annual_return=0.03, annual_volatility=0.07, source="Owner")
    }
    estimate, *_ = estimate_parameters(_request(user_overrides=override))

    assert estimate.annual_returns[0] == pytest.approx(0.03)
    assert estimate.annual_volatilities[0] == pytest.approx(0.07)
    assert estimate.parameter_sources["AAA"]["return_weights"] == {"user_override": 1.0}


def test_external_assumption_csv_import() -> None:
    imported = import_external_assumptions(
        StringIO(
            "symbol,annual_return,annual_volatility,source,publication,as_of_date\n"
            "AAA,0.07,0.16,Publisher,2026 outlook,2026-01-01\n"
            "BBB,0.05,0.10,Owner import,,\n"
        )
    )

    assert imported["AAA"].annual_return == 0.07
    assert imported["AAA"].publication == "2026 outlook"
    with pytest.raises(ValueError, match="duplicate"):
        import_external_assumptions(
            "symbol,annual_return,annual_volatility\nAAA,.1,.2\nAAA,.2,.3\n"
        )


def test_sensitivity_covers_returns_volatility_and_correlations() -> None:
    result = parametric_sensitivity(
        _request(assumptions=ParametricAssumptions(1, simulations=300, seed=8)),
        return_shifts=(-0.02, 0.02),
        volatility_multipliers=(0.8, 1.2),
        correlation_multipliers=(0.5, 1.0),
    )

    assert set(result.parameter) == {"expected_return", "volatility", "correlation"}
    expected = result[result.parameter == "expected_return"].sort_values("value")
    assert expected.iloc[0].median_terminal < expected.iloc[1].median_terminal


def test_rolling_backtest_is_compatible_and_never_self_validates() -> None:
    request = _request(
        frame=_returns(48),
        assumptions=ParametricAssumptions(
            1, simulations=100, seed=4, minimum_history_months=24, parameter_uncertainty=False
        ),
    )
    result = rolling_parametric_backtest(
        request,
        criteria=BacktestCriteria(
            minimum_valid_windows=3,
            interval_coverage_min=0,
            maximum_absolute_median_bias=1,
            minimum_downside_coverage=0,
        ),
    )

    assert result.valid_windows == 13
    assert result.invalid_windows == 0
    assert result.validation_status in {"eligible_for_review", "unvalidated"}
    assert result.validation_status != "validated"


def test_model_comparison_has_like_for_like_outputs() -> None:
    frame = _returns()
    parametric = run_parametric_forecast(_request(frame=frame))
    bootstrap = run_block_bootstrap(
        BootstrapRequest(
            {"AAA": 60.0, "BBB": 40.0},
            frame,
            BootstrapAssumptions(1, simulations=500, seed=17),
        )
    )

    comparison = model_comparison_table(bootstrap, parametric)

    assert list(comparison.model) == [bootstrap.model_type, PARAMETRIC_MODEL_TYPE]
    assert {"p5_terminal", "median_terminal", "p95_terminal", "probability_real_loss"} <= set(
        comparison.columns
    )


def test_calibration_comparison_does_not_rank_models() -> None:
    criteria = BacktestCriteria(minimum_valid_windows=1)
    bootstrap = BacktestResult(
        0.9, 0.01, 0.95, 12, 0, 1, True, "eligible_for_review", pd.DataFrame(), criteria
    )
    parametric = BacktestResult(
        0.85, -0.02, 0.92, 12, 1, 0, False, "unvalidated", pd.DataFrame(), criteria
    )

    comparison = calibration_comparison_table(bootstrap, parametric)

    assert list(comparison.model) == ["historical_block_bootstrap", PARAMETRIC_MODEL_TYPE]
    assert "winner" not in comparison.columns
    assert list(comparison.validation_status) == ["eligible_for_review", "unvalidated"]


def test_persistence_saves_parameters_sources_and_validation(session: Session) -> None:
    portfolio = Portfolio(name="Phase 9", cash=Decimal("100"), account_type="taxable")
    dataset = MarketDataset(
        provider="owner",
        source="fixture",
        timeframe="1Day",
        adjustment_method="split",
        retrieved_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
        coverage_start=date(2018, 1, 1),
        coverage_end=date(2022, 12, 31),
        quality_status="validated",
    )
    session.add_all([portfolio, dataset])
    session.flush()
    result = run_parametric_forecast(
        _request(
            assumptions=ParametricAssumptions(
                1,
                distribution="student_t",
                degrees_of_freedom=5,
                simulations=50,
                seed=41,
            ),
            dataset_ids=(dataset.id,),
            external_assumptions={
                "AAA": CapitalMarketAssumption(0.08, 0.15, "Published owner input")
            },
        )
    )

    run = save_parametric_run(session, [portfolio], result)
    assumptions = json.loads(run.assumptions)
    summary = json.loads(run.summary_outputs)

    assert assumptions["distribution"] == "student_t"
    assert assumptions["degrees_of_freedom"] == 5
    assert assumptions["seed"] == 41
    assert summary["parameter_estimates"]["shrinkage_method"]
    assert summary["parameter_estimates"]["covariance_method"]
    assert summary["parameter_sources"]["AAA"]["external"]["source"] == "Published owner input"
    assert [(item.dataset_id, item.purpose) for item in run.dataset_references] == [
        (dataset.id, "parametric_estimation")
    ]
    assert run.validation_status == "unvalidated"
    assert "terminal_values" not in summary
