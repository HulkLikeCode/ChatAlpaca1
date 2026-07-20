from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from chat_alpaca.bootstrap_forecasting import (
    BOOTSTRAP_MODEL_TYPE,
    BacktestCriteria,
    BootstrapAssumptions,
    BootstrapRequest,
    monthly_returns_from_prices,
    rolling_origin_backtest,
    run_block_bootstrap,
    sample_block_indices,
    save_bootstrap_run,
)
from chat_alpaca.models import MarketDataset, Portfolio


def _returns(months: int = 36, **columns: float | np.ndarray) -> pd.DataFrame:
    index = pd.date_range("2020-01-31", periods=months, freq="ME")
    return pd.DataFrame(
        {
            symbol: np.full(months, value) if np.isscalar(value) else value
            for symbol, value in columns.items()
        },
        index=index,
    )


def _request(
    frame: pd.DataFrame,
    *,
    assumptions: BootstrapAssumptions | None = None,
    values: dict[str, float] | None = None,
    **kwargs: object,
) -> BootstrapRequest:
    return BootstrapRequest(
        values or {str(frame.columns[0]): 100.0},
        frame,
        assumptions or BootstrapAssumptions(1, simulations=100, seed=7),
        **kwargs,
    )


def test_seed_reproducibility_and_block_preservation() -> None:
    frame = _returns(36, AAA=np.linspace(-0.1, 0.1, 36))
    request = _request(frame)

    first = run_block_bootstrap(request)
    second = run_block_bootstrap(request)
    indices = sample_block_indices(36, 12, 6, 10, 7)

    assert np.array_equal(first.terminal_values, second.terminal_values)
    assert first.monthly_percentiles.equals(second.monthly_percentiles)
    for row in indices:
        for start in range(0, 12, 6):
            block = row[start : start + 6]
            assert np.all((np.diff(block) % 36) == 1)


def test_holding_sampling_preserves_perfect_cross_asset_correlation() -> None:
    observations = np.tile(np.array([-0.05, 0.02, 0.08]), 12)
    frame = _returns(36, AAA=observations, BBB=observations)
    combined = run_block_bootstrap(_request(frame, values={"AAA": 60.0, "BBB": 40.0}))
    single = run_block_bootstrap(_request(frame, values={"AAA": 100.0}))

    assert combined.terminal_values == pytest.approx(single.terminal_values)


def test_contributions_inflation_and_fees() -> None:
    frame = _returns(36, AAA=0.0)
    contribution = run_block_bootstrap(
        _request(
            frame,
            assumptions=BootstrapAssumptions(
                1, simulations=5, seed=1, monthly_contribution=10, minimum_history_months=24
            ),
        )
    )
    fee = run_block_bootstrap(
        _request(
            frame,
            assumptions=BootstrapAssumptions(
                1, simulations=5, seed=1, annual_fee=0.12, minimum_history_months=24
            ),
        )
    )
    inflation = run_block_bootstrap(
        _request(
            frame,
            assumptions=BootstrapAssumptions(
                1, simulations=5, seed=1, annual_inflation=0.03, minimum_history_months=24
            ),
        )
    )

    assert contribution.terminal_values == pytest.approx(np.full(5, 220.0))
    assert fee.terminal_values == pytest.approx(np.full(5, 88.0))
    assert contribution.probability_nominal_loss == 0
    assert inflation.probability_real_loss == 1


def test_rebalancing_changes_holding_level_outcomes() -> None:
    a = np.tile([1.0, 0.0, 0.0], 12)
    b = np.tile([0.0, 1.0, 0.0], 12)
    frame = _returns(36, AAA=a, BBB=b)
    common = dict(horizon_years=1, simulations=20, seed=3, minimum_history_months=24)
    monthly = run_block_bootstrap(
        _request(
            frame,
            values={"AAA": 50, "BBB": 50},
            assumptions=BootstrapAssumptions(**common, rebalancing="monthly"),
        )
    )
    never = run_block_bootstrap(
        _request(
            frame,
            values={"AAA": 50, "BBB": 50},
            assumptions=BootstrapAssumptions(**common, rebalancing="never"),
        )
    )

    assert not np.array_equal(monthly.terminal_values, never.terminal_values)


def test_proxy_usage_is_disclosed_and_lowers_sufficiency() -> None:
    frame = _returns(36, NEW=np.r_[np.full(20, np.nan), np.zeros(16)], SPY=0.01)
    result = run_block_bootstrap(_request(frame, values={"NEW": 100}, proxies={"NEW": "SPY"}))

    assert result.proxies == {"NEW": "SPY"}
    assert result.data_coverage["status"] == "limited"
    assert "NEW uses explicit return-history proxy SPY." in result.warnings


def test_insufficient_history_requires_a_usable_explicit_proxy() -> None:
    frame = _returns(18, NEW=0.01, SPY=0.01)
    assumptions = BootstrapAssumptions(1, minimum_history_months=24)

    with pytest.raises(ValueError, match="explicit proxy"):
        run_block_bootstrap(_request(frame, assumptions=assumptions, values={"NEW": 100}))
    with pytest.raises(ValueError, match="also has insufficient"):
        run_block_bootstrap(
            _request(
                frame,
                assumptions=assumptions,
                values={"NEW": 100},
                proxies={"NEW": "SPY"},
            )
        )


def test_outputs_include_target_benchmark_and_downside_contributors() -> None:
    frame = _returns(
        36,
        AAA=np.tile([-0.1, 0.03, 0.02], 12),
        BBB=np.tile([-0.02, 0.01, 0.01], 12),
    )
    benchmark = pd.Series(0.0, index=frame.index)
    result = run_block_bootstrap(
        _request(
            frame,
            values={"AAA": 70, "BBB": 30},
            benchmark_returns=benchmark,
            assumptions=BootstrapAssumptions(1, simulations=100, seed=9, target_value=110),
        )
    )

    assert 0 <= result.target_probability <= 1
    assert 0 <= result.probability_beating_benchmark <= 1
    assert set(result.downside_percentiles) == {"P5", "P10", "P25"}
    assert list(result.downside_contribution_by_holding) == ["AAA", "BBB"]
    assert result.terminal_distribution["count"] == 100


def test_monthly_returns_does_not_forward_fill_missing_months() -> None:
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0]},
        index=pd.to_datetime(["2020-01-31", "2020-03-31"]),
    )

    returns = monthly_returns_from_prices(prices)

    assert returns["AAA"].isna().all()


def test_persistence_saves_summaries_and_not_raw_paths(session: Session) -> None:
    portfolio = Portfolio(name="Phase 8", cash=Decimal("100"), account_type="taxable")
    dataset = MarketDataset(
        provider="test",
        source="fixture",
        timeframe="1Day",
        adjustment_method="split",
        retrieved_at=datetime(2026, 7, 19, tzinfo=timezone.utc),
        coverage_start=date(2020, 1, 1),
        coverage_end=date(2022, 12, 31),
        quality_status="validated",
    )
    session.add_all([portfolio, dataset])
    session.flush()
    result = run_block_bootstrap(
        _request(
            _returns(36, AAA=0.01),
            assumptions=BootstrapAssumptions(1, simulations=25, seed=44),
            dataset_ids=(dataset.id,),
        )
    )

    run = save_bootstrap_run(session, [portfolio], result)
    assumptions = json.loads(run.assumptions)
    summary = json.loads(run.summary_outputs)

    assert run.model_type == BOOTSTRAP_MODEL_TYPE
    assert assumptions["seed"] == 44
    assert assumptions["simulations"] == 25
    assert assumptions["block_length"] == 6
    assert json.loads(run.data_coverage)["data_start"] == "2020-01-31"
    assert [(row.dataset_id, row.purpose) for row in run.dataset_references] == [
        (dataset.id, "bootstrap_returns")
    ]
    assert "terminal_values" not in summary
    assert "paths" not in summary


def test_rolling_origin_backtests_and_never_claims_validation() -> None:
    request = _request(
        _returns(60, AAA=0.01),
        assumptions=BootstrapAssumptions(
            1, simulations=20, seed=5, block_length=3, minimum_history_months=24
        ),
    )
    result = rolling_origin_backtest(
        request,
        criteria=BacktestCriteria(
            minimum_valid_windows=3,
            interval_coverage_min=0.5,
            maximum_absolute_median_bias=0.01,
            minimum_downside_coverage=0.5,
        ),
    )

    assert result.valid_windows == 25
    assert result.invalid_windows == 0
    assert result.forecast_interval_coverage == 1
    assert result.median_forecast_bias == pytest.approx(0)
    assert result.criteria_met
    assert result.validation_status == "eligible_for_review"


def test_rolling_backtest_reports_insufficient_windows() -> None:
    request = _request(
        _returns(30, AAA=0.01),
        assumptions=BootstrapAssumptions(1, simulations=5, minimum_history_months=24),
    )

    result = rolling_origin_backtest(request)

    assert result.valid_windows == 0
    assert result.insufficient_windows > 0
    assert result.validation_status == "unvalidated"
