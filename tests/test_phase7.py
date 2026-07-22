from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_alpaca.models import (
    ForecastRun,
    HoldingLot,
    MarketDataset,
    Portfolio,
    PortfolioTransaction,
)
from chat_alpaca.scenarios import (
    DETERMINISTIC_MODEL_TYPE,
    DETERMINISTIC_MODEL_VERSION,
    DIVIDEND_DISCLOSURE,
    SHOCK_DISCLOSURE,
    DatasetReference,
    ScenarioAssumptions,
    ScenarioType,
    ledger_state_hash,
    record_validation_evidence,
    run_deterministic_scenario,
    save_scenario_run,
    sensitivity_grid,
)


def _portfolio(session: Session, name: str = "Household") -> Portfolio:
    portfolio = Portfolio(name=name, cash=Decimal("100"), account_type="taxable")
    portfolio.holdings = [
        HoldingLot(
            symbol="AAA",
            shares=Decimal("10"),
            acquired_on=date(2025, 1, 1),
            cost_basis=Decimal("8"),
        ),
        HoldingLot(
            symbol="BBB",
            shares=Decimal("5"),
            acquired_on=date(2025, 1, 1),
            cost_basis=Decimal("15"),
        ),
    ]
    session.add(portfolio)
    session.flush()
    return portfolio


def _dataset(session: Session) -> MarketDataset:
    row = MarketDataset(
        provider="test",
        source="fixture",
        feed=None,
        timeframe="1Day",
        adjustment_method="split",
        retrieved_at=datetime(2026, 7, 19, tzinfo=timezone.utc),
        coverage_start=date(2020, 1, 1),
        coverage_end=date(2020, 12, 31),
        quality_status="validated",
    )
    session.add(row)
    session.flush()
    return row


def test_deterministic_reproducibility_and_attribution(session: Session) -> None:
    portfolio = _portfolio(session)
    assumptions = ScenarioAssumptions(ScenarioType.BROAD_MARKET_DECLINE, market_decline=-0.2)
    inputs = dict(sectors={"AAA": "Technology", "BBB": "Energy"})

    first = run_deterministic_scenario([portfolio], {"AAA": 10, "BBB": 20}, assumptions, **inputs)
    second = run_deterministic_scenario([portfolio], {"AAA": 10, "BBB": 20}, assumptions, **inputs)

    assert first == second
    assert first.total_household_impact == pytest.approx(-40)
    assert first.impact_by_holding == {"AAA": -20, "BBB": -20}
    assert first.impact_by_sector == {"Technology": -20, "Energy": -20}
    assert first.account_type_effects == {"taxable": -40}
    assert first.model_version == "1.1.0"
    assert SHOCK_DISCLOSURE in first.warnings


def test_dataframe_prices_resolve_one_common_household_snapshot(session: Session) -> None:
    portfolio = _portfolio(session)
    closes = pd.DataFrame(
        {
            "AAA": [10.0, 11.0, 12.0],
            "BBB": [20.0, 21.0, None],
        },
        index=pd.to_datetime(["2026-07-17", "2026-07-20", "2026-07-21"]),
    )

    result = run_deterministic_scenario(
        [portfolio], closes, ScenarioAssumptions("broad_market_decline", market_decline=-0.10)
    )

    # Both baseline and shock use AAA=11 and BBB=21 from the common July 20 snapshot.
    assert result.baseline_value == pytest.approx(315)
    assert result.total_household_impact == pytest.approx(-21.5)
    assert result.coverage["common_valuation_date"] == "2026-07-20"
    assert result.coverage["price_source_dates"] == {
        "AAA": "2026-07-20",
        "BBB": "2026-07-20",
    }
    assert result.coverage["price_observation_counts"] == {"AAA": 2, "BBB": 2}
    assert result.coverage["jointly_complete_price_observations"] == 2
    assert "Scenario valuation date: 2026-07-20." in result.warnings


@pytest.mark.parametrize("invalid", [-1, 0, float("nan"), float("inf"), True])
def test_mapping_prices_reject_invalid_values(session: Session, invalid: object) -> None:
    portfolio = _portfolio(session)

    with pytest.raises(ValueError, match="numeric, finite, positive, and not boolean"):
        run_deterministic_scenario(
            [portfolio],
            {"AAA": invalid, "BBB": 20},
            ScenarioAssumptions("broad_market_decline"),
        )


def test_mapping_prices_are_an_undated_explicit_snapshot(session: Session) -> None:
    portfolio = _portfolio(session)

    result = run_deterministic_scenario(
        [portfolio], {"AAA": 10, "BBB": 20}, ScenarioAssumptions("broad_market_decline")
    )

    assert result.coverage["common_valuation_date"] is None
    assert result.coverage["price_source_dates"] == {}


def test_missing_data_refusal_and_proxy_disclosure(session: Session) -> None:
    portfolio = _portfolio(session)
    assumptions = ScenarioAssumptions(ScenarioType.BROAD_MARKET_DECLINE)

    with pytest.raises(ValueError, match="missing for: BBB"):
        run_deterministic_scenario([portfolio], {"AAA": 10}, assumptions)

    result = run_deterministic_scenario(
        [portfolio],
        {"AAA": 10, "BBB": 20},
        assumptions,
        proxy_use={"BBB": "SPY"},
    )
    assert result.coverage["proxy_symbols"] == ["BBB"]
    assert "BBB uses proxy SPY." in result.warnings


def test_historical_replay_requires_and_records_dataset_reference(session: Session) -> None:
    portfolio = _portfolio(session)
    dataset = _dataset(session)
    history = pd.DataFrame(
        {"AAA": [100.0, 80.0], "BBB": [50.0, 55.0]},
        index=pd.to_datetime(["2020-01-02", "2020-12-31"]),
    )
    assumptions = ScenarioAssumptions(
        ScenarioType.HISTORICAL_REPLAY,
        historical_start=date(2020, 1, 1),
        historical_end=date(2020, 12, 31),
    )

    result = run_deterministic_scenario(
        [portfolio],
        {"AAA": 10, "BBB": 20},
        assumptions,
        historical_prices=history,
        dataset_references=[DatasetReference(dataset.id, "historical_replay")],
    )

    assert result.impact_by_holding == pytest.approx({"AAA": -20, "BBB": 10})
    assert result.coverage["dataset_ids"] == [dataset.id]
    assert result.coverage["replay_start_date"] == "2020-01-02"
    assert result.coverage["replay_end_date"] == "2020-12-31"
    assert result.coverage["jointly_complete_replay_observations"] == 2
    assert "Historical replay endpoints: 2020-01-02 through 2020-12-31." in result.warnings
    assert "Historical replay observations: 2." in result.warnings


def test_historical_replay_uses_only_jointly_complete_rows(session: Session) -> None:
    portfolio = _portfolio(session)
    dataset = _dataset(session)
    history = pd.DataFrame(
        {
            "AAA": [100.0, 200.0, None, 150.0],
            "BBB": [None, 50.0, 100.0, 75.0],
        },
        index=pd.to_datetime(["2020-01-02", "2020-02-03", "2020-03-02", "2020-04-01"]),
    )

    result = run_deterministic_scenario(
        [portfolio],
        {"AAA": 10, "BBB": 20},
        ScenarioAssumptions("historical_replay"),
        historical_prices=history,
        dataset_references=[DatasetReference(dataset.id, "historical_replay")],
    )

    # Shared endpoints are February and April: AAA loses 25%, while BBB gains 50%.
    assert result.impact_by_holding == pytest.approx({"AAA": -25, "BBB": 50})
    assert result.coverage["replay_start_date"] == "2020-02-03"
    assert result.coverage["replay_end_date"] == "2020-04-01"
    assert result.coverage["jointly_complete_replay_observations"] == 2


def test_historical_replay_rejects_fewer_than_two_joint_observations(session: Session) -> None:
    portfolio = _portfolio(session)
    dataset = _dataset(session)
    history = pd.DataFrame(
        {"AAA": [100.0, None], "BBB": [None, 50.0]},
        index=pd.to_datetime(["2020-01-02", "2020-02-03"]),
    )

    with pytest.raises(ValueError, match="at least two jointly complete observations"):
        run_deterministic_scenario(
            [portfolio],
            {"AAA": 10, "BBB": 20},
            ScenarioAssumptions("historical_replay"),
            historical_prices=history,
            dataset_references=[DatasetReference(dataset.id, "historical_replay")],
        )


def test_scenario_reference_calculations_and_disclosures(session: Session) -> None:
    portfolio = _portfolio(session)
    portfolio.transactions.append(
        PortfolioTransaction(
            transaction_date=date(2026, 6, 1),
            kind="dividend",
            action="Dividend",
            cash_delta=Decimal("24"),
            source="test",
        )
    )

    dividend = run_deterministic_scenario(
        [portfolio],
        {"AAA": 10, "BBB": 20},
        ScenarioAssumptions(
            "dividend_reduction",
            dividend_reduction=0.50,
            expected_return=0,
            horizon_years=2,
        ),
        as_of=date(2026, 7, 19),
    )
    inflation = run_deterministic_scenario(
        [portfolio],
        {"AAA": 10, "BBB": 20},
        ScenarioAssumptions(
            "inflation_increase",
            expected_return=0,
            inflation=0.02,
            inflation_increase=0.01,
            horizon_years=2,
        ),
    )

    assert dividend.baseline_value == pytest.approx(300)
    assert dividend.scenario_value == pytest.approx(276)
    assert dividend.total_household_impact == pytest.approx(-24)
    assert DIVIDEND_DISCLOSURE in dividend.warnings
    assert inflation.baseline_value == pytest.approx(300 / 1.02**2)
    assert inflation.scenario_value == pytest.approx(300 / 1.03**2)
    assert inflation.total_household_impact == pytest.approx(
        inflation.scenario_value - inflation.baseline_value
    )


def test_sensitivity_grid_varies_selected_assumptions(session: Session) -> None:
    portfolio = _portfolio(session)
    frame = sensitivity_grid(
        [portfolio],
        {"AAA": 10, "BBB": 20},
        ScenarioAssumptions(ScenarioType.BROAD_MARKET_DECLINE),
        {"market_decline": [-0.1, -0.2], "expected_return": [0.05, 0.07]},
    )

    assert len(frame) == 4
    assert list(frame.columns) == [
        "market_decline",
        "expected_return",
        "baseline_value",
        "scenario_value",
        "household_impact",
        "impact_percent",
    ]
    assert sorted(frame.household_impact.unique()) == [-40, -20]


@pytest.mark.parametrize(
    ("assumptions", "expected_nonpositive"),
    [
        (ScenarioAssumptions("holding_decline", holding_symbol="AAA"), True),
        (ScenarioAssumptions("sector_decline", sector="Technology"), True),
        (ScenarioAssumptions("dividend_reduction"), True),
        (
            ScenarioAssumptions(
                "contribution_interruption", contribution_amount=100, interruption_months=12
            ),
            True,
        ),
        (ScenarioAssumptions("inflation_increase"), True),
        (ScenarioAssumptions("low_return_period"), True),
        (ScenarioAssumptions("lost_decade"), True),
        (
            ScenarioAssumptions(
                "retirement_date_decline", retirement_date=date(2028, 1, 1), horizon_years=5
            ),
            True,
        ),
    ],
)
def test_all_non_replay_scenario_types_execute(
    session: Session, assumptions: ScenarioAssumptions, expected_nonpositive: bool
) -> None:
    portfolio = _portfolio(session, f"Scenario {assumptions.scenario_type.value}")
    portfolio.transactions.append(
        PortfolioTransaction(
            transaction_date=date(2026, 6, 1),
            kind="dividend",
            action="Dividend",
            description="Fixture dividend",
            cash_delta=Decimal("20"),
            source="manual",
        )
    )

    result = run_deterministic_scenario(
        [portfolio],
        {"AAA": 10, "BBB": 20},
        assumptions,
        sectors={"AAA": "Technology", "BBB": "Energy"},
        as_of=date(2026, 7, 19),
    )

    assert (result.total_household_impact <= 0) == expected_nonpositive
    assert sum(result.impact_by_portfolio.values()) == pytest.approx(result.total_household_impact)


def test_ledger_hash_and_result_persistence(session: Session) -> None:
    portfolio = _portfolio(session)
    transaction = PortfolioTransaction(
        portfolio_id=portfolio.id,
        transaction_date=date(2026, 1, 1),
        kind="cash_adjustment",
        action="Cash Adjustment",
        description="Opening cash",
        cash_delta=Decimal("100"),
        source="manual",
    )
    session.add(transaction)
    session.flush()
    first_hash = ledger_state_hash(session, [portfolio.id])
    result = run_deterministic_scenario(
        [portfolio], {"AAA": 10, "BBB": 20}, ScenarioAssumptions("broad_market_decline")
    )

    run = save_scenario_run(session, [portfolio], result)
    transaction.cash_delta = Decimal("101")
    session.flush()

    assert run.ledger_state_hash == first_hash
    assert ledger_state_hash(session, [portfolio.id]) != first_hash
    stored = session.scalar(select(ForecastRun).where(ForecastRun.id == run.id))
    assert json.loads(stored.assumptions)["market_decline"] == -0.2
    assert json.loads(stored.summary_outputs)["total_household_impact"] == -40
    assert stored.scenario_bands is not None
    assert "paths" not in stored.summary_outputs


def test_stored_dataset_references_and_validation_status_behavior(session: Session) -> None:
    portfolio = _portfolio(session)
    dataset = _dataset(session)
    validation = record_validation_evidence(
        session,
        DETERMINISTIC_MODEL_TYPE,
        DETERMINISTIC_MODEL_VERSION,
        automated_tests_passed=True,
        evidence=["reproducibility suite"],
        limitations=["No stochastic probabilities"],
    )
    result = run_deterministic_scenario(
        [portfolio], {"AAA": 10, "BBB": 20}, ScenarioAssumptions("broad_market_decline")
    )
    run = save_scenario_run(
        session,
        [portfolio],
        result,
        dataset_references=[DatasetReference(dataset.id, "current_valuation")],
    )

    assert validation.automated_tests_passed
    assert validation.status == "in_review"
    assert run.validation_status == "in_review"
    assert [(item.dataset_id, item.purpose) for item in run.dataset_references] == [
        (dataset.id, "current_valuation")
    ]
