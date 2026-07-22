from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from chat_alpaca.hypothetical import (
    CONCENTRATION_DISCLOSURE,
    DRAWDOWN_DISCLOSURE,
    EXPECTED_RETURN_DISCLOSURE,
    HypotheticalAssumptions,
    PortfolioBaseline,
    ProposedAction,
    RetirementAnalysisAssumptions,
    analyze_hypothetical_scenario,
    baseline_from_portfolios,
    load_hypothetical_scenarios,
    prepare_order_ticket_transfer,
    save_hypothetical_scenario,
)
from chat_alpaca.models import (
    HoldingLot,
    HypotheticalScenario,
    LedgerEntry,
    OrderAllocation,
    Portfolio,
    PortfolioTransaction,
)
from chat_alpaca.scenarios import ledger_state_hash


def _portfolios(session) -> tuple[Portfolio, Portfolio]:
    first = Portfolio(name="Growth", cash=Decimal("1000"), account_type="taxable")
    second = Portfolio(name="Retirement", cash=Decimal("500"), account_type="roth_ira")
    session.add_all([first, second])
    session.flush()
    first.holdings.append(
        HoldingLot(
            symbol="AAA",
            shares=Decimal("10"),
            acquired_on=date(2020, 1, 1),
            cost_basis=Decimal("50"),
        )
    )
    second.holdings.append(
        HoldingLot(
            symbol="BBB",
            shares=Decimal("5"),
            acquired_on=date(2021, 1, 1),
            cost_basis=Decimal("10"),
        )
    )
    session.add_all(
        [
            PortfolioTransaction(
                portfolio_id=first.id,
                transaction_date=date(2020, 1, 1),
                kind="opening_position",
                action="Opening position",
                symbol="AAA",
                quantity=Decimal("10"),
                price=Decimal("50"),
                cash_delta=Decimal("0"),
                source="test",
            ),
            PortfolioTransaction(
                portfolio_id=second.id,
                transaction_date=date(2021, 1, 1),
                kind="opening_position",
                action="Opening position",
                symbol="BBB",
                quantity=Decimal("5"),
                price=Decimal("10"),
                cash_delta=Decimal("0"),
                source="test",
            ),
        ]
    )
    session.flush()
    return first, second


def _market_inputs() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        {
            "AAA": rng.normal(0.0005, 0.012, 180),
            "BBB": rng.normal(0.0002, 0.006, 180),
        },
        index=pd.date_range("2025-01-02", periods=180, freq="B"),
    )
    benchmark = returns.AAA * 0.7 + returns.BBB * 0.3
    return returns, benchmark


def _assumptions() -> HypotheticalAssumptions:
    return HypotheticalAssumptions(
        expected_returns={"AAA": 0.09, "BBB": 0.04},
        sectors={"AAA": "Technology", "BBB": "Health Care"},
        benchmark_weights={"AAA": 0.6, "BBB": 0.4},
        stress_shocks={"Technology": -0.5, "BBB": -0.25},
        forecast_horizon_years=3,
        forecast_target=3_000,
        retirement=RetirementAnalysisAssumptions(20, 120, simulations=100, seed=4),
    )


def _run(first: Portfolio, second: Portfolio):
    returns, benchmark = _market_inputs()
    actions = (
        ProposedAction("buy", first.id, symbol="BBB", quantity=10, price=20),
        ProposedAction("add_cash", first.id, amount=300),
        ProposedAction("remove_cash", second.id, amount=50),
        ProposedAction(
            "reassign",
            first.id,
            symbol="AAA",
            quantity=2,
            destination_portfolio_id=second.id,
        ),
    )
    result = analyze_hypothetical_scenario(
        baseline_from_portfolios([first, second]),
        actions,
        {"AAA": 100, "BBB": 20},
        returns,
        _assumptions(),
        market_data_as_of=datetime(2026, 7, 20, 15, tzinfo=timezone.utc),
        benchmark_returns=benchmark,
        common_confirmed_valuation_date=date(2026, 7, 17),
        latest_symbol_dates={"AAA": date(2026, 7, 20), "BBB": date(2026, 7, 17)},
    )
    return actions, result


def test_multiple_trades_before_after_cash_weights_sectors_risk_and_forecast(session) -> None:
    first, second = _portfolios(session)

    actions, result = _run(first, second)

    assert len(actions) == 4
    assert result.before.cash == pytest.approx(1_500)
    assert result.after.cash == pytest.approx(1_550)
    assert result.changes["cash"] == pytest.approx(50)
    assert result.before.market_value == pytest.approx(1_100)
    assert result.after.market_value == pytest.approx(1_300)
    assert result.before.holding_weights["AAA"] == pytest.approx(1000 / 2600)
    assert result.after.holding_weights["BBB"] == pytest.approx(300 / 2850)
    assert (
        result.after.sector_exposure["Health Care"] > result.before.sector_exposure["Health Care"]
    )
    assert result.after.cost_basis == pytest.approx(result.before.cost_basis + 200)
    assert result.before.volatility is not None
    assert result.after.volatility != result.before.volatility
    assert result.after.beta is not None
    assert result.after.risk_contribution
    assert result.after.drawdown_exposure is not None
    assert result.after.expected_return != result.before.expected_return
    assert result.after.forecast_target_probability is not None
    assert set(result.after.downside_percentiles) == {"P5", "P25"}
    assert result.after.deterministic_stress_losses["Technology"] < 0
    assert result.after.depletion_probability is not None
    assert result.after.effective_number_of_holdings > result.before.effective_number_of_holdings
    assert "AAA" in result.after.benchmark_relative_exposure
    assert CONCENTRATION_DISCLOSURE in result.warnings
    assert DRAWDOWN_DISCLOSURE in result.warnings
    assert EXPECTED_RETURN_DISCLOSURE in result.warnings
    assert result.model_version == "1.1.0"
    assert result.common_confirmed_valuation_date == date(2026, 7, 17)
    assert result.confirmed_prices == {"AAA": 100.0, "BBB": 20.0}
    assert result.latest_symbol_dates["AAA"] == date(2026, 7, 20)
    assert "mixed-date values are non-additive" in result.valuation_layers["monitoring_overlay"]


def test_independent_weight_covariance_and_component_risk_reference_case() -> None:
    baseline = PortfolioBaseline(
        1,
        "Taxable",
        100,
        (
            baseline_lot(1, "Taxable", "AAA", 10, 10, 1),
            baseline_lot(1, "Taxable", "BBB", 10, 5, 2),
        ),
    )
    returns = pd.DataFrame(
        {
            "AAA": [0.10, -0.10, 0.08, -0.08, 0.06, -0.06],
            "BBB": [-0.04, 0.04, -0.03, 0.03, -0.02, 0.02],
        },
        index=pd.date_range("2026-01-02", periods=6, freq="B"),
    )
    assumptions = HypotheticalAssumptions(
        expected_returns={"AAA": 0.08, "BBB": 0.04},
        sectors={"AAA": "Technology", "BBB": "Health Care"},
        benchmark_weights={"AAA": 0.40, "BBB": 0.60},
        stress_shocks={},
    )

    result = analyze_hypothetical_scenario(
        (baseline,),
        (ProposedAction("add_cash", 1, amount=1),),
        {"AAA": 20, "BBB": 10},
        returns,
        assumptions,
        market_data_as_of=datetime(2026, 7, 20, 15, tzinfo=timezone.utc),
    )
    snapshot = result.before
    weights = np.array([0.50, 0.25])
    covariance = returns[["AAA", "BBB"]].cov().to_numpy() * 252
    variance = float(weights @ covariance @ weights)
    expected_components = weights * (covariance @ weights) / variance

    assert snapshot.total_value == 400
    assert snapshot.portfolio_values == {"Taxable": 400}
    assert snapshot.cash / snapshot.total_value == pytest.approx(0.25)
    assert snapshot.holding_weights == pytest.approx({"AAA": 0.50, "BBB": 0.25})
    assert snapshot.assignment_weights == pytest.approx(
        {"Taxable · AAA": 0.50, "Taxable · BBB": 0.25}
    )
    assert snapshot.sector_exposure == pytest.approx({"Technology": 0.50, "Health Care": 0.25})
    assert snapshot.benchmark_relative_exposure == pytest.approx({"AAA": 0.10, "BBB": -0.35})
    assert snapshot.concentration == pytest.approx(
        {
            "largest_holding_weight": 0.50,
            "top_five_weight": 0.75,
            "herfindahl_index": 5 / 9,
        }
    )
    assert snapshot.effective_number_of_holdings == pytest.approx(1.8)
    assert snapshot.volatility == pytest.approx(np.sqrt(variance))
    assert snapshot.risk_contribution == pytest.approx(
        dict(zip(("AAA", "BBB"), expected_components, strict=True))
    )
    assert snapshot.risk_contribution["BBB"] < 0
    assert sum(snapshot.risk_contribution.values()) == pytest.approx(1)


def test_analysis_does_not_mutate_accounting_or_submit_orders(session, monkeypatch) -> None:
    first, second = _portfolios(session)
    before = {
        "cash": (first.cash, second.cash),
        "lots": session.scalar(select(func.count(HoldingLot.id))),
        "transactions": session.scalar(select(func.count(PortfolioTransaction.id))),
        "ledger": session.scalar(select(func.count(LedgerEntry.id))),
        "allocations": session.scalar(select(func.count(OrderAllocation.id))),
    }
    submitted = []
    monkeypatch.setattr(
        "chat_alpaca.trading.submit_allocated_order",
        lambda *args, **kwargs: submitted.append((args, kwargs)),
    )

    _run(first, second)
    session.flush()

    assert (first.cash, second.cash) == before["cash"]
    assert session.scalar(select(func.count(HoldingLot.id))) == before["lots"]
    assert session.scalar(select(func.count(PortfolioTransaction.id))) == before["transactions"]
    assert session.scalar(select(func.count(LedgerEntry.id))) == before["ledger"]
    assert session.scalar(select(func.count(OrderAllocation.id))) == before["allocations"]
    assert submitted == []


def test_sell_uses_fifo_basis_and_assignment_preserves_basis() -> None:
    baseline = PortfolioBaseline(
        1,
        "Taxable",
        0,
        (
            # Oldest five shares have the lower basis.
            baseline_lot(1, "Taxable", "AAA", 5, 10, 1),
            baseline_lot(1, "Taxable", "AAA", 5, 30, 2),
        ),
    )
    returns, _ = _market_inputs()
    assumptions = HypotheticalAssumptions({"AAA": 0.05}, {"AAA": "Technology"}, {"AAA": 1}, {})

    result = analyze_hypothetical_scenario(
        (baseline,),
        (ProposedAction("sell", 1, symbol="AAA", quantity=6, price=20),),
        {"AAA": 20},
        returns[["AAA"]],
        assumptions,
        market_data_as_of=datetime.now(timezone.utc),
    )

    assert result.before.cost_basis == pytest.approx(200)
    assert result.after.cost_basis == pytest.approx(120)


@pytest.mark.parametrize(
    "action",
    [
        lambda: ProposedAction("buy", 1, symbol="AAA", quantity=True, price=10),
        lambda: ProposedAction("buy", 1, symbol="AAA", quantity=float("inf"), price=10),
        lambda: ProposedAction("buy", 1, symbol="AAA", quantity=1, price=float("nan")),
        lambda: ProposedAction("buy", 1, symbol="AAA", quantity=1, price=10, fees=True),
        lambda: ProposedAction("sell", 1, symbol="AAA", quantity=1, price=10, fees=float("inf")),
        lambda: ProposedAction("add_cash", 1, amount=float("nan")),
        lambda: ProposedAction("remove_cash", 1, amount=True),
    ],
)
def test_hypothetical_actions_reject_boolean_and_nonfinite_numbers(action) -> None:
    with pytest.raises(ValueError):
        action()


@pytest.mark.parametrize(
    "assumptions",
    [
        lambda: HypotheticalAssumptions(
            {"AAA": float("nan")}, {"AAA": "Technology"}, {"AAA": 1}, {}
        ),
        lambda: HypotheticalAssumptions({"AAA": True}, {"AAA": "Technology"}, {"AAA": 1}, {}),
        lambda: HypotheticalAssumptions(
            {"AAA": 0.05}, {"AAA": "Technology"}, {"AAA": float("inf")}, {}
        ),
        lambda: HypotheticalAssumptions(
            {"AAA": 0.05}, {"AAA": {"Technology": float("nan")}}, {"AAA": 1}, {}
        ),
        lambda: HypotheticalAssumptions(
            {"AAA": 0.05}, {"AAA": "Technology"}, {"AAA": 1}, {"AAA": float("nan")}
        ),
        lambda: HypotheticalAssumptions(
            {"AAA": 0.05},
            {"AAA": "Technology"},
            {"AAA": 1},
            {},
            forecast_target=float("inf"),
        ),
        lambda: RetirementAnalysisAssumptions(20, float("nan")),
        lambda: RetirementAnalysisAssumptions(True, 100),
        lambda: RetirementAnalysisAssumptions(20, 100, simulations=True),
    ],
)
def test_hypothetical_assumptions_reject_boolean_and_nonfinite_numbers(assumptions) -> None:
    with pytest.raises(ValueError):
        assumptions()


@pytest.mark.parametrize("price", [True, float("nan"), float("inf")])
def test_hypothetical_analysis_rejects_invalid_current_prices(price: object) -> None:
    baseline = PortfolioBaseline(
        1,
        "Taxable",
        100,
        (baseline_lot(1, "Taxable", "AAA", 1, 10, 1),),
    )
    assumptions = HypotheticalAssumptions({"AAA": 0.05}, {"AAA": "Technology"}, {"AAA": 1}, {})

    with pytest.raises(ValueError):
        analyze_hypothetical_scenario(
            (baseline,),
            (ProposedAction("add_cash", 1, amount=1),),
            {"AAA": price},
            pd.DataFrame({"AAA": [0.01, -0.01]}),
            assumptions,
            market_data_as_of=datetime.now(timezone.utc),
        )


@pytest.mark.parametrize(
    "lot",
    [
        lambda: baseline_lot(1, "Taxable", "AAA", True, 10, 1),
        lambda: baseline_lot(1, "Taxable", "AAA", 1, float("nan"), 1),
    ],
)
def test_hypothetical_baselines_reject_invalid_numbers(lot) -> None:
    with pytest.raises(ValueError):
        lot()


def baseline_lot(
    portfolio_id: int,
    portfolio_name: str,
    symbol: str,
    shares: float,
    basis: float,
    order: int,
):
    from chat_alpaca.hypothetical import BaselineLot

    return BaselineLot(portfolio_id, portfolio_name, symbol, shares, basis, order)


def test_saved_scenarios_and_stale_baseline_warning(session) -> None:
    first, second = _portfolios(session)
    actions, result = _run(first, second)
    saved = save_hypothetical_scenario(
        session,
        name="Diversify",
        creator="owner",
        portfolios=[first, second],
        market_data_as_of=result.market_data_as_of,
        assumptions=_assumptions(),
        actions=actions,
        result=result,
    )
    session.flush()

    loaded = load_hypothetical_scenarios(session)
    assert saved.id == loaded[0].id
    assert loaded[0].name == "Diversify"
    assert loaded[0].creator == "owner"
    assert loaded[0].market_data_as_of == result.market_data_as_of
    assert len(loaded[0].proposed_trades) == 4
    assert loaded[0].stale_baseline is False
    assert session.scalar(select(func.count(HypotheticalScenario.id))) == 1

    session.add(
        PortfolioTransaction(
            portfolio_id=first.id,
            transaction_date=date(2026, 7, 20),
            kind="cash_adjustment",
            action="Cash adjustment",
            description="Changed after scenario",
            cash_delta=Decimal("1"),
            source="test",
        )
    )
    session.flush()

    assert load_hypothetical_scenarios(session)[0].stale_baseline is True


def test_explicit_transfer_boundary_requires_owner_current_baseline_and_fresh_price(
    session,
) -> None:
    first, second = _portfolios(session)
    ledger_hash = ledger_state_hash(session, [first.id, second.id])
    action = ProposedAction("buy", first.id, symbol="BBB", quantity=2, price=15)
    now = datetime(2026, 7, 20, 15, tzinfo=timezone.utc)

    with pytest.raises(PermissionError, match="owner review"):
        prepare_order_ticket_transfer(
            action,
            owner_review_confirmed=False,
            current_ledger_hash=ledger_hash,
            scenario_ledger_hash=ledger_hash,
            reviewed_market_price=20,
            reviewed_price_as_of=now,
            now=now,
        )
    with pytest.raises(ValueError, match="baseline is stale"):
        prepare_order_ticket_transfer(
            action,
            owner_review_confirmed=True,
            current_ledger_hash="changed",
            scenario_ledger_hash=ledger_hash,
            reviewed_market_price=20,
            reviewed_price_as_of=now,
            now=now,
        )
    with pytest.raises(ValueError, match="price is stale"):
        prepare_order_ticket_transfer(
            action,
            owner_review_confirmed=True,
            current_ledger_hash=ledger_hash,
            scenario_ledger_hash=ledger_hash,
            reviewed_market_price=20,
            reviewed_price_as_of=now - timedelta(minutes=16),
            now=now,
        )

    draft = prepare_order_ticket_transfer(
        action,
        owner_review_confirmed=True,
        current_ledger_hash=ledger_hash,
        scenario_ledger_hash=ledger_hash,
        reviewed_market_price=21,
        reviewed_price_as_of=now,
        now=now,
        source_scenario_id=9,
    )

    assert draft.reviewed_market_price == 21
    assert draft.reviewed_market_price != action.price
    assert draft.source_scenario_id == 9
    assert session.scalar(select(func.count(OrderAllocation.id))) == 0
