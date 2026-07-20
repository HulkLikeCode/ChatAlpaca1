from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from chat_alpaca.classification import (
    UNCLASSIFIED,
    cache_security_metadata,
    portfolio_sector_exposure,
    resolve_security_metadata,
    save_etf_sector_snapshot,
    save_manual_metadata_override,
)
from chat_alpaca.models import HoldingLot, Portfolio
from chat_alpaca.portfolio_configuration import (
    BenchmarkConfiguration,
    benchmark_configurations,
    household_benchmark_series,
    reconstruct_benchmark_series,
    save_benchmark_configuration,
    set_account_type,
    validate_benchmark_weights,
)


def _portfolio(session: Session, name: str = "Phase 6") -> Portfolio:
    portfolio = Portfolio(name=name, cash=Decimal("0"), account_type="unknown")
    session.add(portfolio)
    session.flush()
    return portfolio


def _holding(symbol: str, shares: str) -> HoldingLot:
    return HoldingLot(
        symbol=symbol,
        shares=Decimal(shares),
        acquired_on=date(2026, 1, 1),
        cost_basis=Decimal("1"),
    )


def test_unknown_account_behavior_and_owner_edit(session: Session) -> None:
    portfolio = _portfolio(session)

    assert portfolio.account_type == "unknown"
    set_account_type(session, portfolio.id, "taxable")
    assert portfolio.account_type == "taxable"
    with pytest.raises(ValueError, match="Account type"):
        set_account_type(session, portfolio.id, "brokerage")


def test_benchmark_weight_validation() -> None:
    assert validate_benchmark_weights({"spy": "60", "QQQ": "40"}) == {
        "SPY": Decimal("60"),
        "QQQ": Decimal("40"),
    }
    with pytest.raises(ValueError, match="sum to 100"):
        validate_benchmark_weights({"SPY": "99.9"})
    with pytest.raises(ValueError, match="Invalid benchmark symbol"):
        validate_benchmark_weights({"not a symbol": "100"})


def test_effective_dated_benchmark_changes_are_append_only(session: Session) -> None:
    portfolio = _portfolio(session)
    first = save_benchmark_configuration(
        session, portfolio.id, date(2026, 1, 1), {"SPY": 60, "BND": 40}
    )
    second = save_benchmark_configuration(session, portfolio.id, date(2026, 7, 1), {"QQQ": 100})

    assert first.weights == {"SPY": Decimal("0.6"), "BND": Decimal("0.4")}
    assert second.effective_from == date(2026, 7, 1)
    assert [item.effective_from for item in benchmark_configurations(session, portfolio.id)] == [
        date(2026, 1, 1),
        date(2026, 7, 1),
    ]
    with pytest.raises(ValueError, match="prior history is not rewritten"):
        save_benchmark_configuration(session, portfolio.id, date(2026, 1, 1), {"DIA": 100})


def test_benchmark_reconstruction_and_household_portfolio_identity() -> None:
    index = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    closes = pd.DataFrame({"AAA": [100, 110, 121], "BBB": [100, 100, 120]}, index=index)
    configurations = (
        BenchmarkConfiguration(
            1, date(2026, 1, 1), {"AAA": Decimal("0.5"), "BBB": Decimal("0.5")}, "monthly"
        ),
        BenchmarkConfiguration(1, date(2026, 1, 3), {"BBB": Decimal("1")}, "monthly"),
    )

    result = reconstruct_benchmark_series(
        1, configurations, closes, date(2026, 1, 1), date(2026, 1, 3)
    )
    household = household_benchmark_series(
        {
            1: configurations,
            2: (BenchmarkConfiguration(2, date(2026, 1, 1), {"AAA": Decimal("1")}, "daily"),),
        },
        closes,
        date(2026, 1, 1),
        date(2026, 1, 3),
    )

    assert result.growth.tolist() == [100.0, 105.0, 126.0]
    assert "rebalanced" in result.assumption
    assert set(household.by_portfolio) == {1, 2}
    assert household.by_portfolio[1].configurations != household.by_portfolio[2].configurations


def test_stock_sector_exposure(session: Session) -> None:
    portfolio = _portfolio(session)
    portfolio.holdings = [_holding("AAA", "2")]
    cache_security_metadata(
        session,
        "AAA",
        security_name="Example",
        asset_type="stock",
        sector="Technology",
        industry="Software",
        source="supplemental_free",
    )

    result = portfolio_sector_exposure(session, portfolio, {"AAA": 10})

    assert result.invested_market_value == Decimal("20.00000000")
    assert [(row.sector, row.market_value, row.percentage) for row in result.exposures] == [
        ("Technology", Decimal("20.00000000"), Decimal("100"))
    ]


def test_etf_look_through_keeps_residual_unclassified(session: Session) -> None:
    portfolio = _portfolio(session)
    portfolio.holdings = [_holding("FUND", "10")]
    cache_security_metadata(session, "FUND", asset_type="etf", source="alpaca")
    save_etf_sector_snapshot(
        session,
        "FUND",
        {"Technology": 40, "Financials": 30},
        source="issuer_free",
        effective_date=date(2026, 1, 1),
    )

    result = portfolio_sector_exposure(session, portfolio, {"FUND": 10}, as_of=date(2026, 1, 2))
    values = {row.sector: row.market_value for row in result.exposures}

    assert values == {
        "Technology": Decimal("40.0000000000000000"),
        "Financials": Decimal("30.0000000000000000"),
        UNCLASSIFIED: Decimal("30.0000000000000000"),
    }


def test_mixed_stock_and_etf_portfolio(session: Session) -> None:
    portfolio = _portfolio(session)
    portfolio.holdings = [_holding("STOCK", "5"), _holding("FUND", "10")]
    cache_security_metadata(session, "STOCK", asset_type="stock", sector="Energy", source="free")
    cache_security_metadata(session, "FUND", asset_type="etf", source="alpaca")
    save_etf_sector_snapshot(
        session,
        "FUND",
        {"Technology": 50, "Health Care": 50},
        source="issuer_free",
        effective_date=date(2026, 1, 1),
    )

    result = portfolio_sector_exposure(
        session, portfolio, {"STOCK": 20, "FUND": 10}, as_of=date(2026, 1, 2)
    )

    assert {row.sector: row.percentage for row in result.exposures} == {
        "Energy": Decimal("50.0"),
        "Health Care": Decimal("25.00"),
        "Technology": Decimal("25.00"),
    }


def test_unavailable_metadata_is_explicitly_unclassified(session: Session) -> None:
    portfolio = _portfolio(session)
    portfolio.holdings = [_holding("MYSTERY", "1")]

    metadata = resolve_security_metadata(session, "MYSTERY")
    exposure = portfolio_sector_exposure(session, portfolio, {"MYSTERY": 25})

    assert metadata.unavailable
    assert exposure.exposures[0].sector == UNCLASSIFIED
    assert any("unavailable" in item for item in exposure.disclosures)


def test_manual_metadata_override_wins(session: Session) -> None:
    cache_security_metadata(
        session, "AAA", asset_type="stock", sector="Industrials", source="alpaca"
    )
    save_manual_metadata_override(
        session,
        "AAA",
        asset_type="stock",
        sector="Technology",
        quality_status="owner_confirmed",
    )

    result = resolve_security_metadata(session, "AAA")

    assert result.sector == "Technology"
    assert result.manual_override
    assert result.source == "manual"


def test_stale_metadata_disclosure(session: Session) -> None:
    cache_security_metadata(
        session,
        "OLD",
        asset_type="stock",
        sector="Utilities",
        source="free",
        retrieved_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    result = resolve_security_metadata(session, "OLD", as_of=date(2026, 1, 1), stale_after_days=30)

    assert result.stale
    assert any("stale" in item for item in result.disclosures)
