from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd

from chat_alpaca.analytics import normalized_growth, portfolio_series, summary_metrics
from chat_alpaca.models import HoldingLot, Portfolio


def test_buy_and_hold_series_includes_cash_and_acquisition_date() -> None:
    portfolio = Portfolio(id=1, name="Example", cash=Decimal("100"))
    portfolio.holdings = [
        HoldingLot(
            symbol="ABC",
            shares=Decimal("2"),
            acquired_on=date(2026, 1, 2),
            cost_basis=Decimal("10"),
        )
    ]
    index = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    closes = pd.DataFrame({"ABC": [9.0, 10.0, 12.0]}, index=index)

    values = portfolio_series(portfolio, closes)

    assert values.tolist() == [100.0, 120.0, 124.0]


def test_normalization_and_metrics() -> None:
    values = pd.Series(
        [50.0, 55.0, 60.0],
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        name="ABC",
    )

    normalized = normalized_growth(values)
    metrics = summary_metrics(normalized)

    assert normalized.tolist() == [100.0, 110.00000000000001, 120.0]
    assert round(metrics["Total return"], 6) == 0.2
    assert metrics["Max drawdown"] == 0.0
