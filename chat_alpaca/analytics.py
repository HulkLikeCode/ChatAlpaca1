from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd

from chat_alpaca.models import Portfolio


def portfolio_series(portfolio: Portfolio, closes: pd.DataFrame) -> pd.Series:
    if closes.empty:
        return pd.Series(dtype=float, name=portfolio.name)
    result = pd.Series(float(portfolio.cash), index=closes.index, dtype=float)
    for lot in portfolio.holdings:
        if lot.symbol not in closes:
            continue
        active = closes[lot.symbol].where(closes.index.date >= lot.acquired_on)
        result = result.add(active.fillna(0) * float(lot.shares), fill_value=0)
    result.name = portfolio.name
    return result


def combined_series(series: Iterable[pd.Series]) -> pd.Series:
    values = list(series)
    if not values:
        return pd.Series(dtype=float, name="All portfolios")
    combined = pd.concat(values, axis=1).fillna(0).sum(axis=1)
    combined.name = "All portfolios"
    return combined


def normalized_growth(series: pd.Series) -> pd.Series:
    valid = series.replace(0, np.nan).dropna()
    if valid.empty:
        return pd.Series(dtype=float, name=series.name)
    normalized = valid / valid.iloc[0] * 100
    normalized.name = series.name
    return normalized


def summary_metrics(series: pd.Series) -> dict[str, float]:
    values = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2 or values.iloc[0] == 0:
        return {
            "Total return": 0.0,
            "Annualized return": 0.0,
            "Volatility": 0.0,
            "Max drawdown": 0.0,
        }
    daily_returns = values.pct_change().dropna()
    total_return = values.iloc[-1] / values.iloc[0] - 1
    years = max((values.index[-1] - values.index[0]).days / 365.25, 1 / 365.25)
    annualized = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0.0
    drawdown = values / values.cummax() - 1
    return {
        "Total return": float(total_return),
        "Annualized return": float(annualized),
        "Volatility": float(volatility),
        "Max drawdown": float(drawdown.min()),
    }


def earliest_acquisition(portfolios: Iterable[Portfolio], fallback: date) -> date:
    dates = [lot.acquired_on for portfolio in portfolios for lot in portfolio.holdings]
    return min(dates, default=fallback)


def latest_values(portfolio: Portfolio, closes: pd.DataFrame) -> tuple[Decimal, Decimal]:
    market_value = Decimal("0")
    for lot in portfolio.holdings:
        if lot.symbol not in closes or closes[lot.symbol].dropna().empty:
            continue
        market_value += Decimal(str(closes[lot.symbol].dropna().iloc[-1])) * Decimal(lot.shares)
    return market_value, market_value + Decimal(portfolio.cash)
