from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd

from chat_alpaca.models import Portfolio, PortfolioTransaction

EXTERNAL_CASH_FLOW_KINDS = {"transfer", "cash_adjustment"}


@dataclass(frozen=True)
class GainLossMetrics:
    all_time: float | None
    daily: float | None
    custom: float | None


def _transactions(portfolio: Portfolio) -> list[PortfolioTransaction]:
    return sorted(
        portfolio.transactions,
        key=lambda item: (item.transaction_date, item.id or 0),
    )


def portfolio_series(portfolio: Portfolio, closes: pd.DataFrame) -> pd.Series:
    if closes.empty:
        return pd.Series(dtype=float, name=portfolio.name)

    prices = closes.copy()
    prices.index = pd.to_datetime(prices.index).normalize()
    transactions = _transactions(portfolio)
    if not transactions:
        result = pd.Series(float(portfolio.cash), index=prices.index, dtype=float)
        for lot in portfolio.holdings:
            if lot.symbol not in prices:
                continue
            active = prices[lot.symbol].where(prices.index.date >= lot.acquired_on)
            result = result.add(active.fillna(0) * float(lot.shares), fill_value=0)
        result.name = portfolio.name
        return result

    result = pd.Series(0.0, index=prices.index, dtype=float)
    for transaction in transactions:
        active = prices.index.date >= transaction.transaction_date
        result.loc[active] += float(transaction.cash_delta)
        if (
            transaction.kind not in {"buy", "sell", "opening_position"}
            or transaction.symbol not in prices
            or transaction.quantity is None
        ):
            continue
        direction = -1 if transaction.kind == "sell" else 1
        position_value = prices.loc[active, transaction.symbol].fillna(0)
        result.loc[active] += direction * position_value * float(transaction.quantity)
    result.name = portfolio.name
    return result


def _external_flow(transaction: PortfolioTransaction) -> float:
    if transaction.kind in EXTERNAL_CASH_FLOW_KINDS:
        return float(transaction.cash_delta)
    if (
        transaction.kind == "opening_position"
        and transaction.quantity is not None
        and transaction.price is not None
    ):
        return float(transaction.quantity * transaction.price)
    return 0.0


def _flow_total(portfolio: Portfolio, start: date | None = None, end: date | None = None) -> float:
    transactions = _transactions(portfolio)
    if not transactions:
        if start is None:
            return float(
                sum(
                    (Decimal(lot.shares) * Decimal(lot.cost_basis) for lot in portfolio.holdings),
                    Decimal(portfolio.cash),
                )
            )
        return 0.0
    return sum(
        _external_flow(transaction)
        for transaction in transactions
        if (start is None or transaction.transaction_date >= start)
        and (end is None or transaction.transaction_date <= end)
    )


def _value_on_or_before(series: pd.Series, cutoff: date) -> float | None:
    eligible = series[series.index.date <= cutoff].dropna()
    return float(eligible.iloc[-1]) if not eligible.empty else None


def portfolio_gain_loss(
    portfolio: Portfolio,
    closes: pd.DataFrame,
    custom_start: date,
    custom_end: date,
) -> GainLossMetrics:
    """Calculate dollar P/L while excluding transfers, cash adjustments, and opening assets."""
    if custom_start > custom_end:
        raise ValueError("The custom gain/loss start date must be on or before the end date.")
    series = portfolio_series(portfolio, closes)
    if series.empty:
        return GainLossMetrics(None, None, None)

    all_time_end = float(latest_values(portfolio, closes)[1])
    all_time = all_time_end - _flow_total(portfolio, end=date.today())

    valid = series.dropna()
    if len(valid) >= 2:
        daily_end_date = valid.index[-1].date()
        daily_start_date = valid.index[-2].date()
        daily = (
            float(valid.iloc[-1])
            - float(valid.iloc[-2])
            - _flow_total(
                portfolio,
                start=daily_start_date + pd.Timedelta(days=1),
                end=daily_end_date,
            )
        )
    else:
        daily = None

    custom_end_value = _value_on_or_before(series, custom_end)
    baseline = series[series.index.date < custom_start].dropna()
    custom_start_value = float(baseline.iloc[-1]) if not baseline.empty else 0.0
    custom = None
    if custom_end_value is not None:
        custom = (
            custom_end_value
            - custom_start_value
            - _flow_total(portfolio, start=custom_start, end=custom_end)
        )
    return GainLossMetrics(all_time, daily, custom)


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


def _flow_adjusted_growth(
    values: pd.Series, transactions: Iterable[PortfolioTransaction]
) -> pd.Series:
    """Rebase a value series after removing non-performance portfolio flows."""
    values = values.replace([np.inf, -np.inf], np.nan)
    if values.empty:
        return pd.Series(dtype=float, name=values.name)

    flows_by_date: dict[date, float] = {}
    for transaction in transactions:
        flow = _external_flow(transaction)
        if flow:
            flows_by_date[transaction.transaction_date] = (
                flows_by_date.get(transaction.transaction_date, 0.0) + flow
            )

    growth = pd.Series(np.nan, index=values.index, dtype=float, name=values.name)
    level: float | None = None
    previous_value: float | None = None
    for timestamp, value in values.items():
        if pd.isna(value):
            continue
        current_value = float(value)
        flow = flows_by_date.get(timestamp.date(), 0.0)
        if level is None:
            if current_value != 0 or flow != 0:
                level = 100.0
                growth.loc[timestamp] = level
        elif previous_value not in (None, 0.0):
            level *= (current_value - flow) / previous_value
            growth.loc[timestamp] = level
        elif current_value != 0 or flow != 0:
            level = 100.0
            growth.loc[timestamp] = level
        previous_value = current_value
    return growth


def performance_growth(portfolio: Portfolio, closes: pd.DataFrame) -> pd.Series:
    """Return $100-rebased portfolio performance excluding external flows."""
    return _flow_adjusted_growth(portfolio_series(portfolio, closes), _transactions(portfolio))


def combined_performance_growth(portfolios: Iterable[Portfolio], closes: pd.DataFrame) -> pd.Series:
    """Return $100-rebased combined performance excluding each portfolio's flows."""
    portfolio_list = list(portfolios)
    values = combined_series(portfolio_series(portfolio, closes) for portfolio in portfolio_list)
    values.name = "Selected portfolios"
    return _flow_adjusted_growth(
        values,
        (transaction for portfolio in portfolio_list for transaction in _transactions(portfolio)),
    )


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
    portfolio_list = list(portfolios)
    dates = [lot.acquired_on for portfolio in portfolio_list for lot in portfolio.holdings]
    dates.extend(
        transaction.transaction_date
        for portfolio in portfolio_list
        for transaction in _transactions(portfolio)
    )
    return min(dates, default=fallback)


def latest_values(portfolio: Portfolio, closes: pd.DataFrame) -> tuple[Decimal, Decimal]:
    market_value = Decimal("0")
    for lot in portfolio.holdings:
        if lot.symbol not in closes or closes[lot.symbol].dropna().empty:
            continue
        market_value += Decimal(str(closes[lot.symbol].dropna().iloc[-1])) * Decimal(lot.shares)
    return market_value, market_value + Decimal(portfolio.cash)


def total_portfolio_value(portfolios: Iterable[Portfolio], closes: pd.DataFrame) -> Decimal:
    return sum((latest_values(portfolio, closes)[1] for portfolio in portfolios), Decimal("0"))


def _price_on_or_before(
    closes: pd.DataFrame, symbol: str, cutoff: date | None = None, strict: bool = False
) -> float | None:
    if symbol not in closes:
        return None
    prices = closes[symbol].dropna()
    if cutoff is not None:
        if strict:
            prices = prices[prices.index.date < cutoff]
        else:
            prices = prices[prices.index.date <= cutoff]
    return float(prices.iloc[-1]) if not prices.empty else None


def consolidated_holdings(
    portfolios: Iterable[Portfolio],
    closes: pd.DataFrame,
    custom_start: date,
    custom_end: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return symbol totals and the underlying portfolio/lot detail for current holdings."""
    if custom_start > custom_end:
        raise ValueError("The custom holdings start date must be on or before the end date.")
    rows: list[dict[str, object]] = []
    for portfolio in portfolios:
        for lot in sorted(portfolio.holdings, key=lambda item: (item.symbol, item.acquired_on)):
            shares = float(lot.shares)
            cost_per_share = float(lot.cost_basis)
            cost_basis = shares * cost_per_share
            latest = _price_on_or_before(closes, lot.symbol)
            prices = closes[lot.symbol].dropna() if lot.symbol in closes else pd.Series(dtype=float)
            prior = float(prices.iloc[-2]) if len(prices) >= 2 else None
            custom_end_price = _price_on_or_before(closes, lot.symbol, custom_end)
            if lot.acquired_on > custom_end or custom_end_price is None:
                custom_gain_loss = None
            else:
                custom_start_price = (
                    cost_per_share
                    if lot.acquired_on >= custom_start
                    else _price_on_or_before(closes, lot.symbol, custom_start, strict=True)
                )
                custom_gain_loss = (
                    shares * (custom_end_price - custom_start_price)
                    if custom_start_price is not None
                    else None
                )
            rows.append(
                {
                    "Portfolio": portfolio.name,
                    "Symbol": lot.symbol,
                    "Shares": shares,
                    "Acquired": lot.acquired_on,
                    "Cost / share": cost_per_share,
                    "Cost basis": cost_basis,
                    "Latest price": latest,
                    "Market value": shares * latest if latest is not None else None,
                    "All-time gain/loss": (
                        shares * latest - cost_basis if latest is not None else None
                    ),
                    "Daily gain/loss": (
                        shares * (latest - prior)
                        if latest is not None and prior is not None
                        else None
                    ),
                    "Custom gain/loss": custom_gain_loss,
                }
            )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(), detail
    grouped = detail.groupby("Symbol", as_index=False).agg(
        Portfolios=("Portfolio", "nunique"),
        Shares=("Shares", "sum"),
        **{
            "Total cost basis": ("Cost basis", "sum"),
            "Latest price": ("Latest price", "first"),
            "Market value": ("Market value", lambda values: values.sum(min_count=1)),
            "All-time gain/loss": (
                "All-time gain/loss",
                lambda values: values.sum(min_count=1),
            ),
            "Daily gain/loss": ("Daily gain/loss", lambda values: values.sum(min_count=1)),
            "Custom gain/loss": ("Custom gain/loss", lambda values: values.sum(min_count=1)),
        },
    )
    grouped["Average cost / share"] = grouped["Total cost basis"].div(
        grouped["Shares"].replace(0, np.nan)
    )
    grouped = grouped[
        [
            "Symbol",
            "Portfolios",
            "Shares",
            "Average cost / share",
            "Total cost basis",
            "Latest price",
            "Market value",
            "All-time gain/loss",
            "Daily gain/loss",
            "Custom gain/loss",
        ]
    ].sort_values("Symbol")
    return grouped, detail
