from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from chat_alpaca.historical_data import HistoricalCoverageResult
from chat_alpaca.models import Portfolio, PortfolioTransaction
from chat_alpaca.reconstruction import ReconstructionRequest, reconstruct_from_coverage

EXTERNAL_CASH_FLOW_KINDS = {"transfer", "cash_adjustment", "award"}


@dataclass(frozen=True)
class GainLossMetrics:
    all_time: float | None
    daily: float | None
    custom: float | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValuationResult:
    total_calculated_value: Decimal
    market_value: Decimal
    is_complete: bool
    missing_symbols: tuple[str, ...]
    valued_market_value_percentage: float | None
    warnings: tuple[str, ...]
    common_valuation_date: date | None
    stale_symbols: tuple[str, ...]
    last_price_dates: dict[str, date]


@dataclass(frozen=True)
class HouseholdValuationResult:
    """One additive confirmed valuation plus a separate latest-symbol overlay."""

    valuations: tuple[ValuationResult, ...]
    common_valuation_date: date | None
    confirmed_prices: dict[str, float]
    latest_symbol_prices: dict[str, float]
    latest_symbol_dates: dict[str, date]
    missing_symbols: tuple[str, ...]
    limiting_symbols: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return all(valuation.is_complete for valuation in self.valuations)

    @property
    def total_calculated_value(self) -> Decimal | None:
        if not self.is_complete:
            return None
        return sum(
            (valuation.total_calculated_value for valuation in self.valuations),
            Decimal("0"),
        )


@dataclass(frozen=True)
class AlphaBetaMetrics:
    alpha: float | None
    beta: float | None
    observations: int
    warnings: tuple[str, ...] = ()


class IncompleteValuationError(ValueError):
    """Raised when a compatibility value API cannot return a complete valuation."""


def _transactions(portfolio: Portfolio) -> list[PortfolioTransaction]:
    return sorted(
        portfolio.transactions,
        key=lambda item: (item.transaction_date, item.id or 0),
    )


def _frame_coverage(closes: pd.DataFrame) -> HistoricalCoverageResult:
    """Adapt legacy caller frames to the typed reconstruction boundary."""
    normalized = closes.copy()
    normalized.index = pd.to_datetime(normalized.index).normalize()
    sources = normalized.attrs.get("source", "legacy_frame")
    if isinstance(sources, list):
        sources = tuple(sources)
    return HistoricalCoverageResult(
        data=normalized,
        source=sources,
        feed=normalized.attrs.get("feed"),
        adjustment=normalized.attrs.get("adjustment", "split"),
        coverage_start=normalized.index.min().date() if not normalized.empty else None,
        coverage_end=normalized.index.max().date() if not normalized.empty else None,
        missing_symbols=tuple(symbol for symbol in normalized if normalized[symbol].dropna().empty),
        missing_date_ranges={},
        warnings=tuple(normalized.attrs.get("warnings", ())),
        freshness={symbol: datetime.now() for symbol in normalized},
        usable=not normalized.isna().any().any(),
        usability="legacy in-memory frame supplied by compatibility caller",
    )


def _typed_reconstruction(portfolios: list[Portfolio], closes: pd.DataFrame):
    if closes.empty:
        return None
    start = pd.to_datetime(closes.index).min().date()
    end = pd.to_datetime(closes.index).max().date()
    request = ReconstructionRequest(tuple(portfolio.id for portfolio in portfolios), start, end)
    return reconstruct_from_coverage(portfolios, request, _frame_coverage(closes))


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
                result.loc[prices.index.date >= lot.acquired_on] = np.nan
                continue
            active = prices.loc[prices.index.date >= lot.acquired_on, lot.symbol]
            result.loc[active.index] += active * float(lot.shares)
        result.name = portfolio.name
        return result
    reconstructed = _typed_reconstruction([portfolio], prices)
    if reconstructed is None:
        return pd.Series(dtype=float, name=portfolio.name)
    result = reconstructed.portfolios[portfolio.id].daily.portfolio_value.reindex(prices.index)
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

    valuation = portfolio_valuation(portfolio, closes)
    typed = _typed_reconstruction([portfolio], closes) if _transactions(portfolio) else None
    all_time = (
        typed.portfolios[portfolio.id].gain_loss
        if typed is not None
        else (
            float(valuation.total_calculated_value) - _flow_total(portfolio, end=date.today())
            if valuation.is_complete
            else None
        )
    )

    valid = series.dropna()
    if typed is not None and not typed.portfolios[portfolio.id].daily.gain_loss.dropna().empty:
        daily = float(typed.portfolios[portfolio.id].daily.gain_loss.dropna().iloc[-1])
    elif len(valid) >= 2:
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
    warnings = list(valuation.warnings)
    if typed is not None:
        warnings.extend(typed.warnings)
    custom = None
    if baseline.empty:
        warnings.append(
            "Custom gain/loss is unavailable because no confirmed prior trading close exists."
        )
    elif custom_end_value is not None and not (
        typed is not None and typed.portfolios[portfolio.id].daily.calculation_warnings
    ):
        if typed is not None:
            typed_gain = typed.portfolios[portfolio.id].daily.gain_loss
            selected = typed_gain[
                (typed_gain.index.date >= custom_start) & (typed_gain.index.date <= custom_end)
            ].dropna()
            custom = float(selected.sum()) if not selected.empty else 0.0
        else:
            custom_start_value = float(baseline.iloc[-1])
            custom = (
                custom_end_value
                - custom_start_value
                - _flow_total(portfolio, start=custom_start, end=custom_end)
            )
    return GainLossMetrics(all_time, daily, custom, tuple(warnings))


def combined_series(series: Iterable[pd.Series]) -> pd.Series:
    values = list(series)
    if not values:
        return pd.Series(dtype=float, name="All portfolios")
    combined = pd.concat(values, axis=1).sum(axis=1, min_count=len(values))
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
    if _transactions(portfolio):
        typed = _typed_reconstruction([portfolio], closes)
        if typed is not None:
            growth = (typed.portfolios[portfolio.id].daily.time_weighted_return + 1.0) * 100
            growth.name = portfolio.name
            return growth
    return _flow_adjusted_growth(portfolio_series(portfolio, closes), _transactions(portfolio))


def combined_performance_growth(portfolios: Iterable[Portfolio], closes: pd.DataFrame) -> pd.Series:
    """Return $100-rebased combined performance excluding each portfolio's flows."""
    portfolio_list = list(portfolios)
    if portfolio_list and all(_transactions(portfolio) for portfolio in portfolio_list):
        typed = _typed_reconstruction(portfolio_list, closes)
        if typed is not None:
            growth = (typed.combined.time_weighted_return + 1.0) * 100
            growth.name = "Selected portfolios"
            return growth
    values = combined_series(portfolio_series(portfolio, closes) for portfolio in portfolio_list)
    values.name = "Selected portfolios"
    return _flow_adjusted_growth(
        values,
        (transaction for portfolio in portfolio_list for transaction in _transactions(portfolio)),
    )


def summary_metrics(series: pd.Series) -> dict[str, float | None]:
    values = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2 or values.iloc[0] == 0:
        return {
            "Total return": None,
            "Annualized return": None,
            "Volatility": None,
            "Max drawdown": None,
        }
    daily_returns = values.pct_change().dropna()
    total_return = values.iloc[-1] / values.iloc[0] - 1
    years = max((values.index[-1] - values.index[0]).days / 365.25, 1 / 365.25)
    annualized = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else None
    drawdown = values / values.cummax() - 1
    return {
        "Total return": float(total_return),
        "Annualized return": float(annualized),
        "Volatility": float(volatility) if volatility is not None else None,
        "Max drawdown": float(drawdown.min()),
    }


def alpha_beta_from_returns(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    minimum_observations: int = 60,
) -> AlphaBetaMetrics:
    """Regress daily asset returns on benchmark returns and annualize the intercept."""
    aligned = (
        pd.concat([asset_returns.rename("asset"), benchmark_returns.rename("benchmark")], axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    observations = len(aligned)
    if observations < minimum_observations:
        return AlphaBetaMetrics(
            None,
            None,
            observations,
            (
                f"Alpha/Beta requires at least {minimum_observations} overlapping daily returns; "
                f"{observations} are available.",
            ),
        )
    benchmark_variance = float(aligned["benchmark"].var(ddof=1))
    if not np.isfinite(benchmark_variance) or benchmark_variance <= 0:
        return AlphaBetaMetrics(
            None,
            None,
            observations,
            ("Alpha/Beta is unavailable because benchmark returns have no variance.",),
        )
    covariance = float(aligned[["asset", "benchmark"]].cov().iloc[0, 1])
    beta = covariance / benchmark_variance
    daily_alpha = float((aligned["asset"] - beta * aligned["benchmark"]).mean())
    alpha = (1.0 + daily_alpha) ** 252 - 1.0 if daily_alpha > -1.0 else daily_alpha * 252
    return AlphaBetaMetrics(float(alpha), float(beta), observations)


def alpha_beta_from_levels(
    asset_levels: pd.Series,
    benchmark_levels: pd.Series,
    start: date,
    end: date,
    *,
    minimum_observations: int = 60,
) -> AlphaBetaMetrics:
    """Calculate Alpha/Beta from aligned daily changes within an inclusive date range."""
    asset_returns = asset_levels.sort_index().pct_change(fill_method=None)
    benchmark_returns = benchmark_levels.sort_index().pct_change(fill_method=None)
    selected_asset = asset_returns[
        (asset_returns.index.date >= start) & (asset_returns.index.date <= end)
    ]
    selected_benchmark = benchmark_returns[
        (benchmark_returns.index.date >= start) & (benchmark_returns.index.date <= end)
    ]
    return alpha_beta_from_returns(
        selected_asset,
        selected_benchmark,
        minimum_observations=minimum_observations,
    )


def holding_alpha_beta(
    portfolios: Iterable[Portfolio],
    closes: pd.DataFrame,
    benchmark_levels: pd.Series,
    symbol: str,
    start: date,
    end: date,
    *,
    minimum_observations: int = 60,
) -> AlphaBetaMetrics:
    """Calculate symbol Alpha/Beta, adding attributable ledger dividends to price returns."""
    if symbol not in closes:
        return AlphaBetaMetrics(None, None, 0, (f"No price history is available for {symbol}.",))
    portfolio_list = list(portfolios)
    prices = closes[symbol].sort_index().dropna()
    price_returns = prices.pct_change(fill_method=None)
    position_transactions = sorted(
        (
            transaction
            for portfolio in portfolio_list
            for transaction in _transactions(portfolio)
            if transaction.symbol == symbol
            and transaction.kind in {"buy", "sell", "opening_position"}
            and transaction.quantity is not None
        ),
        key=lambda item: (item.transaction_date, item.id or 0),
    )
    assigned_dividends: dict[date, float] = {}
    for portfolio in portfolio_list:
        for transaction in _transactions(portfolio):
            if transaction.kind == "dividend" and transaction.symbol == symbol:
                assigned_dividends[transaction.transaction_date] = assigned_dividends.get(
                    transaction.transaction_date, 0.0
                ) + float(transaction.cash_delta)

    fallback_shares = sum(
        float(lot.shares)
        for portfolio in portfolio_list
        for lot in portfolio.holdings
        if lot.symbol == symbol
    )
    total_returns = price_returns.copy()
    excluded_dividends = 0
    for timestamp in total_returns.index:
        dividend = assigned_dividends.get(timestamp.date(), 0.0)
        if not dividend:
            continue
        prior_prices = prices[prices.index < timestamp]
        shares = fallback_shares
        if position_transactions:
            shares = sum(
                (-1.0 if transaction.kind == "sell" else 1.0) * float(transaction.quantity)
                for transaction in position_transactions
                if transaction.transaction_date < timestamp.date()
            )
        prior_value = (
            shares * float(prior_prices.iloc[-1]) if shares and not prior_prices.empty else 0
        )
        if prior_value:
            total_returns.loc[timestamp] += dividend / prior_value
        else:
            excluded_dividends += 1

    selected = total_returns[
        (total_returns.index.date >= start) & (total_returns.index.date <= end)
    ]
    benchmark_returns = benchmark_levels.sort_index().pct_change(fill_method=None)
    selected_benchmark = benchmark_returns[
        (benchmark_returns.index.date >= start) & (benchmark_returns.index.date <= end)
    ]
    result = alpha_beta_from_returns(
        selected,
        selected_benchmark,
        minimum_observations=minimum_observations,
    )
    if not excluded_dividends:
        return result
    return AlphaBetaMetrics(
        result.alpha,
        result.beta,
        result.observations,
        (
            *result.warnings,
            f"{excluded_dividends} assigned dividend event(s) could not be attributed.",
        ),
    )


def earliest_acquisition(portfolios: Iterable[Portfolio], fallback: date) -> date:
    portfolio_list = list(portfolios)
    dates = [lot.acquired_on for portfolio in portfolio_list for lot in portfolio.holdings]
    dates.extend(
        transaction.transaction_date
        for portfolio in portfolio_list
        for transaction in _transactions(portfolio)
    )
    return min(dates, default=fallback)


def _last_price_dates(closes: pd.DataFrame, symbols: set[str]) -> dict[str, date]:
    recorded = closes.attrs.get("last_price_dates", {})
    dates: dict[str, date] = {}
    for symbol in symbols:
        if symbol in recorded:
            dates[symbol] = recorded[symbol]
        elif symbol in closes and not closes[symbol].dropna().empty:
            dates[symbol] = closes[symbol].dropna().index[-1].date()
    return dates


def household_valuation(
    portfolios: Iterable[Portfolio], closes: pd.DataFrame
) -> HouseholdValuationResult:
    """Value every selected portfolio at one common confirmed household date."""
    portfolio_list = list(portfolios)
    symbols = {
        lot.symbol
        for portfolio in portfolio_list
        for lot in portfolio.holdings
        if Decimal(lot.shares) != 0
    }
    last_dates = _last_price_dates(closes, symbols)
    missing = tuple(sorted(symbols - last_dates.keys()))
    common_date = min(last_dates.values()) if last_dates and not missing else None
    freshest_date = max(last_dates.values()) if last_dates else None
    limiting = (
        tuple(
            sorted(
                symbol for symbol, price_date in last_dates.items() if price_date < freshest_date
            )
        )
        if freshest_date
        else ()
    )
    latest_prices = {
        symbol: price
        for symbol, price_date in last_dates.items()
        if (price := _price_on_or_before(closes, symbol, price_date)) is not None
    }
    confirmed_prices = (
        {
            symbol: price
            for symbol in symbols
            if (price := _price_on_or_before(closes, symbol, common_date)) is not None
        }
        if common_date is not None
        else {}
    )
    missing_at_common = (
        missing if common_date is None else tuple(sorted(symbols - confirmed_prices.keys()))
    )
    household_complete = not missing_at_common

    household_warnings: list[str] = []
    if missing_at_common:
        household_warnings.append(
            "A common confirmed household valuation is unavailable. Missing prices for held "
            "symbols: " + ", ".join(missing_at_common) + "."
        )
    if limiting:
        details = ", ".join(f"{symbol} ({last_dates[symbol]})" for symbol in limiting)
        household_warnings.append(
            f"The common confirmed household valuation date is {common_date}; limiting symbols: "
            f"{details}."
        )

    valuations: list[ValuationResult] = []
    for portfolio in portfolio_list:
        portfolio_symbols = {lot.symbol for lot in portfolio.holdings if Decimal(lot.shares) != 0}
        portfolio_missing = tuple(sorted(portfolio_symbols - confirmed_prices.keys()))
        market_value = Decimal("0")
        for lot in portfolio.holdings:
            price = confirmed_prices.get(lot.symbol)
            if price is not None:
                market_value += Decimal(str(price)) * Decimal(lot.shares)
        complete = household_complete
        warnings = list(household_warnings)
        if portfolio_missing and not household_warnings:
            warnings.append(
                "Missing prices for held symbols: " + ", ".join(portfolio_missing) + "."
            )
        valuations.append(
            ValuationResult(
                total_calculated_value=market_value + Decimal(portfolio.cash),
                market_value=market_value,
                is_complete=complete,
                missing_symbols=missing_at_common,
                valued_market_value_percentage=(
                    100.0 if complete else (0.0 if symbols and not last_dates else None)
                ),
                warnings=tuple(warnings),
                common_valuation_date=common_date,
                stale_symbols=limiting,
                last_price_dates=dict(last_dates),
            )
        )
    return HouseholdValuationResult(
        valuations=tuple(valuations),
        common_valuation_date=common_date,
        confirmed_prices=confirmed_prices,
        latest_symbol_prices=latest_prices,
        latest_symbol_dates=last_dates,
        missing_symbols=missing_at_common,
        limiting_symbols=limiting,
        warnings=tuple(household_warnings),
    )


def portfolio_valuation(portfolio: Portfolio, closes: pd.DataFrame) -> ValuationResult:
    """Value one portfolio at its common confirmed close."""
    return household_valuation([portfolio], closes).valuations[0]


def latest_values(portfolio: Portfolio, closes: pd.DataFrame) -> tuple[Decimal, Decimal]:
    """Compatibility wrapper returning only complete valuations."""
    valuation = portfolio_valuation(portfolio, closes)
    if not valuation.is_complete:
        raise IncompleteValuationError(" ".join(valuation.warnings))
    return valuation.market_value, valuation.total_calculated_value


def total_portfolio_value(portfolios: Iterable[Portfolio], closes: pd.DataFrame) -> Decimal:
    valuation = household_valuation(portfolios, closes)
    if valuation.total_calculated_value is None:
        raise IncompleteValuationError(" ".join(valuation.warnings))
    return valuation.total_calculated_value


def rebase_comparison_series(series: Iterable[pd.Series]) -> list[pd.Series]:
    """Rebase every already-period-filtered comparison series to exactly $100."""
    return [normalized_growth(item) for item in series]


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
    benchmark_closes: pd.DataFrame | None = None,
    benchmark_symbol: str = "SPY",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return symbol totals and the underlying portfolio/lot detail for current holdings."""
    if custom_start > custom_end:
        raise ValueError("The custom holdings start date must be on or before the end date.")
    portfolio_list = list(portfolios)
    household = household_valuation(portfolio_list, closes)
    risk_metrics: dict[str, AlphaBetaMetrics] = {}
    benchmark_levels = (
        benchmark_closes[benchmark_symbol].dropna()
        if benchmark_closes is not None and benchmark_symbol in benchmark_closes
        else pd.Series(dtype=float)
    )
    if not benchmark_levels.empty:
        for symbol in {lot.symbol for portfolio in portfolio_list for lot in portfolio.holdings}:
            risk_metrics[symbol] = holding_alpha_beta(
                portfolio_list,
                closes,
                benchmark_levels,
                symbol,
                custom_start,
                custom_end,
            )
    rows: list[dict[str, object]] = []
    for portfolio in portfolio_list:
        for lot in sorted(portfolio.holdings, key=lambda item: (item.symbol, item.acquired_on)):
            shares = float(lot.shares)
            cost_per_share = float(lot.cost_basis)
            cost_basis = shares * cost_per_share
            confirmed = household.confirmed_prices.get(lot.symbol)
            latest = household.latest_symbol_prices.get(lot.symbol)
            latest_date = household.latest_symbol_dates.get(lot.symbol)
            prices = closes[lot.symbol].dropna() if lot.symbol in closes else pd.Series(dtype=float)
            prior = float(prices.iloc[-2]) if len(prices) >= 2 else None
            daily_price_dates = (
                f"{prices.index[-2].strftime('%-m/%-d/%y')} → "
                f"{prices.index[-1].strftime('%-m/%-d/%y')}"
                if len(prices) >= 2
                else None
            )
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
                    "Confirmed valuation date": household.common_valuation_date,
                    "Confirmed price": confirmed,
                    "Confirmed value": shares * confirmed if confirmed is not None else None,
                    "Latest symbol price": latest,
                    "Latest symbol date": latest_date,
                    "Latest/indicative value": shares * latest if latest is not None else None,
                    "All-time gain/loss": (
                        shares * confirmed - cost_basis if confirmed is not None else None
                    ),
                    "Daily gain/loss": (
                        shares * (latest - prior)
                        if latest is not None and prior is not None
                        else None
                    ),
                    "Daily price dates": daily_price_dates,
                    "Custom gain/loss": custom_gain_loss,
                    "Alpha": risk_metrics.get(lot.symbol, AlphaBetaMetrics(None, None, 0)).alpha,
                    "Beta": risk_metrics.get(lot.symbol, AlphaBetaMetrics(None, None, 0)).beta,
                    "Alpha/Beta observations": risk_metrics.get(
                        lot.symbol, AlphaBetaMetrics(None, None, 0)
                    ).observations,
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
            "Confirmed valuation date": ("Confirmed valuation date", "first"),
            "Confirmed price": ("Confirmed price", "first"),
            "Confirmed value": ("Confirmed value", lambda values: values.sum(min_count=1)),
            "Latest symbol price": ("Latest symbol price", "first"),
            "Latest symbol date": ("Latest symbol date", "first"),
            "Latest/indicative value": (
                "Latest/indicative value",
                lambda values: values.sum(min_count=1),
            ),
            "All-time gain/loss": (
                "All-time gain/loss",
                lambda values: values.sum(min_count=1),
            ),
            "Daily gain/loss": ("Daily gain/loss", lambda values: values.sum(min_count=1)),
            "Daily price dates": ("Daily price dates", "first"),
            "Custom gain/loss": ("Custom gain/loss", lambda values: values.sum(min_count=1)),
            "Alpha": ("Alpha", "first"),
            "Beta": ("Beta", "first"),
            "Alpha/Beta observations": ("Alpha/Beta observations", "first"),
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
            "Confirmed valuation date",
            "Confirmed price",
            "Confirmed value",
            "Latest symbol price",
            "Latest symbol date",
            "Latest/indicative value",
            "All-time gain/loss",
            "Daily gain/loss",
            "Daily price dates",
            "Custom gain/loss",
            "Alpha",
            "Beta",
            "Alpha/Beta observations",
        ]
    ].sort_values("Symbol")
    return grouped, detail


def adaptive_share_number_format(values: Iterable[object]) -> str:
    """Expose all stored share precision while preserving a numeric column for sorting."""
    precision = 0
    for value in values:
        if pd.isna(value):
            continue
        decimal_value = Decimal(str(value)).normalize()
        precision = max(precision, min(max(-decimal_value.as_tuple().exponent, 0), 8))
    return f"%.{precision}f"
