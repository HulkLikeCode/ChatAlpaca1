from __future__ import annotations

from datetime import date, datetime, time, timezone
from functools import lru_cache

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from chat_alpaca.config import Settings, get_settings
from chat_alpaca.portfolio_service import normalize_symbol


class MarketDataUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client() -> StockHistoricalDataClient:
    settings = get_settings()
    if not settings.alpaca_configured:
        raise MarketDataUnavailable("Alpaca market-data credentials are not configured.")
    return StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )


def _feed(settings: Settings) -> DataFeed:
    feeds = {
        "iex": DataFeed.IEX,
        "sip": DataFeed.SIP,
        "delayed_sip": DataFeed.DELAYED_SIP,
    }
    return feeds.get(settings.alpaca_data_feed, DataFeed.IEX)


def get_daily_closes(
    symbols: list[str],
    start: date,
    end: date | None = None,
    *,
    adjustment: Adjustment = Adjustment.SPLIT,
) -> pd.DataFrame:
    """Return daily closes with accounting-safe adjustment semantics.

    Portfolio reconstruction must use split-only adjustment because dividend cash is
    already represented by canonical transactions. Callers doing benchmark-only
    total-return analysis may explicitly request ``Adjustment.ALL``.
    """
    normalized = sorted({normalize_symbol(symbol) for symbol in symbols})
    if not normalized:
        return pd.DataFrame()
    request = StockBarsRequest(
        symbol_or_symbols=normalized,
        timeframe=TimeFrame.Day,
        start=datetime.combine(start, time.min, tzinfo=timezone.utc),
        end=(
            datetime.combine(end, time.max, tzinfo=timezone.utc)
            if end
            else datetime.now(timezone.utc)
        ),
        adjustment=adjustment,
        feed=_feed(get_settings()),
    )
    frame = _client().get_stock_bars(request).df
    if frame.empty:
        return pd.DataFrame(columns=normalized)
    if isinstance(frame.index, pd.MultiIndex):
        closes = frame["close"].unstack(level="symbol")
    else:
        closes = frame[["close"]]
        closes.columns = normalized[:1]
    closes.index = pd.to_datetime(closes.index, utc=True).tz_convert(None).normalize()
    closes = closes.sort_index()
    closes.attrs["last_price_dates"] = {
        symbol: series.dropna().index[-1].date()
        for symbol, series in closes.items()
        if not series.dropna().empty
    }
    return closes.ffill()


def get_benchmark_daily_closes(
    symbols: list[str], start: date, end: date | None = None
) -> pd.DataFrame:
    """Return dividend-adjusted benchmark closes for total-return comparisons only."""
    return get_daily_closes(symbols, start, end, adjustment=Adjustment.ALL)
