from __future__ import annotations

from datetime import date, datetime, timezone
from functools import lru_cache

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient

from chat_alpaca.config import Settings, get_settings
from chat_alpaca.db import session_scope
from chat_alpaca.historical_data import (
    AlpacaHistoricalProvider,
    HistoricalCoverageResult,
    HistoricalDataService,
    HistoricalRequest,
    PriceAdjustment,
    SqlHistoricalDataRepository,
)
from chat_alpaca.migrations import upgrade_database
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
    result = get_historical_daily_bars(
        normalized,
        start,
        end,
        adjustment={
            Adjustment.RAW: PriceAdjustment.RAW,
            Adjustment.SPLIT: PriceAdjustment.SPLIT,
            Adjustment.DIVIDEND: PriceAdjustment.DIVIDEND,
            Adjustment.ALL: PriceAdjustment.TOTAL_RETURN,
        }[adjustment],
    )
    return result.data


def get_historical_daily_bars(
    symbols: list[str],
    start: date,
    end: date | None = None,
    *,
    adjustment: PriceAdjustment = PriceAdjustment.SPLIT,
    refresh: bool = True,
) -> HistoricalCoverageResult:
    """Return durable closes plus typed provenance and coverage diagnostics."""
    requested_end = end or datetime.now(timezone.utc).date()
    request = HistoricalRequest(tuple(symbols), start, requested_end, adjustment)
    upgrade_database()
    with session_scope() as session:
        repository = SqlHistoricalDataRepository(session)
        provider = AlpacaHistoricalProvider(client=_client(), settings=get_settings())
        return HistoricalDataService(repository, provider).get(request, refresh=refresh)


def get_benchmark_daily_closes(
    symbols: list[str], start: date, end: date | None = None
) -> pd.DataFrame:
    """Return dividend-adjusted benchmark closes for total-return comparisons only."""
    return get_daily_closes(symbols, start, end, adjustment=Adjustment.ALL)
