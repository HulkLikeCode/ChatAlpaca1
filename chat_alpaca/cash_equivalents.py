from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import pandas as pd

from chat_alpaca.market_calendar import market_session_index

# Explicit conventions only. Do not infer stable NAV from a five-letter mutual-fund symbol.
CASH_EQUIVALENT_NAV = {"SPAXX": 1.0}


def split_cash_equivalent_symbols(
    symbols: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized = tuple(sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()}))
    cash_equivalents = tuple(symbol for symbol in normalized if symbol in CASH_EQUIVALENT_NAV)
    market_symbols = tuple(symbol for symbol in normalized if symbol not in CASH_EQUIVALENT_NAV)
    return market_symbols, cash_equivalents


def add_cash_equivalent_closes(
    closes: pd.DataFrame,
    symbols: Iterable[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Add explicitly disclosed fixed-NAV accounting series for supported cash equivalents."""
    requested = tuple(sorted(set(symbols) & CASH_EQUIVALENT_NAV.keys()))
    if not requested:
        return closes
    attrs = dict(closes.attrs)
    sessions = market_session_index(start, end)
    index = pd.DatetimeIndex(closes.index).normalize().union(sessions).sort_values()
    result = closes.reindex(index)
    for symbol in requested:
        result[symbol] = CASH_EQUIVALENT_NAV[symbol]
    warnings = list(attrs.get("warnings", ()))
    warnings.append(
        "SPAXX is a cash-equivalent money-market holding outside Alpaca stock-bar coverage; "
        "it is valued at its disclosed fixed $1.00 NAV convention. Ledger distributions remain "
        "the source of income."
    )
    last_dates = dict(attrs.get("last_price_dates", {}))
    if not index.empty:
        last_dates.update({symbol: index[-1].date() for symbol in requested})
    attrs["warnings"] = tuple(dict.fromkeys(warnings))
    attrs["last_price_dates"] = last_dates
    attrs["cash_equivalent_conventions"] = {
        symbol: CASH_EQUIVALENT_NAV[symbol] for symbol in requested
    }
    result.attrs = attrs
    return result
