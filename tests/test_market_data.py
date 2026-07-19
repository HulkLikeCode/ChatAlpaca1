from datetime import date
from types import SimpleNamespace

import pandas as pd
from alpaca.data.enums import Adjustment

from chat_alpaca import market_data


class _Bars:
    def __init__(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [("ABC", pd.Timestamp("2026-01-02", tz="UTC"))],
            names=["symbol", "timestamp"],
        )
        self.df = pd.DataFrame({"close": [10.0]}, index=index)


class _Client:
    def __init__(self) -> None:
        self.requests = []

    def get_stock_bars(self, request):
        self.requests.append(request)
        return _Bars()


def test_portfolio_closes_default_to_split_only_and_benchmarks_use_total_return(
    monkeypatch,
) -> None:
    client = _Client()
    monkeypatch.setattr(market_data, "_client", lambda: client)
    monkeypatch.setattr(
        market_data, "get_settings", lambda: SimpleNamespace(alpaca_data_feed="iex")
    )

    market_data.get_daily_closes(["ABC"], date(2026, 1, 1), date(2026, 1, 3))
    market_data.get_benchmark_daily_closes(["ABC"], date(2026, 1, 1), date(2026, 1, 3))

    assert client.requests[0].adjustment == Adjustment.SPLIT
    assert client.requests[1].adjustment == Adjustment.ALL
