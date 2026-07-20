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
    requests = []

    def historical(symbols, start, end, *, adjustment):
        requests.append(SimpleNamespace(symbols=symbols, adjustment=adjustment))
        frame = pd.DataFrame({"ABC": [10.0]}, index=pd.to_datetime(["2026-01-02"]))
        return SimpleNamespace(data=frame)

    monkeypatch.setattr(market_data, "get_historical_daily_bars", historical)

    market_data.get_daily_closes(["ABC"], date(2026, 1, 1), date(2026, 1, 3))
    market_data.get_benchmark_daily_closes(["ABC"], date(2026, 1, 1), date(2026, 1, 3))

    assert requests[0].adjustment.value == Adjustment.SPLIT.value
    assert requests[1].adjustment.value == Adjustment.ALL.value


def test_spaxx_uses_disclosed_fixed_nav_without_stock_bar_request(monkeypatch) -> None:
    requests = []

    def historical(symbols, start, end, *, adjustment):
        requests.append(symbols)
        frame = pd.DataFrame({"AAPL": [200.0]}, index=pd.to_datetime(["2026-07-17"]))
        frame.attrs["warnings"] = ()
        frame.attrs["last_price_dates"] = {"AAPL": date(2026, 7, 17)}
        return SimpleNamespace(data=frame)

    monkeypatch.setattr(market_data, "get_historical_daily_bars", historical)

    closes = market_data.get_daily_closes(["AAPL", "SPAXX"], date(2026, 7, 16), date(2026, 7, 20))

    assert requests == [["AAPL"]]
    assert closes["SPAXX"].dropna().eq(1.0).all()
    assert closes.attrs["cash_equivalent_conventions"] == {"SPAXX": 1.0}
    assert any("fixed $1.00 NAV" in warning for warning in closes.attrs["warnings"])
