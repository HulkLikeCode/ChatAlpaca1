from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from chat_alpaca.portfolio_service import list_portfolios, seed_database
from chat_alpaca.realtime import (
    ActiveSessionMonitor,
    ActiveSessionRefreshScheduler,
    ActiveSessionRegistry,
    AlpacaWebSocketSession,
    FreshnessStatus,
    HistoricalGapBackfiller,
    QuoteBook,
    QuoteRecord,
    SlidingWindowRateLimiter,
    SnapshotBatcher,
    SubscriptionInputs,
    alpaca_clients,
    build_portfolio_pulse,
    classify_quote,
    correlation_heuristic,
    market_context_metrics,
    market_hours_state,
    prioritize_subscriptions,
    sample_realized_volatility,
)
from chat_alpaca.trading import submit_allocated_order, sync_allocations

UTC = timezone.utc


def test_subscription_priorities_and_thirty_symbol_cap() -> None:
    inputs = SubscriptionInputs(
        held_symbols=frozenset(f"H{index}" for index in range(40)),
        open_order_symbols=frozenset({"ORD"}),
        selected_symbol="SEL",
        visible_symbols=frozenset({"VIS"}),
        selected_portfolio_symbols=frozenset({"PORT"}),
        risk_contributions={"RISK": 0.8},
        position_values={"BIG": 1_000_000},
        active_alert_symbols=frozenset({"ALERT"}),
        broad_market_proxies=("SPY",),
        sector_proxies=("XLK",),
    )

    plan = prioritize_subscriptions(inputs)

    assert len(plan.streamed) == 30
    assert plan.streamed[:9] == (
        "ORD",
        "SEL",
        "VIS",
        "PORT",
        "RISK",
        "BIG",
        "ALERT",
        "SPY",
        "XLK",
    )
    assert set(plan.snapshot) == set(inputs.held_symbols) - set(plan.streamed)


class FakeSnapshotClient:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, ...]] = []

    def get_stock_snapshot(self, request: object) -> dict[str, object]:
        symbols = tuple(request.symbol_or_symbols)
        self.requests.append(symbols)
        return {symbol: self.responses[symbol] for symbol in symbols if symbol in self.responses}


def _snapshot(price: float, at: datetime) -> object:
    return SimpleNamespace(
        latest_trade=SimpleNamespace(price=price, timestamp=at),
        latest_quote=SimpleNamespace(bid_price=price - 0.1, ask_price=price + 0.1, timestamp=at),
        previous_daily_bar=SimpleNamespace(close=price - 1),
    )


def test_rest_snapshot_batching_fallback_and_missing_quotes() -> None:
    now = datetime(2026, 7, 20, 15, tzinfo=UTC)
    client = FakeSnapshotClient({"A": _snapshot(10, now), "B": _snapshot(20, now)})
    book = QuoteBook()
    batcher = SnapshotBatcher(client, book, batch_size=2, now=lambda: now)

    missing = batcher.refresh(["A", "B", "C"])

    assert client.requests == [("A", "B"), ("C",)]
    assert missing == ("C",)
    records = book.records(["A", "B", "C"], now=now)
    assert records["A"].status == FreshnessStatus.RECENTLY_REFRESHED
    assert records["C"].status == FreshnessStatus.UNAVAILABLE


def test_duplicate_and_out_of_order_stream_events_are_ignored() -> None:
    book = QuoteBook()
    now = datetime(2026, 7, 20, 15, tzinfo=UTC)
    current = QuoteRecord("AAPL", latest_trade=200, as_of_time=now, source="stream")

    assert book.merge(current, "trade")
    assert not book.merge(current, "trade")
    assert not book.merge(
        QuoteRecord(
            "AAPL",
            latest_trade=190,
            as_of_time=now - timedelta(seconds=1),
            source="stream",
        ),
        "trade",
    )
    assert book.records(["AAPL"], streamed_symbols=["AAPL"], now=now)["AAPL"].price == 200

    # A trade may precede a newer quote and is still the newest trade event.
    quote_book = QuoteBook()
    assert quote_book.merge(
        QuoteRecord("MSFT", bid=399, ask=401, as_of_time=now, source="stream"), "quote"
    )
    assert quote_book.merge(
        QuoteRecord(
            "MSFT",
            latest_trade=400,
            as_of_time=now - timedelta(milliseconds=1),
            source="stream",
        ),
        "trade",
    )
    merged = quote_book.records(["MSFT"], streamed_symbols=["MSFT"], now=now)["MSFT"]
    assert merged.latest_trade == 400
    assert merged.midpoint == 400


def test_historical_seed_does_not_replace_snapshot_previous_close() -> None:
    book = QuoteBook()
    receipt = datetime(2026, 7, 20, 21, tzinfo=UTC)
    assert book.merge(
        QuoteRecord(
            "AAPL",
            latest_trade=200,
            previous_close=195,
            receipt_time=receipt,
            as_of_time=receipt,
            source="snapshot",
        ),
        "snapshot",
    )

    book.seed_previous_closes({"AAPL": 199.5})

    assert book.records(["AAPL"], now=receipt)["AAPL"].previous_close == 195


def test_stale_previous_close_and_streaming_classification() -> None:
    now = datetime(2026, 7, 20, 15, tzinfo=UTC)
    stale = classify_quote(
        QuoteRecord(
            "AAPL",
            latest_trade=200,
            as_of_time=now - timedelta(minutes=10),
            source="snapshot",
        ),
        now=now,
    )
    previous = classify_quote(QuoteRecord("MSFT", previous_close=400), now=now)
    streaming = classify_quote(
        QuoteRecord("SPY", latest_trade=600, as_of_time=now, source="stream"),
        now=now,
        streamed_symbols=["SPY"],
    )

    assert stale.status == FreshnessStatus.STALE
    assert "600 seconds" in str(stale.staleness_reason)
    assert previous.status == FreshnessStatus.PREVIOUS_CLOSE
    assert streaming.status == FreshnessStatus.STREAMING


@pytest.mark.parametrize("invalid_trade", [0, -1, float("nan"), float("inf"), True])
def test_quote_price_falls_through_invalid_latest_trade(invalid_trade: object) -> None:
    quote = QuoteRecord("AAA", latest_trade=invalid_trade, bid=99, ask=101, previous_close=90)

    assert quote.price == 100
    assert quote.intraday_price == 100
    assert quote.price_source == "midpoint"


@pytest.mark.parametrize(
    ("bid", "ask"),
    [
        (None, 101),
        (99, None),
        (0, 101),
        (-1, 101),
        (float("nan"), 101),
        (99, float("inf")),
        (True, 101),
    ],
)
def test_quote_price_falls_through_invalid_midpoint_components(bid: object, ask: object) -> None:
    quote = QuoteRecord("AAA", bid=bid, ask=ask, previous_close=90)

    assert quote.midpoint is None
    assert quote.spread is None
    assert quote.price == 90
    assert quote.price_source == "previous_close"


@pytest.mark.parametrize("invalid_close", [0, -1, float("nan"), float("inf"), True])
def test_quote_price_is_unavailable_when_every_source_is_invalid(invalid_close: object) -> None:
    quote = QuoteRecord("AAA", latest_trade=0, bid=0, ask=0, previous_close=invalid_close)

    assert quote.price is None
    assert quote.price_source == "unavailable"


def test_invalid_intraday_values_preserve_previous_close_classification_and_provenance() -> None:
    now = datetime(2026, 7, 20, 15, tzinfo=UTC)

    quote = classify_quote(
        QuoteRecord("AAA", latest_trade=float("nan"), previous_close=90, source="snapshot"),
        now=now,
    )

    assert quote.price == 90
    assert quote.status == FreshnessStatus.PREVIOUS_CLOSE
    assert quote.source == "previous_close"


def test_off_hours_snapshot_uses_receipt_freshness_and_retains_market_as_of() -> None:
    now = datetime(2026, 7, 20, 23, tzinfo=UTC)
    market_time = datetime(2026, 7, 20, 19, 59, tzinfo=UTC)
    refreshed = classify_quote(
        QuoteRecord(
            "AAPL",
            latest_trade=200,
            trade_time=market_time,
            receipt_time=now,
            as_of_time=market_time,
            source="snapshot",
        ),
        now=now,
    )

    assert refreshed.status == FreshnessStatus.RECENTLY_REFRESHED
    assert refreshed.as_of_time == market_time


def test_mixed_stream_and_snapshot_values_build_one_indicative_pulse() -> None:
    portfolio = SimpleNamespace(
        name="Primary",
        cash=100,
        holdings=[
            SimpleNamespace(symbol="AAPL", shares=2),
            SimpleNamespace(symbol="MSFT", shares=1),
        ],
    )
    quotes = {
        "AAPL": QuoteRecord(
            "AAPL",
            latest_trade=200,
            previous_close=195,
            status=FreshnessStatus.STREAMING,
        ),
        "MSFT": QuoteRecord(
            "MSFT",
            bid=399,
            ask=401,
            previous_close=390,
            status=FreshnessStatus.RECENTLY_REFRESHED,
        ),
    }

    pulse = build_portfolio_pulse([portfolio], quotes)

    assert pulse.indicative_total_value == 900
    assert pulse.daily_change == 20
    assert pulse.portfolio_freshness == {"Primary": True}
    assert not pulse.stale_or_missing


def test_cash_only_pulse_has_a_dollar_for_dollar_total() -> None:
    portfolio = SimpleNamespace(name="Cash", cash=125.50, holdings=[])

    pulse = build_portfolio_pulse([portfolio], {})

    assert pulse.indicative_total_value == 125.50
    assert pulse.daily_change is None


@pytest.mark.parametrize(
    ("change", "available"),
    [(0.009, False), (-0.009, False), (0.01, True), (-0.01, True)],
)
def test_pulse_share_of_net_daily_pl_threshold(change: float, available: bool) -> None:
    portfolio = SimpleNamespace(
        name="Threshold",
        cash=0,
        holdings=[SimpleNamespace(symbol="AAA", shares=1)],
    )
    pulse = build_portfolio_pulse(
        [portfolio],
        {
            "AAA": QuoteRecord(
                "AAA",
                latest_trade=100 + change,
                previous_close=100,
                status=FreshnessStatus.STREAMING,
            )
        },
    )

    assert (pulse.holdings[0].contribution is not None) is available


def test_pulse_combines_the_same_symbol_across_portfolios() -> None:
    portfolios = [
        SimpleNamespace(
            name="First",
            cash=0,
            holdings=[SimpleNamespace(symbol="AAPL", shares=2)],
        ),
        SimpleNamespace(
            name="Second",
            cash=0,
            holdings=[SimpleNamespace(symbol="AAPL", shares=3)],
        ),
    ]
    quotes = {
        "AAPL": QuoteRecord(
            "AAPL",
            latest_trade=200,
            previous_close=195,
            status=FreshnessStatus.STREAMING,
        )
    }

    pulse = build_portfolio_pulse(portfolios, quotes)

    assert len(pulse.holdings) == 1
    assert pulse.holdings[0].shares == 5
    assert pulse.holdings[0].value == 1_000
    assert pulse.holdings[0].daily_change == 25
    assert pulse.by_portfolio == {"First": 10, "Second": 15}
    assert pulse.portfolio_freshness == {"First": True, "Second": True}


def test_previous_close_values_do_not_claim_a_zero_daily_move() -> None:
    portfolio = SimpleNamespace(
        name="Primary",
        cash=0,
        holdings=[SimpleNamespace(symbol="AAPL", shares=2)],
    )
    pulse = build_portfolio_pulse(
        [portfolio],
        {"AAPL": QuoteRecord("AAPL", previous_close=195, status=FreshnessStatus.PREVIOUS_CLOSE)},
    )

    assert pulse.indicative_total_value == 390
    assert pulse.daily_change is None
    assert pulse.portfolio_freshness == {"Primary": False}


def test_portfolio_freshness_is_independent_across_selected_portfolios() -> None:
    portfolios = [
        SimpleNamespace(
            name="Fresh",
            cash=0,
            holdings=[SimpleNamespace(symbol="AAPL", shares=1)],
        ),
        SimpleNamespace(
            name="Fallback",
            cash=0,
            holdings=[SimpleNamespace(symbol="MSFT", shares=1)],
        ),
    ]
    quotes = {
        "AAPL": QuoteRecord(
            "AAPL",
            latest_trade=200,
            previous_close=195,
            status=FreshnessStatus.STREAMING,
        ),
        "MSFT": QuoteRecord(
            "MSFT",
            previous_close=400,
            status=FreshnessStatus.PREVIOUS_CLOSE,
        ),
    }

    pulse = build_portfolio_pulse(portfolios, quotes)

    assert pulse.by_portfolio == {"Fresh": 5, "Fallback": None}
    assert pulse.portfolio_freshness == {"Fresh": True, "Fallback": False}


def test_complete_stale_quote_keeps_daily_move_without_claiming_freshness() -> None:
    portfolio = SimpleNamespace(
        name="After close",
        cash=0,
        holdings=[SimpleNamespace(symbol="AAPL", shares=2)],
    )
    pulse = build_portfolio_pulse(
        [portfolio],
        {
            "AAPL": QuoteRecord(
                "AAPL",
                latest_trade=200,
                previous_close=195,
                status=FreshnessStatus.STALE,
            )
        },
    )

    assert pulse.by_portfolio == {"After close": 10}
    assert pulse.daily_change == 10
    assert pulse.portfolio_freshness == {"After close": False}


def test_rate_limiter_and_market_aware_refresh_cadence() -> None:
    elapsed = [0.0]
    limiter = SlidingWindowRateLimiter(2, clock=lambda: elapsed[0])
    assert limiter.acquire()
    assert limiter.acquire()
    assert not limiter.acquire()
    elapsed[0] = 60
    assert limiter.acquire()

    scheduler = ActiveSessionRefreshScheduler(regular_seconds=45, off_hours_seconds=180)
    regular = datetime(2026, 7, 20, 14, tzinfo=UTC)  # 10:00 New York
    assert scheduler.due(["AAPL"], now=regular) == ("AAPL",)
    assert scheduler.due(["AAPL"], now=regular + timedelta(seconds=44)) == ()
    assert scheduler.due(["AAPL"], now=regular + timedelta(seconds=45)) == ("AAPL",)
    scheduler.mark_immediate(["MSFT"])
    assert scheduler.due(["AAPL", "MSFT"], now=regular + timedelta(seconds=46)) == ("MSFT",)


def test_monitor_refreshes_nonstreamed_holdings_and_newly_selected_symbol() -> None:
    refreshed: list[tuple[str, ...]] = []

    class WebSocket:
        def __init__(self) -> None:
            self.started: list[tuple[str, ...]] = []
            self.updated: list[tuple[str, ...]] = []

        def start(self, symbols: object) -> None:
            self.started.append(tuple(symbols))

        def update_subscriptions(self, symbols: object) -> None:
            self.updated.append(tuple(symbols))

        def stop(self) -> None:
            pass

    class Snapshots:
        def refresh(self, symbols: object) -> tuple[str, ...]:
            refreshed.append(tuple(sorted(symbols)))
            return ()

    websocket = WebSocket()
    monitor = ActiveSessionMonitor(
        websocket,
        Snapshots(),
        ActiveSessionRefreshScheduler(),
        QuoteBook(),
        stream_cap=1,
    )
    regular = datetime(2026, 7, 20, 14, tzinfo=UTC)
    monitor.refresh(
        SubscriptionInputs(
            held_symbols=frozenset({"AAPL", "MSFT"}),
            selected_symbol="AAPL",
            broad_market_proxies=(),
            sector_proxies=(),
        ),
        now=regular,
    )
    monitor.refresh(
        SubscriptionInputs(
            held_symbols=frozenset({"AAPL", "MSFT"}),
            selected_symbol="MSFT",
            broad_market_proxies=(),
            sector_proxies=(),
        ),
        now=regular + timedelta(seconds=1),
    )

    assert websocket.started == [("AAPL",), ("MSFT",)]
    assert websocket.updated == [("AAPL",), ("MSFT",)]
    assert refreshed == [("AAPL", "MSFT"), ("MSFT",)]


def test_all_streamed_holdings_receive_an_initial_snapshot_seed() -> None:
    refreshed: list[tuple[str, ...]] = []
    websocket = SimpleNamespace(
        start=lambda symbols: None,
        update_subscriptions=lambda symbols: None,
        stop=lambda: None,
    )
    snapshots = SimpleNamespace(
        refresh=lambda symbols: refreshed.append(tuple(sorted(symbols))) or ()
    )
    monitor = ActiveSessionMonitor(
        websocket,
        snapshots,
        ActiveSessionRefreshScheduler(),
        QuoteBook(),
        stream_cap=2,
    )

    plan = monitor.refresh(
        SubscriptionInputs(
            held_symbols=frozenset({"AAPL", "MSFT"}),
            position_values={"AAPL": 2, "MSFT": 1},
            broad_market_proxies=(),
            sector_proxies=(),
        ),
        now=datetime(2026, 7, 20, 14, tzinfo=UTC),
    )

    assert plan.snapshot == ()
    assert set(plan.streamed) == {"AAPL", "MSFT"}
    assert refreshed == [("AAPL", "MSFT")]


def test_market_hours_regular_closed_and_holiday() -> None:
    regular = market_hours_state(datetime(2026, 7, 20, 14, tzinfo=UTC))
    after_hours = market_hours_state(datetime(2026, 7, 20, 22, tzinfo=UTC))
    holiday = market_hours_state(datetime(2026, 7, 3, 15, tzinfo=UTC))

    assert regular.is_regular_hours
    assert not after_hours.is_regular_hours
    assert not holiday.is_trading_day


class FailingStream:
    def subscribe_quotes(self, handler: object, *symbols: str) -> None:
        pass

    def subscribe_trades(self, handler: object, *symbols: str) -> None:
        pass

    def unsubscribe_quotes(self, *symbols: str) -> None:
        pass

    def unsubscribe_trades(self, *symbols: str) -> None:
        pass

    def run(self) -> None:
        raise ConnectionError("temporary disconnect")

    def stop(self) -> None:
        pass


class StoppingStream(FailingStream):
    def __init__(self, stop: object) -> None:
        self.stop_event = stop

    def run(self) -> None:
        self.stop_event.set()


def test_reconnect_runs_historical_gap_backfill() -> None:
    created = [0]
    backfills: list[tuple[str, ...]] = []
    session: AlpacaWebSocketSession

    def factory() -> object:
        created[0] += 1
        return FailingStream() if created[0] == 1 else StoppingStream(session._stop)

    session = AlpacaWebSocketSession(
        factory,
        QuoteBook(),
        backfill=HistoricalGapBackfiller(lambda symbols: backfills.append(tuple(symbols)) or ()),
        sleeper=lambda seconds: None,
    )
    session._symbols = {"AAPL"}
    session._run()

    assert created[0] == 2
    assert backfills == [("AAPL",)]
    assert "ConnectionError" in str(session.last_error)


def test_session_registry_reuses_and_cleans_abandoned_connections() -> None:
    stopped: list[str] = []

    class Monitor:
        def __init__(self, name: str) -> None:
            self.name = name

        def stop(self) -> None:
            stopped.append(self.name)

    registry = ActiveSessionRegistry(ttl=timedelta(seconds=30))
    start = datetime(2026, 7, 20, 14, tzinfo=UTC)
    first = registry.acquire("one", lambda: Monitor("one"), now=start)
    assert registry.acquire("one", lambda: Monitor("replacement"), now=start) is first
    registry.acquire("two", lambda: Monitor("two"), now=start + timedelta(seconds=31))

    assert stopped == ["one"]
    assert registry.release("two")
    assert stopped == ["one", "two"]


def test_market_context_discloses_components_without_a_score() -> None:
    index = pd.bdate_range("2025-01-01", periods=260)
    closes = pd.DataFrame(
        {
            "SPY": [100 + index for index in range(260)],
            "XLK": [80 + index * 0.5 for index in range(260)],
        },
        index=index,
    )

    result = market_context_metrics(closes)

    assert "Trend" in result
    assert list(result.columns[:2]) == ["Symbol", "Name"]
    assert result["Name"].str.len().max() <= 13
    assert "Drawdown from available-window peak" in result
    assert "Realized volatility" in result
    assert "Correlation heuristic" in result
    assert "21-session SPY correlation" in result
    assert set(result["Correlation observations"]) == {"21/21"}
    assert "Cross-proxy dispersion" not in result
    assert not any("score" in column.lower() for column in result)


def test_market_context_short_history_does_not_reuse_one_return_for_long_horizons() -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05"])

    result = market_context_metrics(pd.DataFrame({"SPY": [100.0, 110.0]}, index=index)).iloc[0]

    assert result["Daily return"] == pytest.approx(0.10)
    assert result["Daily observations"] == "2/2"
    assert result["Daily first date"] == index[0].date()
    assert result["Daily last date"] == index[1].date()
    for horizon, required in (("1M", 22), ("3M", 64), ("12M", 253)):
        assert pd.isna(result[f"{horizon} return"])
        assert result[f"{horizon} observations"] == f"2/{required}"


def test_spy_correlation_uses_exactly_21_complete_pairs_and_discloses_dates() -> None:
    index = pd.bdate_range("2026-01-02", periods=22)
    daily_returns = np.linspace(-0.02, 0.03, 21)
    prices = 100 * np.cumprod(np.r_[1.0, 1 + daily_returns])

    result = market_context_metrics(
        pd.DataFrame({"SPY": prices, "XLK": prices * 0.8}, index=index)
    ).set_index("Symbol")

    assert result.loc["XLK", "21-session SPY correlation"] == pytest.approx(1.0)
    assert result.loc["XLK", "Correlation observations"] == "21/21"
    assert result.loc["XLK", "Correlation first date"] == index[1].date()
    assert result.loc["XLK", "Correlation last date"] == index[-1].date()
    assert result.loc["XLK", "Correlation heuristic"] == "high"


def test_spy_correlation_unavailable_states_are_not_classified_mixed() -> None:
    index = pd.bdate_range("2026-01-02", periods=22)
    variable = 100 * np.cumprod(np.r_[1.0, 1 + np.linspace(-0.02, 0.03, 21)])
    missing_spy = market_context_metrics(pd.DataFrame({"XLK": variable}, index=index)).iloc[0]
    constant_asset = (
        market_context_metrics(
            pd.DataFrame({"SPY": variable, "XLK": np.full(22, 80.0)}, index=index)
        )
        .set_index("Symbol")
        .loc["XLK"]
    )
    insufficient = (
        market_context_metrics(
            pd.DataFrame({"SPY": variable[:10], "XLK": variable[:10]}, index=index[:10])
        )
        .set_index("Symbol")
        .loc["XLK"]
    )

    for row in (missing_spy, constant_asset, insufficient):
        assert pd.isna(row["21-session SPY correlation"])
        assert row["Correlation heuristic"] is None
    assert missing_spy["Correlation observations"] == "0/21"
    assert constant_asset["Correlation observations"] == "21/21"
    assert insufficient["Correlation observations"] == "9/21"


@pytest.mark.parametrize(
    ("correlation", "expected"),
    [(0.699999, "mixed"), (0.700000, "high"), (0.700001, "high")],
)
def test_correlation_heuristic_exact_threshold(correlation: float, expected: str) -> None:
    assert correlation_heuristic(correlation) == expected


def test_sample_realized_volatility_independent_boundaries_and_nonfinite_values() -> None:
    assert sample_realized_volatility(pd.Series([0.02, 0.02])) == 0
    two_returns = sample_realized_volatility(pd.Series([0.01, 0.03]))
    assert two_returns == pytest.approx(np.std([0.01, 0.03], ddof=1) * np.sqrt(252))
    twenty_one = np.linspace(-0.02, 0.02, 21)
    assert sample_realized_volatility(pd.Series(twenty_one)) == pytest.approx(
        np.std(twenty_one, ddof=1) * np.sqrt(252)
    )
    nonfinite = sample_realized_volatility(pd.Series([0.01, np.nan, 0.03, np.inf]))
    assert nonfinite == pytest.approx(np.std([0.01, 0.03], ddof=1) * np.sqrt(252))
    assert pd.isna(sample_realized_volatility(pd.Series([np.nan, np.inf])))


def test_objects_do_not_expose_or_log_credentials(caplog: object) -> None:
    secret = "paper-secret-that-must-not-appear"
    client, factory = alpaca_clients("paper-key-that-must-not-appear", secret, "iex")
    book = QuoteBook()
    session = AlpacaWebSocketSession(factory, book)

    assert secret not in repr(client)
    assert secret not in repr(factory)
    assert secret not in repr(book)
    assert secret not in repr(session)
    assert secret not in getattr(caplog, "text")


def test_fill_updates_are_applied_once_to_the_assigned_portfolio(
    session: object, monkeypatch: object
) -> None:
    seed_database(session)
    target = list_portfolios(session)[3]

    class TradingClient:
        order_id = uuid4()

        def submit_order(self, request: object) -> object:
            return SimpleNamespace(
                id=self.order_id, status="new", filled_qty="0", filled_avg_price=None
            )

        def get_order_by_id(self, order_id: str) -> object:
            assert order_id == str(self.order_id)
            return SimpleNamespace(status="filled", filled_qty="2", filled_avg_price="25")

    client = TradingClient()
    monkeypatch.setattr("chat_alpaca.trading.get_trading_client", lambda: client)
    submit_allocated_order(session, target.id, "MSFT", "buy", 2, "market")

    assert sync_allocations(session) == 1
    assert sync_allocations(session) == 0
    refreshed = next(item for item in list_portfolios(session) if item.id == target.id)
    assert refreshed.cash == Decimal("-50.0000")
    assert [(lot.symbol, lot.shares) for lot in refreshed.holdings] == [
        ("MSFT", Decimal("2.00000000"))
    ]
