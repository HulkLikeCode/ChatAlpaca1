from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from datetime import time as wall_time
from enum import Enum
from typing import Any, Protocol

import numpy as np
import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockSnapshotRequest

from chat_alpaca.analytics import performance_growth, scoped_reconstruction
from chat_alpaca.market_calendar import NEW_YORK, market_session_index
from chat_alpaca.portfolio_service import normalize_symbol

LOGGER = logging.getLogger(__name__)
STREAM_SYMBOL_CAP = 30
BROAD_MARKET_PROXIES = ("SPY", "QQQ", "DIA", "IWM")
SECTOR_PROXIES = ("XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU")
MARKET_CONTEXT_NAMES = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "DIA": "Dow 30",
    "IWM": "Russell 2000",
    "XLC": "Comm Svcs",
    "XLY": "Cons Discret",
    "XLP": "Cons Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLK": "Technology",
    "XLU": "Utilities",
}
OPEN_ORDER_STATUSES = {"new", "accepted", "pending_new", "partially_filled", "held", "replaced"}
CORRELATION_HEURISTIC_DISCLOSURE = (
    "Heuristic correlation label based on a fixed 0.70 threshold; it is descriptive and is not "
    "a statistical-significance test."
)
PERIOD_RETURN_OBSERVATIONS = {"Daily": 2, "1M": 22, "3M": 64, "12M": 253}


class FreshnessStatus(str, Enum):
    STREAMING = "streaming"
    RECENTLY_REFRESHED = "recently refreshed"
    STALE = "stale"
    PREVIOUS_CLOSE = "previous close"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MarketHoursState:
    observed_at: datetime
    is_trading_day: bool
    is_regular_hours: bool
    session_open: datetime | None
    session_close: datetime | None
    label: str


def market_hours_state(now: datetime | None = None) -> MarketHoursState:
    observed = now or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    local = observed.astimezone(NEW_YORK)
    is_session = not market_session_index(
        local.date(), local.date(), now=local, completed_only=False
    ).empty
    opened = datetime.combine(local.date(), wall_time(9, 30), NEW_YORK) if is_session else None
    closed = datetime.combine(local.date(), wall_time(16), NEW_YORK) if is_session else None
    regular = bool(opened and closed and opened <= local < closed)
    if regular:
        label = "regular market hours"
    elif is_session and opened and local < opened:
        label = "pre-market"
    elif is_session:
        label = "after-hours"
    else:
        label = "market closed"
    return MarketHoursState(observed, is_session, regular, opened, closed, label)


@dataclass(frozen=True)
class QuoteRecord:
    symbol: str
    provider: str = "alpaca"
    feed: str = "iex"
    latest_trade: float | None = None
    bid: float | None = None
    ask: float | None = None
    previous_close: float | None = None
    quote_time: datetime | None = None
    trade_time: datetime | None = None
    receipt_time: datetime | None = None
    as_of_time: datetime | None = None
    status: FreshnessStatus = FreshnessStatus.UNAVAILABLE
    staleness_reason: str | None = "no quote, trade, or previous close is available"
    source: str = "unavailable"

    @property
    def midpoint(self) -> float | None:
        bid = _valid_quote_value(self.bid)
        ask = _valid_quote_value(self.ask)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    @property
    def spread(self) -> float | None:
        bid = _valid_quote_value(self.bid)
        ask = _valid_quote_value(self.ask)
        if bid is None or ask is None:
            return None
        return ask - bid

    @property
    def price(self) -> float | None:
        return (
            _valid_quote_value(self.latest_trade)
            or self.midpoint
            or _valid_quote_value(self.previous_close)
        )

    @property
    def intraday_price(self) -> float | None:
        return _valid_quote_value(self.latest_trade) or self.midpoint

    @property
    def price_source(self) -> str:
        if _valid_quote_value(self.latest_trade) is not None:
            return "latest_trade"
        if self.midpoint is not None:
            return "midpoint"
        if _valid_quote_value(self.previous_close) is not None:
            return "previous_close"
        return "unavailable"


def _valid_quote_value(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )


def classify_quote(
    quote: QuoteRecord,
    *,
    now: datetime | None = None,
    streamed_symbols: Iterable[str] = (),
    recent_seconds: int = 75,
    stale_seconds: int = 180,
) -> QuoteRecord:
    observed = _aware(now or datetime.now(timezone.utc))
    as_of = _aware(quote.as_of_time or quote.trade_time or quote.quote_time or quote.receipt_time)
    stream_set = {normalize_symbol(symbol) for symbol in streamed_symbols}
    if quote.intraday_price is None:
        if _valid_quote_value(quote.previous_close) is not None:
            return replace(
                quote,
                status=FreshnessStatus.PREVIOUS_CLOSE,
                staleness_reason="no intraday trade or two-sided quote is available",
                source="previous_close",
            )
        return replace(
            quote,
            status=FreshnessStatus.UNAVAILABLE,
            staleness_reason="no quote, trade, or previous close is available",
            source="unavailable",
        )
    if as_of is None:
        return replace(
            quote,
            status=FreshnessStatus.STALE,
            staleness_reason="provider timestamp is missing",
        )
    hours = market_hours_state(observed)
    freshness_time = as_of
    receipt_time = _aware(quote.receipt_time)
    if quote.source == "snapshot" and not hours.is_regular_hours and receipt_time is not None:
        freshness_time = receipt_time
    age = max(0.0, (observed - freshness_time).total_seconds())
    if quote.symbol in stream_set and quote.source == "stream" and age <= stale_seconds:
        return replace(quote, status=FreshnessStatus.STREAMING, staleness_reason=None)
    if age <= recent_seconds or (not hours.is_regular_hours and age <= stale_seconds * 4):
        return replace(quote, status=FreshnessStatus.RECENTLY_REFRESHED, staleness_reason=None)
    reason = f"last update was {int(age)} seconds ago"
    if not hours.is_regular_hours:
        reason += f"; {hours.label}"
    return replace(quote, status=FreshnessStatus.STALE, staleness_reason=reason)


@dataclass(frozen=True)
class SubscriptionInputs:
    held_symbols: frozenset[str] = frozenset()
    open_order_symbols: frozenset[str] = frozenset()
    selected_symbol: str | None = None
    visible_symbols: frozenset[str] = frozenset()
    selected_portfolio_symbols: frozenset[str] = frozenset()
    risk_contributions: Mapping[str, float] = field(default_factory=dict)
    position_values: Mapping[str, float] = field(default_factory=dict)
    active_alert_symbols: frozenset[str] = frozenset()
    broad_market_proxies: tuple[str, ...] = BROAD_MARKET_PROXIES
    sector_proxies: tuple[str, ...] = SECTOR_PROXIES


@dataclass(frozen=True)
class SubscriptionPlan:
    streamed: tuple[str, ...]
    snapshot: tuple[str, ...]
    priority_reason: Mapping[str, str]


def prioritize_subscriptions(
    inputs: SubscriptionInputs, *, cap: int = STREAM_SYMBOL_CAP
) -> SubscriptionPlan:
    if cap < 1 or cap > STREAM_SYMBOL_CAP:
        raise ValueError(f"Stream cap must be between 1 and {STREAM_SYMBOL_CAP} symbols.")
    held = {normalize_symbol(symbol) for symbol in inputs.held_symbols}
    buckets: list[tuple[str, Sequence[str]]] = [
        ("open order", sorted(inputs.open_order_symbols)),
        ("selected symbol", [inputs.selected_symbol] if inputs.selected_symbol else []),
        ("visible", sorted(inputs.visible_symbols)),
        ("selected portfolio", sorted(inputs.selected_portfolio_symbols)),
        (
            "risk contributor",
            sorted(
                inputs.risk_contributions,
                key=lambda symbol: (-abs(inputs.risk_contributions[symbol]), symbol),
            ),
        ),
        (
            "largest position",
            sorted(
                inputs.position_values,
                key=lambda symbol: (-abs(inputs.position_values[symbol]), symbol),
            ),
        ),
        ("active alert", sorted(inputs.active_alert_symbols)),
        ("broad-market proxy", list(inputs.broad_market_proxies)),
        ("sector proxy", list(inputs.sector_proxies)),
        ("remaining holding", sorted(held)),
    ]
    ordered: list[str] = []
    reasons: dict[str, str] = {}
    for reason, candidates in buckets:
        for candidate in candidates:
            symbol = normalize_symbol(candidate)
            if symbol not in reasons:
                ordered.append(symbol)
                reasons[symbol] = reason
    streamed = tuple(ordered[:cap])
    return SubscriptionPlan(streamed, tuple(sorted(held - set(streamed))), reasons)


class QuoteBook:
    """Thread-safe in-memory intraday state; the ledger remains untouched."""

    def __init__(self) -> None:
        self._quotes: dict[str, QuoteRecord] = {}
        self._events: set[tuple[str, str, datetime | None, float | None, float | None]] = set()
        self._event_times: dict[tuple[str, str], datetime] = {}
        self._lock = threading.RLock()

    def merge(self, incoming: QuoteRecord, event_type: str) -> bool:
        symbol = normalize_symbol(incoming.symbol)
        stamp = _aware(incoming.as_of_time or incoming.trade_time or incoming.quote_time)
        key = (symbol, event_type, stamp, incoming.latest_trade or incoming.bid, incoming.ask)
        with self._lock:
            if key in self._events:
                return False
            existing = self._quotes.get(symbol)
            existing_stamp = _aware(existing.as_of_time) if existing else None
            prior_event_stamp = self._event_times.get((symbol, event_type))
            if prior_event_stamp and stamp and stamp < prior_event_stamp:
                return False
            if event_type == "snapshot" and existing_stamp and stamp and stamp < existing_stamp:
                return False
            self._events.add(key)
            if stamp:
                self._event_times[(symbol, event_type)] = stamp
            if len(self._events) > 10_000:
                self._events = set(list(self._events)[-5_000:])
            if existing:
                incoming = replace(
                    incoming,
                    latest_trade=(
                        incoming.latest_trade
                        if incoming.latest_trade is not None
                        else existing.latest_trade
                    ),
                    bid=incoming.bid if incoming.bid is not None else existing.bid,
                    ask=incoming.ask if incoming.ask is not None else existing.ask,
                    previous_close=(
                        incoming.previous_close
                        if incoming.previous_close is not None
                        else existing.previous_close
                    ),
                    quote_time=incoming.quote_time or existing.quote_time,
                    trade_time=incoming.trade_time or existing.trade_time,
                    receipt_time=max(
                        (
                            item
                            for item in (incoming.receipt_time, existing.receipt_time)
                            if item is not None
                        ),
                        default=None,
                    ),
                    as_of_time=max(
                        (
                            item
                            for item in (incoming.as_of_time, existing.as_of_time)
                            if item is not None
                        ),
                        default=None,
                    ),
                )
            self._quotes[symbol] = replace(incoming, symbol=symbol)
            return True

    def seed_previous_closes(self, closes: Mapping[str, float]) -> None:
        with self._lock:
            for raw_symbol, close in closes.items():
                symbol = normalize_symbol(raw_symbol)
                existing = self._quotes.get(symbol, QuoteRecord(symbol))
                if existing.previous_close is None:
                    self._quotes[symbol] = replace(existing, previous_close=float(close))

    def records(
        self,
        symbols: Iterable[str],
        *,
        streamed_symbols: Iterable[str] = (),
        now: datetime | None = None,
    ) -> dict[str, QuoteRecord]:
        with self._lock:
            return {
                symbol: classify_quote(
                    self._quotes.get(symbol, QuoteRecord(symbol)),
                    streamed_symbols=streamed_symbols,
                    now=now,
                )
                for symbol in sorted({normalize_symbol(item) for item in symbols})
            }


class SlidingWindowRateLimiter:
    def __init__(self, calls_per_minute: int = 180, clock: Callable[[], float] = time.monotonic):
        if calls_per_minute < 1:
            raise ValueError("Rate limit must allow at least one call per minute.")
        self.calls_per_minute = calls_per_minute
        self.clock = clock
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        with self._lock:
            now = self.clock()
            while self._calls and now - self._calls[0] >= 60:
                self._calls.popleft()
            if len(self._calls) >= self.calls_per_minute:
                return False
            self._calls.append(now)
            return True


class SnapshotClient(Protocol):
    def get_stock_snapshot(self, request: StockSnapshotRequest) -> Mapping[str, Any]: ...


def _timestamp(value: object) -> datetime | None:
    timestamp = getattr(value, "timestamp", None)
    return _aware(timestamp) if isinstance(timestamp, datetime) else None


def record_from_snapshot(
    symbol: str, snapshot: object, *, feed: str, receipt: datetime
) -> QuoteRecord:
    trade = getattr(snapshot, "latest_trade", None)
    quote = getattr(snapshot, "latest_quote", None)
    previous = getattr(snapshot, "previous_daily_bar", None)
    trade_time = _timestamp(trade)
    quote_time = _timestamp(quote)
    return QuoteRecord(
        symbol=normalize_symbol(symbol),
        feed=feed,
        latest_trade=float(trade.price) if getattr(trade, "price", None) is not None else None,
        bid=float(quote.bid_price) if getattr(quote, "bid_price", None) is not None else None,
        ask=float(quote.ask_price) if getattr(quote, "ask_price", None) is not None else None,
        previous_close=(
            float(previous.close) if getattr(previous, "close", None) is not None else None
        ),
        quote_time=quote_time,
        trade_time=trade_time,
        receipt_time=receipt,
        as_of_time=max((item for item in (trade_time, quote_time) if item), default=receipt),
        source="snapshot",
    )


class SnapshotBatcher:
    def __init__(
        self,
        client: SnapshotClient,
        quote_book: QuoteBook,
        *,
        feed: str = "iex",
        batch_size: int = 100,
        limiter: SlidingWindowRateLimiter | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.client = client
        self.quote_book = quote_book
        self.feed = feed
        self.batch_size = batch_size
        self.limiter = limiter or SlidingWindowRateLimiter()
        self.now = now

    def refresh(self, symbols: Iterable[str]) -> tuple[str, ...]:
        normalized = sorted({normalize_symbol(symbol) for symbol in symbols})
        missing: list[str] = []
        feed = {"iex": DataFeed.IEX, "sip": DataFeed.SIP, "delayed_sip": DataFeed.DELAYED_SIP}.get(
            self.feed, DataFeed.IEX
        )
        for start in range(0, len(normalized), self.batch_size):
            batch = normalized[start : start + self.batch_size]
            if not self.limiter.acquire():
                missing.extend(batch)
                continue
            response = self.client.get_stock_snapshot(
                StockSnapshotRequest(symbol_or_symbols=batch, feed=feed)
            )
            receipt = self.now()
            for symbol in batch:
                snapshot = response.get(symbol)
                if snapshot is None:
                    missing.append(symbol)
                    continue
                self.quote_book.merge(
                    record_from_snapshot(symbol, snapshot, feed=self.feed, receipt=receipt),
                    "snapshot",
                )
        return tuple(missing)


@dataclass
class ActiveSessionRefreshScheduler:
    regular_seconds: int = 45
    off_hours_seconds: int = 180
    _last_refresh: datetime | None = None
    _urgent: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not 30 <= self.regular_seconds <= 60:
            raise ValueError("Regular-hours snapshot cadence must be between 30 and 60 seconds.")
        if self.off_hours_seconds <= self.regular_seconds:
            raise ValueError("Off-hours cadence must be slower than regular-hours cadence.")

    def mark_immediate(self, symbols: Iterable[str]) -> None:
        self._urgent.update(normalize_symbol(symbol) for symbol in symbols)

    def due(self, symbols: Iterable[str], now: datetime | None = None) -> tuple[str, ...]:
        observed = _aware(now or datetime.now(timezone.utc))
        universe = {normalize_symbol(symbol) for symbol in symbols}
        urgent = tuple(sorted(universe & self._urgent))
        if urgent:
            self._urgent.difference_update(urgent)
            return urgent
        cadence = (
            self.regular_seconds
            if market_hours_state(observed).is_regular_hours
            else self.off_hours_seconds
        )
        if self._last_refresh is None or (observed - self._last_refresh).total_seconds() >= cadence:
            self._last_refresh = observed
            return tuple(sorted(universe))
        return ()


@dataclass(frozen=True)
class ReconnectPolicy:
    initial_seconds: float = 1
    maximum_seconds: float = 30
    multiplier: float = 2

    def delay(self, attempt: int) -> float:
        return min(self.maximum_seconds, self.initial_seconds * self.multiplier ** max(attempt, 0))


@dataclass(frozen=True)
class GapBackfillResult:
    symbols: tuple[str, ...]
    disconnected_at: datetime
    reconnected_at: datetime
    snapshot_missing: tuple[str, ...]


class HistoricalGapBackfiller:
    """Reconcile a disconnected interval without claiming every missed tick was recovered.

    A current snapshot is always fetched. An optional historical loader can cache provider-entitled
    bars for the interval; on Alpaca Basic, the most recent historical window may remain unavailable.
    """

    def __init__(
        self,
        snapshot_refresh: Callable[[Iterable[str]], tuple[str, ...]],
        historical_loader: Callable[[tuple[str, ...], datetime, datetime], object] | None = None,
    ) -> None:
        self.snapshot_refresh = snapshot_refresh
        self.historical_loader = historical_loader
        self.results: deque[GapBackfillResult] = deque(maxlen=100)

    def backfill(
        self, symbols: Iterable[str], disconnected_at: datetime, reconnected_at: datetime
    ) -> GapBackfillResult:
        normalized = tuple(sorted({normalize_symbol(symbol) for symbol in symbols}))
        if self.historical_loader is not None:
            self.historical_loader(normalized, disconnected_at, reconnected_at)
        missing = self.snapshot_refresh(normalized)
        result = GapBackfillResult(normalized, disconnected_at, reconnected_at, tuple(missing))
        self.results.append(result)
        return result


class StreamLike(Protocol):
    def subscribe_quotes(self, handler: Callable[..., Any], *symbols: str) -> None: ...
    def subscribe_trades(self, handler: Callable[..., Any], *symbols: str) -> None: ...
    def unsubscribe_quotes(self, *symbols: str) -> None: ...
    def unsubscribe_trades(self, *symbols: str) -> None: ...
    def run(self) -> None: ...
    def stop(self) -> None: ...


class AlpacaWebSocketSession:
    """A single reconnecting WebSocket lifecycle, safe to reuse across UI reruns."""

    def __init__(
        self,
        stream_factory: Callable[[], StreamLike],
        quote_book: QuoteBook,
        *,
        backfill: HistoricalGapBackfiller | None = None,
        feed: str = "iex",
        reconnect: ReconnectPolicy = ReconnectPolicy(),
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.stream_factory = stream_factory
        self.quote_book = quote_book
        self.backfill = backfill
        self.feed = feed
        self.reconnect = reconnect
        self.sleeper = sleeper
        self._symbols: set[str] = set()
        self._stream: StreamLike | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self.connected = False
        self.last_error: str | None = None
        self.disconnected_at: datetime | None = None

    async def _quote_handler(self, event: object) -> None:
        received = datetime.now(timezone.utc)
        stamp = _timestamp(event)
        self.quote_book.merge(
            QuoteRecord(
                symbol=str(getattr(event, "symbol")),
                feed=self.feed,
                bid=float(getattr(event, "bid_price", 0)) or None,
                ask=float(getattr(event, "ask_price", 0)) or None,
                quote_time=stamp,
                receipt_time=received,
                as_of_time=stamp or received,
                source="stream",
            ),
            "quote",
        )

    async def _trade_handler(self, event: object) -> None:
        received = datetime.now(timezone.utc)
        stamp = _timestamp(event)
        self.quote_book.merge(
            QuoteRecord(
                symbol=str(getattr(event, "symbol")),
                feed=self.feed,
                latest_trade=float(getattr(event, "price")),
                trade_time=stamp,
                receipt_time=received,
                as_of_time=stamp or received,
                source="stream",
            ),
            "trade",
        )

    def start(self, symbols: Iterable[str]) -> None:
        requested = {normalize_symbol(symbol) for symbol in symbols}
        with self._lock:
            if self._thread and self._thread.is_alive():
                self.update_subscriptions(requested)
                return
            self._symbols = requested
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="alpaca-active-session", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            try:
                stream = self.stream_factory()
                with self._lock:
                    self._stream = stream
                    symbols = tuple(sorted(self._symbols))
                if symbols:
                    stream.subscribe_quotes(self._quote_handler, *symbols)
                    stream.subscribe_trades(self._trade_handler, *symbols)
                self.connected = True
                if self.disconnected_at is not None and self.backfill and symbols:
                    self.backfill.backfill(
                        symbols, self.disconnected_at, datetime.now(timezone.utc)
                    )
                self.disconnected_at = None
                stream.run()
                if not self._stop.is_set():
                    raise ConnectionError("Alpaca stream ended unexpectedly")
            except Exception as exc:  # pragma: no branch - recovery boundary
                self.connected = False
                self.last_error = type(exc).__name__
                self.disconnected_at = self.disconnected_at or datetime.now(timezone.utc)
                if self._stop.is_set():
                    break
                delay = self.reconnect.delay(attempt)
                if self.sleeper is time.sleep:
                    self._stop.wait(delay)
                else:
                    self.sleeper(delay)
                attempt += 1
        self.connected = False

    def update_subscriptions(self, symbols: Iterable[str]) -> None:
        requested = {normalize_symbol(symbol) for symbol in symbols}
        with self._lock:
            added = requested - self._symbols
            removed = self._symbols - requested
            self._symbols = requested
            stream = self._stream
            if stream and self.connected:
                if removed:
                    stream.unsubscribe_quotes(*sorted(removed))
                    stream.unsubscribe_trades(*sorted(removed))
                if added:
                    stream.subscribe_quotes(self._quote_handler, *sorted(added))
                    stream.subscribe_trades(self._trade_handler, *sorted(added))

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            stream = self._stream
        if stream is not None:
            try:
                stream.stop()
            except (RuntimeError, asyncio.InvalidStateError):
                LOGGER.info("Alpaca stream was already stopped")
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    @property
    def symbols(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._symbols))


@dataclass
class SessionLease:
    monitor: Any
    touched_at: datetime


class ActiveSessionRegistry:
    """Owns bounded browser-session leases and reaps them during active reruns."""

    def __init__(self, ttl: timedelta = timedelta(minutes=3)) -> None:
        self.ttl = ttl
        self._leases: dict[str, SessionLease] = {}
        self._lock = threading.RLock()

    def acquire(
        self,
        session_id: str,
        factory: Callable[[], Any],
        *,
        now: datetime | None = None,
    ) -> Any:
        observed = _aware(now or datetime.now(timezone.utc))
        self.cleanup(now=observed, exclude={session_id})
        with self._lock:
            lease = self._leases.get(session_id)
            if lease is None:
                lease = SessionLease(factory(), observed)
                self._leases[session_id] = lease
            else:
                lease.touched_at = observed
            return lease.monitor

    def cleanup(self, *, now: datetime | None = None, exclude: set[str] | None = None) -> int:
        observed = _aware(now or datetime.now(timezone.utc))
        protected = exclude or set()
        with self._lock:
            expired = [
                key
                for key, lease in self._leases.items()
                if key not in protected and observed - lease.touched_at > self.ttl
            ]
            monitors = [self._leases.pop(key).monitor for key in expired]
        for monitor in monitors:
            monitor.stop()
        return len(monitors)

    def release(self, session_id: str) -> bool:
        with self._lock:
            lease = self._leases.pop(session_id, None)
        if lease:
            lease.monitor.stop()
            return True
        return False


class ActiveSessionMonitor:
    """Coordinates priority, streaming, backfill, and REST fallback for one active UI session."""

    def __init__(
        self,
        websocket: AlpacaWebSocketSession,
        snapshots: SnapshotBatcher,
        scheduler: ActiveSessionRefreshScheduler,
        quote_book: QuoteBook,
        *,
        stream_cap: int = STREAM_SYMBOL_CAP,
    ) -> None:
        self.websocket = websocket
        self.snapshots = snapshots
        self.scheduler = scheduler
        self.quote_book = quote_book
        self.stream_cap = stream_cap
        self.plan = SubscriptionPlan((), (), {})
        self._selected_or_visible: set[str] = set()
        self._initialized = False

    def refresh(
        self,
        inputs: SubscriptionInputs,
        *,
        previous_closes: Mapping[str, float] | None = None,
        now: datetime | None = None,
    ) -> SubscriptionPlan:
        observed = _aware(now or datetime.now(timezone.utc))
        plan = prioritize_subscriptions(inputs, cap=self.stream_cap)
        self.quote_book.seed_previous_closes(previous_closes or {})
        held = {normalize_symbol(symbol) for symbol in inputs.held_symbols}
        selected_or_visible = {
            *({normalize_symbol(inputs.selected_symbol)} if inputs.selected_symbol else set()),
            *(normalize_symbol(symbol) for symbol in inputs.visible_symbols),
        }
        newly_urgent = selected_or_visible - self._selected_or_visible
        self._selected_or_visible = selected_or_visible
        self.websocket.start(plan.streamed)
        self.websocket.update_subscriptions(plan.streamed)
        refresh_universe = set(plan.snapshot)
        current = self.quote_book.records(held, streamed_symbols=plan.streamed, now=observed)
        if not self._initialized or not market_hours_state(observed).is_regular_hours:
            refresh_universe.update(held)
        else:
            refresh_universe.update(
                symbol
                for symbol in plan.streamed
                if symbol in held
                and current[symbol].status
                in {
                    FreshnessStatus.STALE,
                    FreshnessStatus.PREVIOUS_CLOSE,
                    FreshnessStatus.UNAVAILABLE,
                }
            )
        due = set(self.scheduler.due(refresh_universe, now=observed)) | newly_urgent
        if due:
            self.snapshots.refresh(due)
        self._initialized = True
        self.plan = plan
        return plan

    def records(
        self, symbols: Iterable[str], *, now: datetime | None = None
    ) -> dict[str, QuoteRecord]:
        return self.quote_book.records(symbols, streamed_symbols=self.plan.streamed, now=now)

    def stop(self) -> None:
        self.websocket.stop()


@dataclass(frozen=True)
class PulseHolding:
    symbol: str
    shares: float
    price: float | None
    value: float | None
    daily_change: float | None
    contribution: float | None
    status: FreshnessStatus


@dataclass(frozen=True)
class PortfolioPulse:
    indicative_total_value: float | None
    daily_change: float | None
    holdings: tuple[PulseHolding, ...]
    by_portfolio: Mapping[str, float | None]
    portfolio_values: Mapping[str, float | None]
    portfolio_contributions: Mapping[str, float | None]
    portfolio_staleness_reasons: Mapping[str, str | None]
    portfolio_freshness: Mapping[str, bool]
    stale_or_missing: tuple[str, ...]
    indicative_as_of: datetime | None = None
    indicative_provenance: str | None = None


def build_portfolio_pulse(
    portfolios: Iterable[object], quotes: Mapping[str, QuoteRecord]
) -> PortfolioPulse:
    portfolio_list = list(portfolios)
    shares_by_symbol: dict[str, float] = {}
    portfolio_changes: dict[str, float | None] = {}
    portfolio_values: dict[str, float | None] = {}
    portfolio_staleness_reasons: dict[str, str | None] = {}
    portfolio_freshness: dict[str, bool] = {}
    cash_total = 0.0
    for portfolio in portfolio_list:
        name = str(getattr(portfolio, "name"))
        cash_total += float(getattr(portfolio, "cash"))
        changes: list[float] = []
        values: list[float] = []
        statuses: list[FreshnessStatus] = []
        staleness_reasons: set[str] = set()
        holdings = list(getattr(portfolio, "holdings"))
        for lot in holdings:
            symbol = normalize_symbol(getattr(lot, "symbol"))
            shares = float(getattr(lot, "shares"))
            shares_by_symbol[symbol] = shares_by_symbol.get(symbol, 0.0) + shares
            quote = quotes.get(symbol, QuoteRecord(symbol))
            statuses.append(quote.status)
            if quote.staleness_reason and quote.status not in {
                FreshnessStatus.STREAMING,
                FreshnessStatus.RECENTLY_REFRESHED,
            }:
                staleness_reasons.add(quote.staleness_reason)
            if quote.price is not None:
                values.append(shares * quote.price)
            intraday_price = quote.intraday_price
            previous_close = _valid_quote_value(quote.previous_close)
            change = (
                shares * (intraday_price - previous_close)
                if intraday_price is not None and previous_close is not None
                else None
            )
            if change is not None:
                changes.append(change)
        portfolio_changes[name] = (
            sum(changes) if holdings and len(changes) == len(holdings) else None
        )
        portfolio_values[name] = (
            float(getattr(portfolio, "cash")) + sum(values)
            if len(values) == len(holdings)
            else None
        )
        portfolio_staleness_reasons[name] = (
            "; ".join(sorted(staleness_reasons)) if staleness_reasons else None
        )
        portfolio_freshness[name] = bool(
            holdings
            and len(changes) == len(holdings)
            and all(
                status in {FreshnessStatus.STREAMING, FreshnessStatus.RECENTLY_REFRESHED}
                for status in statuses
            )
        )
    rows = []
    for symbol, shares in sorted(shares_by_symbol.items()):
        quote = quotes.get(symbol, QuoteRecord(symbol))
        price = quote.price
        intraday_price = quote.intraday_price
        previous_close = _valid_quote_value(quote.previous_close)
        change = (
            shares * (intraday_price - previous_close)
            if intraday_price is not None and previous_close is not None
            else None
        )
        rows.append(
            PulseHolding(
                symbol,
                shares,
                price,
                shares * price if price is not None else None,
                change,
                None,
                quote.status,
            )
        )
    market_value = sum(row.value for row in rows if row.value is not None)
    total = cash_total + market_value if all(row.value is not None for row in rows) else None
    total_change = (
        sum(row.daily_change for row in rows if row.daily_change is not None)
        if rows and all(row.daily_change is not None for row in rows)
        else None
    )
    portfolio_contributions = {
        name: (
            change / total_change
            if change is not None and total_change is not None and abs(total_change) >= 0.01
            else None
        )
        for name, change in portfolio_changes.items()
    }
    changed_rows = tuple(
        replace(
            row,
            contribution=(
                row.daily_change / total_change
                if total_change is not None and abs(total_change) >= 0.01
                else None
            ),
        )
        for row in rows
    )
    stale = tuple(
        sorted(
            {
                row.symbol
                for row in rows
                if row.status
                in {
                    FreshnessStatus.STALE,
                    FreshnessStatus.PREVIOUS_CLOSE,
                    FreshnessStatus.UNAVAILABLE,
                }
            }
        )
    )
    overlay_quotes = [
        quotes[row.symbol]
        for row in changed_rows
        if row.daily_change is not None and row.symbol in quotes
    ]
    indicative_as_of = max(
        (quote.as_of_time for quote in overlay_quotes if quote.as_of_time is not None),
        default=None,
    )
    sources = sorted({quote.source for quote in overlay_quotes if quote.source != "unavailable"})
    feeds = sorted({quote.feed.upper() for quote in overlay_quotes if quote.feed})
    indicative_provenance = (
        "Alpaca " + ("/".join(feeds) + " " if feeds else "") + "/".join(sources or ["quote"])
        if overlay_quotes
        else None
    )
    return PortfolioPulse(
        total,
        total_change,
        changed_rows,
        portfolio_changes,
        portfolio_values,
        portfolio_contributions,
        portfolio_staleness_reasons,
        portfolio_freshness,
        stale,
        indicative_as_of,
        indicative_provenance,
    )


def correlation_heuristic(correlation: float | None) -> str | None:
    """Apply the fixed descriptive threshold only when correlation is available."""
    if correlation is None or not np.isfinite(correlation):
        return None
    return "high" if correlation >= 0.70 else "mixed"


def sample_realized_volatility(returns: pd.Series, window: int = 21) -> float:
    """Annualize sample volatility from up to the latest window of finite daily returns."""
    finite = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    finite = finite.tail(window)
    return float(finite.std(ddof=1) * np.sqrt(252)) if len(finite) >= 2 else np.nan


def _usable_close_series(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    series = series[series > 0].sort_index()
    return series[~series.index.duplicated(keep="last")]


def _period_return(
    series: pd.Series, required_observations: int
) -> tuple[float, str, object | None, object | None]:
    selected = series.tail(required_observations)
    count = len(selected)
    first = selected.index[0].date() if count else None
    last = selected.index[-1].date() if count else None
    value = (
        float(selected.iloc[-1] / selected.iloc[0] - 1)
        if count == required_observations
        else np.nan
    )
    return value, f"{count}/{required_observations}", first, last


def _spy_correlation(
    symbol_returns: pd.Series, spy_returns: pd.Series | None
) -> tuple[float, str | None, str, object | None, object | None]:
    if spy_returns is None:
        return np.nan, None, "0/21", None, None
    aligned = pd.concat(
        [symbol_returns.rename("symbol"), spy_returns.rename("SPY")], axis=1, join="inner"
    ).replace([np.inf, -np.inf], np.nan)
    aligned = aligned.dropna().tail(21)
    count = len(aligned)
    first = aligned.index[0].date() if count else None
    last = aligned.index[-1].date() if count else None
    if count < 21:
        return np.nan, None, f"{count}/21", first, last
    symbol_values = aligned["symbol"].to_numpy(dtype=float)
    spy_values = aligned["SPY"].to_numpy(dtype=float)
    if np.var(symbol_values, ddof=1) == 0 or np.var(spy_values, ddof=1) == 0:
        return np.nan, None, "21/21", first, last
    correlation = float(np.corrcoef(symbol_values, spy_values)[0, 1])
    if not np.isfinite(correlation):
        return np.nan, None, "21/21", first, last
    return correlation, correlation_heuristic(correlation), "21/21", first, last


@dataclass(frozen=True)
class PortfolioMonitorMetric:
    portfolio: str
    realized_volatility: float | None
    spy_correlation: float | None
    correlation_observations: str


def portfolio_monitor_metrics(
    portfolios: Sequence[object],
    closes: pd.DataFrame,
    spy_levels: pd.Series | None,
) -> tuple[PortfolioMonitorMetric, ...]:
    """Apply Monitor's existing daily volatility and raw 21-pair correlation conventions."""
    portfolio_list = list(portfolios)
    reconstruction = scoped_reconstruction(portfolio_list, closes) if not closes.empty else None
    spy_returns = (
        _usable_close_series(spy_levels).pct_change(fill_method=None).dropna()
        if spy_levels is not None
        else None
    )
    metrics = []
    for portfolio in portfolio_list:
        levels = performance_growth(portfolio, closes, reconstruction)
        returns = levels.pct_change(fill_method=None).dropna()
        volatility = sample_realized_volatility(returns)
        correlation, _, observations, _, _ = _spy_correlation(returns, spy_returns)
        metrics.append(
            PortfolioMonitorMetric(
                str(getattr(portfolio, "name")),
                float(volatility) if np.isfinite(volatility) else None,
                float(correlation) if np.isfinite(correlation) else None,
                observations,
            )
        )
    return tuple(metrics)


def market_context_metrics(closes: pd.DataFrame) -> pd.DataFrame:
    """Transparent proxy components; intentionally does not synthesize a market score."""
    rows: list[dict[str, object]] = []
    series_by_symbol = {
        str(symbol): _usable_close_series(closes[symbol]) for symbol in closes.columns
    }
    returns_by_symbol = {
        symbol: series.pct_change(fill_method=None).dropna()
        for symbol, series in series_by_symbol.items()
    }
    spy_returns = returns_by_symbol.get("SPY")
    for symbol in closes.columns:
        symbol = str(symbol)
        series = series_by_symbol[symbol]
        returns = returns_by_symbol[symbol]
        if series.empty:
            continue
        peak = series.cummax()
        period_metrics = {
            label: _period_return(series, observations)
            for label, observations in PERIOD_RETURN_OBSERVATIONS.items()
        }
        correlation, heuristic, correlation_count, correlation_first, correlation_last = (
            _spy_correlation(returns, spy_returns)
        )
        row: dict[str, object] = {
            "Symbol": symbol,
            "Name": MARKET_CONTEXT_NAMES.get(symbol, symbol[:13]),
            **{f"{label} return": values[0] for label, values in period_metrics.items()},
            **{f"{label} observations": values[1] for label, values in period_metrics.items()},
            **{f"{label} first date": values[2] for label, values in period_metrics.items()},
            **{f"{label} last date": values[3] for label, values in period_metrics.items()},
            "Trend": "above 50-day average"
            if len(series) >= 50 and series.iloc[-1] >= series.tail(50).mean()
            else ("below 50-day average" if len(series) >= 50 else "insufficient history"),
            "Drawdown from available-window peak": series.iloc[-1] / peak.iloc[-1] - 1,
            "Realized volatility": sample_realized_volatility(returns),
            "21-session SPY correlation": correlation,
            "Correlation observations": correlation_count,
            "Correlation first date": correlation_first,
            "Correlation last date": correlation_last,
            "Correlation heuristic": heuristic,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def position_risk_contributions(
    closes: pd.DataFrame, position_values: Mapping[str, float]
) -> dict[str, float]:
    symbols = [symbol for symbol in position_values if symbol in closes]
    if not symbols:
        return {}
    returns = closes[symbols].pct_change(fill_method=None).dropna(how="any").tail(252)
    total = sum(abs(position_values[symbol]) for symbol in symbols)
    if len(returns) < 20 or total <= 0:
        return {}
    weights = np.array([position_values[symbol] / total for symbol in symbols], dtype=float)
    covariance = returns.cov().to_numpy(dtype=float) * 252
    marginal = covariance @ weights
    variance = float(weights @ marginal)
    if not np.isfinite(variance) or variance <= 0:
        return {}
    contributions = weights * marginal / variance
    return {symbol: float(contributions[index]) for index, symbol in enumerate(symbols)}


def alpaca_clients(
    api_key: str, secret_key: str, feed: str
) -> tuple[StockHistoricalDataClient, Callable[[], StockDataStream]]:
    historical = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    data_feed = {"iex": DataFeed.IEX, "sip": DataFeed.SIP, "delayed_sip": DataFeed.DELAYED_SIP}.get(
        feed, DataFeed.IEX
    )
    return historical, lambda: StockDataStream(api_key, secret_key, feed=data_feed)
