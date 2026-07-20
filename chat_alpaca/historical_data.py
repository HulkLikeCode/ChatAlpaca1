from __future__ import annotations

import hashlib
import io
import json
import math
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from datetime import time as datetime_time
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Protocol

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from chat_alpaca.config import Settings, get_settings
from chat_alpaca.models import DailyBar, Instrument, MarketDataset, ProxyAssignment
from chat_alpaca.portfolio_service import normalize_symbol


class PriceAdjustment(str, Enum):
    RAW = "raw"
    SPLIT = "split"
    DIVIDEND = "dividend"
    TOTAL_RETURN = "all"


PORTFOLIO_ACCOUNTING_ADJUSTMENT = PriceAdjustment.SPLIT
BENCHMARK_TOTAL_RETURN_ADJUSTMENT = PriceAdjustment.TOTAL_RETURN
ALLOWED_ADJUSTMENTS = {item.value for item in PriceAdjustment}


class HistoricalDataError(RuntimeError):
    pass


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True)
class HistoricalRequest:
    symbols: tuple[str, ...]
    start: date
    end: date
    adjustment: PriceAdjustment = PORTFOLIO_ACCOUNTING_ADJUSTMENT
    timeframe: str = "1Day"

    def __post_init__(self) -> None:
        normalized = tuple(sorted({normalize_symbol(symbol) for symbol in self.symbols}))
        if not normalized:
            raise ValueError("At least one symbol is required.")
        if self.start > self.end:
            raise ValueError("Historical-data start date must be on or before the end date.")
        if self.timeframe != "1Day":
            raise ValueError("Phase 3 supports daily bars only.")
        object.__setattr__(self, "symbols", normalized)
        object.__setattr__(self, "adjustment", PriceAdjustment(self.adjustment))


@dataclass(frozen=True)
class BarValue:
    symbol: str
    bar_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None = None
    trade_count: int | None = None
    vwap: Decimal | None = None


@dataclass(frozen=True)
class ProviderDataset:
    provider: str
    source: str
    feed: str | None
    timeframe: str
    adjustment: PriceAdjustment
    retrieved_at: datetime
    bars: tuple[BarValue, ...]
    request_metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    override_priority: int = 100
    imported_file_hash: str | None = None


@dataclass(frozen=True)
class PersistedDataset:
    dataset_id: int
    duplicate: bool = False


@dataclass(frozen=True)
class HistoricalCoverageResult:
    data: pd.DataFrame
    source: str | tuple[str, ...]
    feed: str | tuple[str, ...] | None
    adjustment: str
    coverage_start: date | None
    coverage_end: date | None
    missing_symbols: tuple[str, ...]
    missing_date_ranges: Mapping[str, tuple[tuple[date, date], ...]]
    warnings: tuple[str, ...]
    freshness: Mapping[str, datetime | None]
    usable: bool
    usability: str


class HistoricalDataProvider(Protocol):
    name: str

    def fetch(self, request: HistoricalRequest) -> ProviderDataset: ...


class HistoricalDataRepository(Protocol):
    def persist(self, dataset: ProviderDataset) -> PersistedDataset: ...

    def coverage(self, request: HistoricalRequest) -> HistoricalCoverageResult: ...


def _json_default(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


_SENSITIVE_FRAGMENTS = ("api_key", "apikey", "secret", "password", "authorization", "token")


def sanitize_request_metadata(
    metadata: Mapping[str, Any], *, sensitive_values: Iterable[str] = ()
) -> dict[str, Any]:
    """Return JSON-safe request metadata with credential-shaped data removed."""
    secrets = {value for value in sensitive_values if value}

    def clean(value: Any, key: str = "") -> Any:
        lowered = key.lower()
        if any(fragment in lowered for fragment in _SENSITIVE_FRAGMENTS):
            return "[REDACTED]"
        if isinstance(value, Mapping):
            return {
                str(child_key): clean(child, str(child_key)) for child_key, child in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [clean(child, key) for child in value]
        rendered = str(value)
        if rendered in secrets:
            return "[REDACTED]"
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return rendered

    return clean(metadata)


def _decimal(value: Any, field_name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise DatasetValidationError(f"{field_name} must be numeric.") from exc
    if not parsed.is_finite():
        raise DatasetValidationError(f"{field_name} must be finite.")
    return parsed


class DailyBarValidator:
    """Validate and normalize provider-neutral daily bars before persistence."""

    def validate(self, bars: Sequence[BarValue]) -> tuple[BarValue, ...]:
        normalized: list[BarValue] = []
        seen: set[tuple[str, date]] = set()
        for source_bar in bars:
            symbol = normalize_symbol(source_bar.symbol)
            if not symbol:
                raise DatasetValidationError("Every daily bar requires a symbol.")
            values = {
                name: _decimal(getattr(source_bar, name), name)
                for name in ("open", "high", "low", "close")
            }
            if any(value <= 0 for value in values.values()):
                raise DatasetValidationError(
                    f"{symbol} {source_bar.bar_date}: prices must be positive."
                )
            if values["high"] < max(values.values()) or values["low"] > min(values.values()):
                raise DatasetValidationError(
                    f"{symbol} {source_bar.bar_date}: OHLC values fall outside high/low bounds."
                )
            key = (symbol, source_bar.bar_date)
            if key in seen:
                raise DatasetValidationError(
                    f"Duplicate daily bar for {symbol} on {source_bar.bar_date}."
                )
            seen.add(key)
            volume = (
                _decimal(source_bar.volume, "volume") if source_bar.volume is not None else None
            )
            if volume is not None and volume < 0:
                raise DatasetValidationError("Volume cannot be negative.")
            vwap = _decimal(source_bar.vwap, "vwap") if source_bar.vwap is not None else None
            if vwap is not None and vwap <= 0:
                raise DatasetValidationError("VWAP must be positive.")
            if source_bar.trade_count is not None and source_bar.trade_count < 0:
                raise DatasetValidationError("Trade count cannot be negative.")
            normalized.append(
                BarValue(
                    symbol=symbol,
                    bar_date=source_bar.bar_date,
                    open=values["open"],
                    high=values["high"],
                    low=values["low"],
                    close=values["close"],
                    volume=volume,
                    trade_count=source_bar.trade_count,
                    vwap=vwap,
                )
            )
        return tuple(sorted(normalized, key=lambda bar: (bar.symbol, bar.bar_date)))


def _ranges(dates: Iterable[date]) -> tuple[tuple[date, date], ...]:
    values = sorted(set(dates))
    if not values:
        return ()
    ranges: list[tuple[date, date]] = []
    start = previous = values[0]
    for current in values[1:]:
        expected = (pd.Timestamp(previous) + pd.offsets.BDay()).date()
        if current != expected:
            ranges.append((start, previous))
            start = current
        previous = current
    ranges.append((start, previous))
    return tuple(ranges)


def _collapsed(values: Iterable[str | None]) -> str | tuple[str, ...] | None:
    unique = tuple(sorted({value for value in values if value}))
    if not unique:
        return None
    return unique[0] if len(unique) == 1 else unique


class SqlHistoricalDataRepository:
    def __init__(self, session: Session, validator: DailyBarValidator | None = None) -> None:
        self.session = session
        self.validator = validator or DailyBarValidator()

    def instrument(self, symbol: str, *, asset_type: str = "equity") -> Instrument:
        canonical = normalize_symbol(symbol)
        existing = self.session.scalar(
            select(Instrument).where(Instrument.canonical_symbol == canonical)
        )
        if existing is not None:
            return existing
        item = Instrument(
            canonical_symbol=canonical,
            asset_type=asset_type,
            is_active=True,
            base_currency="USD",
        )
        self.session.add(item)
        self.session.flush()
        return item

    def persist(self, dataset: ProviderDataset) -> PersistedDataset:
        adjustment = PriceAdjustment(dataset.adjustment)
        if dataset.imported_file_hash:
            duplicate = self.session.scalar(
                select(MarketDataset).where(
                    MarketDataset.imported_file_hash == dataset.imported_file_hash
                )
            )
            if duplicate is not None:
                return PersistedDataset(duplicate.id, duplicate=True)

        bars = self.validator.validate(dataset.bars)
        warnings = tuple(dataset.warnings)
        coverage_dates = [bar.bar_date for bar in bars]
        metadata = sanitize_request_metadata(dataset.request_metadata)
        record = MarketDataset(
            provider=dataset.provider,
            source=dataset.source,
            feed=dataset.feed,
            timeframe=dataset.timeframe,
            adjustment_method=adjustment.value,
            retrieved_at=dataset.retrieved_at,
            coverage_start=min(coverage_dates, default=None),
            coverage_end=max(coverage_dates, default=None),
            quality_status="warning" if warnings else "valid",
            validation_warnings=json.dumps(warnings),
            override_priority=dataset.override_priority,
            imported_file_hash=dataset.imported_file_hash,
            request_metadata=json.dumps(metadata, default=_json_default, sort_keys=True),
        )
        self.session.add(record)
        self.session.flush()
        instruments = {symbol: self.instrument(symbol) for symbol in {bar.symbol for bar in bars}}
        self.session.add_all(
            DailyBar(
                instrument_id=instruments[bar.symbol].id,
                bar_date=bar.bar_date,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                trade_count=bar.trade_count,
                vwap=bar.vwap,
                dataset_id=record.id,
            )
            for bar in bars
        )
        self.session.flush()
        return PersistedDataset(record.id)

    def _candidate_query(
        self, request: HistoricalRequest
    ) -> Select[tuple[DailyBar, Instrument, MarketDataset]]:
        return (
            select(DailyBar, Instrument, MarketDataset)
            .join(Instrument, DailyBar.instrument_id == Instrument.id)
            .join(MarketDataset, DailyBar.dataset_id == MarketDataset.id)
            .where(
                Instrument.canonical_symbol.in_(request.symbols),
                DailyBar.bar_date.between(request.start, request.end),
                MarketDataset.timeframe == request.timeframe,
                MarketDataset.adjustment_method == request.adjustment.value,
                MarketDataset.quality_status.in_(("valid", "warning")),
            )
            .order_by(
                Instrument.canonical_symbol,
                DailyBar.bar_date,
                MarketDataset.override_priority.desc(),
                MarketDataset.retrieved_at.desc(),
                MarketDataset.id.desc(),
            )
        )

    def coverage(self, request: HistoricalRequest) -> HistoricalCoverageResult:
        rows = self.session.execute(self._candidate_query(request)).all()
        selected: dict[tuple[str, date], tuple[DailyBar, MarketDataset]] = {}
        warnings: list[str] = []
        for bar, instrument, dataset in rows:
            key = (instrument.canonical_symbol, bar.bar_date)
            existing = selected.get(key)
            if existing is None:
                selected[key] = (bar, dataset)
                continue
            chosen_bar, chosen_dataset = existing
            if bar.close != chosen_bar.close:
                warnings.append(
                    f"Conflicting {request.adjustment.value} close for {key[0]} on {key[1]}; "
                    f"selected {chosen_dataset.source} priority "
                    f"{chosen_dataset.override_priority} over {dataset.source} priority "
                    f"{dataset.override_priority}."
                )

        by_symbol: dict[str, dict[date, tuple[DailyBar, MarketDataset]]] = {
            symbol: {} for symbol in request.symbols
        }
        for (symbol, bar_date), value in selected.items():
            by_symbol[symbol][bar_date] = value
        expected = {timestamp.date() for timestamp in pd.bdate_range(request.start, request.end)}
        missing_symbols = tuple(
            sorted(symbol for symbol, values in by_symbol.items() if not values)
        )
        missing_ranges = {
            symbol: _ranges(expected - set(values))
            for symbol, values in by_symbol.items()
            if expected - set(values)
        }
        for symbol, ranges in missing_ranges.items():
            rendered = ", ".join(
                str(start) if start == end else f"{start} to {end}" for start, end in ranges
            )
            warnings.append(f"Missing daily observations for {symbol}: {rendered}.")

        index = sorted({bar_date for values in by_symbol.values() for bar_date in values})
        frame = pd.DataFrame(index=pd.DatetimeIndex(index), columns=request.symbols, dtype=float)
        chosen_datasets: dict[int, MarketDataset] = {}
        freshness: dict[str, datetime | None] = {}
        for symbol, values in by_symbol.items():
            for bar_date, (bar, dataset) in values.items():
                frame.loc[pd.Timestamp(bar_date), symbol] = float(bar.close)
                chosen_datasets[dataset.id] = dataset
            retrievals = [dataset.retrieved_at for _, dataset in values.values()]
            freshness[symbol] = max(retrievals, default=None)
        frame = frame.sort_index()
        frame.attrs["last_price_dates"] = {
            symbol: max(values) for symbol, values in by_symbol.items() if values
        }
        frame.attrs["adjustment"] = request.adjustment.value
        coverage_dates = list(index)
        sources = [dataset.source for dataset in chosen_datasets.values()]
        feeds = [dataset.feed for dataset in chosen_datasets.values()]
        for dataset in chosen_datasets.values():
            try:
                warnings.extend(json.loads(dataset.validation_warnings))
            except (json.JSONDecodeError, TypeError):
                warnings.append(f"Dataset {dataset.id} has unreadable stored validation warnings.")
        complete = not missing_symbols and not missing_ranges
        usability = (
            "complete for the requested calculation"
            if complete
            else "not complete for calculations requiring every requested daily observation"
        )
        return HistoricalCoverageResult(
            data=frame,
            source=_collapsed(sources) or "none",
            feed=_collapsed(feeds),
            adjustment=request.adjustment.value,
            coverage_start=min(coverage_dates, default=None),
            coverage_end=max(coverage_dates, default=None),
            missing_symbols=missing_symbols,
            missing_date_ranges=missing_ranges,
            warnings=tuple(dict.fromkeys(warnings)),
            freshness=freshness,
            usable=complete,
            usability=usability,
        )

    def record_proxy(
        self,
        *,
        target_symbol: str,
        effective_from: date,
        reason: str,
        data_sufficiency_rationale: str,
        assignment_source: str,
        proxy_symbol: str | None = None,
        proxy_series: str | None = None,
        effective_to: date | None = None,
        confidence: Decimal | None = None,
    ) -> ProxyAssignment:
        if (proxy_symbol is None) == (proxy_series is None):
            raise ValueError("Specify exactly one proxy symbol or proxy series.")
        if assignment_source not in {"manual", "automatic"}:
            raise ValueError("Proxy assignment source must be manual or automatic.")
        if confidence is not None and not Decimal("0") <= confidence <= Decimal("1"):
            raise ValueError("Proxy confidence must be between zero and one.")
        target = self.instrument(target_symbol)
        proxy = self.instrument(proxy_symbol) if proxy_symbol else None
        assignment = ProxyAssignment(
            target_instrument_id=target.id,
            proxy_instrument_id=proxy.id if proxy else None,
            proxy_series=proxy_series,
            effective_from=effective_from,
            effective_to=effective_to,
            reason=reason,
            confidence=confidence,
            data_sufficiency_rationale=data_sufficiency_rationale,
            assignment_source=assignment_source,
        )
        self.session.add(assignment)
        self.session.flush()
        return assignment


class _RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60 / requests_per_minute if requests_per_minute > 0 else 0
        self._last_request = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self.interval - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()


_DEFAULT_ALPACA_RATE_LIMITER = _RateLimiter(200)


class AlpacaHistoricalProvider:
    name = "alpaca"

    def __init__(
        self,
        client: StockHistoricalDataClient | None = None,
        settings: Settings | None = None,
        *,
        requests_per_minute: int = 200,
    ) -> None:
        self.settings = settings or get_settings()
        if client is None:
            if not self.settings.alpaca_configured:
                raise HistoricalDataError("Alpaca market-data credentials are not configured.")
            client = StockHistoricalDataClient(
                api_key=self.settings.alpaca_api_key,
                secret_key=self.settings.alpaca_secret_key,
            )
        self.client = client
        self.rate_limiter = (
            _DEFAULT_ALPACA_RATE_LIMITER
            if requests_per_minute == 200
            else _RateLimiter(requests_per_minute)
        )

    def fetch(self, request: HistoricalRequest) -> ProviderDataset:
        feed_name = self.settings.alpaca_data_feed
        feed = {
            "iex": DataFeed.IEX,
            "sip": DataFeed.SIP,
            "delayed_sip": DataFeed.DELAYED_SIP,
        }.get(feed_name, DataFeed.IEX)
        adjustment = {
            PriceAdjustment.RAW: Adjustment.RAW,
            PriceAdjustment.SPLIT: Adjustment.SPLIT,
            PriceAdjustment.DIVIDEND: Adjustment.DIVIDEND,
            PriceAdjustment.TOTAL_RETURN: Adjustment.ALL,
        }[request.adjustment]
        provider_request = StockBarsRequest(
            symbol_or_symbols=list(request.symbols),
            timeframe=TimeFrame.Day,
            start=datetime.combine(request.start, datetime_time.min, tzinfo=timezone.utc),
            end=datetime.combine(request.end, datetime_time.max, tzinfo=timezone.utc),
            adjustment=adjustment,
            feed=feed,
        )
        self.rate_limiter.wait()
        frame = self.client.get_stock_bars(provider_request).df
        bars = self._bars(frame, request.symbols)
        returned = {bar.symbol for bar in bars}
        warnings = tuple(
            f"Alpaca returned no {request.adjustment.value} daily bars for {symbol}."
            for symbol in request.symbols
            if symbol not in returned
        )
        metadata = sanitize_request_metadata(
            {
                "symbols": request.symbols,
                "start": request.start,
                "end": request.end,
                "timeframe": request.timeframe,
                "adjustment": request.adjustment.value,
                "feed": feed_name,
            },
            sensitive_values=(
                getattr(self.settings, "alpaca_api_key", ""),
                getattr(self.settings, "alpaca_secret_key", ""),
            ),
        )
        return ProviderDataset(
            provider=self.name,
            source="alpaca_historical",
            feed=feed_name,
            timeframe=request.timeframe,
            adjustment=request.adjustment,
            retrieved_at=datetime.now(timezone.utc),
            bars=bars,
            request_metadata=metadata,
            warnings=warnings,
            override_priority=100,
        )

    @staticmethod
    def _bars(frame: pd.DataFrame, requested_symbols: Sequence[str]) -> tuple[BarValue, ...]:
        if frame.empty:
            return ()
        rows = frame.reset_index()
        if "symbol" not in rows:
            rows["symbol"] = requested_symbols[0]
        timestamp_column = "timestamp" if "timestamp" in rows else rows.columns[0]
        result: list[BarValue] = []
        for row in rows.to_dict(orient="records"):
            close = row["close"]
            result.append(
                BarValue(
                    symbol=str(row["symbol"]),
                    bar_date=pd.Timestamp(row[timestamp_column]).date(),
                    open=_decimal(row.get("open", close), "open"),
                    high=_decimal(row.get("high", close), "high"),
                    low=_decimal(row.get("low", close), "low"),
                    close=_decimal(close, "close"),
                    volume=(
                        _decimal(row.get("volume"), "volume")
                        if row.get("volume") is not None and not pd.isna(row.get("volume"))
                        else None
                    ),
                    trade_count=(
                        int(row["trade_count"])
                        if row.get("trade_count") is not None
                        and not pd.isna(row.get("trade_count"))
                        else None
                    ),
                    vwap=(
                        _decimal(row.get("vwap"), "vwap")
                        if row.get("vwap") is not None and not pd.isna(row.get("vwap"))
                        else None
                    ),
                )
            )
        return tuple(result)


class HistoricalDataService:
    """Resolve durable data first and fetch only absent ranges from a provider."""

    def __init__(
        self,
        repository: SqlHistoricalDataRepository,
        provider: HistoricalDataProvider,
        *,
        stale_after: timedelta = timedelta(hours=12),
    ) -> None:
        self.repository = repository
        self.provider = provider
        self.stale_after = stale_after

    def get(self, request: HistoricalRequest, *, refresh: bool = True) -> HistoricalCoverageResult:
        cached = self.repository.coverage(request)
        if not refresh:
            return cached
        now = datetime.now(timezone.utc)
        stale_symbols = {
            symbol
            for symbol, fetched_at in cached.freshness.items()
            if fetched_at is None or now - _aware(fetched_at) > self.stale_after
        }
        symbols_by_range: dict[tuple[date, date], set[str]] = {}
        for symbol, missing_ranges in cached.missing_date_ranges.items():
            for missing_range in missing_ranges:
                symbols_by_range.setdefault(missing_range, set()).add(symbol)
        for symbol in stale_symbols - set(cached.missing_date_ranges):
            if cached.coverage_end is not None:
                symbols_by_range.setdefault((cached.coverage_end, request.end), set()).add(symbol)
        for (range_start, range_end), symbols in sorted(symbols_by_range.items()):
            response = self.provider.fetch(
                HistoricalRequest(
                    tuple(symbols),
                    max(range_start, request.start),
                    min(range_end, request.end),
                    request.adjustment,
                    request.timeframe,
                )
            )
            self.repository.persist(response)
        return self.repository.coverage(request)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class CsvHistoricalImporter:
    """Validate Stooq-like daily CSV without inferring dividend adjustment."""

    def __init__(self, repository: SqlHistoricalDataRepository) -> None:
        self.repository = repository

    def import_bytes(
        self,
        content: bytes,
        *,
        adjustment: PriceAdjustment | str | None,
        source: str = "stooq_csv",
        feed: str | None = None,
        override_priority: int = 50,
        imported_at: datetime | None = None,
    ) -> PersistedDataset:
        if adjustment is None:
            raise DatasetValidationError(
                "CSV adjustment metadata is required; Stooq dividend adjustment is not assumed."
            )
        try:
            parsed_adjustment = PriceAdjustment(adjustment)
        except ValueError as exc:
            raise DatasetValidationError(
                f"Unsupported adjustment metadata; expected one of {sorted(ALLOWED_ADJUSTMENTS)}."
            ) from exc
        file_hash = hashlib.sha256(content).hexdigest()
        existing = self.repository.session.scalar(
            select(MarketDataset).where(MarketDataset.imported_file_hash == file_hash)
        )
        if existing is not None:
            return PersistedDataset(existing.id, duplicate=True)
        try:
            frame = pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise DatasetValidationError("CSV could not be parsed.") from exc
        columns = {str(column).strip().lower().strip("<>"): column for column in frame.columns}
        required = {"symbol", "date", "open", "high", "low", "close"}
        missing = sorted(required - set(columns))
        if missing:
            raise DatasetValidationError(f"CSV is missing required columns: {', '.join(missing)}.")
        if frame.empty:
            raise DatasetValidationError("CSV must contain at least one daily bar.")
        bars: list[BarValue] = []
        for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
            try:
                parsed_date = pd.to_datetime(row[columns["date"]], errors="raise").date()
                symbol = normalize_symbol(str(row[columns["symbol"]]))
                if not symbol or symbol in {"NAN", "NONE"}:
                    raise DatasetValidationError("symbol is required")
                trade_value = row.get(columns.get("trade count"))
                bars.append(
                    BarValue(
                        symbol=symbol,
                        bar_date=parsed_date,
                        open=_decimal(row[columns["open"]], "open"),
                        high=_decimal(row[columns["high"]], "high"),
                        low=_decimal(row[columns["low"]], "low"),
                        close=_decimal(row[columns["close"]], "close"),
                        volume=_optional_decimal(row.get(columns.get("volume")), "volume"),
                        trade_count=(
                            int(trade_value)
                            if trade_value is not None and not pd.isna(trade_value)
                            else None
                        ),
                        vwap=_optional_decimal(row.get(columns.get("vwap")), "vwap"),
                    )
                )
            except (DatasetValidationError, ValueError, TypeError) as exc:
                raise DatasetValidationError(f"Invalid CSV row {row_number}: {exc}") from exc
        validated = self.repository.validator.validate(bars)
        return self.repository.persist(
            ProviderDataset(
                provider="csv",
                source=source,
                feed=feed,
                timeframe="1Day",
                adjustment=parsed_adjustment,
                retrieved_at=imported_at or datetime.now(timezone.utc),
                bars=validated,
                request_metadata={
                    "format": "stooq-like-daily-csv",
                    "row_count": len(validated),
                    "adjustment_supplied_by_importer": parsed_adjustment.value,
                },
                override_priority=override_priority,
                imported_file_hash=file_hash,
            )
        )


def _optional_decimal(value: Any, field_name: str) -> Decimal | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    return _decimal(value, field_name)
