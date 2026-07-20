from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import func, select

from chat_alpaca.historical_data import (
    AlpacaHistoricalProvider,
    BarValue,
    CsvHistoricalImporter,
    DatasetValidationError,
    HistoricalDataService,
    HistoricalRequest,
    PriceAdjustment,
    ProviderDataset,
    SqlHistoricalDataRepository,
)
from chat_alpaca.models import DailyBar, MarketDataset, ProxyAssignment


def _bar(symbol: str, day: date, close: str = "10") -> BarValue:
    value = Decimal(close)
    return BarValue(
        symbol=symbol,
        bar_date=day,
        open=value,
        high=value + 1,
        low=value - 1,
        close=value,
        volume=Decimal("100"),
        trade_count=12,
        vwap=value,
    )


def _dataset(
    bars: tuple[BarValue, ...],
    *,
    adjustment: PriceAdjustment = PriceAdjustment.SPLIT,
    priority: int = 100,
    source: str = "alpaca_historical",
    metadata: dict[str, object] | None = None,
) -> ProviderDataset:
    return ProviderDataset(
        provider="alpaca",
        source=source,
        feed="iex",
        timeframe="1Day",
        adjustment=adjustment,
        retrieved_at=datetime.now(timezone.utc),
        bars=bars,
        request_metadata=metadata or {},
        override_priority=priority,
    )


class _Provider:
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[HistoricalRequest] = []

    def fetch(self, request: HistoricalRequest) -> ProviderDataset:
        self.requests.append(request)
        bars = tuple(
            _bar(symbol, timestamp.date())
            for symbol in request.symbols
            for timestamp in pd.bdate_range(request.start, request.end)
        )
        return _dataset(bars)


def test_alpaca_dataset_persistence_and_provenance(session) -> None:
    index = pd.MultiIndex.from_tuples(
        [("ABC", pd.Timestamp("2026-01-05", tz="UTC"))],
        names=["symbol", "timestamp"],
    )
    frame = pd.DataFrame(
        {
            "open": [9.0],
            "high": [11.0],
            "low": [8.0],
            "close": [10.0],
            "volume": [1000],
            "trade_count": [50],
            "vwap": [9.8],
        },
        index=index,
    )
    client = SimpleNamespace(get_stock_bars=lambda _request: SimpleNamespace(df=frame))
    settings = SimpleNamespace(
        alpaca_data_feed="iex",
        alpaca_configured=True,
        alpaca_api_key="key-do-not-store",
        alpaca_secret_key="secret-do-not-store",
    )
    provider = AlpacaHistoricalProvider(client, settings, requests_per_minute=0)
    repository = SqlHistoricalDataRepository(session)

    persisted = repository.persist(
        provider.fetch(HistoricalRequest(("ABC",), date(2026, 1, 5), date(2026, 1, 5)))
    )

    dataset = session.get(MarketDataset, persisted.dataset_id)
    bar = session.scalar(select(DailyBar))
    assert dataset is not None
    assert dataset.provider == "alpaca"
    assert dataset.feed == "iex"
    assert dataset.adjustment_method == "split"
    assert dataset.coverage_start == dataset.coverage_end == date(2026, 1, 5)
    assert bar is not None
    assert bar.trade_count == 50
    assert bar.vwap == Decimal("9.80000000")
    persisted_text = dataset.request_metadata
    assert "key-do-not-store" not in persisted_text
    assert "secret-do-not-store" not in persisted_text


def test_incremental_refresh_fetches_only_missing_range(session) -> None:
    repository = SqlHistoricalDataRepository(session)
    provider = _Provider()
    service = HistoricalDataService(repository, provider)
    request = HistoricalRequest(("ABC",), date(2026, 1, 5), date(2026, 1, 7))

    first = service.get(request)
    second = service.get(request)

    assert first.usable
    assert second.usable
    assert [(item.start, item.end) for item in provider.requests] == [
        (date(2026, 1, 5), date(2026, 1, 7))
    ]
    assert session.scalar(select(func.count()).select_from(MarketDataset)) == 1


def test_adjustment_datasets_remain_separate(session) -> None:
    repository = SqlHistoricalDataRepository(session)
    day = date(2026, 1, 5)
    repository.persist(_dataset((_bar("ABC", day, "10"),)))
    repository.persist(
        _dataset(
            (_bar("ABC", day, "12"),),
            adjustment=PriceAdjustment.TOTAL_RETURN,
        )
    )

    split = repository.coverage(HistoricalRequest(("ABC",), day, day))
    total_return = repository.coverage(
        HistoricalRequest(("ABC",), day, day, PriceAdjustment.TOTAL_RETURN)
    )

    assert split.data.loc[pd.Timestamp(day), "ABC"] == 10
    assert total_return.data.loc[pd.Timestamp(day), "ABC"] == 12
    assert split.adjustment == "split"
    assert total_return.adjustment == "all"


def test_duplicate_csv_import_returns_original_dataset(session) -> None:
    content = b"Symbol,Date,Open,High,Low,Close,Volume\nABC,2026-01-05,9,11,8,10,100\n"
    importer = CsvHistoricalImporter(SqlHistoricalDataRepository(session))

    first = importer.import_bytes(content, adjustment=PriceAdjustment.RAW)
    second = importer.import_bytes(content, adjustment=PriceAdjustment.RAW)

    assert not first.duplicate
    assert second == type(second)(first.dataset_id, duplicate=True)
    assert session.scalar(select(func.count()).select_from(MarketDataset)) == 1


def test_precedence_is_deterministic_and_conflicts_warn(session) -> None:
    repository = SqlHistoricalDataRepository(session)
    day = date(2026, 1, 5)
    repository.persist(_dataset((_bar("ABC", day, "9"),), priority=50, source="csv_import"))
    repository.persist(
        _dataset((_bar("ABC", day, "10"),), priority=100, source="alpaca_historical")
    )

    result = repository.coverage(HistoricalRequest(("ABC",), day, day))

    assert result.data.loc[pd.Timestamp(day), "ABC"] == 10
    assert result.source == "alpaca_historical"
    assert any("selected alpaca_historical priority 100" in item for item in result.warnings)
    assert session.scalar(select(func.count()).select_from(DailyBar)) == 2


def test_partial_symbol_coverage_and_missing_dates_are_explicit(session) -> None:
    repository = SqlHistoricalDataRepository(session)
    repository.persist(
        _dataset(
            (
                _bar("ABC", date(2026, 1, 5)),
                _bar("ABC", date(2026, 1, 7)),
            )
        )
    )

    result = repository.coverage(
        HistoricalRequest(("ABC", "MISSING"), date(2026, 1, 5), date(2026, 1, 7))
    )

    assert result.missing_symbols == ("MISSING",)
    assert result.missing_date_ranges["ABC"] == ((date(2026, 1, 6), date(2026, 1, 6)),)
    assert result.missing_date_ranges["MISSING"] == ((date(2026, 1, 5), date(2026, 1, 7)),)
    assert not result.usable
    assert pd.isna(result.data.loc[pd.Timestamp("2026-01-05"), "MISSING"])


def test_incremental_refresh_does_not_refetch_covered_peer_symbols(session) -> None:
    repository = SqlHistoricalDataRepository(session)
    repository.persist(
        _dataset(
            tuple(
                _bar("ABC", timestamp.date())
                for timestamp in pd.bdate_range(date(2026, 1, 5), date(2026, 1, 7))
            )
        )
    )
    provider = _Provider()
    service = HistoricalDataService(repository, provider)

    result = service.get(HistoricalRequest(("ABC", "XYZ"), date(2026, 1, 5), date(2026, 1, 7)))

    assert result.usable
    assert [item.symbols for item in provider.requests] == [("XYZ",)]


@pytest.mark.parametrize(
    "content, message",
    [
        (b"Date,Open,High,Low,Close\n2026-01-05,9,11,8,10\n", "symbol"),
        (b"Symbol,Date,Open,High,Low,Close\nABC,2026-01-05,0,11,8,10\n", "positive"),
        (b"Symbol,Date,Open,High,Low,Close\nABC,2026-01-05,9,8,7,10\n", "bounds"),
        (
            b"Symbol,Date,Open,High,Low,Close\nABC,2026-01-05,9,11,8,10\n"
            b"ABC,2026-01-05,9,11,8,10\n",
            "Duplicate",
        ),
    ],
)
def test_invalid_csv_is_rejected(session, content: bytes, message: str) -> None:
    importer = CsvHistoricalImporter(SqlHistoricalDataRepository(session))

    with pytest.raises(DatasetValidationError, match=message):
        importer.import_bytes(content, adjustment=PriceAdjustment.RAW)


def test_csv_requires_explicit_adjustment_metadata(session) -> None:
    importer = CsvHistoricalImporter(SqlHistoricalDataRepository(session))

    with pytest.raises(DatasetValidationError, match="Stooq dividend adjustment is not assumed"):
        importer.import_bytes(b"ignored", adjustment=None)


def test_proxy_assignment_records_rationale(session) -> None:
    repository = SqlHistoricalDataRepository(session)

    created = repository.record_proxy(
        target_symbol="NEW",
        proxy_symbol="SPY",
        effective_from=date(2026, 1, 1),
        reason="Limited issuer history",
        confidence=Decimal("0.75"),
        data_sufficiency_rationale="Only 40 daily observations are available.",
        assignment_source="manual",
    )

    stored = session.scalar(select(ProxyAssignment))
    assert stored is not None
    assert stored.id == created.id
    assert stored.proxy_instrument.canonical_symbol == "SPY"
    assert stored.data_sufficiency_rationale.startswith("Only 40")
    assert stored.assignment_source == "manual"


def test_sensitive_request_metadata_is_redacted_and_never_logged(session, caplog) -> None:
    repository = SqlHistoricalDataRepository(session)
    secret = "especially-sensitive-secret"

    persisted = repository.persist(
        _dataset(
            (_bar("ABC", date(2026, 1, 5)),),
            metadata={"api_key": "public-looking-key", "nested": {"token": secret}},
        )
    )

    dataset = session.get(MarketDataset, persisted.dataset_id)
    assert dataset is not None
    metadata = json.loads(dataset.request_metadata)
    assert metadata["api_key"] == "[REDACTED]"
    assert metadata["nested"]["token"] == "[REDACTED]"
    assert secret not in dataset.request_metadata
    assert secret not in caplog.text


def test_stale_complete_coverage_refreshes_only_trailing_range(session) -> None:
    repository = SqlHistoricalDataRepository(session)
    old = datetime.now(timezone.utc) - timedelta(days=2)
    repository.persist(
        ProviderDataset(
            **{
                **_dataset((_bar("ABC", date(2026, 1, 5)),)).__dict__,
                "retrieved_at": old,
            }
        )
    )
    provider = _Provider()
    service = HistoricalDataService(repository, provider, stale_after=timedelta(hours=1))

    service.get(HistoricalRequest(("ABC",), date(2026, 1, 5), date(2026, 1, 5)))

    assert [(item.start, item.end) for item in provider.requests] == [
        (date(2026, 1, 5), date(2026, 1, 5))
    ]
