from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation

from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_alpaca.models import EtfSectorWeight, Instrument, Portfolio, SecurityMetadata

UNCLASSIFIED = "Unclassified"
DEFAULT_STALE_AFTER_DAYS = 180
WEIGHT_TOLERANCE = Decimal("0.0001")  # percentage points


@dataclass(frozen=True)
class MetadataResolution:
    symbol: str
    security_name: str | None
    asset_type: str | None
    sector: str | None
    industry: str | None
    source: str | None
    effective_date: date | None
    retrieved_at: datetime | None
    confidence: Decimal | None
    quality_status: str
    manual_override: bool
    unavailable: bool
    stale: bool
    disclosures: tuple[str, ...]


@dataclass(frozen=True)
class EtfSectorSnapshot:
    symbol: str
    weights: Mapping[str, Decimal]
    source: str | None
    effective_date: date | None
    retrieved_at: datetime | None
    quality_status: str
    stale: bool
    disclosures: tuple[str, ...]


@dataclass(frozen=True)
class SectorExposure:
    sector: str
    market_value: Decimal
    percentage: Decimal


@dataclass(frozen=True)
class SectorExposureResult:
    exposures: tuple[SectorExposure, ...]
    invested_market_value: Decimal
    missing_price_symbols: tuple[str, ...]
    stale_metadata_symbols: tuple[str, ...]
    disclosures: tuple[str, ...]


def _instrument(session: Session, symbol: str, *, asset_type: str = "unknown") -> Instrument:
    canonical = symbol.strip().upper()
    if not canonical or len(canonical) > 32:
        raise ValueError("A valid security symbol is required.")
    instrument = session.scalar(select(Instrument).where(Instrument.canonical_symbol == canonical))
    if instrument is None:
        instrument = Instrument(canonical_symbol=canonical, asset_type=asset_type)
        session.add(instrument)
        session.flush()
    return instrument


def security_symbol_labels(session: Session) -> dict[str, str]:
    """Return active cached symbols with their newest available security names."""
    rows = session.execute(
        select(Instrument.canonical_symbol, SecurityMetadata.security_name)
        .outerjoin(SecurityMetadata, SecurityMetadata.instrument_id == Instrument.id)
        .where(Instrument.is_active.is_(True))
        .order_by(
            Instrument.canonical_symbol,
            SecurityMetadata.manual_override.desc(),
            SecurityMetadata.retrieved_at.desc(),
        )
    )
    labels: dict[str, str] = {}
    for symbol, security_name in rows:
        if symbol not in labels:
            labels[symbol] = security_name or symbol
    return labels


def _confidence(value: object | None) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("Metadata confidence must be numeric.") from exc
    if not parsed.is_finite() or parsed < 0 or parsed > 1:
        raise ValueError("Metadata confidence must be between 0 and 1.")
    return parsed


def cache_security_metadata(
    session: Session,
    symbol: str,
    *,
    security_name: str | None = None,
    asset_type: str | None = None,
    sector: str | None = None,
    industry: str | None = None,
    source: str,
    effective_date: date | None = None,
    retrieved_at: datetime | None = None,
    confidence: object | None = None,
    quality_status: str = "available",
    manual_override: bool = False,
) -> SecurityMetadata:
    clean_source = source.strip()
    clean_status = quality_status.strip()
    if not clean_source or not clean_status:
        raise ValueError("Metadata source and quality status are required.")
    normalized_asset_type = asset_type.strip().lower() if asset_type else None
    instrument = _instrument(session, symbol, asset_type=normalized_asset_type or "unknown")
    if normalized_asset_type:
        instrument.asset_type = normalized_asset_type
    row = SecurityMetadata(
        instrument_id=instrument.id,
        security_name=security_name.strip() if security_name else None,
        asset_type=normalized_asset_type,
        sector=sector.strip() if sector else None,
        industry=industry.strip() if industry else None,
        source=clean_source,
        effective_date=effective_date,
        retrieved_at=retrieved_at or datetime.now(timezone.utc),
        confidence=_confidence(confidence),
        quality_status=clean_status,
        manual_override=manual_override,
    )
    session.add(row)
    session.flush()
    return row


def save_manual_metadata_override(
    session: Session,
    symbol: str,
    **fields: object,
) -> SecurityMetadata:
    """Persist an owner override without requiring any external metadata service."""
    return cache_security_metadata(
        session,
        symbol,
        source="manual",
        manual_override=True,
        **fields,
    )


def cache_alpaca_asset_metadata(
    session: Session,
    asset: object,
    *,
    retrieved_at: datetime | None = None,
) -> SecurityMetadata:
    """Cache fields exposed by an Alpaca asset object or mapping; sector may remain unknown."""

    def value(name: str) -> object | None:
        return asset.get(name) if isinstance(asset, Mapping) else getattr(asset, name, None)

    asset_class = str(value("asset_class") or value("class") or "equity").lower()
    asset_type = "etf" if str(value("name") or "").lower().endswith(" etf") else asset_class
    return cache_security_metadata(
        session,
        str(value("symbol") or ""),
        security_name=str(value("name")) if value("name") else None,
        asset_type=asset_type,
        source="alpaca",
        retrieved_at=retrieved_at,
        confidence=Decimal("0.85"),
        quality_status=str(value("status") or "available").lower(),
    )


def resolve_security_metadata(
    session: Session,
    symbol: str,
    *,
    as_of: date | None = None,
    stale_after_days: int = DEFAULT_STALE_AFTER_DAYS,
) -> MetadataResolution:
    canonical = symbol.strip().upper()
    instrument = session.scalar(select(Instrument).where(Instrument.canonical_symbol == canonical))
    effective_as_of = as_of or date.today()
    if instrument is None:
        return MetadataResolution(
            canonical,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "unavailable",
            False,
            True,
            False,
            (f"Metadata is unavailable for {canonical}.",),
        )
    rows = list(
        session.scalars(
            select(SecurityMetadata)
            .where(
                SecurityMetadata.instrument_id == instrument.id,
                (
                    SecurityMetadata.effective_date.is_(None)
                    | (SecurityMetadata.effective_date <= effective_as_of)
                ),
            )
            .order_by(SecurityMetadata.manual_override.desc(), SecurityMetadata.retrieved_at.desc())
        )
    )
    if not rows:
        return MetadataResolution(
            canonical,
            None,
            instrument.asset_type if instrument.asset_type != "unknown" else None,
            None,
            None,
            None,
            None,
            None,
            None,
            "unavailable",
            False,
            True,
            False,
            (f"Metadata is unavailable for {canonical}.",),
        )
    selected = rows[0]
    stale = (effective_as_of - selected.retrieved_at.date()).days > stale_after_days
    disclosures = []
    if stale:
        disclosures.append(
            f"Metadata for {canonical} is stale (retrieved {selected.retrieved_at.date()})."
        )
    if selected.manual_override:
        disclosures.append(f"Metadata for {canonical} includes an owner manual override.")
    return MetadataResolution(
        canonical,
        selected.security_name,
        selected.asset_type or instrument.asset_type,
        selected.sector,
        selected.industry,
        selected.source,
        selected.effective_date,
        selected.retrieved_at,
        Decimal(selected.confidence) if selected.confidence is not None else None,
        selected.quality_status,
        selected.manual_override,
        False,
        stale,
        tuple(disclosures),
    )


def save_etf_sector_snapshot(
    session: Session,
    symbol: str,
    weights: Mapping[str, object],
    *,
    source: str,
    effective_date: date,
    retrieved_at: datetime | None = None,
    quality_status: str = "available",
) -> EtfSectorSnapshot:
    normalized: dict[str, Decimal] = {}
    for sector, value in weights.items():
        clean_sector = sector.strip()
        try:
            percentage = Decimal(str(value))
        except (InvalidOperation, TypeError) as exc:
            raise ValueError("ETF sector weights must be numeric percentages.") from exc
        if not clean_sector or not percentage.is_finite() or percentage < 0 or percentage > 100:
            raise ValueError("ETF sector weights must be between 0% and 100%.")
        normalized[clean_sector] = percentage
    total = sum(normalized.values(), Decimal("0"))
    if total > Decimal("100") + WEIGHT_TOLERANCE:
        raise ValueError(f"ETF sector weights cannot exceed 100%; received {total}%.")
    instrument = _instrument(session, symbol, asset_type="etf")
    instrument.asset_type = "etf"
    timestamp = retrieved_at or datetime.now(timezone.utc)
    for sector, percentage in normalized.items():
        session.add(
            EtfSectorWeight(
                instrument_id=instrument.id,
                sector=sector,
                weight=percentage / Decimal("100"),
                source=source.strip(),
                effective_date=effective_date,
                retrieved_at=timestamp,
                quality_status=quality_status.strip(),
            )
        )
    session.flush()
    return resolve_etf_sector_snapshot(session, symbol, as_of=effective_date)


def resolve_etf_sector_snapshot(
    session: Session,
    symbol: str,
    *,
    as_of: date | None = None,
    stale_after_days: int = DEFAULT_STALE_AFTER_DAYS,
) -> EtfSectorSnapshot:
    canonical = symbol.strip().upper()
    instrument = session.scalar(select(Instrument).where(Instrument.canonical_symbol == canonical))
    effective_as_of = as_of or date.today()
    if instrument is None:
        return EtfSectorSnapshot(
            canonical,
            {},
            None,
            None,
            None,
            "unavailable",
            False,
            (f"ETF sector weights are unavailable for {canonical}.",),
        )
    candidates = list(
        session.scalars(
            select(EtfSectorWeight)
            .where(
                EtfSectorWeight.instrument_id == instrument.id,
                EtfSectorWeight.effective_date <= effective_as_of,
            )
            .order_by(EtfSectorWeight.effective_date.desc(), EtfSectorWeight.retrieved_at.desc())
        )
    )
    if not candidates:
        return EtfSectorSnapshot(
            canonical,
            {},
            None,
            None,
            None,
            "unavailable",
            False,
            (f"ETF sector weights are unavailable for {canonical}.",),
        )
    selected = candidates[0]
    rows = [
        row
        for row in candidates
        if row.effective_date == selected.effective_date
        and row.retrieved_at == selected.retrieved_at
        and row.source == selected.source
    ]
    weights = {row.sector: Decimal(row.weight) for row in rows}
    classified = sum(weights.values(), Decimal("0"))
    if classified < Decimal("1"):
        weights[UNCLASSIFIED] = Decimal("1") - classified
    stale = (effective_as_of - selected.retrieved_at.date()).days > stale_after_days
    disclosures = []
    if UNCLASSIFIED in weights:
        disclosures.append(f"{canonical} has explicit unclassified ETF exposure.")
    if stale:
        disclosures.append(
            f"ETF sector weights for {canonical} are stale "
            f"(retrieved {selected.retrieved_at.date()})."
        )
    return EtfSectorSnapshot(
        canonical,
        weights,
        selected.source,
        selected.effective_date,
        selected.retrieved_at,
        selected.quality_status,
        stale,
        tuple(disclosures),
    )


def portfolio_sector_exposure(
    session: Session,
    portfolio: Portfolio,
    prices: Mapping[str, object],
    *,
    as_of: date | None = None,
    stale_after_days: int = DEFAULT_STALE_AFTER_DAYS,
) -> SectorExposureResult:
    shares: dict[str, Decimal] = {}
    for lot in portfolio.holdings:
        shares[lot.symbol] = shares.get(lot.symbol, Decimal("0")) + Decimal(lot.shares)
    sector_values: dict[str, Decimal] = {}
    missing_prices: list[str] = []
    stale_symbols: list[str] = []
    disclosures: list[str] = []
    invested = Decimal("0")
    for symbol, quantity in shares.items():
        if symbol not in prices or prices[symbol] is None:
            missing_prices.append(symbol)
            continue
        market_value = quantity * Decimal(str(prices[symbol]))
        invested += market_value
        metadata = resolve_security_metadata(
            session, symbol, as_of=as_of, stale_after_days=stale_after_days
        )
        disclosures.extend(metadata.disclosures)
        if metadata.stale:
            stale_symbols.append(symbol)
        snapshot = resolve_etf_sector_snapshot(
            session, symbol, as_of=as_of, stale_after_days=stale_after_days
        )
        # A stored ETF snapshot is itself positive ETF classification evidence,
        # including when a primary provider exposes only a generic equity class.
        if (metadata.asset_type or "").lower() == "etf" or snapshot.weights:
            disclosures.extend(snapshot.disclosures)
            if snapshot.stale:
                stale_symbols.append(symbol)
            allocations = snapshot.weights or {UNCLASSIFIED: Decimal("1")}
            for sector, weight in allocations.items():
                sector_values[sector] = sector_values.get(sector, Decimal("0")) + (
                    market_value * weight
                )
        else:
            sector = metadata.sector or UNCLASSIFIED
            sector_values[sector] = sector_values.get(sector, Decimal("0")) + market_value
    if missing_prices:
        disclosures.append(
            "Sector exposure excludes symbols without prices: "
            + ", ".join(sorted(missing_prices))
            + "."
        )
    exposures = tuple(
        SectorExposure(
            sector,
            value,
            (value / invested * Decimal("100")) if invested else Decimal("0"),
        )
        for sector, value in sorted(sector_values.items(), key=lambda item: (-item[1], item[0]))
    )
    return SectorExposureResult(
        exposures,
        invested,
        tuple(sorted(missing_prices)),
        tuple(sorted(set(stale_symbols))),
        tuple(dict.fromkeys(disclosures)),
    )
