# Data-source policy

## Primary source and entitlement

Use the individual Alpaca Trading API Basic subscription as the primary source for current and recent equity data, IEX real-time data, historical bars within entitlement, orders, fills, brokerage positions, account information, and available news data. Respect current entitlement, coverage, and rate limits. Do not assume a paid supplemental market-data service.

IEX-based real-time values are indicative rather than consolidated-market values. The UI must state the freshness and provenance of displayed values.

## Supplemental and proxy data

The architecture may use free supplemental metadata APIs, occasional validated Stooq or similar CSV imports, and proxy histories for securities without adequate history. Supplemental data must be clearly distinguished from Alpaca data and must not silently override a higher-priority validated source.

Every stored market or metadata dataset must retain:

- Source and feed
- Retrieval or import date
- Coverage dates
- Adjustment method
- Imported-file hash, when applicable
- Quality status and validation warnings
- Override priority

This provenance is required for reproducible forecasts, market-data diagnosis, and safe reconciliation. Imported or proxy datasets must record their limitations and selection rationale.

## Historical price semantics and precedence

Portfolio reconstruction and accounting use split-adjusted, non-dividend-adjusted prices. Dividend
cash remains explicit in the transaction ledger and must not be counted again through adjusted
prices. Benchmark total-return calculations may explicitly select dividend-adjusted data. Raw,
split-adjusted, dividend-adjusted, and total-return datasets are stored and queried separately.

Historical datasets are immutable acquisition records. Resolution selects the greatest explicit
override priority and then the newest retrieval and dataset identifiers as deterministic tie
breakers. A conflicting lower-priority observation remains stored and produces a coverage warning;
it never silently overwrites the selected observation. Missing observations remain null rather than
being forward-filled or converted to zero.

Alpaca historical data is cached durably and refreshed only for missing or stale ranges. Requests
use the configured Basic-plan feed (IEX by default), are rate constrained, and persist only
credential-free request metadata. Occasional Stooq-like CSV imports require explicit adjustment
metadata, validation, and a SHA-256 file hash; Stooq dividend adjustment is never inferred.

## Sector data

For individual stocks, retain sector and industry classifications from the best available cached metadata source. For ETFs, prefer look-through sector allocations over assigning the entire ETF to a single sector.

Portfolio sector exposure combines direct-stock exposure and proportional ETF look-through exposure. Keep unclassified and unusual instruments explicit; do not force them into an inaccurate sector.

## Shared analytics use

Market, sector, risk, forecasting, and security views consume one shared analytics foundation and its provenance records. This ensures that forecasts can identify their source data, adjustment methods, coverage, and proxies, and that valuation and risk displays can expose data quality consistently.
