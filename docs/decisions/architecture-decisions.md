# Architecture decisions

## ADR-001: Extend DashApp incrementally

**Decision:** Extend and incrementally refactor the existing DashApp instead of starting a separate rebuild.

**Rationale:** The repository already contains an accounting model, portfolio workflow, and paper-trading safety controls that should be preserved. A rebuild needs code-audit evidence that an incremental migration is unsafe or impractical, plus explicit approval.

## ADR-002: Preserve Streamlit as the UI layer

**Decision:** Retain Streamlit for presentation and interaction.

**Rationale:** Financial calculations, acquisition, reconstruction, forecasts, scenarios, and validation must be reusable modules outside Streamlit pages, keeping the UI replaceable and testable.

## ADR-003: Ledger is the accounting authority

**Decision:** Preserve the transaction ledger as the source of truth for cash, holdings, FIFO lots, cost basis, portfolio history, contributions, withdrawals, income, fees, and taxes.

**Rationale:** Derived balances, valuations, and analytics must remain reproducible from dated transactions.

## ADR-004: Use active-session monitoring now

**Decision:** Use tiered active-session streaming and batched refreshes for monitoring; do not require persistent monitoring when the application is closed.

**Rationale:** Broad coverage of fewer than 400 held symbols is preferred to continuous streaming for only a few. Streamlit is not a guaranteed background worker, but interfaces should permit a future worker.

## ADR-005: Forecasts must be reproducible and backtested

**Decision:** Save forecast model/version, assumptions, coverage, sources, adjustment methods, simulation count, random seed, timestamp, and validation status; support rolling historical backtests.

**Rationale:** Successful execution is not validation. Outputs must disclose uncertainty, downside, assumptions, limitations, and data sufficiency.

## ADR-006: Use look-through ETF sector exposure

**Decision:** Prefer ETF look-through sector allocations and combine them proportionally with direct-stock exposure. Keep unclassified instruments explicit.

**Rationale:** Classifying an ETF as one sector produces misleading concentration and risk analysis.

## ADR-007: No paid secondary data API requirement

**Decision:** Alpaca Basic is the primary service; optional free metadata, validated occasional CSV imports, and documented proxy history may supplement it.

**Rationale:** The design must fit the current account entitlement while preserving dataset provenance and explicit data-quality status.

## ADR-008: Per-portfolio benchmark blends

**Decision:** Each internal portfolio supports a configurable benchmark blend, rather than only one benchmark symbol.

**Rationale:** Portfolio objectives and allocations differ, so performance and risk comparisons need portfolio-specific reference blends.

## ADR-009: One shared analytics foundation

**Decision:** Use one shared analytics foundation for forecasting, risk, market, sector, and security views.

**Rationale:** Shared reconstruction, classifications, return series, assumptions, and provenance prevent contradictory results across features.

## ADR-010: Hypothetical trades remain non-executable analysis

**Decision:** Keep hypothetical-trade analysis separate from order submission until a user explicitly transfers a proposal into an order.

**Rationale:** This preserves a clear review boundary while enabling before/after analysis of allocation, risk, forecast, and stress effects.

## ADR-011: Retirement taxes are transparent planning assumptions

**Decision:** Phase 10 uses configurable effective rates and explicit account treatment for a
first-pass retirement tax estimate. It must be labeled as planning analysis rather than tax advice,
and it does not infer unsupported jurisdiction-specific rules.

**Rationale:** Account-aware withdrawals materially affect long-horizon funding, but tax brackets,
deductions, RMDs, filing status, state/local law, detailed future lots, and Social Security
provisional-income calculations require facts and legal scope the application does not possess.
Keeping rates, realization, withdrawal order, and limitations visible is more reproducible than
implying tax-return precision.
