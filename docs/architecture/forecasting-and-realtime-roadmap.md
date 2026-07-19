# Forecasting and real-time roadmap

## Direction and boundaries

DashApp will be extended and incrementally refactored. A separate rebuild is not the default; it requires a later code audit showing that safe incremental migration is impractical. The transaction ledger remains the accounting authority. Streamlit remains the presentation and interaction layer, while financial calculations, market-data acquisition, portfolio reconstruction, forecasting, scenarios, and validation reside in reusable modules independent of Streamlit pages.

Build one shared analytics foundation for portfolio reconstruction, risk, forecasting, market, sector, and security views. This avoids competing calculations and permits each view to disclose the same data coverage, assumptions, and limitations.

## Forecasting objectives

Prioritize forecasting work in this order:

1. Retirement planning
2. Portfolio risk management
3. Security selection

The platform will support both 1–10-year investment and portfolio forecasts and 20–40-year retirement accumulation and withdrawal forecasts. Where adequate history is available, models should use correlated holding-level returns. For limited-history securities, use documented sector, style, factor, or broad-market proxy fallbacks.

Expected-return inputs must support a blend of historical estimates, external published capital-market assumptions, and user overrides. Unadjusted historical average returns must not be the sole default estimate.

Every output must communicate probability bands, downside scenarios, assumption sensitivity, model limitations, data sufficiency, and reproducibility. A model is not validated merely because it executes: forecast models must support rolling historical backtesting.

## Delivery sequence

1. Historical-data and portfolio-reconstruction foundation
2. Deterministic scenario analysis
3. Historical block-bootstrap simulation
4. Correlated parametric Monte Carlo
5. Long-horizon retirement modeling
6. Hypothetical-trade analysis
7. Regime-based modeling

The retirement model will account for Traditional IRA, Roth IRA, and taxable-account treatment; Social Security; pensions and other outside income; fixed inflation-adjusted spending; contributions and withdrawals; account-aware withdrawal sequencing; inflation; fees; rebalancing; target and depletion probabilities; and sequence-of-returns risk.

Each saved forecast run must retain model type and version, assumptions, data coverage, data sources, adjustment methods, simulation count, random seed, creation timestamp, and validation status.

## Real-time monitoring

The current roadmap targets fewer than 400 distinct held symbols, with an expected stock-to-ETF ratio near 6:1. Favor broad monitoring across held symbols over exact continuous streaming for a small subset.

Use a tiered active-session design:

- Stream the highest-priority symbols permitted by the Alpaca subscription.
- Refresh remaining held symbols through efficient batched snapshot requests.
- Prioritize open orders, visible symbols, selected portfolios, large positions, large risk contributors, active alerts, broad-market proxies, and sector proxies.
- Present whether a value is streamed, recently refreshed, stale, or previous close.
- Label IEX-derived real-time portfolio valuations as **indicative**, not consolidated-market values.

Persistent monitoring while the app is closed is out of scope for now. Streamlit is not a guaranteed persistent worker; interfaces should allow a future separate worker without requiring one in this phase.

## Benchmarks and hypothetical trades

Each internal portfolio should have a configurable benchmark blend, rather than only one benchmark symbol. Hypothetical trades remain separate from order submission unless the user explicitly transfers a proposal into an order.

Before-versus-after hypothetical analysis should address portfolio weights, cash, sector exposure, concentration, volatility, beta, risk contribution, drawdown exposure, expected return, forecast success probability, downside percentiles, and stress losses.
