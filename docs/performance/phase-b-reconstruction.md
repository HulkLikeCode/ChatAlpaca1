# Phase B reconstruction performance evidence

## Scope and method

Baseline: `performance-phase-a-bootstrap` at
`18f18c281c2aed7cd410a5a1882ffee6a41a0480`.

Optimized branch: `performance-phase-b-reconstruction`.

The local synthetic fixture used 130 business-day closes and three ledger transactions per
portfolio (transfer, buy, and dividend). Overview assembled the combined performance report,
portfolio cards, and consolidated holdings. Compare assembled the combined performance and
comparison reports. Both paths used one SPY benchmark series.

Each case had three warm-ups. The medium fixture used 10 measured repetitions and the stress
fixture used 5. Reported wall and CPU values are medians. Peak Python allocation was measured
separately with `tracemalloc`, so it should not be compared with process RSS. Call counts were
collected by monkeypatching `_typed_reconstruction`, `reconstruct_from_coverage`, and
`household_valuation`; no production counter or long-lived cache was added.

These are local synthetic-fixture measurements, not claims about production latency.

## Results

| Operation | Portfolios | Transactions | History | Reconstruction calls, before → after | Household calls, before → after | Median wall, before → after | Median CPU, before → after | Peak allocation, before → after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overview | 2 | 6 | 130 | 7 → 1 | 5 → 1 | 1.194 s → 0.331 s | 1.169 s → 0.324 s | 272 KB → 332 KB |
| Compare | 2 | 6 | 130 | 10 → 1 | 3 → 1 | 1.770 s → 0.358 s | 1.715 s → 0.339 s | 270 KB → 259 KB |
| Overview | 8 | 24 | 130 | 25 → 1 | 11 → 1 | 6.480 s → 1.255 s | 6.017 s → 1.227 s | 777 KB → 833 KB |
| Compare | 8 | 24 | 130 | 34 → 1 | 9 → 1 | 11.241 s → 1.189 s | 9.590 s → 1.167 s | 769 KB → 805 KB |
| Overview | 20 | 60 | 130 | 61 → 1 | 23 → 1 | 13.931 s → 4.212 s | 13.802 s → 3.790 s | 1.794 MB → 1.840 MB |
| Compare | 20 | 60 | 130 | 82 → 1 | 21 → 1 | 20.311 s → 3.276 s | 20.050 s → 3.238 s | 1.903 MB → 1.896 MB |

The after path makes one typed multi-portfolio reconstruction call. Internally,
`reconstruct_from_coverage` still reconstructs each selected portfolio exactly once, so work scales
linearly with the selected portfolios rather than repeating the same portfolio for each consumer.

Peak traced allocation increased by roughly 36–60 KB in four cases because the scoped context
retains one accepted multi-portfolio result until report assembly completes. The absolute change is
small, is bounded by one render/report operation, and does not create retained application state.
Compare stress allocation decreased slightly.

## Equivalence and invalidation evidence

Tests compare context-backed and independently assembled outputs exactly for:

- household and per-portfolio confirmed valuation;
- all-time, daily, and custom gain/loss;
- per-portfolio and household performance growth;
- benchmark-normalized comparison series;
- Alpha/Beta values and observation counts;
- Overview performance rows and Compare metrics tables;
- warnings, coverage text, missing-price states, and common-date metadata.

Existing analytics, reconstruction, report, and app tests continue to cover buys, sells, multiple
lots, dividends, interest, quantity awards, partial and missing history, staggered price dates,
weekend boundaries, cash-only portfolios, latest-symbol overlay metadata, and accepted common-date
valuation.

The context is an immutable, explicit argument owned by one report or active-tab render. It is not
stored in a module global or `st.cache_data`. It validates portfolio selection and the exact
call-scoped close-frame identity before reuse. A new render or standalone report assembly builds a
new context, so changed selections, ledger state, holdings, cash, closes, ranges, benchmarks, or
valuation inputs naturally receive a new reconstruction. Custom ranges and benchmark inputs are
not cached in the context; consumers apply them to the accepted reconstruction as before.
