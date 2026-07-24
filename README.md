# ChatAlpaca1 aka DashApp

A private, multi-portfolio personal portfolio manager that brings together portfolio-specific and broader-market analysis, current valuations, historical performance, and planning/forecasting tools in one dashboard. It supports internal portfolio tracking, transaction-ledger accounting, benchmarking, historical analysis, projections, automatic database persistence, and Alpaca order allocation. The first release is paper-first and keeps up to 20 internal portfolios separate even though Alpaca holds their combined positions in one brokerage account.

## Included

- Up to 20 portfolios with an uncapped number of tracked symbols
- Transaction-backed cash, FIFO lots, cost basis, and market value
- Durable, provider-neutral daily market-data cache with typed coverage and provenance
- Typed ledger reconstruction for daily values, positions, flows, returns, benchmarks, and coverage
- Owner-editable Traditional IRA, Roth IRA, taxable, and unknown account classifications
- Effective-dated per-portfolio benchmark blends with explicit rebalancing assumptions
- Cached security metadata and dated ETF sector look-through classifications with provenance
- Described, searchable buy-and-hold benchmark comparisons against SPY, QQQ, IWM, DIA, VTI, VT,
  EFA, EEM, AGG, BND, TLT, IEF, SHY, and LQD, plus arbitrary stock or ETF symbols
- Growth-of-$100 chart plus return, volatility, and drawdown statistics
- Read-only-user 1–10 year planning projections with editable session-only scenarios, monthly contributions, percentile bands, and target probabilities
- Reproducible 1–10 year historical block-bootstrap forecasts with correlated monthly returns,
  explicit proxies, inflation, fees, rebalancing, downside attribution, and rolling backtests
- Reproducible correlated parametric forecasts using multivariate normal or Student's t returns,
  shrunk and owner-blended parameters, matrix validation, sensitivity analysis, and backtests
- Reproducible 20–40-year retirement forecasts with accumulation, fixed real spending,
  outside income, account-aware withdrawals, transparent tax estimates, sequence-risk diagnostics,
  sensitivities, and historical sequence replay
- Saved, reproducible deterministic stress scenarios with household, portfolio, holding, sector,
  account-type, baseline, coverage, warning, and sensitivity output
- Named, non-executable hypothetical scenarios with multiple proposed trades, cash and internal
  assignment changes, before/after allocation, sector, benchmark, concentration, risk, forecast,
  stress, and optional retirement analysis, plus ledger-baseline staleness warnings
- Separate admin and read-only user passwords; only admins can mutate data, upload files, or act on brokerage orders
- Manual transaction entry, brokerage CSV preview/import, duplicate protection, and rebuild-from-statement
- Combined sortable transaction management with portfolio/type/date filters and guarded edits/deletes
- Sticky, batched portfolio and master-date controls with immediate 5D, 1M, 6M, YTD, 1Y, and 5Y
  calendar-aware presets plus a compact selected-scope TPV, Holdings, and Cash strip
- Compact portfolio-income reporting for realized dividends and interest, with master-end-aware
  YTD, trailing-365-day, selected-range, normalized-quarterly, monthly, and source views
- Transaction-aware confirmed all-time, daily, and custom-range portfolio gain/loss excluding
  contributions and awards, with a separately labeled, timestamped, provenance-disclosed
  indicative active-session IEX overlay and all-time independence from Custom Start/End
- Consolidated exact holdings with weighted cost basis, symbol-level gain/loss, lot drilldown,
  Eastern retrieval timestamps, and row-level relative age
- Per-portfolio TPV, Holdings, and Cash in both Overview and Compare
- Top-right-toolbar-only Streamlit header rollover and dark theme-consistent notifications
- Selected-range cumulative dividends on portfolio value cards
- Portfolio and holding Alpha/Beta against SPY total return, using at least 60 overlapping daily returns
- Market and limit orders, cancellation, fill synchronization, and an auditable internal transaction ledger
- Tiered active-session monitoring for fewer than 400 held symbols, with capped IEX streaming,
  batched snapshot fallback, reconnect gap backfill, explicit freshness, portfolio pulse, market
  context, portfolio value/contribution/staleness/volatility/correlation fields, and symbol detail;
  period returns require their full horizon, drawdown is measured from
  the available-window peak, and raw 21-session SPY correlation requires 21 complete pairs before
  its secondary fixed-threshold heuristic is shown; no closed-app monitoring claim or persistent
  worker
- Automatic persistence through SQLite locally or hosted PostgreSQL in production
- Architecture hooks for strategies, short positions, options, and separately gated live trading
- Dark black/blue/purple/white theme with no green or red status colors

## Data scope and roadmap

The dashboard uses an individual Alpaca Trading API account on the free Basic subscription. For US stocks and ETFs, the plan provides IEX real-time market data, historical data since 2016 with the most recent 15 minutes unavailable through the historical API, and 200 historical API calls per minute. See Alpaca's [Trading API subscription documentation](https://docs.alpaca.markets/us/docs/about-market-data-api#trading-api-subscriptions) for current plan details.

Development will continually expand historical analysis and forecasting capabilities, using as much Alpaca API data as is reasonable within the account's data entitlement, coverage, and rate-limit constraints. For critical historical-market-data gaps, manually maintained Stooq uploads may be used as an occasional contingency source.

Historical portfolio accounting uses split-adjusted, non-dividend-adjusted prices because dividends
are explicit ledger transactions. Benchmark total-return comparisons may request dividend-adjusted
data. Raw, split-adjusted, dividend-adjusted, and total-return datasets remain separate, and missing
daily observations remain missing. Alpaca responses and occasional CSV imports are stored with
source, feed, coverage, retrieval time, adjustment, quality, priority, warnings, and credential-free
request metadata; imports additionally retain a SHA-256 file hash.

A future state may add an Alpaca paper account that mirrors portfolios currently held at Schwab and Fidelity, streamlining real-time updates. Until then, Schwab and Fidelity CSV transaction imports are available as occasional reconciliation patches rather than the preferred operating model.

The ultimate goal is a private, comprehensive manager for multiple personal portfolios, combining portfolio-specific and general-market historical, real-time, and forecasting capabilities.

## Safety model

The dashboard is intended for private access. Every browser session must authenticate with either
`ADMIN_PASSWORD` or `USER_PASSWORD`. The admin role has full application access. The user role can
view Overview, Compare, Forecast, Manage, Trade, and Architecture and can run session-only forecast
and deterministic-scenario calculations, but cannot edit transactions or portfolio settings,
upload files, save scenario runs, submit or cancel orders, synchronize fills, or otherwise change
permanent data. Deployment access controls must still prevent unauthorized users from reaching the
application. Credentials are never exposed or stored in the database.

Paper mode is the default. Live mode requires all three of the following:

1. Separate live Alpaca credentials.
2. `TRADING_MODE=live`.
3. `ALLOW_LIVE_TRADING=true`.

Do not enable live mode until authentication, risk limits, audit review, and deployment access controls have been strengthened. Never commit an API key, secret key, database password, or owner password.

## Local setup

Python 3.10 through 3.12 are supported; Python 3.12 is the recommended local version. Python 3.13
is intentionally excluded until the pinned binary dependencies provide compatible wheels on Intel
macOS. The checked-in `.python-version` selects Python 3.12.10 for pyenv-compatible tools.

Do not reuse a `.venv` made with another Python version. If your existing virtual environment uses
Python 3.13, recreate it with Python 3.12 before installing dependencies.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python scripts/verify_environment.py --python-only
python -m pip install -r requirements-dev.txt
python scripts/verify_environment.py
cp .env.example .env
python -m streamlit run streamlit_app.py
```

Set newly rotated paper credentials in `.env`:

```dotenv
DATABASE_URL=sqlite:///data/chat_alpaca.db
ADMIN_PASSWORD=choose-a-long-password
USER_PASSWORD=choose-a-different-long-password
ALPACA_API_KEY=your-rotated-paper-key
ALPACA_SECRET_KEY=your-rotated-paper-secret
ALPACA_DATA_FEED=iex
REALTIME_STREAM_CAP=30
REALTIME_REGULAR_SECONDS=45
REALTIME_OFF_HOURS_SECONDS=180
REALTIME_CALLS_PER_MINUTE=180
TRADING_MODE=paper
ALLOW_LIVE_TRADING=false
```

The ignored `data/chat_alpaca.db` file is created automatically. Alembic schema upgrades run before
seed data or one-time data migrations. Seed data is inserted when the portfolio table is empty,
while the existing `data_migrations` markers continue to make data-level changes run exactly once.

## Database migrations

Application startup performs a guarded upgrade automatically. A fresh database is migrated to the
current schema. An existing pre-Alembic DashApp database is inspected without reading transaction
values; if its tables, columns, keys, constraints, and indexes match the baseline, it is stamped at
the baseline revision without recreating records. An incompatible schema is rejected with the
differences listed so it can be backed up and reconciled first.

Run the same guarded upgrade manually from the repository root:

```bash
python -m chat_alpaca.migrations upgrade
```

Developers may use Alembic directly for an already-versioned database:

```bash
alembic upgrade head
alembic downgrade -1
```

The first revision is the safely adoptable Phase 2 baseline. Phase 3 adds the historical market-data
tables, Phase 6 adds account classifications, effective-dated benchmark components, security
metadata, and ETF sector snapshots, and Phase 7 adds saved forecast summaries, exact dataset
references, and model-validation evidence. Back up the database before downgrading. Set `DATABASE_URL` to select
the SQLite or PostgreSQL target; PostgreSQL URLs are normalized to the installed psycopg 3 driver.

## Portfolio transactions

Owner controls support manual entries for buys, sells, dividends, interest, transfers, awards, fees,
taxes, and cash adjustments. Transactions are the source of truth: cash, FIFO lots, and ledger rows
can be rebuilt deterministically after a guarded service-level update or deletion. A fresh database
creates five empty, generically named portfolios. Existing local or hosted database records remain
authoritative and are not replaced by tracked statement files or private hard-coded holdings.

Manage shows the master-selected portfolios in one sortable transaction table. The master date
range and transaction-type filter drive the displayed rows. Select a row to edit or delete that
transaction in its own portfolio; separate target selectors are used for new entries and CSV
imports. Compact add and edit controls follow the transaction-table field order, while new-entry
action text is derived from its transaction type. Manage keeps Transactions first, followed by Add
Transaction, Brokerage CSV, and consolidated portfolio administration. Transaction entry and
posted dates use `M/D/YY`.

Every transaction edit or deletion requires a transaction-specific typed confirmation. Imported,
seeded, and Alpaca-generated transactions show an additional divergence warning. Updates are marked
with the `manual_override` source, and immutable before/after audit snapshots retain the original
source and values; deletion audits retain the complete removed transaction snapshot.

## Portfolio valuation

Historical portfolio values are reconstructed from dated transactions and adjusted daily market
closes rather than projecting today's cash and holdings backward. Portfolio gain/loss excludes
external transfers, cash adjustments, awards, and the fair value of contributed opening positions;
market movement, dividends, and interest remain part of performance. A quantity award requires an
explicit stored fair value; no close or quote is inferred, and affected gain/return outputs are
unavailable with a warning when it is absent. Daily gain/loss compares the
two latest market closes, while custom gain/loss uses the close before the applied master start date
through the master end date. The same range controls dividend custom totals, Compare charts and
metrics, exact holdings, and transaction filtering.

During an active authenticated session, complete per-portfolio IEX quote moves may overlay the
latest confirmed close for indicative gain/loss. Fresh and complete stale quote moves refresh the
display every 30 seconds; stale-derived amounts use muted blue, while portfolios with no calculable
quote move retain confirmed-close values. Muted blue is limited to stale real-time amounts during
regular market hours; after-hours, weekend, and holiday values use the normal theme colors.
Extended-hours trades continue to compare with the previous completed session close. Combined
metrics may therefore contain fresh, stale, and fallback portfolio rows, with compact coverage
disclosing that mix. Custom gain/loss receives the overlay only when the applied master end date is
today; a historical end date remains fixed.
Confirmed all-time performance always runs from inception through the latest complete confirmed
valuation date and is not truncated by Custom Start or Custom End. Any permitted live overlay is
separately labeled indicative with its timestamp and provenance and does not alter the confirmed
historical series. Comparison metrics are unavailable, rather than numeric zero, when history is
insufficient. Alpha is displayed as `Alpha`, with its annualized market-model-intercept and
zero-risk-free-rate assumption disclosed below the table. Missing quote moves remain unavailable
rather than zero. Overview and Compare omit redundant summary metric tiles while retaining their
tables, warnings, coverage, and Overview portfolio value cards. Their tables expose per-portfolio
TPV, Holdings, and Cash from the same common-date valuation contract used by the master strip;
TPV equals Holdings plus ledger Cash before display-only nearest-$100 half-up rounding.
Because the strip is global, every page resolves one selection-scoped historical request through
the existing 15-minute data cache; it never issues a second strip-only request. Transaction-ledger
reconstruction remains lazy and is added to the request context only for analytics views that need
it. Applying a new portfolio selection creates a new cache key rather than showing the prior
selection's session value.
Exact holdings retain numeric share values and adapt display precision through eight decimal
places. The confirmed valuation display uses the authoritative dataset retrieval timestamp in
daylight-aware Eastern Time and derives `As of` from that same row-level timestamp without
refreshing market data. Current-open-lot fields are labeled `Unrealized gain/loss`, `Latest close
change` with the actual observation dates, and `Current-lot unrealized custom change`; the latter
excludes sold lots and income and is distinct from portfolio Custom gain/loss.

Portfolio cards show cumulative dividend ledger credits in the inclusive applied master range;
interest is excluded. Portfolio and
holding Alpha/Beta use the applied master range and require at least 60 overlapping daily returns
against SPY total-return data. Alpha is the compounded annualized daily regression intercept.
Portfolio returns are ledger-aware. Holding returns add symbol-assigned dividend credits to
split-adjusted price returns; unassigned dividends and dividend events without an attributable
prior holding value are excluded and disclosed.

Compare opens Performance Comparison by default. Its hover-only Plotly trace presents the available
series from highest to lowest at each hovered date without reordering the legend, and its retained
series table formats total return to one decimal place. Notes below the chart disclose that
annualized return compounds over elapsed calendar days using 365.25 days per year and that
volatility is the sample standard deviation of daily returns annualized by the square root of 252.

The reusable Phase 4 service in `chat_alpaca.reconstruction` is the shared analytics boundary. It
consumes canonical transactions and confirmed split-adjusted, non-dividend-adjusted repository
closes; it never replaces missing prices with zero or a silent forward fill. Typed results preserve
individual portfolios and a combined household view, disclose common-date freshness and provenance,
separate external flows and ledger income or expense categories, and report price, income, total,
time-weighted, and money-weighted returns. Its transparent data-sufficiency score is a bounded
status, not a claim of precision. Proxy assignments are disclosed but never used to value holdings.

Phase 5 keeps Streamlit as the forms-and-presentation layer. `chat_alpaca.commands` constructs and
validates transaction commands, `chat_alpaca.reports` owns symbol universes, baseline windows,
market-data adjustment policy, comparison assembly, combined performance reports, coverage, and
warnings, and `chat_alpaca.forecasting` constructs explicit forecast requests. When market prices
are unavailable, a planning scenario may retain the existing cost-basis-plus-cash workflow only
with that fallback and its coverage disclosed. Application database migration, seeding, and initial
portfolio loading are coordinated by `chat_alpaca.bootstrap.initialize_application` outside the UI.
The session-only legacy planning projection has the explicit contract `legacy_projection / 1.0.0`.
Its scope is exactly the applied master-selected portfolios; there is no second projection-scope
selector or scenario preset. Expected return and volatility are direct model inputs, and the
initial forecast horizon defaults to 10 years without replacing an existing session selection.
Each result records its assumptions, seed, simulation count, common confirmed source valuation date
or disclosed fallback, valuation methodology, and generation timestamp. Boolean and nonfinite
numeric inputs are rejected before simulation. Calendar labels begin at that explicit source date,
or at the UTC generation date for the disclosed fallback, and label each modeled monthly step at
the end of the corresponding following calendar month. These labels do not change forecast paths.

Phase 7 adds deterministic scenario analysis in `chat_alpaca.scenarios`. The V.15.2 UI defaults to
the first-class `As Is` branch and offers Market Decline, Holding Decline, Dividend Reduction,
Inflation Increase, Low Return, and Retirement-Date Decline as mutually exclusive alternatives.
Sector decline, contribution interruption, lost decade, and historical replay remain readable for
legacy saved records but are absent from new user-generated runs. Runs persist the model/version, exact
ledger hash and dataset references, assumptions, coverage, proxy disclosure, validation state, and
summary outputs. The compact always-visible input grid defaults to 0% shocks, no selected holding,
$0 monthly contribution, 3% inflation, $0 annual spending, 7% expected return, 4% low return, and a
10-year horizon. Its uneditable Retirement Date is derived safely as `Month YYYY` from today plus
the 1–40-year horizon. The UI derives its scenario explanation and four-column assumption/default/delta
comparison from the structured assumptions used by the calculation. Results use one compact
Baseline / Scenario / Household Impact table and disclose whether the selected branch is an
immediate current-value shock or terminal forecast value. Deterministic scenarios
generate no raw paths. Automated test success is retained
as validation evidence but cannot by itself label a model validated.
DataFrame inputs resolve every nonzero held symbol to one common household valuation date: the
oldest latest usable symbol date, with each positive finite price taken on or before that date.
Mapping inputs are treated as an explicitly supplied undated snapshot and reject booleans,
nonfinite values, zero, and negative prices. Historical replay uses only jointly complete rows,
never forward-fills, requires at least two observations, and persists its shared endpoints and
observation count. Deterministic model version `1.2.0` applies one resolved snapshot throughout each
run. It retains the first-order shock, fixed trailing-dividend, nominal-to-real inflation, and
mutually exclusive branch limitations; saved `1.1.0` records remain readable.

Phase 8 adds historical block-bootstrap forecasting in `chat_alpaca.bootstrap_forecasting`. It
samples observed, jointly aligned monthly holding or portfolio returns in circular 3-, 6-, or
12-month blocks and does not impose an expected-return estimate. Runs support deterministic seeds,
configurable simulation counts, 1–10 year horizons, month-end contributions, constant inflation,
annual fees, monthly/quarterly/annual/no rebalancing, target and benchmark probabilities, nominal
and real loss probabilities, terminal distributions, downside percentiles, and holding-level
downside contribution. Holdings with inadequate histories require explicit documented proxies;
proxy use lowers forecast sufficiency and joint row sampling preserves the proxy's observed market
relationships.
Boolean and nonfinite bootstrap assumptions, starting values, cash, inflation, fees, contributions,
targets, counts, and seeds are rejected before sampling or loss-probability calculation.

Circular sampling preserves within-block order and cross-asset row dependence and wraps from the
end of observed history to its beginning; that boundary is a modeling assumption, not a claim that
the historical endpoints were economically adjacent.

Rolling-origin backtests report 5th–95th percentile interval coverage, median forecast bias,
downside-band coverage, and valid, invalid, and insufficient windows. Passing configured criteria
makes a model eligible for review, never automatically validated. Saved runs retain model/version,
seed, simulation count, block length, data period, immutable dataset references, proxies,
assumptions, backtest summaries, percentile bands, and summarized terminal distributions. Raw
simulated paths and terminal samples are not persisted.

Phase 9 adds correlated parametric Monte Carlo in `chat_alpaca.parametric_forecasting` alongside,
not in place of, the bootstrap model. Multivariate normal and variance-normalized multivariate
Student's t returns share the bootstrap model's core output contract. Expected returns use
annualized compounded log-return estimates with `cross-sectional median shrinkage` rather than
unquestioned raw arithmetic means. Covariance estimates use `fixed diagonal covariance shrinkage`,
and every external or override
correlation matrix is checked for labels, dimensions, finite values, symmetry, unit diagonal,
bounded entries, and positive semidefiniteness.

Historical estimates can be blended with owner-entered or CSV-imported published capital-market
assumptions; no paid data source is required. Explicit user overrides take precedence for the fields
they set and retain their source disclosure. Optional mean-parameter uncertainty, expected-return,
volatility, and correlation sensitivities, normal-versus-fat-tail downside comparison, rolling
backtests, and like-for-like bootstrap comparison are supported. Saved runs retain distribution,
degrees of freedom, seed, model version, parameter sources and estimates, shrinkage and covariance
methods, exact datasets, proxies, assumptions, validation result, output bands, and summarized
terminal results. Raw paths and terminal samples remain excluded.

The legacy planning Monte Carlo chart keeps its visible percentile bands, fills, median, colors,
and legend. One invisible hover trace sorts the actual percentile values from highest to lowest at
each date, uses percentile rank as the deterministic tie-breaker, and displays the date once.
Streamlit's otherwise transparent header accepts no full-width pointer events; only the stable
top-right toolbar target reveals on hover or keyboard focus. Central alert styling keeps info,
warning, error, success, validation, saved-run, and stale-data notices within the black, blue,
purple, cyan, and white palette.

Phase 10 adds long-horizon retirement planning in `chat_alpaca.retirement`. It reuses the Phase 8
historical block-bootstrap and Phase 9 correlated parametric return engines for 20–40-year
accumulation and withdrawal paths. Profiles support retirement by age or date, monthly or annual
pre-retirement contributions, fixed inflation-adjusted real spending, one-time spending events,
Social Security, pensions, other outside income, fees, rebalancing, and an optional real target
estate value.

Traditional IRA, Roth IRA, taxable, and unknown balances remain explicit, with optional
account-specific allocations. Monthly contributions are deposited at month-end; configured annual
amounts are divided into 12 equal month-end deposits. Roth withdrawals are assumed qualified and
tax free. Traditional IRAs assume no nondeductible basis. Owner RMD starting age is selected from
date of birth (70½, 72, 73, or 75), uses versioned IRS Publication 590-B Table III by default and
Table II only for a sole-beneficiary spouse more than 10 years younger, calculates each owner IRA
from its prior December 31 balance, permits aggregate satisfaction across owner IRAs, and posts at
December month-end. Inherited-account RMDs remain out of scope. Social Security uses a fixed
configured taxable fraction rather than provisional-income rules.

Taxable basis is the aggregate approximation of remaining FIFO security basis plus taxable cash,
which can exceed market value, and is reduced proportionally on withdrawals. Unspent outside income
and net RMD surplus remain zero-return household cash until the next configured rebalance, remain
spendable before additional account withdrawals, and are included in retirement-date and terminal
household value. Each applicable RMD is calculated, withdrawn, taxed, and added net to household
cash once before any additional withdrawal. Current net outside income and net RMD proceeds fund
spending before previously retained cash. The tax calculation is a transparent planning estimate,
not tax advice.

Outputs include nominal and real percentile paths, retirement-date and terminal distributions,
full-horizon funding and depletion probabilities, depletion ages, lifetime taxes, withdrawals by
account type, outside-income funding, shortfalls, target-estate probability, early-retirement
sequence diagnostics, and worst-decile scenarios. Sensitivity covers retirement age, spending,
inflation, Social Security timing, contributions, expected returns, tax rates, and withdrawal order.
Rolling historical sequence replay is validation evidence only and never self-validates the model.
Saved runs retain summaries and annual bands but exclude raw paths and per-scenario arrays.
Retirement model version `1.3.0` reports separate household-asset and unpaid-shortfall
reconciliations; unpaid shortfall is an unmet obligation and is never counted as an asset. It
refuses withdrawal and tax calculations until every in-scope account is classified as taxable,
Traditional IRA, or Roth IRA. One-time spending outputs retain the resolved nearest monthly model
step and disclose that intra-month timing is not modeled. Boolean and nonfinite numerical profile
inputs are rejected. Existing saved `1.2.0` results retain their original version and outputs.

Phase 11 adds non-executable hypothetical trade analysis in `chat_alpaca.hypothetical`. It copies
ledger-derived cash and FIFO lots into isolated in-memory state, applies any number of proposed
buys, sells, cash changes, or internal portfolio reassignments, and compares allocation, basis,
look-through sectors, benchmark-relative exposure, concentration, historical risk, assumptions,
forecast downside and target probability, deterministic stress loss, and optional `Depletion
Probability`. That adjacent simplified metric uses only fixed spending and seeded lognormal returns
and omits the full retirement engine's inflation, taxes, income, fees, contributions, account types,
and withdrawal order. Named saved scenarios retain creator/time, portfolio scope, exact baseline
ledger hash, market-data as-of time, assumptions, proposals, and summarized results. Loading a
scenario recomputes the ledger hash and warns when its baseline is stale.

Hypothetical analysis never writes transactions, lots, cash, ledger entries, Alpaca allocations,
or orders. The separate order-ticket copy boundary accepts only a buy or sell after fresh owner
confirmation, an unchanged ledger hash, and a newly reviewed non-stale market price; it returns
reviewed ticket data and does not submit an order.

Exact Holdings combines the same symbol across every selected portfolio. It shows total shares to
zero display decimals, weighted-average cost, total basis, and all-time/daily/custom gain or loss.
Confirmed prices and values use one additive common date across the selected household. A separate
latest-symbol monitoring layer shows each symbol's latest date, price, and indicative value without
summing mixed-date values. Currency values also use zero display decimals; stored shares, prices,
and calculations retain their full precision. Its Summary and By Portfolio / Lot views preserve
each acquisition date and original cost basis without nesting collapsible sections. Saved
hypothetical model `1.1.0` records the common confirmed date, confirmed prices, latest-symbol dates,
and the valuation-layer semantics; saved `1.0.0` records remain unchanged.
Hypothetical actions, baselines, prices, assumptions, stress magnitudes, and adjacent retirement
inputs reject Boolean and nonfinite numerical values at the reusable service boundary.
When a symbol has both positive and negative open lots, Exact Holdings preserves those lot rows but
keeps average cost per share numerically unavailable instead of netting a misleading signed value.
Normalized quarterly income scales the selected period to 91.3125 days and is disclosed as a
non-forecast that may be unstable for periods shorter than 30 days.

## Portfolio configuration and classification

Every portfolio has an explicit owner-editable account type: Traditional IRA, Roth IRA, taxable,
or unknown. The Phase 6 migration classifies only names containing the unambiguous phrases
“Traditional IRA” or “Roth IRA”; every other existing portfolio remains unknown. In particular,
the absence of IRA wording never implies taxable status.

Benchmark configurations are append-only effective periods made from arbitrary stock or ETF
symbols whose percentage weights total 100% within a small decimal tolerance. Benchmark history
uses component total returns. Each configuration records a daily, monthly, quarterly, annual, or
never-rebalance assumption; monthly is the owner-form default. A later configuration takes effect
on its date without changing earlier periods. Combined household analytics retain each portfolio's
own benchmark rather than silently substituting a single household reference.

Security name, asset type, sector, industry, source, effective/retrieval time, confidence/status,
and manual-override state are cached independently of the ledger. Alpaca asset metadata can be
cached when available, but the app has no required supplemental metadata subscription. ETF sector
snapshots retain source, dates, and quality status. Reusable sector analytics combine direct-stock
classification with proportional ETF look-through; incomplete ETF weights and unavailable
classifications remain explicitly Unclassified, and stale inputs are disclosed. Sector dashboards
remain deferred beyond Phase 6.

Brokerage CSV imports are read in memory and preview every row before posting. Re-importing a
statement skips transactions already recorded, while the **Rebuild portfolio from statement**
action deliberately replaces that portfolio's lots, cash, transaction history, legacy ledger rows,
and saved Alpaca allocations after the owner types `REBUILD`. Private statements are not runtime
seed dependencies and must not be tracked. Runtime upload, temporary, export, snapshot, cache, and
local database paths are Git-ignored. Removing a statement from the current tree does not purge it
from earlier Git history.

## How order allocation works

1. The owner assigns every order to one of the internal portfolios.
2. The app submits the order to the single Alpaca account with a unique client order ID.
3. The assignment is saved before the UI reports success.
4. **Sync fills** retrieves Alpaca's current filled quantity and average price.
5. Only newly filled shares are applied to the assigned internal portfolio, making repeated synchronization idempotent.
6. Buys reduce that portfolio's cash; sells increase it. Sells beyond internal long shares create a negative internal position.

Alpaca remains the authority for brokerage orders and combined positions. This application's database is the authority for the internal portfolio split.

## Production deployment

GitHub Pages cannot run this Python application. Deploy the private GitHub repository through Streamlit Community Cloud or another private-capable host, and restrict application access to authorized users.

Before deployment:

1. Create a hosted PostgreSQL database with a provider such as Supabase or Neon.
2. Copy its connection URL into the Streamlit app's secret settings as `DATABASE_URL`.
3. Add distinct `ADMIN_PASSWORD` and `USER_PASSWORD` values, rotated Alpaca paper credentials, and the remaining values from `.env.example` to Streamlit secrets.
4. Keep the GitHub repository private and deploy `streamlit_app.py` as the entry point.
5. Configure the host's access controls and confirm that only authorized users can reach the application before sharing it.

The production database must use TLS according to the database provider's connection instructions. SQLite is intended for local development only because a hosted Streamlit filesystem is not durable.

## Verification

```bash
.venv/bin/python scripts/verify_environment.py
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

Tests cover generic non-private startup, synthetic statement rebuilding, duplicate-safe imports,
runtime-path ignore guards, FIFO sales, cash ledger entries,
uncapped holdings, analytics, historical-data provenance and precedence, adjustment separation,
incremental refresh, CSV validation, proxy records, idempotent order-fill allocation, and a
credential-free Streamlit render.

The merged Phase 2 calculation-remediation evidence is preserved in the
[Phase 2 acceptance record](docs/manual-tests/phase-2-calculation-remediation.md).
The completed 127-row calculation audit, final evidence, limitations, and acceptance boundary are
preserved in the
[final calculation-audit acceptance record](docs/manual-tests/final-calculation-audit-acceptance.md).

## Deliberately deferred

The initial architecture models automated strategies, short positions, options, and live mode, but the following production features remain later work:

- Background strategy scheduling and automatic order submission
- Strategy-level risk budgets and emergency shutdown
- Option contract discovery and multi-leg order tickets
- Borrow checks and portfolio exposure limits for short sales
- Strong identity-provider login, roles, rate limiting, and a full audit console
- Live-trading operational review and activation

Paper trading is a simulation and does not reproduce every live-market condition. Review every order before submission.
