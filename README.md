# ChatAlpaca1 aka DashApp

A private, multi-portfolio personal portfolio manager that brings together portfolio-specific and broader-market analysis, current valuations, historical performance, and planning/forecasting tools in one dashboard. It supports internal portfolio tracking, transaction-ledger accounting, benchmarking, historical analysis, projections, automatic database persistence, and Alpaca order allocation. The first release is paper-first and keeps up to 20 internal portfolios separate even though Alpaca holds their combined positions in one brokerage account.

## Included

- Up to 20 portfolios with an uncapped number of tracked symbols
- Transaction-backed cash, FIFO lots, cost basis, and market value
- Durable, provider-neutral daily market-data cache with typed coverage and provenance
- Typed ledger reconstruction for daily values, positions, flows, returns, benchmarks, and coverage
- Seeded `KCs Traditional IRA`, `KCs Roth IRA`, and `KC and Papa` portfolios
- Owner-editable Traditional IRA, Roth IRA, taxable, and unknown account classifications
- Effective-dated per-portfolio benchmark blends with explicit rebalancing assumptions
- Cached security metadata and dated ETF sector look-through classifications with provenance
- Buy-and-hold comparisons against SPY, QQQ, DIA, IWM, and arbitrary stock or ETF symbols
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
- Separate admin and read-only user passwords; only admins can mutate data, upload files, or act on brokerage orders
- Manual transaction entry, brokerage CSV preview/import, duplicate protection, and rebuild-from-statement
- Combined sortable transaction management with portfolio/type/date filters, totals, and guarded edits/deletes
- Sticky, batched portfolio and master-date controls shared by Overview, Compare, and Manage
- Compact portfolio-income reporting for realized dividends and interest, with master-end-aware
  YTD, trailing-365-day, selected-range, normalized-quarterly, monthly, and source views
- Transaction-aware all-time, daily, and custom-range portfolio gain/loss excluding contributions
- Consolidated exact holdings with weighted cost basis, symbol-level gain/loss, and lot drilldown
- Selected-portfolio total value in both Overview and Compare
- Selected-range cumulative dividends on portfolio value cards
- Portfolio and holding Alpha/Beta against SPY total return, using at least 60 overlapping daily returns
- Market and limit orders, cancellation, fill synchronization, and an auditable internal transaction ledger
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
can be rebuilt deterministically after a guarded service-level update or deletion. The seeded
Traditional and Roth holdings are recorded as cash-neutral opening-position transactions, and the
three initial cash balances are recorded as Phase 1 cash adjustments effective `5/15/26`.

Manage shows the master-selected portfolios in one sortable transaction table. The master date
range and transaction-type filter also drive the displayed type totals and grand total. Select a
row to edit or delete that transaction in its own portfolio; separate target selectors are used for
new entries and CSV imports. Cash changes are totaled by transaction type, while share quantities
are totaled separately by symbol and type. Manage keeps Transactions first, followed by Add
Transaction, Brokerage CSV, and consolidated portfolio administration. Transaction entry and
posted dates use `M/D/YY`.

Every transaction edit or deletion requires a transaction-specific typed confirmation. Imported,
seeded, and Alpaca-generated transactions show an additional divergence warning. Updates are marked
with the `manual_override` source, and immutable before/after audit snapshots retain the original
source and values; deletion audits retain the complete removed transaction snapshot.

## Portfolio valuation

Historical portfolio values are reconstructed from dated transactions and adjusted daily market
closes rather than projecting today's cash and holdings backward. Portfolio gain/loss excludes
external transfers, cash adjustments, and the cost basis of contributed opening positions; market
movement, dividends, interest, and awards remain part of performance. Daily gain/loss compares the
two latest market closes, while custom gain/loss uses the close before the applied master start date
through the master end date. The same range controls dividend custom totals, Compare charts and
metrics, exact holdings, and transaction filtering.

Portfolio cards show cumulative dividend ledger credits in the inclusive applied master range;
interest is excluded. Portfolio and
holding Alpha/Beta use the applied master range and require at least 60 overlapping daily returns
against SPY total-return data. Alpha is the compounded annualized daily regression intercept.
Portfolio returns are ledger-aware. Holding returns add symbol-assigned dividend credits to
split-adjusted price returns; unassigned dividends and dividend events without an attributable
prior holding value are excluded and disclosed.

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

Phase 7 adds deterministic scenario analysis in `chat_alpaca.scenarios`. It supports broad-market,
holding, sector, dividend, contribution, inflation, low-return, lost-decade, retirement-date, and
historical-replay stresses plus tabular sensitivity grids. Runs persist the model/version, exact
ledger hash and dataset references, assumptions, coverage, proxy disclosure, validation state, and
summary outputs. Deterministic scenarios generate no raw paths. Automated test success is retained
as validation evidence but cannot by itself label a model validated.

Phase 8 adds historical block-bootstrap forecasting in `chat_alpaca.bootstrap_forecasting`. It
samples observed, jointly aligned monthly holding or portfolio returns in circular 3-, 6-, or
12-month blocks and does not impose an expected-return estimate. Runs support deterministic seeds,
configurable simulation counts, 1–10 year horizons, month-end contributions, constant inflation,
annual fees, monthly/quarterly/annual/no rebalancing, target and benchmark probabilities, nominal
and real loss probabilities, terminal distributions, downside percentiles, and holding-level
downside contribution. Holdings with inadequate histories require explicit documented proxies;
proxy use lowers forecast sufficiency and joint row sampling preserves the proxy's observed market
relationships.

Rolling-origin backtests report 5th–95th percentile interval coverage, median forecast bias,
downside-band coverage, and valid, invalid, and insufficient windows. Passing configured criteria
makes a model eligible for review, never automatically validated. Saved runs retain model/version,
seed, simulation count, block length, data period, immutable dataset references, proxies,
assumptions, backtest summaries, percentile bands, and summarized terminal distributions. Raw
simulated paths and terminal samples are not persisted.

Phase 9 adds correlated parametric Monte Carlo in `chat_alpaca.parametric_forecasting` alongside,
not in place of, the bootstrap model. Multivariate normal and variance-normalized multivariate
Student's t returns share the bootstrap model's core output contract. Expected returns use
annualized compounded log-return estimates with cross-sectional shrinkage rather than unquestioned
raw arithmetic means. Covariance estimates use diagonal shrinkage, and every external or override
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

Phase 10 adds long-horizon retirement planning in `chat_alpaca.retirement`. It reuses the Phase 8
historical block-bootstrap and Phase 9 correlated parametric return engines for 20–40-year
accumulation and withdrawal paths. Profiles support retirement by age or date, monthly or annual
pre-retirement contributions, fixed inflation-adjusted real spending, one-time spending events,
Social Security, pensions, other outside income, fees, rebalancing, and an optional real target
estate value.

Traditional IRA, Roth IRA, taxable, and unknown balances remain explicit. Configurable withdrawal
ordering applies tax-free Roth, ordinary-income Traditional IRA, taxable realization/capital-gain,
dividend, and conservative unknown-account assumptions. The tax calculation is a transparent first
planning estimate, not tax advice or a tax-return calculation. It intentionally excludes tax
brackets, deductions, required minimum distributions, state/local rules, filing status, detailed
tax lots, loss harvesting, and jurisdiction-specific complexity. Social Security uses a configured
taxable fraction rather than provisional-income rules.

Outputs include nominal and real percentile paths, retirement-date and terminal distributions,
full-horizon funding and depletion probabilities, depletion ages, lifetime taxes, withdrawals by
account type, outside-income funding, shortfalls, target-estate probability, early-retirement
sequence diagnostics, and worst-decile scenarios. Sensitivity covers retirement age, spending,
inflation, Social Security timing, contributions, expected returns, tax rates, and withdrawal order.
Rolling historical sequence replay is validation evidence only and never self-validates the model.
Saved runs retain summaries and annual bands but exclude raw paths and per-scenario arrays.

Exact Holdings combines the same symbol across every selected portfolio. It shows total shares to
two decimals, weighted-average cost, total basis, market value, and all-time/daily/custom gain or
loss. Its Summary and By Portfolio / Lot views preserve each acquisition date and original cost
basis without nesting collapsible sections.

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

Brokerage CSV imports preview every row before posting. The included `KC and Papa.csv` format supports its current buy, sell, dividend, interest, transfer, award, fee, and foreign-tax rows. Re-importing a statement skips transactions already recorded, while the **Rebuild portfolio from statement** action deliberately replaces that portfolio's lots, cash, transaction history, legacy ledger rows, and saved Alpaca allocations after the owner types `REBUILD`.

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

Tests cover seeded statement rebuilding, duplicate-safe imports, FIFO sales, cash ledger entries,
uncapped holdings, analytics, historical-data provenance and precedence, adjustment separation,
incremental refresh, CSV validation, proxy records, idempotent order-fill allocation, and a
credential-free Streamlit render.

## Deliberately deferred

The initial architecture models automated strategies, short positions, options, and live mode, but the following production features remain later work:

- Background strategy scheduling and automatic order submission
- Strategy-level risk budgets and emergency shutdown
- Option contract discovery and multi-leg order tickets
- Borrow checks and portfolio exposure limits for short sales
- Strong identity-provider login, roles, rate limiting, and a full audit console
- Live-trading operational review and activation

Paper trading is a simulation and does not reproduce every live-market condition. Review every order before submission.
