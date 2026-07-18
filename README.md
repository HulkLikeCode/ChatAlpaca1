# ChatAlpaca1

A compact personal portfolio dashboard with public benchmark views, password-protected owner controls, automatic database persistence, and Alpaca order allocation. The first release is paper-first and keeps up to 20 internal portfolios separate even though Alpaca holds their combined positions in one brokerage account.

## Included

- Up to 20 portfolios with an uncapped number of tracked symbols
- Transaction-backed cash, FIFO lots, cost basis, and market value
- Seeded `KCs Traditional IRA`, `KCs Roth IRA`, and `KC and Papa` portfolios
- Buy-and-hold comparisons against SPY, QQQ, DIA, IWM, and arbitrary stock or ETF symbols
- Growth-of-$100 chart plus return, volatility, and drawdown statistics
- Password-protected portfolio editor and assigned Alpaca order ticket
- Manual transaction entry, brokerage CSV preview/import, duplicate protection, and rebuild-from-statement
- Combined sortable transaction management with portfolio/type/date filters, totals, and guarded edits/deletes
- Portfolio-scoped YTD, trailing-365-day, and custom-range dividend totals
- Transaction-aware all-time, daily, and custom-range portfolio gain/loss excluding contributions
- Consolidated exact holdings with weighted cost basis, symbol-level gain/loss, and lot drilldown
- Selected-portfolio total value in both Overview and Compare
- Market and limit orders, cancellation, fill synchronization, and an auditable internal transaction ledger
- Automatic persistence through SQLite locally or hosted PostgreSQL in production
- Architecture hooks for strategies, short positions, options, and separately gated live trading
- Dark black/blue/purple/white theme with no green or red status colors

## Safety model

The public link exposes the exact internally tracked portfolios but never exposes credentials or owner controls. `ADMIN_PASSWORD` unlocks owner features for a browser session.

Paper mode is the default. Live mode requires all three of the following:

1. Separate live Alpaca credentials.
2. `TRADING_MODE=live`.
3. `ALLOW_LIVE_TRADING=true`.

Do not enable live mode until authentication, risk limits, audit review, and deployment access controls have been strengthened. Never commit an API key, secret key, database password, or owner password.

## Local setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
streamlit run streamlit_app.py
```

Set newly rotated paper credentials in `.env`:

```dotenv
DATABASE_URL=sqlite:///data/chat_alpaca.db
ADMIN_PASSWORD=choose-a-long-password
ALPACA_API_KEY=your-rotated-paper-key
ALPACA_SECRET_KEY=your-rotated-paper-secret
ALPACA_DATA_FEED=iex
TRADING_MODE=paper
ALLOW_LIVE_TRADING=false
```

The ignored `data/chat_alpaca.db` file is created automatically. Seed data is inserted when the
portfolio table is empty, while versioned data migrations also run against existing databases
exactly once.

## Portfolio transactions

Owner controls support manual entries for buys, sells, dividends, interest, transfers, awards, fees,
taxes, and cash adjustments. Transactions are the source of truth: cash, FIFO lots, and ledger rows
can be rebuilt deterministically after a guarded service-level update or deletion. The seeded
Traditional and Roth holdings are recorded as cash-neutral opening-position transactions, and the
three initial cash balances are recorded as Phase 1 cash adjustments effective `5/15/26`.

Manage can show multiple portfolios in one sortable transaction table. Its portfolio, transaction
type, and date-range filters also drive the displayed type totals and grand total. Select a row to
edit or delete that transaction in its own portfolio; the separate **Target portfolio** selector is
used only for new entries and CSV imports. Cash changes are totaled by transaction type, while share
quantities are totaled separately by symbol and type. Transaction entry and posted dates use
`M/D/YY`.

Every transaction edit or deletion requires a transaction-specific typed confirmation. Imported,
seeded, and Alpaca-generated transactions show an additional divergence warning. Updates are marked
with the `manual_override` source, and immutable before/after audit snapshots retain the original
source and values; deletion audits retain the complete removed transaction snapshot.

## Portfolio valuation

Historical portfolio values are reconstructed from dated transactions and adjusted daily market
closes rather than projecting today's cash and holdings backward. Portfolio gain/loss excludes
external transfers, cash adjustments, and the cost basis of contributed opening positions; market
movement, dividends, interest, and awards remain part of performance. Daily gain/loss compares the
two latest market closes, while custom gain/loss uses the close before the selected start date
through the selected end date.

Exact Holdings combines the same symbol across every selected portfolio. It shows total shares,
weighted-average cost, total basis, market value, and all-time/daily/custom gain or loss. The
expandable portfolio/lot table preserves each acquisition date and original cost basis.

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

GitHub Pages cannot run this Python application. Deploy the private GitHub repository through Streamlit Community Cloud and make the Streamlit app public for view-only access.

Before deployment:

1. Create a hosted PostgreSQL database with a provider such as Supabase or Neon.
2. Copy its connection URL into the Streamlit app's secret settings as `DATABASE_URL`.
3. Add `ADMIN_PASSWORD`, rotated Alpaca paper credentials, and the remaining values from `.env.example` to Streamlit secrets.
4. Keep the GitHub repository private and deploy `streamlit_app.py` as the entry point.
5. Confirm the public view does not show Manage or Trade tabs before sharing the link.

The production database must use TLS according to the database provider's connection instructions. SQLite is intended for local development only because a hosted Streamlit filesystem is not durable.

## Verification

```bash
ruff format --check .
ruff check .
pytest -q
```

Tests cover seeded statement rebuilding, duplicate-safe imports, FIFO sales, cash ledger entries, uncapped holdings, analytics, idempotent order-fill allocation, and a credential-free Streamlit render.

## Deliberately deferred

The initial architecture models automated strategies, short positions, options, and live mode, but the following production features remain later work:

- Background strategy scheduling and automatic order submission
- Strategy-level risk budgets and emergency shutdown
- Option contract discovery and multi-leg order tickets
- Borrow checks and portfolio exposure limits for short sales
- Strong identity-provider login, roles, rate limiting, and a full audit console
- Live-trading operational review and activation

Paper trading is a simulation and does not reproduce every live-market condition. Review every order before submission.
