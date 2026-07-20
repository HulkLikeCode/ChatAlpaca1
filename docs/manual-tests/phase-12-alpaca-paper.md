# Phase 12 Alpaca paper manual tests

Use rotated paper credentials only. Keep `TRADING_MODE=paper` and
`ALLOW_LIVE_TRADING=false`. Run these checks while the US equity market is open and repeat the
freshness checks once outside regular hours.

## Setup

1. Set `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and `ALPACA_DATA_FEED=iex` locally.
2. Leave the default `REALTIME_STREAM_CAP=30`, `REALTIME_REGULAR_SECONDS=45`,
   `REALTIME_OFF_HOURS_SECONDS=180`, and `REALTIME_CALLS_PER_MINUTE=180` settings.
3. Start `python -m streamlit run streamlit_app.py`, authenticate, and open **Monitor**.
4. Confirm the page explicitly says values are indicative IEX values and that monitoring stops
   when the browser session closes or the machine sleeps.

## Streaming and REST tiers

1. Use a paper account/database scope containing more than 30 unique held symbols.
2. Confirm exactly 30 or fewer symbols are reported as streamed and all other held symbols are
   assigned to REST fallback.
3. Select a nonstreamed symbol. Confirm an immediate snapshot is requested and it moves into the
   streamed priority set on the next subscription update.
4. Change the selected portfolio scope and confirm visible/selected holdings refresh immediately.
5. During regular hours, observe nonstreamed held symbols refresh every configured 30–60 seconds.
6. Outside regular hours, confirm refreshes use the slower configured cadence.

## Freshness and valuation

1. For a liquid held symbol, compare latest trade, bid, ask, midpoint, spread, quote/trade/receipt
   timestamps, feed, and as-of time with the Alpaca paper dashboard. Confirm timestamps use the
   compact daylight-aware format `7/20 3:42 PM ET`.
2. Confirm streaming and recently refreshed mover rows display `Fresh`; delayed, previous-close,
   and unavailable rows display `Stale`. Confirm stale/missing symbols use the light-pink themed
   alert and retain a diagnostic reason.
3. Confirm Portfolio pulse includes indicative total value, daily change, holding and portfolio
   contribution, one combined largest-mover row per symbol, and stale/missing symbols without
   replacing missing values by zero. Confirm the mover table has no Portfolio column.
4. Confirm broad-market and sector rows disclose each return, trend, drawdown, volatility,
   correlation regime, and 21-day rolling SPY correlation component; confirm proxy dispersion is
   absent and there is no unexplained composite market score.
5. On Overview, confirm indicative all-time and daily gain/loss refresh with quotes. Confirm custom
   gain/loss refreshes only when Custom End is today and remains fixed for an earlier end date.

## Orders, fills, reconnect, and cleanup

1. Submit a small paper limit order away from market. Confirm its symbol receives top stream
   priority and appears under Open orders and in Symbol detail order state.
2. Cancel it and use **Sync fills** on the Trade page; confirm the status updates without a ledger
   fill transaction.
3. Submit a marketable paper order, sync fills, and confirm the incremental fill appears once in
   Recent fills and once in the assigned internal portfolio ledger. Sync again and confirm no
   duplicate transaction is created.
4. Disable networking briefly, then restore it. Confirm the connection reports reconnecting,
   reconnects without spawning duplicate sessions, and snapshot-backfills streamed symbols across
   the gap.
5. Rerun the app repeatedly and confirm only one `alpaca-active-session` thread exists for the
   browser session. Log out and confirm the connection stops. Close an unlogged-out browser tab,
   wait more than three minutes, then open/rerun another session and confirm the abandoned lease is
   cleaned up.
6. Review terminal and application logs. API keys and secret keys must never appear.

Record the date/time, paper account, market-hours state, symbol count, browser, observed cadences,
and any discrepancies. Do not use live-trading credentials for this test.
