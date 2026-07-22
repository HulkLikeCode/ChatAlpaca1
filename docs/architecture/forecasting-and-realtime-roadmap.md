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

## Phase 4 reconstruction foundation

`chat_alpaca.reconstruction.PortfolioReconstructionService` is the shared typed boundary between
the canonical ledger, historical-data repository, and analytics consumers. It replays dated
transactions deterministically for each portfolio, produces a separate household aggregate, and
uses only confirmed split-adjusted, non-dividend-adjusted closes for accounting. Explicit ledger
dividends are therefore not duplicated through adjusted valuation prices.

Results include daily value, cash, positions, external flows, income and expense categories, return
attribution, TWR, XIRR, gain/loss, optional total-return benchmarks, common as-of status, missing and
stale symbols, assumptions, warnings, and forecast suitability. Missing observations remain unknown.
The sufficiency status scores history length, observation completeness, freshness, proxy use,
adjustment quality, and common-date completeness, retaining its components so the status cannot
imply more precision than the underlying data supports.

Forecast models and real-time streaming remain outside Phase 4.

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

The earlier session-only planning projection is separately identified as
`legacy_projection / 1.0.0`; it is not a bootstrap or parametric model. Its result contract retains
the assumptions, seed, simulation count, common confirmed source valuation date or disclosed
cost-basis fallback, valuation methodology, and generation timestamp. All numeric service inputs
must be finite and non-Boolean before paths are generated. Presentation month zero uses the explicit
source valuation date when available, otherwise the UTC generation date; modeled month `n` is
labeled at the end of the `n`th following calendar month. Annual calendar-year labels come from the
exact month-12 endpoints and never participate in simulation arithmetic.

## Phase 7 deterministic scenarios

`chat_alpaca.scenarios` is the reusable Phase 7 boundary. It applies fixed, explicit assumptions to
ledger-derived holdings and cash and refuses scenarios when required current or replay prices are
missing. Results retain household, internal-portfolio, holding, sector, and account-type effects,
largest loss contributors, assumptions, coverage, proxy warnings, and baseline comparisons.
Model `1.1.0` resolves DataFrame inputs at one common household valuation date and persists that
date, per-symbol source dates, and completeness counts. Mapping values form one undated explicit
snapshot and must be numeric, finite, positive, and non-boolean. Historical replay retains only
jointly complete required-symbol rows, requires two or more, persists shared endpoints and count,
and never forward-fills.

Saved deterministic runs contain summary outputs and deterministic baseline/scenario bands only;
there is no raw path storage. Market inputs are linked to immutable `market_datasets`, and a
canonical SHA-256 hash identifies the exact scoped transaction-ledger state. The generic validation
record and interface are intentionally usable by later stochastic models. Passing automated tests
may move a model into review, but only an explicit reviewed governance decision can label it
validated.

Historical block bootstrap and correlated parametric Monte Carlo remain outside Phase 7.

## Phase 8 historical block bootstrap

`chat_alpaca.bootstrap_forecasting` consumes monthly returns derived from the shared immutable
market-data coverage and can construct its starting holdings and cash from the shared reconstruction
result. It jointly samples holding and benchmark rows in circular 3-, 6-, or 12-month blocks,
preserving within-block time order and observed cross-asset dependence. Portfolio-level sampling is
also supported. The model bootstraps observed returns directly; it does not impose the legacy
planning projection's lognormal expected return.

The simulation supports deterministic seeds, configurable counts, 1–10 year horizons, monthly
contributions, constant inflation, annual fees, and monthly, quarterly, annual, or no rebalancing.
Boolean and nonfinite numerical assumptions and starting values are rejected before sampling or
nominal/real loss calculation.
Insufficient holding histories require an explicit proxy with adequate overlapping history. Proxy
substitution is disclosed and lowers sufficiency even when the joint history remains usable.

Outputs include monthly and annual percentile bands, a summarized terminal distribution, target
probability, downside percentiles, nominal and real contributed-capital loss probabilities,
probability of beating a jointly sampled benchmark, and average holding return-P&L contribution in
downside terminal outcomes. Assumptions and limitations define these measures explicitly.

Rolling-origin expanding-window backtests report 5th–95th interval coverage, median bias,
5th-percentile downside coverage, and valid, invalid, and insufficient windows. Meeting configured
criteria yields only `eligible_for_review`; it does not set model validation. Persistence reuses the
Phase 7 forecast-run and immutable dataset-reference records and saves reproducibility inputs,
coverage, proxy use, percentile bands, summarized terminal output, and optional backtest summaries.
Raw simulated paths and terminal samples are excluded by default.

## Phase 9 correlated parametric Monte Carlo

`chat_alpaca.parametric_forecasting` implements a second stochastic model alongside the historical
block bootstrap. It uses correlated monthly simple returns drawn from either a multivariate normal
distribution or a variance-normalized multivariate Student's t distribution with explicit degrees
of freedom. Both models expose compatible percentile bands, terminal summaries, target and loss
probabilities, benchmark comparison, downside percentiles, and holding-level downside attribution,
so their results can be shown in one comparison table without ranking either model universally.

Historical expected returns are annualized from mean log returns using `cross-sectional median
shrinkage`. They are not raw historical arithmetic means. Historical covariance uses `fixed
diagonal covariance shrinkage` before conversion to marginal volatilities and correlation.
Owner-entered or
CSV-imported published capital-market assumptions can be blended with historical return and
volatility estimates using explicit weights, while field-level user overrides take precedence.
Every non-historical assumption retains source, publication, and as-of disclosure. A paid data
source is neither required nor assumed.

External and override correlation matrices are explicitly checked for symbol labels and order,
dimensions, finite values, symmetry, unit diagonal, correlation bounds, and positive
semidefiniteness. Invalid matrices are rejected rather than silently repaired. Optional uncertainty
in expected-return estimates is propagated by simulation-level mean draws. Sensitivity tables vary
expected returns, volatility, and the strength of off-diagonal correlations; a separate comparison
shows normal and fat-tailed downside results.

Rolling-origin expanding-window backtests use the same interval coverage, median bias, downside
coverage, and valid/invalid/insufficient-window contract as Phase 8. Calibration results compare
models without claiming universal superiority, and criteria success yields only eligibility for
review. Generic Phase 7 forecast persistence already accommodates Phase 9 without a schema change:
saved runs include model/version, distribution, degrees of freedom, seed, assumptions, parameter
sources and estimates, shrinkage and covariance methods, dataset references, proxies, coverage,
validation evidence, percentile bands, and summarized output. Raw paths and terminal samples are
not saved.

Retirement cash-flow and transparent first-pass tax modeling begin in Phase 10.

## Phase 10 long-horizon retirement modeling

`chat_alpaca.retirement` composes the shared Phase 8 block-bootstrap and Phase 9 parametric return
engines with a separate monthly retirement cash-flow layer. It covers 20–40-year accumulation and
withdrawal horizons, retirement by age or date, monthly or annual contributions, fixed real
spending, one-time spending, inflation, fees, rebalancing, Social Security, pensions, other income,
and optional estate targets expressed in real dollars. It does not add guardrail or
percentage-spending strategies.

Account balances retain Traditional IRA, Roth IRA, taxable, and unknown classifications and may use
account-specific allocations. Contributions post at month-end; annual amounts are split into 12
equal deposits. Roth withdrawals assume qualified tax-free treatment, and Traditional IRAs assume
no nondeductible basis. Owner Traditional IRA RMDs use the date-of-birth-dependent federal starting
age, versioned IRS Publication 590-B Table III by default and Table II only for a sole-beneficiary
spouse more than 10 years younger, each IRA's prior December 31 balance, aggregate owner-IRA
satisfaction, and December month-end timing. Inherited-account RMDs remain out of scope.

Taxable basis is approximated as remaining FIFO security basis plus taxable cash and may exceed
market value; withdrawals reduce it proportionally. Social Security uses a fixed configured taxable
fraction rather than provisional-income rules. Unspent outside income and net RMD surplus remain
zero-return taxable household cash until the next configured rebalance. Model `1.3.0` keeps that
cash spendable: current net outside income and net RMD proceeds fund spending first, then prior
retained cash, then additional withdrawals. RMDs are calculated, executed, taxed, and deposited net
into household cash once before that additional-withdrawal loop. Retained cash remains in
retirement-date, path, and terminal household values. This is a transparent planning estimate, not
tax advice; brackets, deductions, filing status, state/local rules, detailed future lots, loss
harvesting, and unsupported jurisdiction-specific rules remain out of scope.

Results disclose funding and depletion probabilities, depletion ages, nominal and real percentile
paths, retirement-date value, lifetime taxes, withdrawals by account type, outside-income funding,
shortfalls, real target-estate probability, early-retirement sequence risk, and worst-decile
scenarios. Sensitivities vary retirement timing, spending, inflation, Social Security timing,
contributions, returns, taxes, and withdrawal order with a stable seed. Complete rolling historical
sequences can be replayed as backtest evidence. Neither replay nor automated tests grant validated
status. Generic forecast persistence stores reproducibility inputs, summarized results, and annual
bands without raw stochastic paths.
Household assets reconcile separately from unpaid shortfall. Gross account withdrawals are internal
transfers and therefore cancel from the household identity; withdrawal tax and spending actually
funded reduce household assets. Unpaid shortfall equals required obligations less obligations
actually funded and is not included in ending household value.

Model `1.3.0` refuses withdrawal-sensitive results while any in-scope account remains `unknown`;
the account must first be classified as taxable, Traditional IRA, or Roth IRA. Boolean and
nonfinite numerical profile, account, tax, spending-event, income, fee, and return-assumption inputs
are rejected. Each one-time event exposes its resolved nearest monthly model step and discloses that
intra-month timing is not modeled. Existing saved `1.2.0` results retain their original version.

## Real-time monitoring

The current roadmap targets fewer than 400 distinct held symbols, with an expected stock-to-ETF ratio near 6:1. Favor broad monitoring across held symbols over exact continuous streaming for a small subset.

Use a tiered active-session design:

- Stream the highest-priority symbols permitted by the Alpaca subscription.
- Refresh remaining held symbols through efficient batched snapshot requests.
- Prioritize open orders, visible symbols, selected portfolios, large positions, large risk contributors, active alerts, broad-market proxies, and sector proxies.
- Present whether a value is streamed, recently refreshed, stale, or previous close.
- Label IEX-derived real-time portfolio valuations as **indicative**, not consolidated-market values.

Persistent monitoring while the app is closed is out of scope for now. Streamlit is not a guaranteed persistent worker; interfaces should allow a future separate worker without requiring one in this phase.

## Phase 12 tiered active-session monitoring

`chat_alpaca.realtime` is the reusable monitoring boundary. It owns the 30-symbol Basic-plan
stream cap, ordered subscription planning, one WebSocket lifecycle per active browser-session
lease, bounded reconnect backoff, explicit disconnected-interval reconciliation, batched REST
snapshots, a shared sliding-window request limiter, market-hours-aware refresh cadence, duplicate
event rejection, and typed quote freshness/provenance. Selected or newly visible symbols receive an
immediate snapshot even when subscription changes are still connecting.

Intraday state is deliberately session-scoped and never updates holdings or accounting balances.
Durable split-adjusted closes provide previous-close fallback through the existing historical-data
repository. Abandoned leases are reaped on active reruns and explicit logout stops the connection;
there is no persistent background worker and no uninterrupted closed-app or sleeping-device claim.
Complete per-portfolio indicative quote changes may be overlaid on ledger-derived close-based
all-time and daily gain/loss without waiting for every selected portfolio to become fresh. The
overlay is separately labeled indicative with timestamp and provenance and never alters the
confirmed inception-through-latest-complete-valuation series. A
complete stale quote move may remain visible; muted blue applies only during regular market hours.
After-hours values may continue to reflect extended-hours trades against the previous completed
session close, while weekends and holidays retain the most recent completed session without stale
coloring. Portfolios without a calculable quote move retain their confirmed-close values. Mixed
aggregates disclose available, fresh, and fallback portfolio coverage. Custom gain/loss receives the
overlay only when its selected end date is the current day. This presentation overlay never mutates
the ledger or durable historical datasets.
Quote values follow latest trade, valid two-sided midpoint, valid previous close, then unavailable.
Boolean, nonfinite, zero, and negative values fall through without changing freshness or provenance
classification.

The Monitor view presents an indicative IEX portfolio pulse, holding and portfolio daily
contribution, symbol-consolidated largest movers, stale/missing symbols, assigned open orders and
recent fills, symbol quote/trade detail, and controlled broad-market and sector proxy components.
Daily, 1-month, 3-month, and 12-month returns require their full observation windows and disclose
counts and endpoints; shorter windows remain unavailable. Drawdown is labeled from the
available-window peak. The 50-day trend is a descriptive comparison of the final finite positive
level with the arithmetic mean of the final 50 usable dated levels; equality is labeled above, and
fewer than 50 levels is insufficient history. It is not a prediction or expanded technical-analysis
model. The primary correlation output uses exactly 21 complete aligned daily-return
pairs against SPY and discloses n/21 plus endpoints; missing SPY, insufficient pairs, zero variance,
or nonfinite correlation is unavailable. A secondary `high` at 0.70 or above and `mixed` otherwise
label is explicitly disclosed as a fixed descriptive heuristic, not a significance test. Phase 12
does not create a proprietary composite market score.
Cash-only selections retain their cash value. Holding share of net daily P/L is unavailable when
the absolute net daily P/L denominator is below $0.01; exactly $0.01 remains calculable.

## Benchmarks and hypothetical trades

Each internal portfolio should have a configurable benchmark blend, rather than only one benchmark symbol. Hypothetical trades remain separate from order submission unless the user explicitly transfers a proposal into an order.

Before-versus-after hypothetical analysis should address portfolio weights, cash, sector exposure, concentration, volatility, beta, risk contribution, drawdown exposure, expected return, forecast success probability, downside percentiles, and stress losses.

## Phase 11 hypothetical trade analysis

`chat_alpaca.hypothetical` is a non-executable domain boundary. It copies the selected
ledger-derived cash and FIFO lots, then applies multiple proposed buys, sells, cash additions or
removals, and internal portfolio assignment changes without mutating ORM portfolio state. Results
compare cash, market value, cost basis, holding and assignment weights, look-through sectors,
benchmark-relative exposure, concentration, effective holdings, volatility, beta, component risk,
historical drawdown, expected-return assumptions, forecast target probability and downside bands,
deterministic stress losses, and optional retirement success probability.
Largest and top-five weights use total household value including cash; HHI and effective holdings
use invested assets. Historical drawdown is a constant-weight exposure statistic, not the realized
path of current or proposed trades. Expected return gives uninvested cash a zero return, and the
current UI applies one common entered return to all involved securities.

Named scenario persistence retains creator and creation time, portfolio scope, the canonical
baseline ledger hash, market-data as-of time, assumptions, proposals, and summary results. Reads
recompute the scoped ledger hash and expose a stale-baseline warning. Hypothetical records have no
foreign-key or service path to transactions, lots, ledger entries, order allocations, or Alpaca
submission. A separate ticket-copy function requires explicit owner confirmation, an unchanged
baseline, and a newly reviewed price inside the freshness window; it creates review data only and
never submits an order.

Hypothetical model `1.1.0` values both baseline and proposed snapshots from one common confirmed
date across the selected household. Saved results retain that date, the confirmed prices, any
included latest-symbol dates, and explicit metadata distinguishing the additive confirmed layer
from the non-additive mixed-date monitoring overlay. Existing saved `1.0.0` results are read as
their original snapshots and are not reinterpreted. Action quantities, prices, fees, cash amounts,
baseline values, expected returns, targets, stress magnitudes, allocation inputs, and adjacent
retirement assumptions must be finite and non-Boolean before analysis.
