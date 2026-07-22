from __future__ import annotations

import hmac
from datetime import date, timedelta, timezone
from html import escape
from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chat_alpaca.analytics import (
    adaptive_share_number_format,
    consolidated_holdings,
    household_valuation,
)
from chat_alpaca.bootstrap import initialize_application
from chat_alpaca.classification import (
    resolve_etf_sector_snapshot,
    resolve_security_metadata,
    security_symbol_labels,
)
from chat_alpaca.commands import (
    TransactionCommand,
    build_transaction_draft,
    calculated_trade_cash,
    transaction_kind_label,
    validate_transaction_symbol,
)
from chat_alpaca.config import get_settings
from chat_alpaca.db import session_scope
from chat_alpaca.forecasting import (
    ForecastAssumptions,
    ForecastRequest,
    ProjectionResult,
    build_forecast_request,
    run_forecast,
)
from chat_alpaca.hypothetical import (
    HypotheticalActionType,
    HypotheticalAssumptions,
    ProposedAction,
    RetirementAnalysisAssumptions,
    analyze_hypothetical_scenario,
    baseline_from_portfolios,
    load_hypothetical_scenarios,
    save_hypothetical_scenario,
)
from chat_alpaca.market_calendar import format_eastern_timestamp
from chat_alpaca.models import OrderAllocation, Portfolio, PortfolioTransaction
from chat_alpaca.portfolio_configuration import (
    ACCOUNT_TYPE_LABELS,
    ACCOUNT_TYPES,
    REBALANCING_FREQUENCIES,
    benchmark_configurations,
    parse_benchmark_components,
    save_benchmark_configuration,
    set_account_type,
)
from chat_alpaca.portfolio_service import (
    MANUAL_KINDS,
    create_portfolio,
    delete_portfolio,
    delete_transaction,
    format_short_date,
    import_statement,
    list_transactions_for_portfolios,
    parse_short_date,
    parse_statement_csv,
    portfolio_income_events,
    portfolio_income_summary,
    rebuild_portfolio_from_csv,
    record_transaction,
    rename_portfolio,
    update_transaction,
)
from chat_alpaca.realtime import (
    BROAD_MARKET_PROXIES,
    CORRELATION_HEURISTIC_DISCLOSURE,
    OPEN_ORDER_STATUSES,
    SECTOR_PROXIES,
    ActiveSessionMonitor,
    ActiveSessionRefreshScheduler,
    ActiveSessionRegistry,
    AlpacaWebSocketSession,
    FreshnessStatus,
    HistoricalGapBackfiller,
    QuoteBook,
    SlidingWindowRateLimiter,
    SnapshotBatcher,
    SubscriptionInputs,
    alpaca_clients,
    build_portfolio_pulse,
    market_context_metrics,
    market_hours_state,
    position_risk_contributions,
)
from chat_alpaca.reports import (
    HistoricalDataRequest,
    acquire_historical_data,
    assemble_combined_performance_report,
    assemble_comparison_report,
    assemble_portfolio_card_reports,
    comparison_acquisition_plan,
    overlay_intraday_performance,
    portfolio_acquisition_request,
)
from chat_alpaca.scenarios import (
    DatasetReference,
    ScenarioAssumptions,
    ScenarioType,
    run_deterministic_scenario,
    save_scenario_run,
    sensitivity_grid,
)
from chat_alpaca.theme import PLOT_COLORS, THEME_CSS
from chat_alpaca.trading import (
    cancel_order,
    get_trading_client,
    list_allocations,
    submit_allocated_order,
    sync_allocations,
)

BENCHMARKS = {
    "SPY": "S&P 500 large-cap U.S. equity benchmark ETF",
    "QQQ": "Nasdaq-100 large growth and technology-heavy benchmark ETF",
    "IWM": "Russell 2000 small-cap U.S. equity benchmark ETF",
    "DIA": "Dow Jones Industrial Average blue-chip U.S. equity benchmark ETF",
    "VTI": "Total U.S. stock market benchmark ETF",
    "VT": "Vanguard total world stock market benchmark ETF",
    "EFA": "Developed markets ex-U.S. equity benchmark ETF",
    "EEM": "Emerging markets equity benchmark ETF",
    "AGG": "U.S. aggregate bond market benchmark ETF",
    "BND": "Total U.S. bond market benchmark ETF",
    "TLT": "Long-term U.S. Treasury bond benchmark ETF",
    "IEF": "Intermediate-term U.S. Treasury bond benchmark ETF",
    "SHY": "Short-term U.S. Treasury bond benchmark ETF",
    "LQD": "Investment-grade U.S. corporate bond benchmark ETF",
}
ALL_PORTFOLIOS_OPTION = "__all_portfolios__"
SELECT_PORTFOLIO_OPTION = "__select_portfolio__"
EDITABLE_KINDS = (*MANUAL_KINDS, "opening_position")
TRADE_KINDS = {"buy", "sell"}
SYMBOL_CASH_KINDS = {"dividend", "fee", "tax"}
STALE_VALUE_COLOR = "#86A7D8"
MASTER_DEFAULT_START = date(2026, 5, 15)
CSV_TEMPLATE = """Date,Action,Symbol,Description,Quantity,Price,Fees & Comm,Amount
7/15/2026,Buy,AAPL,Apple Inc,10,$210.00,$0.00,"($2,100.00)"
7/16/2026,Cash Dividend,AAPL,Apple dividend,,,,15.50
7/17/2026,MoneyLink Transfer,,Cash contribution,,,,1000.00
"""

st.set_page_config(
    page_title="Let’s Go Blue!",
    page_icon=str(Path(__file__).with_name("assets") / "favicon.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(THEME_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def cached_historical_data(request: HistoricalDataRequest) -> pd.DataFrame:
    return acquire_historical_data(request)


@st.cache_data(show_spinner=False)
def cached_projection(
    request: ForecastRequest,
) -> ProjectionResult:
    return run_forecast(request)


@st.cache_resource(show_spinner=False)
def active_session_registry() -> ActiveSessionRegistry:
    return ActiveSessionRegistry()


def _new_active_monitor() -> ActiveSessionMonitor:
    settings = get_settings()
    client, stream_factory = alpaca_clients(
        settings.alpaca_api_key, settings.alpaca_secret_key, settings.alpaca_data_feed
    )
    book = QuoteBook()
    snapshots = SnapshotBatcher(
        client,
        book,
        feed=settings.alpaca_data_feed,
        limiter=SlidingWindowRateLimiter(settings.realtime_calls_per_minute),
    )
    websocket = AlpacaWebSocketSession(
        stream_factory,
        book,
        backfill=HistoricalGapBackfiller(snapshots.refresh),
        feed=settings.alpaca_data_feed,
    )
    return ActiveSessionMonitor(
        websocket,
        snapshots,
        ActiveSessionRefreshScheduler(
            settings.realtime_regular_seconds,
            settings.realtime_off_hours_seconds,
        ),
        book,
        stream_cap=settings.realtime_stream_cap,
    )


def dollars(value: object) -> str:
    return f"${float(value):,.0f}"


def whole_dollars(value: object) -> str:
    return f"${float(value):,.0f}"


def freshness_label(status: FreshnessStatus) -> str:
    return (
        "Fresh"
        if status in {FreshnessStatus.STREAMING, FreshnessStatus.RECENTLY_REFRESHED}
        else "Stale"
    )


def quantity(value: object) -> str:
    numeric = float(value)
    return f"{numeric:,.2f}"


def kind_label(value: str) -> str:
    return transaction_kind_label(value)


def portfolio_has_data(portfolio: Portfolio) -> bool:
    """Return whether a portfolio has reportable cash, holdings, or activity."""
    return bool(portfolio.holdings or portfolio.transactions or float(portfolio.cash) != 0)


def render_master_controls(
    portfolios: list[Portfolio],
) -> tuple[list[Portfolio], date, date]:
    """Render batched global reporting filters and return their applied state."""
    today = date.today()
    available_ids = [portfolio.id for portfolio in portfolios]
    available_id_set = set(available_ids)

    applied_ids = [
        portfolio_id
        for portfolio_id in st.session_state.get("master_portfolio_ids", available_ids)
        if portfolio_id in available_id_set
    ]
    applied_all = set(applied_ids) == available_id_set

    applied_start = st.session_state.get("master_start_date", min(MASTER_DEFAULT_START, today))
    applied_end = st.session_state.get("master_end_date", today)
    if applied_start > applied_end:
        applied_start = applied_end

    default_draft_ids = [ALL_PORTFOLIOS_OPTION] if applied_all else applied_ids
    draft_ids = [
        portfolio_id
        for portfolio_id in st.session_state.get("master_portfolio_draft", default_draft_ids)
        if portfolio_id == ALL_PORTFOLIOS_OPTION or portfolio_id in available_id_set
    ]
    st.session_state.master_portfolio_draft = draft_ids
    st.session_state.setdefault("master_start_draft", applied_start)
    st.session_state.setdefault("master_end_draft", applied_end)

    id_to_portfolio = {portfolio.id: portfolio for portfolio in portfolios}
    with st.container(key="master_controls"):
        with st.form("master_filters", border=False):
            columns = st.columns([3.2, 1.35, 1.35, 0.9], vertical_alignment="bottom")
            scope_label = "All Portfolios" if applied_all else f"{len(applied_ids)} portfolios"
            portfolio_label = (
                f"Portfolios · Applied: {scope_label} · "
                f"{applied_start:%m/%d/%y}–{applied_end:%m/%d/%y}"
            )
            selected_ids = columns[0].multiselect(
                portfolio_label,
                [ALL_PORTFOLIOS_OPTION, *available_ids],
                key="master_portfolio_draft",
                format_func=lambda portfolio_id: (
                    "All Portfolios"
                    if portfolio_id == ALL_PORTFOLIOS_OPTION
                    else id_to_portfolio[portfolio_id].name
                ),
                placeholder="Choose portfolios",
            )
            master_start = columns[1].date_input(
                "Custom Start",
                key="master_start_draft",
                max_value=today,
                format="MM/DD/YYYY",
            )
            master_end = columns[2].date_input(
                "Custom End",
                key="master_end_draft",
                max_value=today,
                format="MM/DD/YYYY",
            )
            applied = columns[3].form_submit_button("Apply", width="stretch", type="primary")

        validation_message = None
        if applied:
            all_selected = ALL_PORTFOLIOS_OPTION in selected_ids
            next_ids = available_ids if all_selected else list(selected_ids)
            if portfolios and not next_ids:
                validation_message = "Select at least one portfolio."
            elif master_start > master_end:
                validation_message = "Custom Start must be on or before Custom End."
            else:
                applied_ids = next_ids
                applied_start = master_start
                applied_end = master_end
                st.session_state.master_portfolio_ids = applied_ids
                st.session_state.master_start_date = applied_start
                st.session_state.master_end_date = applied_end
        if validation_message:
            st.error(validation_message)

    selected = [portfolio for portfolio in portfolios if portfolio.id in set(applied_ids)]
    return selected, applied_start, applied_end


def get_prices(
    portfolios: list[Portfolio], report_start: date, report_end: date
) -> tuple[pd.DataFrame, str | None]:
    inception_candidates = [
        transaction.transaction_date
        for portfolio in portfolios
        for transaction in portfolio.transactions
    ] + [lot.acquired_on for portfolio in portfolios for lot in portfolio.holdings]
    confirmed_start = min([report_start, *inception_candidates])
    request = portfolio_acquisition_request(portfolios, confirmed_start, date.today())
    if not request.symbols:
        return pd.DataFrame(), None
    try:
        return cached_historical_data(request), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def get_alpha_beta_benchmark(
    report_start: date, report_end: date
) -> tuple[pd.DataFrame, str | None]:
    request = HistoricalDataRequest(
        ("SPY",),
        report_start - timedelta(days=7),
        report_end,
        "benchmark_total_return",
    )
    try:
        return cached_historical_data(request), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def authenticate_access() -> str | None:
    """Authenticate this browser session as an admin or read-only user."""
    settings = get_settings()
    role = st.session_state.get("access_role")
    if st.session_state.get("owner_authenticated"):
        role = "admin"
        st.session_state.access_role = role
    if role in {"admin", "user"}:
        with st.sidebar:
            label = "Admin" if role == "admin" else "Read-only user"
            st.markdown("### Access")
            st.caption(f"{label} session active.")
            if st.button("Log out", width="stretch"):
                session_id = st.session_state.get("realtime_session_id")
                if session_id:
                    active_session_registry().release(session_id)
                st.session_state.access_role = None
                st.session_state.owner_authenticated = False
                st.rerun()
        return role

    if not settings.admin_password and not settings.user_password:
        st.error("Set ADMIN_PASSWORD and USER_PASSWORD to enable application access.")
        return None
    if (
        settings.admin_password
        and settings.user_password
        and hmac.compare_digest(settings.admin_password, settings.user_password)
    ):
        st.error("ADMIN_PASSWORD and USER_PASSWORD must be different.")
        return None

    st.markdown("### Sign in")
    st.caption("Enter the admin or read-only user password.")
    with st.form("access_login", border=True):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", width="stretch", type="primary")
    if not submitted:
        return None
    if settings.admin_password and hmac.compare_digest(password, settings.admin_password):
        role = "admin"
    elif settings.user_password and hmac.compare_digest(password, settings.user_password):
        role = "user"
    else:
        st.error("That password did not match.")
        return None
    st.session_state.access_role = role
    st.session_state.owner_authenticated = role == "admin"
    st.rerun()
    return None


def render_access_status(role: str) -> None:
    """Describe the active role without exposing credentials."""
    with st.sidebar:
        if role == "user":
            st.caption("Viewing and forecasts are enabled. Permanent changes are disabled.")


def render_portfolio_cards(
    portfolios: list[Portfolio], closes: pd.DataFrame, custom_start: date, custom_end: date
) -> None:
    cards = []
    for report in assemble_portfolio_card_reports(portfolios, closes, custom_start, custom_end):
        cards.append(
            "".join(
                (
                    '<div class="portfolio-card">',
                    f'<div class="eyebrow">{escape(report.name)}</div>',
                    f'<div class="value">{report.value_label}</div>',
                    f'<div class="detail">Cash: {whole_dollars(report.cash)} · '
                    f"CDT Div: {whole_dollars(report.cumulative_dividends)}</div>",
                    "</div>",
                )
            )
        )
    st.markdown(
        f'<div class="portfolio-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _metric_dollars(value: float | None) -> str:
    return dollars(value) if value is not None else "—"


def _render_metric(
    column: object,
    label: str,
    value: object,
    *,
    key: str,
    stale: bool = False,
) -> None:
    """Render a metric with the theme's quiet stale-value treatment when needed."""
    with column:
        if stale:
            with st.container(key=f"stale_metric_{key}"):
                st.metric(label, value)
        else:
            st.metric(label, value)


def _render_warning(message: str, *, subdued: bool = False) -> None:
    if subdued or message.startswith("Historical data note:"):
        st.caption(message)
    else:
        st.warning(message)


def _style_stale_values(
    frame: pd.DataFrame,
    stale_rows: pd.Series,
    columns: tuple[str, ...],
) -> object:
    """Color only stale numeric cells while preserving the surrounding table styling."""
    styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
    available = [column for column in columns if column in frame]
    styles.loc[stale_rows.fillna(False), available] = f"color: {STALE_VALUE_COLOR}"
    return frame.style.apply(lambda _: styles, axis=None)


def _indicative_performance_pulse(portfolios: list[Portfolio], closes: pd.DataFrame):
    settings = get_settings()
    symbols = sorted({lot.symbol for portfolio in portfolios for lot in portfolio.holdings})
    if not settings.alpaca_configured or not symbols:
        return None
    previous_closes = _latest_closes(closes)
    position_values: dict[str, float] = {}
    for portfolio in portfolios:
        for lot in portfolio.holdings:
            position_values[lot.symbol] = position_values.get(lot.symbol, 0.0) + float(
                lot.shares
            ) * previous_closes.get(lot.symbol, float(lot.cost_basis))
    session_id = st.session_state.setdefault("realtime_session_id", uuid4().hex)
    monitor = active_session_registry().acquire(session_id, _new_active_monitor)
    try:
        monitor.refresh(
            SubscriptionInputs(
                held_symbols=frozenset(symbols),
                visible_symbols=frozenset(symbols[:12]),
                selected_portfolio_symbols=frozenset(symbols),
                position_values=position_values,
            ),
            previous_closes=previous_closes,
        )
        return build_portfolio_pulse(portfolios, monitor.records(symbols))
    except Exception:
        return None


@st.fragment(run_every="30s")
def render_performance_summary(
    portfolios: list[Portfolio],
    closes: pd.DataFrame,
    custom_start: date,
    custom_end: date,
    key_prefix: str,
    benchmark_closes: pd.DataFrame,
    expanded: bool = False,
    show_portfolio_cards: bool = False,
    data_note: str | None = None,
) -> None:
    with st.expander("Portfolio value and gain/loss", expanded=expanded):
        regular_market_hours = market_hours_state().is_regular_hours
        report = assemble_combined_performance_report(
            portfolios,
            closes,
            custom_start,
            custom_end,
            benchmark_closes,
        )
        pulse = _indicative_performance_pulse(portfolios, closes)
        fresh_portfolios = (
            {name for name, fresh in pulse.portfolio_freshness.items() if fresh}
            if pulse is not None
            else set()
        )
        available_portfolios = (
            {name for name, change in pulse.by_portfolio.items() if change is not None}
            if pulse is not None
            else set()
        )
        live_count = len(fresh_portfolios)
        available_count = len(available_portfolios)
        overlay_is_fresh = bool(pulse is not None and live_count == len(portfolios))
        if pulse is not None:
            report = overlay_intraday_performance(
                report,
                dict(pulse.by_portfolio),
                include_custom=custom_end == date.today(),
                indicative_total_value=pulse.indicative_total_value,
            )

        metrics = st.columns(6)
        _render_metric(
            metrics[0],
            report.total_label,
            dollars(report.total_value) if report.total_value is not None else "—",
            key=f"{key_prefix}_selected_totals",
            stale=bool(
                regular_market_hours
                and pulse is not None
                and pulse.indicative_total_value is not None
                and not overlay_is_fresh
            ),
        )
        _render_metric(
            metrics[1],
            (
                "All-time gain/loss (indicative overlay)"
                if pulse is not None and available_count > 0
                else "All-time gain/loss (confirmed)"
            ),
            _metric_dollars(report.all_time),
            key=f"{key_prefix}_all_time",
            stale=bool(
                regular_market_hours
                and pulse is not None
                and available_count > 0
                and not overlay_is_fresh
            ),
        )
        _render_metric(
            metrics[2],
            "Daily gain/loss",
            _metric_dollars(report.daily),
            key=f"{key_prefix}_daily",
            stale=bool(
                regular_market_hours
                and pulse is not None
                and not overlay_is_fresh
                and report.daily is not None
            ),
        )
        _render_metric(
            metrics[3],
            "Custom gain/loss",
            _metric_dollars(report.custom),
            key=f"{key_prefix}_custom",
            stale=bool(
                regular_market_hours
                and pulse is not None
                and custom_end == date.today()
                and available_count > 0
                and not overlay_is_fresh
            ),
        )
        metrics[4].metric(
            "Annualized market-model intercept, RF assumed 0%",
            f"{report.alpha:.2%}" if report.alpha is not None else "—",
        )
        metrics[5].metric("Beta", f"{report.beta:.2f}" if report.beta is not None else "—")
        if show_portfolio_cards:
            render_portfolio_cards(portfolios, closes, custom_start, custom_end)

        status_parts = [report.coverage.removesuffix(".")]
        if closes.empty:
            status_parts.append("Alpha/Beta unavailable")
        else:
            status_parts.append(
                f"Alpha/Beta {report.alpha_beta_observations}/60 overlapping SPY returns"
            )
        if pulse is None:
            status_parts.append("Live quotes unavailable")
        elif overlay_is_fresh:
            status_parts.append(f"Live daily {live_count}/{len(portfolios)} · 30s refresh")
        else:
            quote_status = (
                f"Daily quotes {available_count}/{len(portfolios)} · "
                f"fresh {live_count}/{len(portfolios)}"
            )
            if available_count < len(portfolios):
                quote_status += f" · {len(portfolios) - available_count} confirmed-close fallback"
            status_parts.append(quote_status + " · 30s refresh")
        if pulse is not None and available_count > 0:
            timestamp = (
                format_eastern_timestamp(pulse.indicative_as_of)
                if pulse.indicative_as_of is not None
                else "timestamp unavailable"
            )
            status_parts.append(
                f"Indicative overlay: {pulse.indicative_provenance or 'quote source unavailable'} "
                f"· {timestamp}"
            )
        if custom_end != date.today():
            status_parts.append(f"Custom fixed at {custom_end:%-m/%-d/%y}")
        st.markdown(
            '<div class="performance-status">'
            + "".join(f"<span>{escape(item)}</span>" for item in status_parts)
            + "</div>",
            unsafe_allow_html=True,
        )
        if data_note:
            st.info(f"Live market values are unavailable, so cost basis is shown. {data_note}")
        for warning in report.warnings:
            _render_warning(warning, subdued=closes.empty)
        performance = pd.DataFrame(
            [
                {
                    "Portfolio": row.portfolio,
                    "Cash": row.cash,
                    "All-time gain/loss": row.all_time,
                    "Daily gain/loss": row.daily,
                    "Custom gain/loss": row.custom,
                    "Annualized market-model intercept, RF assumed 0%": (
                        row.alpha * 100 if row.alpha is not None else None
                    ),
                    "Beta": row.beta,
                    "Observations": row.alpha_beta_observations,
                    "_stale": regular_market_hours
                    and pulse is not None
                    and row.portfolio not in fresh_portfolios,
                }
                for row in report.rows
            ]
        )
        stale_performance_rows = performance.pop("_stale")
        performance_display = (
            _style_stale_values(
                performance,
                stale_performance_rows,
                ("All-time gain/loss", "Daily gain/loss", "Custom gain/loss"),
            )
            if regular_market_hours and pulse is not None
            else performance
        )
        st.dataframe(
            performance_display,
            hide_index=True,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="$%,.0f")
                for column in (
                    "Cash",
                    "All-time gain/loss",
                    "Daily gain/loss",
                    "Custom gain/loss",
                )
            }
            | {
                "Annualized market-model intercept, RF assumed 0%": (
                    st.column_config.NumberColumn(format="%.2f%%")
                ),
                "Beta": st.column_config.NumberColumn(format="%.2f"),
                "Observations": st.column_config.NumberColumn(format="%d"),
            },
            key=f"{key_prefix}_portfolio_gain_loss",
        )
        st.caption(
            "Confirmed all-time gain/loss runs from inception through the latest complete confirmed "
            "valuation date and is independent of Custom End. Any live amount is separately labeled "
            "as an indicative overlay with quote provenance and time. Gain/loss excludes transfers, "
            "cash adjustments, awards, and contributed opening positions. "
            "Daily uses the two latest market closes. Alpha/Beta uses daily ledger-aware returns "
            "against SPY total return over the applied range and requires 60 overlapping days."
        )


def render_consolidated_holdings(
    portfolios: list[Portfolio],
    closes: pd.DataFrame,
    custom_start: date,
    custom_end: date,
    benchmark_closes: pd.DataFrame,
) -> None:
    with st.expander("Exact holdings", expanded=False):
        summary, detail = consolidated_holdings(
            portfolios, closes, custom_start, custom_end, benchmark_closes
        )
        if summary.empty:
            st.caption("No holdings yet.")
            return

        view = st.radio(
            "Holdings view",
            ["Summary", "By portfolio / lot"],
            horizontal=True,
            key="holdings_view",
            label_visibility="collapsed",
        )
        common_renames = {
            "Average cost / share": "Avg/share",
            "Cost / share": "Avg/share",
            "Total cost basis": "Cost basis",
            "All-time gain/loss": "Unrealized gain/loss",
            "Daily gain/loss": "Latest close change",
            "Daily price dates": "Change dates",
            "Custom gain/loss": "Current-lot unrealized custom change",
            "Alpha": "Annualized market-model intercept, RF assumed 0%",
        }
        money_columns = (
            "Avg/share",
            "Confirmed price",
            "Latest symbol price",
            "Cost basis",
            "Confirmed value",
            "Latest/indicative value",
            "Unrealized gain/loss",
            "Latest close change",
            "Current-lot unrealized custom change",
        )
        if view == "Summary":
            summary_columns = [
                "Symbol",
                "Avg/share",
                "Confirmed valuation date",
                "Confirmed price",
                "Confirmed value",
                "Latest symbol price",
                "Latest symbol date",
                "Latest/indicative value",
                "Cost basis",
                "Unrealized gain/loss",
                "Latest close change",
                "Change dates",
                "Current-lot unrealized custom change",
                "Annualized market-model intercept, RF assumed 0%",
                "Beta",
                "Alpha/Beta observations",
                "Shares",
                "Portfolios",
            ]
            summary_view = summary.rename(columns=common_renames)[summary_columns].copy()
            summary_view["Annualized market-model intercept, RF assumed 0%"] *= 100
            st.dataframe(
                summary_view,
                hide_index=True,
                width="stretch",
                column_order=summary_columns,
                column_config={
                    "Shares": st.column_config.NumberColumn(
                        format=adaptive_share_number_format(summary_view["Shares"])
                    ),
                    "Annualized market-model intercept, RF assumed 0%": (
                        st.column_config.NumberColumn(format="%.2f%%")
                    ),
                    "Beta": st.column_config.NumberColumn(format="%.2f"),
                    "Confirmed valuation date": st.column_config.DateColumn(format="M/D/YY"),
                    "Latest symbol date": st.column_config.DateColumn(format="M/D/YY"),
                    **{
                        column: st.column_config.NumberColumn(format="$%,.0f")
                        for column in money_columns
                    },
                    "Avg/share": st.column_config.NumberColumn(format="$%,.2f"),
                    "Confirmed price": st.column_config.NumberColumn(format="$%,.2f"),
                    "Latest symbol price": st.column_config.NumberColumn(format="$%,.2f"),
                },
            )
        else:
            detail_columns = [
                "Symbol",
                "Avg/share",
                "Confirmed valuation date",
                "Confirmed price",
                "Confirmed value",
                "Latest symbol price",
                "Latest symbol date",
                "Latest/indicative value",
                "Cost basis",
                "Unrealized gain/loss",
                "Latest close change",
                "Change dates",
                "Current-lot unrealized custom change",
                "Annualized market-model intercept, RF assumed 0%",
                "Beta",
                "Alpha/Beta observations",
                "Shares",
                "Portfolio",
                "Acquired",
            ]
            detail_view = detail.rename(columns=common_renames)[detail_columns].copy()
            detail_view["Annualized market-model intercept, RF assumed 0%"] *= 100
            st.dataframe(
                detail_view,
                hide_index=True,
                width="stretch",
                column_order=detail_columns,
                column_config={
                    "Shares": st.column_config.NumberColumn(
                        format=adaptive_share_number_format(detail_view["Shares"])
                    ),
                    "Annualized market-model intercept, RF assumed 0%": (
                        st.column_config.NumberColumn(format="%.2f%%")
                    ),
                    "Beta": st.column_config.NumberColumn(format="%.2f"),
                    "Acquired": st.column_config.DateColumn(format="M/D/YY"),
                    "Confirmed valuation date": st.column_config.DateColumn(format="M/D/YY"),
                    "Latest symbol date": st.column_config.DateColumn(format="M/D/YY"),
                    **{
                        column: st.column_config.NumberColumn(format="$%,.0f")
                        for column in money_columns
                    },
                    "Avg/share": st.column_config.NumberColumn(format="$%,.2f"),
                    "Confirmed price": st.column_config.NumberColumn(format="$%,.2f"),
                    "Latest symbol price": st.column_config.NumberColumn(format="$%,.2f"),
                },
            )
        insufficient = summary[summary["Alpha"].isna()]
        st.caption(
            "Holding Alpha/Beta uses daily price returns plus symbol-assigned ledger dividends "
            "against SPY total return. Unassigned dividends are excluded. "
            f"{len(insufficient)} holding(s) lack the required 60 overlapping daily returns."
        )
        st.caption(
            "Current-lot unrealized custom change uses only open lots and price movement. It "
            "excludes sold lots and income and is not the portfolio Custom gain/loss measure."
        )
        st.caption(
            "Monitoring overlay — mixed-date values are non-additive unless all symbol dates "
            "match. Latest/indicative values are shown per symbol and are not totaled."
        )


def render_portfolio_income(
    portfolios: list[Portfolio], custom_start: date, custom_end: date
) -> None:
    portfolio_ids = [portfolio.id for portfolio in portfolios]
    with session_scope() as session:
        summary = portfolio_income_summary(session, portfolio_ids, custom_start, custom_end)
        events = portfolio_income_events(session, portfolio_ids, custom_start, custom_end)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected-range income", dollars(summary.selected_range))
    metric_columns[1].metric("YTD through end date", dollars(summary.year_to_date))
    metric_columns[2].metric("Trailing 365 through end date", dollars(summary.trailing_365_days))
    metric_columns[3].metric(
        "Normalized quarterly average", dollars(summary.normalized_quarterly_average)
    )
    st.caption(
        "Cash received from gross dividend and interest credits. The quarterly average "
        "normalizes the selected range to 91.3125 days; it is not a forecast."
    )
    if not events:
        st.caption("No dividend or interest income was received in the selected range.")
        return

    names_by_id = {portfolio.id: portfolio.name for portfolio in portfolios}
    rows = [
        {
            "Date": transaction.transaction_date,
            "Month": pd.Timestamp(transaction.transaction_date).to_period("M").to_timestamp(),
            "Portfolio": names_by_id[transaction.portfolio_id],
            "Income type": kind_label(transaction.kind),
            "Source": (
                transaction.symbol or "Unassigned dividend"
                if transaction.kind == "dividend"
                else "Interest"
            ),
            "Cash received": float(transaction.cash_delta),
        }
        for transaction in events
    ]
    income = pd.DataFrame(rows)
    monthly = income.groupby(["Month", "Income type"], as_index=False)["Cash received"].sum()
    figure = go.Figure()
    income_colors = (PLOT_COLORS[2], PLOT_COLORS[3])
    for index, income_type in enumerate(("Dividend", "Interest")):
        values = monthly[monthly["Income type"] == income_type]
        figure.add_trace(
            go.Bar(
                x=values["Month"],
                y=values["Cash received"],
                name=income_type,
                marker_color=income_colors[index],
                hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>" + income_type + "</extra>",
            )
        )
    figure.update_layout(
        height=300,
        barmode="stack",
        margin={"l": 12, "r": 12, "t": 18, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        xaxis={"title": None, "gridcolor": "rgba(105,126,255,.08)"},
        yaxis={"title": "Cash received", "tickprefix": "$", "gridcolor": "rgba(105,126,255,.14)"},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    sources = (
        income.groupby(["Portfolio", "Income type", "Source"], as_index=False)["Cash received"]
        .sum()
        .sort_values(["Portfolio", "Income type", "Source"])
    )
    st.dataframe(
        sources,
        hide_index=True,
        width="stretch",
        column_config={"Cash received": st.column_config.NumberColumn(format="$%,.0f")},
    )


def render_overview(
    portfolios: list[Portfolio],
    closes: pd.DataFrame,
    data_note: str | None,
    custom_start: date,
    custom_end: date,
    benchmark_closes: pd.DataFrame,
) -> None:
    if not portfolios:
        st.caption("No non-blank portfolios are available for the selected scope.")
        return
    render_performance_summary(
        portfolios,
        closes,
        custom_start,
        custom_end,
        "overview",
        benchmark_closes,
        expanded=True,
        show_portfolio_cards=True,
        data_note=data_note,
    )
    with st.expander("Portfolio income", expanded=False):
        render_portfolio_income(portfolios, custom_start, custom_end)
    render_consolidated_holdings(portfolios, closes, custom_start, custom_end, benchmark_closes)


def render_compare(
    portfolios: list[Portfolio],
    portfolio_closes: pd.DataFrame,
    custom_start: date,
    custom_end: date,
    benchmark_closes: pd.DataFrame,
) -> None:
    if not portfolios:
        st.caption("No portfolios are available to compare.")
        return
    render_performance_summary(
        portfolios,
        portfolio_closes,
        custom_start,
        custom_end,
        "compare",
        benchmark_closes,
        expanded=True,
    )
    with st.expander("Performance comparison", expanded=False):
        controls = st.columns([1.4, 2.2])
        with controls[0]:
            selected_benchmarks = st.multiselect(
                "Benchmark ETFs",
                tuple(BENCHMARKS),
                default=["SPY"],
                format_func=lambda symbol: f"{symbol} — {BENCHMARKS[symbol]}",
                key="compare_benchmarks",
            )
        with session_scope() as session:
            symbol_labels = security_symbol_labels(session)
        for portfolio in portfolios:
            for lot in portfolio.holdings:
                symbol_labels.setdefault(lot.symbol, lot.symbol)
        with controls[1]:
            extra_choices = st.multiselect(
                "Additional stocks or ETFs",
                tuple(sorted(symbol_labels)),
                key="compare_extras",
                format_func=lambda symbol: (
                    f"{symbol} — {symbol_labels[symbol]}"
                    if symbol_labels.get(symbol) not in {None, symbol}
                    else symbol
                ),
                placeholder="Start typing a ticker or security name",
                help="Choose a suggestion or enter a new stock or ETF ticker.",
                accept_new_options=True,
                filter_mode="fuzzy",
            )
        extras: list[str] = []
        for choice in extra_choices:
            try:
                extras.append(validate_transaction_symbol(str(choice)))
            except ValueError as exc:
                st.warning(str(exc))
        extras = list(dict.fromkeys(extras))
        benchmark_symbols = tuple(dict.fromkeys([*selected_benchmarks, *extras]))
        acquisition = comparison_acquisition_plan(
            portfolios, custom_start, custom_end, benchmark_symbols
        )
        try:
            closes = portfolio_closes
            comparison_benchmark_closes = (
                benchmark_closes
                if set(benchmark_symbols).issubset(benchmark_closes.columns)
                else cached_historical_data(acquisition.benchmark)
            )
        except Exception as exc:
            st.info(f"Comparison data is unavailable. Configure rotated Alpaca credentials. {exc}")
            return
        if closes.empty:
            st.info("No market data was returned for this comparison.")
            return

        report = assemble_comparison_report(
            portfolios,
            closes,
            comparison_benchmark_closes,
            custom_start,
            custom_end,
            benchmark_symbols,
        )
        for warning in report.warnings:
            _render_warning(warning)
        st.caption(report.coverage)
        if not report.series:
            st.info("The selected series do not share usable data in this date range.")
            return
        figure = go.Figure()
        for index, item in enumerate(report.series):
            figure.add_trace(
                go.Scatter(
                    x=item.index,
                    y=item.values,
                    name=item.name,
                    mode="lines",
                    line={
                        "width": 2.3,
                        "color": PLOT_COLORS[index % len(PLOT_COLORS)],
                    },
                )
            )
        figure.update_layout(
            height=520,
            margin={"l": 12, "r": 12, "t": 24, "b": 12},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(6,8,16,.62)",
            font={"color": "#F7F8FF"},
            legend={"orientation": "h", "y": 1.08},
            hovermode="x unified",
            yaxis={"title": "Growth of $100", "gridcolor": "rgba(105,126,255,.14)"},
            xaxis={"gridcolor": "rgba(105,126,255,.10)"},
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        st.dataframe(
            report.metrics,
            hide_index=True,
            width="stretch",
            column_config={
                key: st.column_config.NumberColumn(format="%.0f%%")
                for key in (
                    "Total return",
                    "Annualized return",
                    "Volatility",
                    "Max drawdown",
                )
            },
        )
        st.caption(
            "All series are rebased to $100. Portfolio performance excludes transfers, cash "
            "adjustments, and contributed opening positions. Portfolio prices are split-adjusted "
            "only; benchmark series use dividend-adjusted total-return closes."
        )


FORECAST_PRESETS = {
    "Conservative": (0.05, 0.08),
    "Baseline": (0.07, 0.12),
    "Growth": (0.09, 0.16),
}


def _apply_forecast_preset() -> None:
    annual_return, annual_volatility = FORECAST_PRESETS[st.session_state.forecast_preset]
    st.session_state.forecast_annual_return = annual_return * 100
    st.session_state.forecast_annual_volatility = annual_volatility * 100


def render_forecast(
    portfolios: list[Portfolio], closes: pd.DataFrame, *, owner: bool = False
) -> None:
    """Render public, assumption-driven long-term portfolio planning scenarios."""
    if not portfolios:
        st.caption("No portfolios are available for a projection.")
        return

    scope_options = ["Selected portfolios", *(portfolio.name for portfolio in portfolios)]
    st.session_state.setdefault("forecast_preset", "Baseline")
    st.session_state.setdefault("forecast_annual_return", 7.0)
    st.session_state.setdefault("forecast_annual_volatility", 12.0)
    st.session_state.setdefault("forecast_monthly_contribution", 0.0)
    st.session_state.setdefault("forecast_horizon", 10)
    st.session_state.setdefault("forecast_target_value", 1_000_000.0)

    first_row = st.columns(3)
    scope = first_row[0].selectbox("Projection scope", scope_options, key="forecast_scope")
    first_row[1].selectbox(
        "Scenario preset",
        list(FORECAST_PRESETS),
        key="forecast_preset",
        on_change=_apply_forecast_preset,
    )
    horizon_years = first_row[2].select_slider(
        "Forecast horizon (years)", options=list(range(1, 11)), key="forecast_horizon"
    )
    scoped_portfolios = (
        portfolios
        if scope == "Selected portfolios"
        else [next(portfolio for portfolio in portfolios if portfolio.name == scope)]
    )
    second_row = st.columns(3)
    annual_return = second_row[0].number_input(
        "Expected annual return (%)",
        min_value=-99.0,
        max_value=50.0,
        step=0.25,
        key="forecast_annual_return",
    )
    annual_volatility = second_row[1].number_input(
        "Annual volatility (%)",
        min_value=0.0,
        max_value=100.0,
        step=0.25,
        key="forecast_annual_volatility",
    )
    monthly_contribution = second_row[2].number_input(
        "Monthly contribution ($)", min_value=0.0, step=100.0, key="forecast_monthly_contribution"
    )
    target_row = st.columns([1.0, 1.5, 1.5], vertical_alignment="bottom")
    include_target = target_row[0].checkbox("Set a target value", key="forecast_include_target")
    entered_target = target_row[1].number_input(
        "Target portfolio value ($)",
        min_value=1.0,
        step=1_000.0,
        format="%.0f",
        key="forecast_target_value",
        disabled=not include_target,
    )
    target_probability_slot = target_row[2].empty()
    target_value = entered_target if include_target else None
    assumptions = ForecastAssumptions(
        annual_return=annual_return / 100,
        annual_volatility=annual_volatility / 100,
        monthly_contribution=monthly_contribution,
        horizon_years=horizon_years,
        target_value=target_value,
    )
    try:
        request = build_forecast_request(scoped_portfolios, closes, assumptions)
    except ValueError as exc:
        st.warning(str(exc))
        return
    st.caption(f"Starting value for {scope}: {dollars(request.current_value)}.")
    st.caption(request.coverage)
    for warning in request.warnings:
        _render_warning(warning)
    result = cached_projection(request)
    contract = result.contract
    source_date = (
        contract.source_valuation_date.strftime("%-m/%-d/%y")
        if contract.source_valuation_date is not None
        else "unavailable (disclosed fallback)"
    )
    st.caption(
        f"Model {contract.model_type} / {contract.model_version} · seed {contract.seed} · "
        f"{contract.simulation_count:,} simulations · source valuation date {source_date} · "
        f"method {contract.source_valuation_methodology} · generated "
        f"{contract.result_generated_at.isoformat()}."
    )
    dates = pd.date_range(
        pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(1),
        periods=horizon_years * 12,
        freq="ME",
    ).insert(0, pd.Timestamp.today().normalize())
    percentile_data = result.monthly_percentiles
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=percentile_data["P95"],
            line={"width": 0, "color": "rgba(126,105,255,0)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=percentile_data["P5"],
            fill="tonexty",
            fillcolor="rgba(126,105,255,.16)",
            line={"width": 0, "color": "rgba(126,105,255,0)"},
            name="5th–95th percentile",
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>5th percentile</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=percentile_data["P75"],
            line={"width": 0, "color": "rgba(105,126,255,0)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=percentile_data["P25"],
            fill="tonexty",
            fillcolor="rgba(105,126,255,.28)",
            line={"width": 0, "color": "rgba(105,126,255,0)"},
            name="25th–75th percentile",
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>25th percentile</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=percentile_data["P50"],
            mode="lines",
            name="Median scenario",
            line={"width": 3, "color": PLOT_COLORS[0]},
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>Median scenario</extra>",
        )
    )
    figure.update_layout(
        height=480,
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(6,8,16,.62)",
        font={"color": "#F7F8FF"},
        legend={"orientation": "h", "y": 1.08},
        hovermode="x unified",
        yaxis={"title": "Portfolio value", "tickprefix": "$", "gridcolor": "rgba(105,126,255,.14)"},
        xaxis={"gridcolor": "rgba(105,126,255,.10)"},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Methodology: 10,000 reproducible monthly lognormal simulations use the selected "
        "expected return for drift and volatility for outcome dispersion; contributions are "
        "added at month-end. The chart shows the median, 25th–75th, and 5th–95th percentile "
        "ranges. This is assumption-driven planning analysis, not a prediction based on recent "
        "portfolio performance."
    )

    annual = result.annual_percentiles.iloc[1:].reset_index()
    annual.insert(1, "Calendar year", [date.today().year + year for year in annual["Year"]])
    st.dataframe(
        annual,
        hide_index=True,
        width="stretch",
        column_config={
            "Year": st.column_config.NumberColumn("Year from now", format="%d"),
            "Calendar year": st.column_config.NumberColumn(format="%d"),
            **{
                percentile: st.column_config.NumberColumn(format="$%,.0f")
                for percentile in ("P5", "P25", "P50", "P75", "P95")
            },
        },
    )
    if result.target_probability is not None:
        target_probability_slot.metric(
            f"Chance of reaching {dollars(target_value)} in {horizon_years} years",
            f"{result.target_probability:.0%}",
        )
    render_deterministic_scenarios(scoped_portfolios, closes, owner=owner)


SCENARIO_LABELS = {
    ScenarioType.BROAD_MARKET_DECLINE: "Immediate broad-market decline",
    ScenarioType.HOLDING_DECLINE: "Holding-specific decline",
    ScenarioType.SECTOR_DECLINE: "Sector decline",
    ScenarioType.DIVIDEND_REDUCTION: "Dividend reduction",
    ScenarioType.CONTRIBUTION_INTERRUPTION: "Contribution interruption",
    ScenarioType.INFLATION_INCREASE: "Inflation increase",
    ScenarioType.LOW_RETURN_PERIOD: "Low-return period",
    ScenarioType.LOST_DECADE: "Lost decade",
    ScenarioType.RETIREMENT_DATE_DECLINE: "Retirement-date decline",
    ScenarioType.HISTORICAL_REPLAY: "Historical replay",
}


def render_deterministic_scenarios(
    portfolios: list[Portfolio], closes: pd.DataFrame, *, owner: bool
) -> None:
    """Render the focused Phase 7 scenario view; calculations remain in the service module."""
    with st.expander("Deterministic scenario analysis", expanded=False):
        st.caption(
            "One fixed set of assumptions is applied reproducibly. This view does not run "
            "bootstrap or Monte Carlo models."
        )
        selected_type = st.selectbox(
            "Deterministic scenario",
            list(SCENARIO_LABELS),
            format_func=lambda value: SCENARIO_LABELS[value],
            key="deterministic_scenario_type",
        )
        symbols = sorted({lot.symbol for portfolio in portfolios for lot in portfolio.holdings})
        shock_label = (
            "Dividend reduction (%)"
            if selected_type == ScenarioType.DIVIDEND_REDUCTION
            else "Market decline (%)"
        )
        market_decline = (
            -st.number_input(
                shock_label,
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                key="scenario_market_decline",
            )
            / 100
        )
        inflation_increase = (
            st.number_input(
                "Additional inflation (%)",
                min_value=0.0,
                max_value=25.0,
                value=2.0,
                step=0.25,
                key="scenario_inflation_increase",
            )
            / 100
            if selected_type == ScenarioType.INFLATION_INCREASE
            else 0.02
        )
        low_return = (
            st.number_input(
                "Low-period annual return (%)",
                min_value=-99.0,
                max_value=25.0,
                value=1.0,
                step=0.25,
                key="scenario_low_return",
            )
            / 100
            if selected_type in {ScenarioType.LOW_RETURN_PERIOD, ScenarioType.LOST_DECADE}
            else 0.01
        )
        interruption_months = (
            int(
                st.number_input(
                    "Contribution interruption (months)",
                    min_value=0,
                    max_value=480,
                    value=12,
                    step=1,
                    key="scenario_interruption_months",
                )
            )
            if selected_type == ScenarioType.CONTRIBUTION_INTERRUPTION
            else 12
        )
        holding_symbol = (
            st.selectbox("Holding", symbols, key="scenario_holding")
            if selected_type == ScenarioType.HOLDING_DECLINE and symbols
            else None
        )
        sector = None
        sectors: dict[str, str] = {}
        with session_scope() as session:
            for symbol in symbols:
                metadata = resolve_security_metadata(session, symbol)
                sectors[symbol] = metadata.sector or "Unclassified"
        available_sectors = sorted(set(sectors.values()))
        if selected_type == ScenarioType.SECTOR_DECLINE:
            sector = st.selectbox("Sector", available_sectors, key="scenario_sector")
        assumptions = ScenarioAssumptions(
            selected_type,
            market_decline=market_decline,
            holding_symbol=holding_symbol,
            holding_decline=market_decline,
            sector=sector,
            sector_decline=market_decline,
            dividend_reduction=abs(market_decline),
            contribution_amount=float(st.session_state.get("forecast_monthly_contribution", 0)),
            interruption_months=interruption_months,
            inflation_increase=inflation_increase,
            expected_return=float(st.session_state.get("forecast_annual_return", 7)) / 100,
            low_return=low_return,
            horizon_years=int(st.session_state.get("forecast_horizon", 10)),
            retirement_date=date.today().replace(
                year=date.today().year + min(5, int(st.session_state.get("forecast_horizon", 10)))
            ),
        )
        dataset_references = [
            DatasetReference(dataset_id, "historical_replay")
            for dataset_id in closes.attrs.get("dataset_ids", ())
        ]
        try:
            result = run_deterministic_scenario(
                portfolios,
                closes,
                assumptions,
                sectors=sectors,
                historical_prices=(
                    closes if selected_type == ScenarioType.HISTORICAL_REPLAY else None
                ),
                dataset_references=dataset_references,
            )
        except ValueError as exc:
            st.warning(str(exc))
            return
        metrics = st.columns(3)
        metrics[0].metric("Baseline", dollars(result.baseline_value))
        metrics[1].metric("Scenario", dollars(result.scenario_value))
        metrics[2].metric("Household impact", dollars(result.total_household_impact))
        portfolio_impact = pd.DataFrame(
            [
                {"Portfolio": name, "Impact": impact}
                for name, impact in result.impact_by_portfolio.items()
            ]
        )
        detail = pd.DataFrame(
            [
                {"Dimension": "Holding", "Name": name, "Impact": impact}
                for name, impact in result.impact_by_holding.items()
            ]
            + [
                {"Dimension": "Sector", "Name": name, "Impact": impact}
                for name, impact in result.impact_by_sector.items()
            ]
            + [
                {"Dimension": "Account type", "Name": name, "Impact": impact}
                for name, impact in result.account_type_effects.items()
            ]
        )
        assumption_frame = pd.DataFrame(
            [
                {"Assumption": name.replace("_", " ").title(), "Value": str(value)}
                for name, value in result.assumptions.items()
            ]
        )
        impact_height = min(
            360,
            max(
                120,
                36 * (max(len(portfolio_impact), len(detail), len(assumption_frame)) + 1),
            ),
        )
        impact_columns = st.columns(3)
        with impact_columns[0]:
            st.markdown("#### Portfolio")
            st.dataframe(
                portfolio_impact,
                hide_index=True,
                width="stretch",
                height=impact_height,
                column_config={"Impact": st.column_config.NumberColumn(format="$%,.0f")},
            )
        with impact_columns[1]:
            st.markdown("#### Dimension")
            st.dataframe(
                detail,
                hide_index=True,
                width="stretch",
                height=impact_height,
                column_config={"Impact": st.column_config.NumberColumn(format="$%,.0f")},
            )
        with impact_columns[2]:
            st.markdown("#### Assumptions")
            st.dataframe(
                assumption_frame,
                hide_index=True,
                width="stretch",
                height=impact_height,
            )
        st.markdown("#### Sensitivity")
        sensitivity_values: dict[str, list[object]]
        if selected_type == ScenarioType.HISTORICAL_REPLAY:
            sensitivity_values = {}
        elif selected_type == ScenarioType.INFLATION_INCREASE:
            sensitivity_values = {"inflation": [0.02, 0.03, 0.04]}
        elif selected_type == ScenarioType.CONTRIBUTION_INTERRUPTION:
            contribution = assumptions.contribution_amount
            sensitivity_values = {"contribution_amount": [0.0, contribution, contribution * 1.25]}
        elif selected_type in {ScenarioType.LOW_RETURN_PERIOD, ScenarioType.LOST_DECADE}:
            sensitivity_values = {"expected_return": [0.04, 0.07, 0.10]}
        else:
            sensitivity_values = {"market_decline": [-0.10, -0.20, -0.30]}
        if sensitivity_values:
            sensitivity = sensitivity_grid(
                portfolios,
                closes,
                assumptions,
                sensitivity_values,
                sectors=sectors,
                historical_prices=(
                    closes if selected_type == ScenarioType.HISTORICAL_REPLAY else None
                ),
                dataset_references=dataset_references,
            )
            sensitivity = sensitivity.copy()
            percentage_columns = [
                column
                for column in (
                    "market_decline",
                    "inflation",
                    "expected_return",
                    "impact_percent",
                )
                if column in sensitivity
            ]
            sensitivity[percentage_columns] *= 100
            usd_columns = [
                column
                for column in (
                    "contribution_amount",
                    "spending",
                    "baseline_value",
                    "scenario_value",
                    "household_impact",
                )
                if column in sensitivity
            ]
            st.dataframe(
                sensitivity,
                hide_index=True,
                width="stretch",
                column_config={
                    **{
                        column: st.column_config.NumberColumn(
                            column.replace("_", " ").title(), format="%.1f%%"
                        )
                        for column in percentage_columns
                    },
                    **{
                        column: st.column_config.NumberColumn(
                            column.replace("_", " ").title(), format="$%,.0f"
                        )
                        for column in usd_columns
                    },
                },
            )
        else:
            st.caption("Historical replay uses its fixed available prior-period observations.")
        st.caption(
            f"Coverage: {result.coverage['priced_holdings']} priced lots. "
            f"Model {result.model_version}; validation status is unvalidated unless an explicit "
            "model review records otherwise."
        )
        for warning in result.warnings:
            _render_warning(warning)
        if owner and st.button("Save scenario run", key="save_deterministic_scenario"):
            with session_scope() as session:
                saved = save_scenario_run(
                    session,
                    portfolios,
                    result,
                    dataset_references=dataset_references,
                )
            st.info(f"Saved forecast run #{saved.id}.")


def transaction_frame(
    transactions: list[PortfolioTransaction], names_by_id: dict[int, str]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "_transaction_id": entry.id,
                "_portfolio_id": entry.portfolio_id,
                "Portfolio": names_by_id.get(entry.portfolio_id, str(entry.portfolio_id)),
                "Date": entry.transaction_date,
                "Action": entry.action,
                "Type": kind_label(entry.kind),
                "Symbol": entry.symbol,
                "Quantity": float(entry.quantity) if entry.quantity is not None else None,
                "Price": float(entry.price) if entry.price is not None else None,
                "Fees": float(entry.fees) if entry.fees is not None else None,
                "Cash change": float(entry.cash_delta),
                "Description": entry.description,
                "Source": entry.source,
            }
            for entry in transactions
        ]
    )


def render_transaction_editor(transaction: PortfolioTransaction, portfolio_name: str) -> None:
    with st.container(key=f"compact_transaction_editor_{transaction.id}"):
        st.markdown("### Edit or delete selected transaction")
        st.caption(f"Transaction #{transaction.id} · Portfolio: **{portfolio_name}**")
        if transaction.source != "manual":
            st.warning(
                f"This transaction originated from `{transaction.source}`. A manual override may "
                "diverge from the source; the original remains in the audit history."
            )
        else:
            st.warning(
                "Editing or deleting recalculates cash and holdings; the original remains in the "
                "audit history."
            )
        with st.form(f"edit_transaction_{transaction.id}"):
            first_row = st.columns(4)
            edit_date = first_row[0].text_input(
                "Date (M/D/YY)", value=format_short_date(transaction.transaction_date)
            )
            edit_action = first_row[1].text_input("Action", value=transaction.action, max_chars=80)
            edit_kind = first_row[2].selectbox(
                "Type",
                EDITABLE_KINDS,
                index=EDITABLE_KINDS.index(transaction.kind),
                format_func=kind_label,
            )
            edit_symbol = first_row[3].text_input("Symbol", value=transaction.symbol or "")
            value_row = st.columns(4)
            edit_quantity = value_row[0].number_input(
                "Quantity",
                min_value=0.0,
                value=float(transaction.quantity or 0),
                format="%.2f",
            )
            edit_price = value_row[1].number_input(
                "Price",
                min_value=0.0,
                value=float(transaction.price or 0),
                format="%.6f",
            )
            edit_fees = value_row[2].number_input(
                "Fees",
                min_value=0.0,
                value=float(transaction.fees or 0),
                format="%.4f",
            )
            edit_cash = value_row[3].number_input(
                "Cash change",
                value=float(transaction.cash_delta),
                step=1.0,
                format="%.4f",
                help="Buy and sell cash changes are recalculated from quantity, price, and fees.",
            )
            edit_description = st.text_input(
                "Description", value=transaction.description, max_chars=500
            )
            update_row = st.columns([3, 1], vertical_alignment="bottom")
            update_phrase = update_row[0].text_input(f'Type "UPDATE {transaction.id}" to confirm')
            updated = update_row[1].form_submit_button("Save changes", width="stretch")
        if updated:
            try:
                draft = build_transaction_draft(
                    TransactionCommand(
                        edit_date,
                        edit_kind,
                        edit_symbol,
                        edit_description,
                        edit_quantity,
                        edit_price,
                        edit_fees,
                        edit_cash,
                        action=edit_action,
                    )
                )
                with session_scope() as session:
                    update_transaction(
                        session,
                        transaction.portfolio_id,
                        transaction.id,
                        draft,
                        confirmation=update_phrase,
                    )
                cached_historical_data.clear()
                st.session_state.flash = f"Transaction #{transaction.id} updated."
                st.rerun()
            except Exception as exc:
                st.info(f"Transaction was not updated: {exc}")

        with st.form(f"delete_transaction_{transaction.id}"):
            delete_row = st.columns([3, 1], vertical_alignment="bottom")
            delete_phrase = delete_row[0].text_input(f'Type "DELETE {transaction.id}" to confirm')
            deleted = delete_row[1].form_submit_button(
                "Delete transaction", type="secondary", width="stretch"
            )
        if deleted:
            try:
                with session_scope() as session:
                    delete_transaction(
                        session,
                        transaction.portfolio_id,
                        transaction.id,
                        confirmation=delete_phrase,
                    )
                cached_historical_data.clear()
                st.session_state.flash = f"Transaction #{transaction.id} deleted."
                st.rerun()
            except Exception as exc:
                st.info(f"Transaction was not deleted: {exc}")


def render_csv_import(portfolio: Portfolio) -> None:
    st.markdown(
        """
**Required column names:** `Date`, `Action`, `Symbol`, `Description`, `Quantity`, `Price`,
`Fees & Comm`, and `Amount`.

**Accepted actions:** Buy, Sell, Cash Dividend, Qualified Dividend, Non-Qualified Div,
Pr Yr Cash Div, Credit Interest, MoneyLink Transfer, Promotional Award, ADR Mgmt Fee,
and Foreign Tax Paid. Other action text is retained and posted as a cash adjustment.

**Date and currency conventions:** Dates may be `M/D/YYYY` or `YYYY-MM-DD`. Currency may
include `$`, commas, or parentheses for negatives. `Amount` is required; buys must be negative,
sells positive, and blank quantity/price/fees cells are allowed for cash-only activity.

**Example rows:**
"""
    )
    st.code(CSV_TEMPLATE, language="csv")
    st.download_button(
        "Download Brokerage CSV template",
        data=CSV_TEMPLATE,
        file_name="brokerage_transactions_template.csv",
        mime="text/csv",
    )
    st.caption(f"Import target: {portfolio.name}")
    uploaded_statement = st.file_uploader(
        "Brokerage CSV", type=["csv"], key=f"statement_{portfolio.id}"
    )
    if uploaded_statement is None:
        return
    statement_content = uploaded_statement.getvalue()
    parsed_statement = parse_statement_csv(statement_content)
    if parsed_statement.errors:
        st.dataframe(pd.DataFrame({"Import issues": parsed_statement.errors}), hide_index=True)
        return
    preview = pd.DataFrame(
        [
            {
                "Date": item.transaction_date,
                "Action": item.action,
                "Category": kind_label(item.kind),
                "Symbol": item.symbol,
                "Quantity": float(item.quantity) if item.quantity is not None else None,
                "Price": float(item.price) if item.price is not None else None,
                "Fees": float(item.fees) if item.fees is not None else None,
                "Cash change": float(item.cash_delta),
                "Description": item.description,
            }
            for item in parsed_statement.transactions
        ]
    )
    st.dataframe(
        preview,
        hide_index=True,
        width="stretch",
        column_config={
            "Date": st.column_config.DateColumn(format="M/D/YY"),
            "Quantity": st.column_config.NumberColumn(format="%.2f"),
            "Price": st.column_config.NumberColumn(format="$%,.2f"),
            "Fees": st.column_config.NumberColumn(format="$%,.2f"),
            "Cash change": st.column_config.NumberColumn(format="$%,.2f"),
        },
    )
    import_columns = st.columns([1, 1])
    if import_columns[0].button("Import new transactions", key=f"import_{portfolio.id}"):
        try:
            with session_scope() as session:
                added, duplicates = import_statement(session, portfolio.id, parsed_statement)
            cached_historical_data.clear()
            st.session_state.flash = (
                f"Imported {added} transaction(s); skipped {duplicates} duplicate(s)."
            )
            st.rerun()
        except Exception as exc:
            st.info(f"Statement was not imported: {exc}")
    rebuild_phrase = import_columns[1].text_input(
        "Type REBUILD to replace this portfolio", key=f"rebuild_phrase_{portfolio.id}"
    )
    if import_columns[1].button("Rebuild portfolio from statement", key=f"rebuild_{portfolio.id}"):
        if rebuild_phrase != "REBUILD":
            st.info("Type REBUILD before replacing this portfolio.")
        else:
            try:
                with session_scope() as session:
                    rebuilt = rebuild_portfolio_from_csv(session, portfolio.id, statement_content)
                cached_historical_data.clear()
                st.session_state.flash = f"Portfolio rebuilt from {rebuilt} transaction(s)."
                st.rerun()
            except Exception as exc:
                st.info(f"Portfolio was not rebuilt: {exc}")


def render_transactions(
    portfolios: list[Portfolio],
    names_by_id: dict[int, str],
    custom_start: date,
    custom_end: date,
    *,
    editable: bool,
) -> None:
    with st.expander("Transactions", expanded=True):
        st.caption(
            f"Showing activity from {custom_start:%m/%d/%y} through {custom_end:%m/%d/%y} "
            "for the applied portfolio scope."
        )
        with session_scope() as session:
            transactions = list_transactions_for_portfolios(
                session, [portfolio.id for portfolio in portfolios]
            )
        if not transactions:
            st.caption("No transactions match the applied portfolios and master date range.")
            return

        type_options = sorted({transaction.kind for transaction in transactions})
        selected_types = st.multiselect(
            "Transaction type filter",
            type_options,
            default=type_options,
            format_func=kind_label,
            key="manage_transaction_types",
        )
        filtered = [
            transaction
            for transaction in transactions
            if transaction.kind in selected_types
            and custom_start <= transaction.transaction_date <= custom_end
        ]
        frame = transaction_frame(filtered, names_by_id)
        if frame.empty:
            st.caption("No transactions match the type filter and master date range.")
            return

        st.caption(
            "Select any row to edit or delete it. Click a column header to sort."
            if editable
            else "Read-only transaction history. Click a column header to sort."
        )
        table_scope = "_".join(
            [
                *[str(portfolio.id) for portfolio in portfolios],
                *selected_types,
                custom_start.isoformat(),
                custom_end.isoformat(),
            ]
        )
        table_options = {
            "hide_index": True,
            "width": "stretch",
            "column_order": [
                "Portfolio",
                "Date",
                "Action",
                "Type",
                "Symbol",
                "Quantity",
                "Price",
                "Fees",
                "Cash change",
                "Description",
                "Source",
            ],
            "column_config": {
                "Date": st.column_config.DateColumn(format="M/D/YY"),
                "Quantity": st.column_config.NumberColumn(format="%.2f"),
                "Price": st.column_config.NumberColumn(format="$%,.2f"),
                "Fees": st.column_config.NumberColumn(format="$%,.2f"),
                "Cash change": st.column_config.NumberColumn(format="$%,.2f"),
            },
            "key": f"manage_transaction_table_{table_scope}",
        }
        if editable:
            table_options.update(on_select="rerun", selection_mode="single-row")
        table_event = st.dataframe(frame, **table_options)
        selected_rows = table_event.selection.rows if editable else []
        if editable and selected_rows:
            selected_transaction = filtered[selected_rows[0]]
            render_transaction_editor(
                selected_transaction, names_by_id[selected_transaction.portfolio_id]
            )


def _target_portfolio_selector(
    label: str,
    key: str,
    portfolios: list[Portfolio],
    selected_portfolios: list[Portfolio],
) -> Portfolio | None:
    """Default an action target only when the master scope has one portfolio."""
    available_ids = {portfolio.id for portfolio in portfolios}
    master_scope = tuple(portfolio.id for portfolio in selected_portfolios)
    scope_key = f"_{key}_master_scope"
    default_id = selected_portfolios[0].id if len(selected_portfolios) == 1 else None
    if st.session_state.get(scope_key) != master_scope or st.session_state.get(key) not in {
        SELECT_PORTFOLIO_OPTION,
        *available_ids,
    }:
        st.session_state[key] = default_id or SELECT_PORTFOLIO_OPTION
        st.session_state[scope_key] = master_scope

    portfolio_by_id = {portfolio.id: portfolio for portfolio in portfolios}
    target_id = st.selectbox(
        label,
        [SELECT_PORTFOLIO_OPTION, *portfolio_by_id],
        key=key,
        format_func=lambda portfolio_id: (
            "Select portfolio"
            if portfolio_id == SELECT_PORTFOLIO_OPTION
            else portfolio_by_id[portfolio_id].name
        ),
    )
    return None if target_id == SELECT_PORTFOLIO_OPTION else portfolio_by_id[target_id]


def _validate_manual_trade_symbol() -> None:
    """Validate and normalize a buy/sell symbol as soon as it changes."""
    key = "add_transaction_symbol"
    error_key = "add_transaction_symbol_error"
    value = st.session_state.get(key, "")
    try:
        st.session_state[key] = validate_transaction_symbol(value)
        st.session_state.pop(error_key, None)
    except ValueError as exc:
        st.session_state[error_key] = str(exc)


def render_add_transaction(
    portfolios: list[Portfolio], selected_portfolios: list[Portfolio]
) -> None:
    with st.expander("Add transaction", expanded=False):
        if not portfolios:
            st.caption("Create a portfolio before recording a transaction.")
            return
        with st.container(key="compact_transaction_add"):
            first_row = st.columns(4)
            with first_row[0]:
                target = _target_portfolio_selector(
                    "Portfolio",
                    "transaction_target",
                    portfolios,
                    selected_portfolios,
                )
            transaction_date_text = first_row[1].text_input(
                "Date (M/D/YY)",
                value=format_short_date(date.today()),
                key="add_transaction_date",
            )
            kind = first_row[2].selectbox(
                "Type",
                MANUAL_KINDS,
                format_func=kind_label,
                key="add_transaction_kind",
            )
            symbol = ""
            symbol_error = ""
            if kind in SYMBOL_CASH_KINDS:
                symbol = first_row[3].text_input(
                    "Symbol (optional)", max_chars=16, key="add_transaction_symbol"
                )
            elif kind in TRADE_KINDS:
                symbol = first_row[3].text_input(
                    "Symbol",
                    max_chars=16,
                    key="add_transaction_symbol",
                    on_change=_validate_manual_trade_symbol,
                )
                symbol_error = st.session_state.get("add_transaction_symbol_error", "")
            else:
                first_row[3].text_input(
                    "Symbol", value="", key="add_transaction_symbol_disabled", disabled=True
                )

            manual_quantity = 0.0
            manual_price = 0.0
            manual_fees = 0.0
            value_row = st.columns(4)
            if kind in TRADE_KINDS or kind == "award":
                manual_quantity = value_row[0].number_input(
                    "Quantity" if kind in TRADE_KINDS else "Award quantity (optional)",
                    min_value=0.0,
                    value=0.0,
                    format="%.2f",
                    key="add_transaction_quantity",
                )
                manual_price = value_row[1].number_input(
                    "Price" if kind in TRADE_KINDS else "Recorded fair value / share",
                    min_value=0.0,
                    value=0.0,
                    format="%.6f",
                    key="add_transaction_price",
                )
                if kind in TRADE_KINDS:
                    manual_fees = value_row[2].number_input(
                        "Fees",
                        min_value=0.0,
                        value=0.0,
                        format="%.4f",
                        key="add_transaction_fees",
                    )
                    st.session_state.add_transaction_calculated_cash = calculated_trade_cash(
                        kind, manual_quantity, manual_price, manual_fees
                    )
                    cash_delta = value_row[3].number_input(
                        "Cash change",
                        key="add_transaction_calculated_cash",
                        format="%.4f",
                        disabled=True,
                        help="Calculated from quantity, price, and fees.",
                    )
                else:
                    value_row[2].caption(
                        "A quantity award requires this stored fair value; no market price is inferred."
                    )
                    cash_delta = value_row[3].number_input(
                        "Cash award amount",
                        value=0.0,
                        step=1.0,
                        format="%.4f",
                        key="add_transaction_cash",
                    )
            else:
                cash_delta = value_row[3].number_input(
                    "Cash change",
                    value=0.0,
                    step=1.0,
                    format="%.4f",
                    key="add_transaction_cash",
                )
            action_row = st.columns([4, 1], vertical_alignment="bottom")
            description = action_row[0].text_input(
                "Description", max_chars=500, key="add_transaction_description"
            )
            recorded = action_row[1].button(
                "Record transaction",
                key="record_manual_transaction",
                disabled=bool(symbol_error) or target is None,
                width="stretch",
            )
            if symbol_error:
                st.error(symbol_error)
            if target is None:
                st.caption("Select a portfolio before recording a transaction.")
        if recorded:
            try:
                draft = build_transaction_draft(
                    TransactionCommand(
                        transaction_date_text,
                        kind,
                        symbol,
                        description,
                        manual_quantity,
                        manual_price,
                        manual_fees,
                        cash_delta,
                    )
                )
                with session_scope() as session:
                    record_transaction(session, target.id, draft)
                cached_historical_data.clear()
                st.session_state.flash = f"Transaction recorded in {target.name}."
                st.rerun()
            except Exception as exc:
                st.info(f"Transaction was not recorded: {exc}")


def render_csv_section(portfolios: list[Portfolio], selected_portfolios: list[Portfolio]) -> None:
    with st.expander("Brokerage CSV", expanded=False):
        if not portfolios:
            st.caption("Create a portfolio before importing a brokerage CSV.")
            return
        target = _target_portfolio_selector(
            "Import target portfolio",
            "csv_target",
            portfolios,
            selected_portfolios,
        )
        if target is None:
            st.caption("Select a portfolio before importing a brokerage CSV.")
            return
        render_csv_import(target)


def render_portfolio_actions(portfolios: list[Portfolio]) -> None:
    with st.expander("Portfolio administration", expanded=False):
        actions = (
            ["Edit portfolio", "Add portfolio", "Delete portfolio"]
            if portfolios
            else ["Add portfolio"]
        )
        action = st.selectbox("Portfolio action", actions, key="portfolio_action")

        if action == "Edit portfolio":
            target_name = st.selectbox(
                "Portfolio to edit",
                [portfolio.name for portfolio in portfolios],
                key="rename_target",
            )
            target = next(portfolio for portfolio in portfolios if portfolio.name == target_name)
            with st.form(f"rename_portfolio_{target.id}"):
                renamed_name = st.text_input("Portfolio name", value=target.name, max_chars=80)
                selected_account_type = st.selectbox(
                    "Account type",
                    ACCOUNT_TYPES,
                    index=ACCOUNT_TYPES.index(target.account_type),
                    format_func=lambda value: ACCOUNT_TYPE_LABELS[value],
                )
                renamed = st.form_submit_button("Save portfolio settings")
            if renamed:
                try:
                    with session_scope() as session:
                        rename_portfolio(session, target.id, renamed_name)
                        set_account_type(session, target.id, selected_account_type)
                    st.session_state.flash = "Portfolio settings saved."
                    st.rerun()
                except Exception as exc:
                    st.info(f"Portfolio settings were not saved: {exc}")

            with session_scope() as session:
                configurations = benchmark_configurations(session, target.id)
            if configurations:
                st.caption("Effective-dated benchmark history (weights are percentages).")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Effective": configuration.effective_from,
                                "Components": ", ".join(
                                    f"{symbol} {float(weight) * 100:g}%"
                                    for symbol, weight in configuration.weights.items()
                                ),
                                "Rebalancing": configuration.rebalancing_frequency.title(),
                            }
                            for configuration in configurations
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
            st.caption(
                "New benchmark configurations apply only from their effective date forward. "
                "Total-return components rebalance at the selected frequency."
            )
            with st.form(f"benchmark_configuration_{target.id}"):
                benchmark_effective_text = st.text_input(
                    "Benchmark effective date (M/D/YY)", value=format_short_date(date.today())
                )
                benchmark_components = st.text_input(
                    "Benchmark blend",
                    value="SPY:100",
                    help="Comma-separated SYMBOL:percentage entries; weights must total 100%.",
                )
                benchmark_frequency = st.selectbox(
                    "Benchmark rebalancing", REBALANCING_FREQUENCIES, index=1
                )
                benchmark_saved = st.form_submit_button("Add benchmark configuration")
            if benchmark_saved:
                try:
                    weights = parse_benchmark_components(benchmark_components)
                    benchmark_effective = parse_short_date(benchmark_effective_text)
                    with session_scope() as session:
                        save_benchmark_configuration(
                            session,
                            target.id,
                            benchmark_effective,
                            weights,
                            rebalancing_frequency=benchmark_frequency,
                        )
                    st.session_state.flash = "Benchmark configuration added."
                    st.rerun()
                except Exception as exc:
                    st.info(f"Benchmark configuration was not added: {exc}")

        elif action == "Add portfolio":
            with st.form("create_portfolio"):
                new_name = st.text_input(
                    "New portfolio name", value="", max_chars=80, placeholder="Portfolio name"
                )
                created = st.form_submit_button("Add portfolio")
            if created:
                try:
                    with session_scope() as session:
                        portfolio = create_portfolio(session, new_name)
                    st.session_state.flash = (
                        f"Portfolio '{portfolio.name}' added. "
                        "Record or import its first transaction."
                    )
                    st.rerun()
                except Exception as exc:
                    st.info(f"Portfolio was not added: {exc}")

        else:
            target_name = st.selectbox(
                "Portfolio to delete",
                [portfolio.name for portfolio in portfolios],
                key="delete_target",
            )
            target = next(portfolio for portfolio in portfolios if portfolio.name == target_name)
            deletion_phrase = st.text_input(
                "Type DELETE to permanently remove this portfolio and all of its data",
                key=f"delete_phrase_{target.id}",
            )
            if st.button("Delete portfolio", key=f"delete_{target.id}", type="secondary"):
                if deletion_phrase != "DELETE":
                    st.info("Type DELETE before permanently removing this portfolio.")
                else:
                    try:
                        with session_scope() as session:
                            delete_portfolio(session, target.id)
                        cached_historical_data.clear()
                        st.session_state.flash = (
                            "Portfolio and all of its data were permanently deleted."
                        )
                        st.rerun()
                    except Exception as exc:
                        st.info(f"Portfolio was not deleted: {exc}")


def render_portfolio_admin(
    reporting_portfolios: list[Portfolio],
    action_portfolios: list[Portfolio],
    custom_start: date,
    custom_end: date,
    *,
    editable: bool,
) -> None:
    names_by_id = {portfolio.id: portfolio.name for portfolio in action_portfolios}
    render_transactions(
        reporting_portfolios,
        names_by_id,
        custom_start,
        custom_end,
        editable=editable,
    )
    if editable:
        render_add_transaction(action_portfolios, reporting_portfolios)
        render_csv_section(action_portfolios, reporting_portfolios)
        render_portfolio_actions(action_portfolios)
        return
    with st.expander("Portfolio administration", expanded=False):
        st.caption("Read-only portfolio settings and effective-dated benchmark history.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Portfolio": portfolio.name,
                        "Account type": ACCOUNT_TYPE_LABELS[portfolio.account_type],
                        "Cash": float(portfolio.cash),
                    }
                    for portfolio in action_portfolios
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={"Cash": st.column_config.NumberColumn(format="$%,.0f")},
        )
        benchmark_rows = []
        with session_scope() as session:
            for portfolio in action_portfolios:
                for configuration in benchmark_configurations(session, portfolio.id):
                    benchmark_rows.append(
                        {
                            "Portfolio": portfolio.name,
                            "Effective": configuration.effective_from,
                            "Components": ", ".join(
                                f"{symbol} {float(weight) * 100:g}%"
                                for symbol, weight in configuration.weights.items()
                            ),
                            "Rebalancing": configuration.rebalancing_frequency.title(),
                        }
                    )
        if benchmark_rows:
            st.dataframe(pd.DataFrame(benchmark_rows), hide_index=True, width="stretch")
        else:
            st.caption("No benchmark configurations have been recorded.")


def allocation_frame(
    allocations: list[OrderAllocation], portfolios: list[Portfolio]
) -> pd.DataFrame:
    names = {item.id: item.name for item in portfolios}
    return pd.DataFrame(
        [
            {
                "Portfolio": names.get(order.portfolio_id, str(order.portfolio_id)),
                "Symbol": order.symbol,
                "Side": order.side,
                "Type": order.order_type,
                "Requested": float(order.requested_qty),
                "Filled": float(order.filled_qty),
                "Avg fill": float(order.filled_avg_price) if order.filled_avg_price else None,
                "Status": order.status,
                "Submitted": order.submitted_at,
                "Alpaca order ID": order.alpaca_order_id,
            }
            for order in allocations
        ]
    )


def render_trade_admin(portfolios: list[Portfolio], *, editable: bool) -> None:
    settings = get_settings()
    with st.expander("Assigned order ticket", expanded=editable):
        if not editable:
            st.caption("Order entry is disabled for read-only users.")
        elif not portfolios:
            st.caption("No non-blank portfolios are available for order assignment.")
        else:
            if not settings.alpaca_configured:
                st.info("Add rotated Alpaca credentials to enable paper order submission.")
            mode_text = "Paper execution" if settings.paper else "Live execution"
            st.caption(f"{mode_text}. Every order must be assigned to one internal portfolio.")
            with st.form("trade_ticket"):
                columns = st.columns(3)
                portfolio_name = columns[0].selectbox(
                    "Portfolio", [item.name for item in portfolios]
                )
                symbol = columns[1].text_input("Symbol", max_chars=16)
                side = columns[2].selectbox("Side", ["Buy", "Sell"])
                columns = st.columns(3)
                order_type = columns[0].selectbox("Order type", ["Market", "Limit"])
                qty = columns[1].number_input(
                    "Shares", min_value=0.00000001, value=1.0, format="%.2f"
                )
                limit_price = columns[2].number_input(
                    "Limit price", min_value=0.0, value=0.0, format="%.4f"
                )
                confirmed = st.checkbox("I reviewed this order and its assigned portfolio.")
                submitted = st.form_submit_button(
                    "Submit assigned order",
                    width="stretch",
                    disabled=not settings.alpaca_configured,
                )
            if submitted:
                if not confirmed:
                    st.info("Review and confirm the order before submitting.")
                else:
                    selected = next(item for item in portfolios if item.name == portfolio_name)
                    try:
                        with session_scope() as session:
                            order = submit_allocated_order(
                                session,
                                selected.id,
                                symbol,
                                side,
                                qty,
                                order_type,
                                limit_price if order_type == "Limit" else None,
                            )
                        st.session_state.flash = (
                            f"Order submitted and assigned: {order.alpaca_order_id}"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.info(f"Order was not submitted: {exc}")

    with st.expander("Alpaca account and orders", expanded=False):
        if settings.alpaca_configured:
            try:
                account = get_trading_client().get_account()
                columns = st.columns(4)
                columns[0].metric("Equity", dollars(account.equity))
                columns[1].metric("Cash", dollars(account.cash))
                columns[2].metric("Buying power", dollars(account.buying_power))
                columns[3].metric("Status", str(getattr(account.status, "value", account.status)))
            except Exception as exc:
                st.info(f"Alpaca account data is unavailable: {exc}")
        with session_scope() as session:
            allocations = list_allocations(session)
        if editable:
            action_columns = st.columns([1, 1, 3])
            if action_columns[0].button(
                "Sync fills",
                width="stretch",
                disabled=not settings.alpaca_configured,
            ):
                try:
                    with session_scope() as session:
                        changed = sync_allocations(session)
                    cached_historical_data.clear()
                    st.session_state.flash = f"Fill sync complete. {changed} allocation(s) updated."
                    st.rerun()
                except Exception as exc:
                    st.info(f"Orders could not be synchronized: {exc}")
            open_orders = [
                item
                for item in allocations
                if item.status not in {"filled", "canceled", "expired", "rejected"}
            ]
            cancel_id = action_columns[1].selectbox(
                "Cancel",
                ["", *[item.alpaca_order_id for item in open_orders]],
                label_visibility="collapsed",
            )
            if action_columns[2].button(
                "Cancel selected order",
                disabled=not cancel_id or not settings.alpaca_configured,
            ):
                try:
                    cancel_order(cancel_id)
                    st.session_state.flash = "Cancellation requested. Sync fills to refresh status."
                    st.rerun()
                except Exception as exc:
                    st.info(f"Cancellation could not be requested: {exc}")
        else:
            st.caption("Fill synchronization and order cancellation are admin-only.")
        if allocations:
            st.dataframe(
                allocation_frame(allocations, portfolios),
                hide_index=True,
                width="stretch",
                column_config={
                    "Requested": st.column_config.NumberColumn(format="%.2f"),
                    "Filled": st.column_config.NumberColumn(format="%.2f"),
                    "Avg fill": st.column_config.NumberColumn(format="$%,.2f"),
                },
            )
        else:
            st.caption("No assigned orders have been submitted yet.")


def render_architecture() -> None:
    with st.expander("Extension architecture", expanded=True):
        st.caption(
            "These capabilities are designed into the service boundaries but intentionally "
            "inactive in the first release."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Capability": "Automated strategies",
                        "Initial state": "Strategy interface only",
                        "Activation requirement": "Scheduler, approvals, risk limits",
                    },
                    {
                        "Capability": "Short selling",
                        "Initial state": "Ledger supports negative shares",
                        "Activation requirement": "Borrow and exposure controls",
                    },
                    {
                        "Capability": "Options",
                        "Initial state": "Asset and position intents modeled",
                        "Activation requirement": "Contract picker and option order validation",
                    },
                    {
                        "Capability": "Live trading",
                        "Initial state": "Locked by configuration",
                        "Activation requirement": "Separate live keys and explicit safety review",
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def _hypothetical_sector_inputs(
    portfolios: list[Portfolio], proposed_symbols: set[str]
) -> dict[str, str | dict[str, float]]:
    symbols = {
        lot.symbol.upper() for portfolio in portfolios for lot in portfolio.holdings
    } | proposed_symbols
    resolved: dict[str, str | dict[str, float]] = {}
    with session_scope() as session:
        for symbol in symbols:
            metadata = resolve_security_metadata(session, symbol)
            if metadata.asset_type == "etf":
                snapshot = resolve_etf_sector_snapshot(session, symbol)
                if snapshot.weights:
                    resolved[symbol] = {
                        sector: float(weight) / 100 for sector, weight in snapshot.weights.items()
                    }
                    continue
            resolved[symbol] = metadata.sector or "Unclassified"
    return resolved


def _hypothetical_benchmark_weights(
    portfolios: list[Portfolio], prices: dict[str, float], as_of: date
) -> dict[str, float]:
    """Value-weight each portfolio's effective benchmark blend for scope comparison."""
    portfolio_values = {
        portfolio.id: float(portfolio.cash)
        + sum(float(lot.shares) * prices.get(lot.symbol.upper(), 0.0) for lot in portfolio.holdings)
        for portfolio in portfolios
    }
    total = sum(portfolio_values.values())
    combined: dict[str, float] = {}
    with session_scope() as session:
        for portfolio in portfolios:
            effective = [
                config
                for config in benchmark_configurations(session, portfolio.id)
                if config.effective_from <= as_of
            ]
            weights = effective[-1].weights if effective else {"SPY": 1.0}
            scope_weight = portfolio_values[portfolio.id] / total if total > 0 else 0.0
            for symbol, weight in weights.items():
                combined[symbol] = combined.get(symbol, 0.0) + scope_weight * float(weight)
    return combined or {"SPY": 1.0}


def render_hypothetical_analysis(
    portfolios: list[Portfolio],
    closes: pd.DataFrame,
    benchmark_closes: pd.DataFrame,
    *,
    editable: bool,
) -> None:
    st.subheader("Hypothetical trade analysis")
    st.caption(
        "Analysis only. This workflow cannot create accounting records, change internal "
        "allocations, or submit an Alpaca order. Any later ticket copy requires a new owner "
        "review and a fresh price."
    )
    actions: list[ProposedAction] = st.session_state.setdefault("hypothetical_actions", [])
    action_label = st.selectbox(
        "Proposed action",
        options=[item.value for item in HypotheticalActionType],
        format_func=lambda value: value.replace("_", " ").title(),
        key="hypothetical_action_type",
    )
    portfolio = st.selectbox(
        "Scenario portfolio", portfolios, format_func=lambda item: item.name, key="hypo_portfolio"
    )
    destination = st.selectbox(
        "Assignment destination",
        portfolios,
        format_func=lambda item: item.name,
        key="hypo_destination",
    )
    symbol = st.text_input("Hypothetical symbol", key="hypo_symbol").strip().upper()
    quantity_value = st.number_input(
        "Hypothetical quantity", min_value=0.0, value=0.0, key="hypo_quantity"
    )
    price_value = st.number_input(
        "Hypothetical analysis price", min_value=0.0, value=0.0, key="hypo_price"
    )
    amount_value = st.number_input(
        "Hypothetical cash amount", min_value=0.0, value=0.0, key="hypo_amount"
    )
    fees_value = st.number_input("Hypothetical fees", min_value=0.0, value=0.0, key="hypo_fees")
    add_column, clear_column = st.columns(2)
    if add_column.button("Add proposed action"):
        try:
            action_type = HypotheticalActionType(action_label)
            actions.append(
                ProposedAction(
                    action_type,
                    portfolio.id,
                    symbol=symbol or None,
                    quantity=quantity_value or None,
                    price=price_value or None,
                    amount=amount_value or None,
                    destination_portfolio_id=(
                        destination.id if action_type == HypotheticalActionType.REASSIGN else None
                    ),
                    fees=fees_value,
                )
            )
            st.session_state.hypothetical_actions = actions
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if clear_column.button("Clear proposed actions", disabled=not actions):
        st.session_state.hypothetical_actions = []
        st.session_state.pop("hypothetical_result", None)
        st.rerun()

    if actions:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Action": action.action.value.replace("_", " ").title(),
                        "Portfolio": next(
                            item.name for item in portfolios if item.id == action.portfolio_id
                        ),
                        "Symbol": action.symbol,
                        "Quantity": action.quantity,
                        "Price": action.price,
                        "Amount": action.amount,
                        "Destination": next(
                            (
                                item.name
                                for item in portfolios
                                if item.id == action.destination_portfolio_id
                            ),
                            None,
                        ),
                    }
                    for action in actions
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    expected_return = st.number_input(
        "Annual expected return assumption (%)",
        min_value=-99.0,
        max_value=100.0,
        value=7.0,
        key="hypo_expected_return",
    )
    target_enabled = st.checkbox("Set a hypothetical forecast target", key="hypo_target_on")
    target_value = (
        st.number_input(
            "Hypothetical forecast target",
            min_value=1.0,
            value=100000.0,
            key="hypo_target",
        )
        if target_enabled
        else None
    )
    retirement_enabled = st.checkbox(
        "Include retirement success analysis", key="hypo_retirement_on"
    )
    retirement_horizon = 20
    retirement_spending = 0.0
    if retirement_enabled:
        retirement_horizon = st.select_slider(
            "Retirement analysis horizon (years)",
            options=list(range(20, 41)),
            value=20,
            key="hypo_retirement_horizon",
        )
        retirement_spending = st.number_input(
            "Annual retirement spending assumption",
            min_value=0.0,
            value=40000.0,
            key="hypo_retirement_spending",
        )
    if st.button("Run hypothetical analysis", disabled=not actions):
        try:
            if closes.empty:
                raise ValueError(
                    "Confirmed market prices are required for hypothetical trade analysis."
                )
            household = household_valuation(portfolios, closes)
            if not household.is_complete or household.common_valuation_date is None:
                raise ValueError(
                    "A common confirmed household valuation is required for hypothetical analysis."
                )
            prices = dict(household.confirmed_prices)
            proposed_symbols = {action.symbol for action in actions if action.symbol}
            expected = {symbol: expected_return / 100 for symbol in set(prices) | proposed_symbols}
            returns = closes.pct_change(fill_method=None)
            benchmark = None
            if "SPY" in benchmark_closes and not benchmark_closes["SPY"].dropna().empty:
                benchmark = benchmark_closes["SPY"].pct_change(fill_method=None)
            as_of = pd.Timestamp(closes.index.max()).to_pydatetime()
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=timezone.utc)
            assumptions = HypotheticalAssumptions(
                expected,
                _hypothetical_sector_inputs(portfolios, proposed_symbols),
                _hypothetical_benchmark_weights(portfolios, prices, as_of.date()),
                {"Broad market": -0.20},
                forecast_target=float(target_value) if target_value else None,
                retirement=(
                    RetirementAnalysisAssumptions(retirement_horizon, float(retirement_spending))
                    if retirement_enabled
                    else None
                ),
            )
            result = analyze_hypothetical_scenario(
                baseline_from_portfolios(portfolios),
                tuple(actions),
                prices,
                returns,
                assumptions,
                market_data_as_of=as_of,
                benchmark_returns=benchmark,
                common_confirmed_valuation_date=household.common_valuation_date,
                latest_symbol_dates=household.latest_symbol_dates,
            )
            st.session_state.hypothetical_result = (result, assumptions)
        except ValueError as exc:
            st.error(str(exc))

    stored = st.session_state.get("hypothetical_result")
    if stored:
        result, assumptions = stored
        before, after = result.before, result.after
        columns = st.columns(4)
        columns[0].metric("Before value", dollars(before.total_value))
        columns[1].metric("After value", dollars(after.total_value))
        columns[2].metric("Before cash", dollars(before.cash))
        columns[3].metric("After cash", dollars(after.cash))
        st.dataframe(
            pd.DataFrame(
                {
                    "Before": {
                        "Cash": before.cash,
                        "Market value": before.market_value,
                        "Cost basis": before.cost_basis,
                    },
                    "After": {
                        "Cash": after.cash,
                        "Market value": after.market_value,
                        "Cost basis": after.cost_basis,
                    },
                }
            ),
            width="stretch",
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Before": before.holding_weights,
                    "After": after.holding_weights,
                }
            ).fillna(0),
            width="stretch",
        )
        st.markdown("#### Portfolio assignment and benchmark-relative exposure")
        st.dataframe(
            pd.DataFrame(
                {
                    "Before assignment": before.assignment_weights,
                    "After assignment": after.assignment_weights,
                }
            ).fillna(0),
            width="stretch",
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Before relative": before.benchmark_relative_exposure,
                    "After relative": after.benchmark_relative_exposure,
                }
            ).fillna(0),
            width="stretch",
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Before sector": before.sector_exposure,
                    "After sector": after.sector_exposure,
                }
            ).fillna(0),
            width="stretch",
        )
        risk_rows = {
            "Volatility": (before.volatility, after.volatility),
            "Beta": (before.beta, after.beta),
            "Maximum drawdown exposure": (
                before.drawdown_exposure,
                after.drawdown_exposure,
            ),
            "Effective holdings": (
                before.effective_number_of_holdings,
                after.effective_number_of_holdings,
            ),
            "Expected return": (before.expected_return, after.expected_return),
            "Forecast target probability": (
                before.forecast_target_probability,
                after.forecast_target_probability,
            ),
            "Depletion Probability": (
                before.depletion_probability,
                after.depletion_probability,
            ),
        }
        st.dataframe(
            pd.DataFrame.from_dict(risk_rows, orient="index", columns=["Before", "After"]),
            width="stretch",
        )
        st.caption(
            "Depletion Probability is from the adjacent simplified hypothetical model: fixed "
            "monthly spending and seeded lognormal returns only. It omits inflation, taxes, "
            "outside income, fees, contributions, account types, and withdrawal ordering and is "
            "not the full retirement engine."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Before concentration": before.concentration,
                    "After concentration": after.concentration,
                }
            ),
            width="stretch",
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Before risk contribution": before.risk_contribution,
                    "After risk contribution": after.risk_contribution,
                }
            ).fillna(0),
            width="stretch",
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Before downside": before.downside_percentiles,
                    "After downside": after.downside_percentiles,
                }
            ),
            width="stretch",
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Before stress loss": before.deterministic_stress_losses,
                    "After stress loss": after.deterministic_stress_losses,
                }
            ),
            width="stretch",
        )
        for warning in result.warnings:
            _render_warning(warning)
        if editable:
            scenario_name = st.text_input("Saved scenario name", key="hypo_scenario_name")
            creator = st.text_input("Scenario creator", value="owner", key="hypo_creator")
            if st.button("Save hypothetical scenario"):
                try:
                    with session_scope() as session:
                        save_hypothetical_scenario(
                            session,
                            name=scenario_name,
                            creator=creator,
                            portfolios=portfolios,
                            market_data_as_of=result.market_data_as_of,
                            assumptions=assumptions,
                            actions=actions,
                            result=result,
                        )
                    st.success("Hypothetical scenario saved without changing accounting or orders.")
                except ValueError as exc:
                    st.error(str(exc))

    with session_scope() as session:
        saved = load_hypothetical_scenarios(session)
    if saved:
        st.markdown("#### Saved scenarios")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Name": item.name,
                        "Creator": item.creator,
                        "Created": item.created_at,
                        "Market data as of": item.market_data_as_of,
                        "Stale baseline": item.stale_baseline,
                    }
                    for item in saved
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        if stale_names := [item.name for item in saved if item.stale_baseline]:
            st.warning(
                "Saved scenario baseline is stale and must be rerun before any ticket review: "
                + ", ".join(stale_names)
            )


def _latest_closes(closes: pd.DataFrame) -> dict[str, float]:
    return {
        symbol: float(values.iloc[-1])
        for symbol in closes
        if not (values := closes[symbol].dropna()).empty
    }


@st.fragment(run_every="30s")
def render_active_monitoring(
    portfolios: list[Portfolio], closes: pd.DataFrame, *, editable: bool
) -> None:
    settings = get_settings()
    symbols = sorted({lot.symbol for portfolio in portfolios for lot in portfolio.holdings})
    hours = market_hours_state()
    st.caption(
        f"Active browser-session monitoring · {hours.label}. "
        "Monitoring pauses when this app session closes or the device sleeps; no closed-app alerts are claimed."
    )
    if not symbols:
        if session_id := st.session_state.get("realtime_session_id"):
            active_session_registry().release(session_id)
        st.caption("No held symbols are available to monitor.")
        return

    previous_closes = _latest_closes(closes)
    position_values: dict[str, float] = {}
    selected_portfolio_symbols = {
        lot.symbol for portfolio in portfolios for lot in portfolio.holdings
    }
    for portfolio in portfolios:
        for lot in portfolio.holdings:
            position_values[lot.symbol] = position_values.get(lot.symbol, 0.0) + float(
                lot.shares
            ) * previous_closes.get(lot.symbol, float(lot.cost_basis))
    visible = set(
        sorted(position_values, key=lambda symbol: (-abs(position_values[symbol]), symbol))[:12]
    )
    selected_symbol = st.session_state.get("monitor_selected_symbol", symbols[0])
    if selected_symbol not in symbols:
        selected_symbol = symbols[0]
        st.session_state.monitor_selected_symbol = selected_symbol
    with session_scope() as session:
        allocations = list_allocations(session)
    open_allocations = [item for item in allocations if item.status in OPEN_ORDER_STATUSES]
    open_order_symbols = {item.symbol for item in open_allocations}

    if settings.alpaca_configured:
        session_id = st.session_state.setdefault("realtime_session_id", uuid4().hex)
        monitor = active_session_registry().acquire(session_id, _new_active_monitor)
        inputs = SubscriptionInputs(
            held_symbols=frozenset(symbols),
            open_order_symbols=frozenset(open_order_symbols),
            selected_symbol=selected_symbol,
            visible_symbols=frozenset(visible),
            selected_portfolio_symbols=frozenset(selected_portfolio_symbols),
            risk_contributions=position_risk_contributions(closes, position_values),
            position_values=position_values,
        )
        try:
            plan = monitor.refresh(inputs, previous_closes=previous_closes)
            records = monitor.records(symbols)
            stream_state = (
                "stream session active"
                if monitor.websocket.connected
                else "stream session starting/reconnecting"
            )
            st.caption(
                f"Alpaca {settings.alpaca_data_feed.upper()} · {stream_state} · "
                f"{len(plan.streamed)} stream-priority / "
                f"{len(plan.snapshot)} scheduled REST fallback. "
                "Initial and stale stream symbols use snapshot reconciliation. "
                "IEX-derived values are indicative, not consolidated-market values."
            )
            if monitor.websocket.last_error:
                st.caption(
                    "The stream is reconnecting after a transient provider connection error."
                )
        except Exception as exc:
            st.info(f"Active quotes are temporarily unavailable: {type(exc).__name__}.")
            records = monitor.records(symbols)
    else:
        if session_id := st.session_state.get("realtime_session_id"):
            active_session_registry().release(session_id)
        book = QuoteBook()
        book.seed_previous_closes(previous_closes)
        records = book.records(symbols)
        st.info(
            "Configure rotated Alpaca credentials to start active-session streaming and snapshots. "
            "Durable previous closes are shown where available."
        )

    pulse = build_portfolio_pulse(portfolios, records)
    metrics = st.columns(4)
    _render_metric(
        metrics[0],
        "Selected Totals",
        dollars(pulse.indicative_total_value)
        if pulse.indicative_total_value is not None
        else "Unavailable",
        key="monitor_selected_totals",
        stale=bool(
            hours.is_regular_hours
            and pulse.stale_or_missing
            and pulse.indicative_total_value is not None
        ),
    )
    _render_metric(
        metrics[1],
        "Daily change",
        dollars(pulse.daily_change) if pulse.daily_change is not None else "Unavailable",
        key="monitor_daily_change",
        stale=bool(
            hours.is_regular_hours and pulse.stale_or_missing and pulse.daily_change is not None
        ),
    )
    metrics[2].metric("Held symbols", len(symbols))
    metrics[3].metric("Stale / missing", len(pulse.stale_or_missing))

    holding_rows = pd.DataFrame(
        [
            {
                "Symbol": row.symbol,
                "Price": row.price,
                "Value": row.value,
                "Daily change": row.daily_change,
                "Share of net daily P/L": row.contribution,
                "Freshness": freshness_label(row.status),
                "As of": format_eastern_timestamp(records[row.symbol].as_of_time),
                "Feed": records[row.symbol].feed,
                "Provider": records[row.symbol].provider,
                "Staleness reason": records[row.symbol].staleness_reason,
                "_stale": hours.is_regular_hours
                and row.status
                not in {FreshnessStatus.STREAMING, FreshnessStatus.RECENTLY_REFRESHED},
            }
            for row in pulse.holdings
        ]
    )
    if not holding_rows.empty:
        st.markdown("#### Largest movers and holding contribution")
        holding_rows["Share of net daily P/L"] *= 100
        holding_rows["_absolute mover"] = pd.to_numeric(
            holding_rows["Daily change"], errors="coerce"
        ).abs()
        holding_rows = holding_rows.sort_values(
            "_absolute mover", ascending=False, na_position="last"
        ).drop(columns="_absolute mover")
        stale_holding_rows = holding_rows.pop("_stale")
        styled_holdings = (
            _style_stale_values(
                holding_rows,
                stale_holding_rows,
                ("Price", "Value", "Daily change", "Share of net daily P/L"),
            )
            if hours.is_regular_hours
            else holding_rows
        )
        st.dataframe(
            styled_holdings,
            hide_index=True,
            width="stretch",
            column_config={
                "Price": st.column_config.NumberColumn(format="$%,.2f"),
                "Value": st.column_config.NumberColumn(format="$%,.0f"),
                "Daily change": st.column_config.NumberColumn(format="$%,.0f"),
                "Share of net daily P/L": st.column_config.NumberColumn(format="%.0f%%"),
            },
        )
    portfolio_rows = pd.DataFrame(
        [
            {
                "Portfolio": name,
                "Daily contribution": change,
                "_stale": hours.is_regular_hours
                and any(
                    records[lot.symbol].status
                    not in {FreshnessStatus.STREAMING, FreshnessStatus.RECENTLY_REFRESHED}
                    for portfolio in portfolios
                    if portfolio.name == name
                    for lot in portfolio.holdings
                ),
            }
            for name, change in pulse.by_portfolio.items()
        ]
    )
    if not portfolio_rows.empty:
        stale_portfolio_rows = portfolio_rows.pop("_stale")
        st.dataframe(
            (
                _style_stale_values(portfolio_rows, stale_portfolio_rows, ("Daily contribution",))
                if hours.is_regular_hours
                else portfolio_rows
            ),
            hide_index=True,
            width="stretch",
            column_config={"Daily contribution": st.column_config.NumberColumn(format="$%,.0f")},
        )
    if pulse.stale_or_missing and hours.is_regular_hours:
        stale_symbols = escape(", ".join(pulse.stale_or_missing))
        st.markdown(
            f'<div class="stale-symbol-alert">Stale or missing symbols: {stale_symbols}</div>',
            unsafe_allow_html=True,
        )
    elif pulse.stale_or_missing:
        st.caption("Stale or missing symbols: " + ", ".join(pulse.stale_or_missing))

    quote = records[selected_symbol]
    exposure_shares = sum(
        float(lot.shares)
        for portfolio in portfolios
        for lot in portfolio.holdings
        if lot.symbol == selected_symbol
    )
    state = [item.status for item in allocations if item.symbol == selected_symbol]
    st.markdown("#### Symbol detail")
    detail_columns = st.columns([1, 4], vertical_alignment="center")
    with detail_columns[0]:
        st.selectbox("Symbol", symbols, key="monitor_selected_symbol")
    with detail_columns[1]:
        symbol_detail = pd.DataFrame(
            [
                {
                    "Symbol": selected_symbol,
                    "Latest trade": quote.latest_trade,
                    "Bid": quote.bid,
                    "Ask": quote.ask,
                    "Midpoint": quote.midpoint,
                    "Spread": quote.spread,
                    "Quote time": format_eastern_timestamp(quote.quote_time),
                    "Trade time": format_eastern_timestamp(quote.trade_time),
                    "Receipt time": format_eastern_timestamp(quote.receipt_time),
                    "As of": format_eastern_timestamp(quote.as_of_time),
                    "Feed": quote.feed,
                    "Freshness": freshness_label(quote.status),
                    "Exposure shares": exposure_shares,
                    "Exposure value": exposure_shares * quote.price if quote.price else None,
                    "Order state": ", ".join(state) if state else "none",
                }
            ]
        )
        styled_symbol_detail = (
            _style_stale_values(
                symbol_detail,
                pd.Series(
                    [
                        quote.status
                        not in {
                            FreshnessStatus.STREAMING,
                            FreshnessStatus.RECENTLY_REFRESHED,
                        }
                    ]
                ),
                ("Latest trade", "Bid", "Ask", "Midpoint", "Spread", "Exposure value"),
            )
            if hours.is_regular_hours
            else symbol_detail
        )
        st.dataframe(
            styled_symbol_detail,
            hide_index=True,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="$%,.2f")
                for column in ("Latest trade", "Bid", "Ask", "Midpoint", "Spread")
            }
            | {
                "Exposure shares": st.column_config.NumberColumn(format="%.2f"),
                "Exposure value": st.column_config.NumberColumn(format="$%,.0f"),
            },
        )

    order_columns = st.columns(2)
    with order_columns[0]:
        st.markdown("#### Open orders")
        if open_allocations:
            st.dataframe(
                allocation_frame(open_allocations, portfolios), hide_index=True, width="stretch"
            )
        else:
            st.caption("No open assigned orders.")
    with order_columns[1]:
        st.markdown("#### Recent fills")
        recent_fills = [item for item in allocations if float(item.filled_qty) > 0][:20]
        if recent_fills:
            st.dataframe(
                allocation_frame(recent_fills, portfolios), hide_index=True, width="stretch"
            )
        else:
            st.caption("No recent assigned fills.")

    st.markdown("#### Market context")
    context_request = HistoricalDataRequest(
        tuple(dict.fromkeys((*BROAD_MARKET_PROXIES, *SECTOR_PROXIES))),
        date.today() - timedelta(days=400),
        date.today(),
        "benchmark_total_return",
    )
    try:
        context = market_context_metrics(cached_historical_data(context_request))
    except Exception:
        context = pd.DataFrame()
    if context.empty:
        st.caption("Market-context history is unavailable.")
    else:
        percentage_columns = (
            "Daily return",
            "1M return",
            "3M return",
            "12M return",
            "Drawdown from available-window peak",
            "Realized volatility",
            "21-session SPY correlation",
        )
        context = context.copy()
        context[list(percentage_columns)] *= 100
        st.dataframe(
            context,
            hide_index=True,
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(format="%.0f%%") for name in percentage_columns
            },
        )
        st.caption(
            "Components are disclosed individually: horizon-specific returns and coverage, "
            "50-day trend, drawdown from the available-window peak, 21-session realized "
            "volatility, and raw 21-session SPY correlation with n/21 and aligned endpoint dates. "
            "Unavailable horizons and correlations remain unavailable. No composite market score "
            "is used."
        )
        st.caption(CORRELATION_HEURISTIC_DISCLOSURE)


def main() -> None:
    role = authenticate_access()
    if role is None:
        return
    owner = role == "admin"
    render_access_status(role)
    all_portfolios = initialize_application()
    reporting_portfolios = [
        portfolio for portfolio in all_portfolios if portfolio_has_data(portfolio)
    ]
    selected_portfolios, master_start, master_end = render_master_controls(reporting_portfolios)
    if flash := st.session_state.pop("flash", None):
        st.info(flash)
    labels = [
        "Overview",
        "Monitor",
        "Compare",
        "Forecast",
        "Hypothetical",
        "Manage",
        "Trade",
        "Architecture",
    ]
    if st.session_state.get("active_page") not in {None, *labels}:
        st.session_state.active_page = labels[0]
    tabs = st.tabs(labels, key="active_page", on_change="rerun")
    closes = pd.DataFrame()
    data_note = None
    benchmark_closes = pd.DataFrame()
    benchmark_note = None
    if any(tab.open for tab in tabs[:5]):
        closes, data_note = get_prices(selected_portfolios, master_start, master_end)
    if any(tabs[index].open for index in (0, 2, 4)):
        benchmark_closes, benchmark_note = get_alpha_beta_benchmark(master_start, master_end)
    if tabs[0].open:
        with tabs[0]:
            render_overview(
                selected_portfolios,
                closes,
                data_note,
                master_start,
                master_end,
                benchmark_closes,
            )
    elif tabs[1].open:
        with tabs[1]:
            render_active_monitoring(selected_portfolios, closes, editable=owner)
    elif tabs[2].open:
        with tabs[2]:
            render_compare(
                selected_portfolios,
                closes,
                master_start,
                master_end,
                benchmark_closes,
            )
    elif tabs[3].open:
        with tabs[3]:
            render_forecast(selected_portfolios, closes, owner=owner)
    elif tabs[4].open:
        with tabs[4]:
            render_hypothetical_analysis(
                selected_portfolios,
                closes,
                benchmark_closes,
                editable=owner,
            )
    elif tabs[5].open:
        with tabs[5]:
            render_portfolio_admin(
                selected_portfolios,
                all_portfolios,
                master_start,
                master_end,
                editable=owner,
            )
    elif tabs[6].open:
        with tabs[6]:
            render_trade_admin(reporting_portfolios, editable=owner)
    elif tabs[7].open:
        with tabs[7]:
            render_architecture()
    if benchmark_note:
        st.caption(f"SPY Alpha/Beta data is unavailable: {benchmark_note}")


if __name__ == "__main__":
    main()
