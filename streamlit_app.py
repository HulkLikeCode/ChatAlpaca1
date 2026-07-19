from __future__ import annotations

import hmac
from datetime import date, timedelta
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chat_alpaca.analytics import (
    combined_performance_growth,
    consolidated_holdings,
    performance_growth,
    portfolio_gain_loss,
    portfolio_valuation,
    rebase_comparison_series,
    summary_metrics,
)
from chat_alpaca.config import get_settings
from chat_alpaca.db import init_database, session_scope
from chat_alpaca.forecasting import ProjectionResult, simulate_portfolio_projection
from chat_alpaca.market_data import get_benchmark_daily_closes, get_daily_closes
from chat_alpaca.models import OrderAllocation, Portfolio, PortfolioTransaction
from chat_alpaca.portfolio_service import (
    MANUAL_KINDS,
    TransactionDraft,
    create_portfolio,
    delete_portfolio,
    delete_transaction,
    format_short_date,
    import_statement,
    list_portfolios,
    list_transactions_for_portfolios,
    money,
    normalize_symbol,
    parse_short_date,
    parse_statement_csv,
    portfolio_cost,
    portfolio_income_events,
    portfolio_income_summary,
    rebuild_portfolio_from_csv,
    record_transaction,
    rename_portfolio,
    seed_database,
    shares,
    update_transaction,
)
from chat_alpaca.theme import PLOT_COLORS, THEME_CSS
from chat_alpaca.trading import (
    cancel_order,
    get_trading_client,
    list_allocations,
    submit_allocated_order,
    sync_allocations,
)

BENCHMARKS = ("SPY", "QQQ", "DIA", "IWM")
ALL_PORTFOLIOS_OPTION = "__all_portfolios__"
SELECT_PORTFOLIO_OPTION = "__select_portfolio__"
EDITABLE_KINDS = (*MANUAL_KINDS, "opening_position")
TRADE_KINDS = {"buy", "sell"}
SYMBOL_CASH_KINDS = {"dividend", "fee", "tax"}
MASTER_DEFAULT_START = date(2026, 5, 15)
CSV_TEMPLATE = """Date,Action,Symbol,Description,Quantity,Price,Fees & Comm,Amount
7/15/2026,Buy,AAPL,Apple Inc,10,$210.00,$0.00,"($2,100.00)"
7/16/2026,Cash Dividend,AAPL,Apple dividend,,,,15.50
7/17/2026,MoneyLink Transfer,,Cash contribution,,,,1000.00
"""

st.set_page_config(
    page_title="ChatAlpaca · Portfolio Command Center",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(THEME_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def cached_closes(symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    return get_daily_closes(list(symbols), start, end)


@st.cache_data(ttl=900, show_spinner=False)
def cached_benchmark_closes(symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    return get_benchmark_daily_closes(list(symbols), start, end)


@st.cache_data(show_spinner=False)
def cached_projection(
    current_value: float,
    annual_return: float,
    annual_volatility: float,
    monthly_contribution: float,
    horizon_years: int,
    target_value: float | None,
) -> ProjectionResult:
    return simulate_portfolio_projection(
        current_value=current_value,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        monthly_contribution=monthly_contribution,
        horizon_years=horizon_years,
        target_value=target_value,
    )


def dollars(value: object) -> str:
    return f"${float(value):,.2f}"


def quantity(value: object) -> str:
    numeric = float(value)
    return f"{numeric:,.2f}"


def kind_label(value: str) -> str:
    return value.replace("_", " ").title()


def transaction_draft(
    transaction_date_text: str,
    kind: str,
    symbol: str,
    description: str,
    raw_quantity: float,
    raw_price: float,
    raw_fees: float,
    raw_cash_delta: float,
    action: str | None = None,
) -> TransactionDraft:
    position_kind = kind in {"buy", "sell", "opening_position"}
    parsed_quantity = shares(raw_quantity) if position_kind else None
    parsed_price = money(raw_price) if position_kind else None
    parsed_fees = money(raw_fees) if raw_fees else None
    if kind == "buy":
        assert parsed_quantity is not None and parsed_price is not None
        parsed_cash_delta = -(parsed_quantity * parsed_price + (parsed_fees or 0))
    elif kind == "sell":
        assert parsed_quantity is not None and parsed_price is not None
        parsed_cash_delta = parsed_quantity * parsed_price - (parsed_fees or 0)
    elif kind == "opening_position":
        parsed_cash_delta = money(0)
    else:
        parsed_cash_delta = money(raw_cash_delta)
    return TransactionDraft(
        transaction_date=parse_short_date(transaction_date_text),
        action=(action or kind_label(kind)).strip(),
        kind=kind,
        symbol=normalize_symbol(symbol) if symbol.strip() else None,
        description=description.strip(),
        quantity=parsed_quantity,
        price=parsed_price,
        fees=parsed_fees,
        cash_delta=parsed_cash_delta,
    )


def load_portfolios() -> list[Portfolio]:
    with session_scope() as session:
        return list_portfolios(session)


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
            selected_ids = columns[0].multiselect(
                "Portfolios",
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

        scope_label = "All Portfolios" if applied_all else f"{len(applied_ids)} portfolios"
        st.caption(f"Applied: {scope_label} · {applied_start:%m/%d/%y}–{applied_end:%m/%d/%y}")

    selected = [portfolio for portfolio in portfolios if portfolio.id in set(applied_ids)]
    return selected, applied_start, applied_end


def get_prices(
    portfolios: list[Portfolio], start: date, extra: tuple[str, ...] = ()
) -> tuple[pd.DataFrame, str | None]:
    symbols = (
        {lot.symbol for portfolio in portfolios for lot in portfolio.holdings}
        | {
            transaction.symbol
            for portfolio in portfolios
            for transaction in portfolio.transactions
            if transaction.symbol
        }
        | set(extra)
    )
    if not symbols:
        return pd.DataFrame(), None
    try:
        return cached_closes(tuple(sorted(symbols)), start, date.today()), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def authenticate_owner() -> bool:
    settings = get_settings()
    with st.sidebar:
        st.markdown("### Owner access")
        if st.session_state.get("owner_authenticated"):
            st.caption("Owner controls unlocked for this browser session.")
            if st.button("Lock owner controls", width="stretch"):
                st.session_state.owner_authenticated = False
                st.rerun()
            return True
        if not settings.admin_password:
            st.caption("Set ADMIN_PASSWORD to enable owner controls.")
            return False
        with st.form("owner_login", border=False):
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Unlock", width="stretch")
        if submitted:
            authenticated = hmac.compare_digest(password, settings.admin_password)
            st.session_state.owner_authenticated = authenticated
            if authenticated:
                st.rerun()
            st.info("That password did not match.")
        return False


def render_header() -> None:
    settings = get_settings()
    mode = "PAPER MODE" if settings.paper else "LIVE MODE"
    st.markdown(f'<span class="mode-chip">{mode}</span>', unsafe_allow_html=True)
    st.title("KC's Retirement Dough, Let's GO!!!")


def render_portfolio_cards(portfolios: list[Portfolio], closes: pd.DataFrame) -> None:
    cards = []
    for portfolio in portfolios:
        if closes.empty:
            total = portfolio_cost(portfolio)
            value_label = f"Cost basis {dollars(total)}"
        else:
            valuation = portfolio_valuation(portfolio, closes)
            value_label = (
                dollars(valuation.total_calculated_value)
                if valuation.is_complete
                else "Incomplete valuation"
            )
        unique_symbols = len({lot.symbol for lot in portfolio.holdings})
        cards.append(
            "".join(
                (
                    '<div class="portfolio-card">',
                    f'<div class="eyebrow">{escape(portfolio.name)}</div>',
                    f'<div class="value">{value_label}</div>',
                    f'<div class="detail">{unique_symbols} symbols · '
                    f"{dollars(portfolio.cash)} cash</div>",
                    "</div>",
                )
            )
        )
    st.markdown(
        f'<div class="portfolio-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _summed_metric(values: list[float | None]) -> float | None:
    return sum(values) if values and all(value is not None for value in values) else None


def _metric_dollars(value: float | None) -> str:
    return dollars(value) if value is not None else "—"


def render_performance_summary(
    portfolios: list[Portfolio],
    closes: pd.DataFrame,
    custom_start: date,
    custom_end: date,
    key_prefix: str,
    expanded: bool = False,
) -> None:
    with st.expander("Portfolio value and gain/loss", expanded=expanded):
        if closes.empty:
            total_value = sum(
                (portfolio_cost(portfolio) for portfolio in portfolios), start=money(0)
            )
            rows = []
        else:
            valuations = [portfolio_valuation(portfolio, closes) for portfolio in portfolios]
            total_value = (
                sum(
                    (valuation.total_calculated_value for valuation in valuations),
                    start=money(0),
                )
                if all(valuation.is_complete for valuation in valuations)
                else None
            )
            rows = [
                {
                    "Portfolio": portfolio.name,
                    **portfolio_gain_loss(portfolio, closes, custom_start, custom_end).__dict__,
                }
                for portfolio in portfolios
            ]

        metrics = st.columns(4)
        total_label = "Selected cost basis + cash" if closes.empty else "Total selected value"
        metrics[0].metric(total_label, dollars(total_value) if total_value is not None else "—")
        metrics[1].metric(
            "All-time gain/loss",
            _metric_dollars(_summed_metric([row["all_time"] for row in rows])),
        )
        metrics[2].metric(
            "Daily gain/loss",
            _metric_dollars(_summed_metric([row["daily"] for row in rows])),
        )
        metrics[3].metric(
            "Custom gain/loss",
            _metric_dollars(_summed_metric([row["custom"] for row in rows])),
        )
        if closes.empty:
            st.caption("Cost basis plus cash is shown; gain/loss requires market data.")
            return
        warnings = sorted({warning for row in rows for warning in row["warnings"]})
        for warning in warnings:
            st.warning(warning)

        performance = pd.DataFrame(
            [
                {
                    "Portfolio": row["Portfolio"],
                    "All-time gain/loss": row["all_time"],
                    "Daily gain/loss": row["daily"],
                    "Custom gain/loss": row["custom"],
                }
                for row in rows
            ]
        )
        st.dataframe(
            performance,
            hide_index=True,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="$%,.2f")
                for column in (
                    "All-time gain/loss",
                    "Daily gain/loss",
                    "Custom gain/loss",
                )
            },
            key=f"{key_prefix}_portfolio_gain_loss",
        )
        st.caption(
            "Gain/loss excludes transfers, cash adjustments, and contributed opening positions. "
            "Daily uses the two latest market closes."
        )


def render_consolidated_holdings(
    portfolios: list[Portfolio],
    closes: pd.DataFrame,
    custom_start: date,
    custom_end: date,
) -> None:
    with st.expander("Exact holdings", expanded=False):
        summary, detail = consolidated_holdings(portfolios, closes, custom_start, custom_end)
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
            "Latest price": "Current",
            "Market value": "Value",
            "All-time gain/loss": "All time",
            "Daily gain/loss": "Day",
            "Custom gain/loss": "Custom",
        }
        money_columns = (
            "Avg/share",
            "Current",
            "Cost basis",
            "Value",
            "All time",
            "Day",
            "Custom",
        )
        if view == "Summary":
            summary_columns = [
                "Symbol",
                "Avg/share",
                "Current",
                "Cost basis",
                "Value",
                "All time",
                "Day",
                "Custom",
                "Shares",
                "Portfolios",
            ]
            summary_view = summary.rename(columns=common_renames)[summary_columns]
            st.dataframe(
                summary_view,
                hide_index=True,
                width="stretch",
                column_order=summary_columns,
                column_config={
                    "Shares": st.column_config.NumberColumn(format="%.2f"),
                    **{
                        column: st.column_config.NumberColumn(format="$%,.2f")
                        for column in money_columns
                    },
                },
            )
        else:
            detail_columns = [
                "Symbol",
                "Avg/share",
                "Current",
                "Cost basis",
                "Value",
                "All time",
                "Day",
                "Custom",
                "Shares",
                "Portfolio",
                "Acquired",
            ]
            detail_view = detail.rename(columns=common_renames)[detail_columns]
            st.dataframe(
                detail_view,
                hide_index=True,
                width="stretch",
                column_order=detail_columns,
                column_config={
                    "Shares": st.column_config.NumberColumn(format="%.2f"),
                    "Acquired": st.column_config.DateColumn(format="M/D/YY"),
                    **{
                        column: st.column_config.NumberColumn(format="$%,.2f")
                        for column in money_columns
                    },
                },
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
    for index, income_type in enumerate(("Dividend", "Interest")):
        values = monthly[monthly["Income type"] == income_type]
        figure.add_trace(
            go.Bar(
                x=values["Month"],
                y=values["Cash received"],
                name=income_type,
                marker_color=PLOT_COLORS[index % len(PLOT_COLORS)],
                hovertemplate="%{x|%b %Y}<br>$%{y:,.2f}<extra>" + income_type + "</extra>",
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
        column_config={"Cash received": st.column_config.NumberColumn(format="$%,.2f")},
    )


def render_cash_positions(portfolios: list[Portfolio]) -> None:
    rows = [
        {"Portfolio": portfolio.name, "Cash": float(portfolio.cash)} for portfolio in portfolios
    ]
    rows.append({"Portfolio": "Total", "Cash": sum(row["Cash"] for row in rows)})
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"Cash": st.column_config.NumberColumn(format="$%,.2f")},
    )


def render_overview(
    portfolios: list[Portfolio],
    closes: pd.DataFrame,
    data_note: str | None,
    custom_start: date,
    custom_end: date,
) -> None:
    if not portfolios:
        st.caption("No non-blank portfolios are available for the selected scope.")
        return
    render_performance_summary(
        portfolios, closes, custom_start, custom_end, "overview", expanded=True
    )
    with st.expander("Portfolio values", expanded=False):
        render_portfolio_cards(portfolios, closes)
        if data_note:
            st.info(f"Live market values are unavailable, so cost basis is shown. {data_note}")
    with st.expander("Portfolio income", expanded=False):
        render_portfolio_income(portfolios, custom_start, custom_end)
    render_consolidated_holdings(portfolios, closes, custom_start, custom_end)
    with st.expander("Cash positions", expanded=False):
        render_cash_positions(portfolios)


def render_compare(
    portfolios: list[Portfolio],
    portfolio_closes: pd.DataFrame,
    custom_start: date,
    custom_end: date,
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
        expanded=True,
    )
    with st.expander("Performance comparison", expanded=False):
        controls = st.columns([1.4, 2.2])
        with controls[0]:
            selected_benchmarks = st.multiselect(
                "Benchmark ETFs",
                BENCHMARKS,
                default=["SPY"],
                key="compare_benchmarks",
            )
        with controls[1]:
            extra_text = st.text_input(
                "Additional stocks or ETFs",
                placeholder="AAPL, MSFT, VTI",
                key="compare_extras",
                help="Enter one or more symbols, separated by commas.",
            )
        extras = tuple(
            dict.fromkeys(
                symbol.strip().upper() for symbol in extra_text.split(",") if symbol.strip()
            )
        )
        all_symbols = tuple(
            sorted(
                {lot.symbol for portfolio in portfolios for lot in portfolio.holdings}
                | {
                    transaction.symbol
                    for portfolio in portfolios
                    for transaction in portfolio.transactions
                    if transaction.symbol
                }
                | set(selected_benchmarks)
                | set(extras)
            )
        )
        try:
            closes = cached_closes(all_symbols, custom_start - timedelta(days=7), custom_end)
            benchmark_symbols = tuple(dict.fromkeys([*selected_benchmarks, *extras]))
            benchmark_closes = cached_benchmark_closes(benchmark_symbols, custom_start, custom_end)
        except Exception as exc:
            st.info(f"Comparison data is unavailable. Configure rotated Alpaca credentials. {exc}")
            return
        if closes.empty:
            st.info("No market data was returned for this comparison.")
            return

        per_portfolio = {
            portfolio.name: performance_growth(portfolio, closes) for portfolio in portfolios
        }
        combined = combined_performance_growth(portfolios, closes)
        series: list[pd.Series] = [combined, *per_portfolio.values()]
        series = [
            item[(item.index.date >= custom_start) & (item.index.date <= custom_end)]
            for item in series
        ]
        for symbol in [*selected_benchmarks, *extras]:
            if symbol in benchmark_closes:
                stock_series = benchmark_closes.loc[
                    (benchmark_closes.index.date >= custom_start)
                    & (benchmark_closes.index.date <= custom_end),
                    symbol,
                ].copy()
                stock_series.name = symbol
                series.append(stock_series)
        normalized = rebase_comparison_series(series)
        normalized = [item for item in normalized if not item.empty]
        if not normalized:
            st.info("The selected series do not share usable data in this date range.")
            return
        figure = go.Figure()
        for index, item in enumerate(normalized):
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
        rows = []
        for item in normalized:
            metrics = {key: value * 100 for key, value in summary_metrics(item).items()}
            rows.append({"Series": item.name, **metrics})
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                key: st.column_config.NumberColumn(format="%.2f%%")
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


def render_forecast(portfolios: list[Portfolio], closes: pd.DataFrame) -> None:
    """Render public, assumption-driven long-term portfolio planning scenarios."""
    if not portfolios:
        st.caption("No portfolios are available for a projection.")
        return

    scope_options = ["Selected portfolios", *(portfolio.name for portfolio in portfolios)]
    scope = st.selectbox("Projection scope", scope_options, key="forecast_scope")
    scoped_portfolios = (
        portfolios
        if scope == "Selected portfolios"
        else [next(portfolio for portfolio in portfolios if portfolio.name == scope)]
    )
    if not closes.empty:
        valuations = [portfolio_valuation(portfolio, closes) for portfolio in scoped_portfolios]
        if not all(valuation.is_complete for valuation in valuations):
            st.warning("A projection is unavailable until every held symbol has a usable price.")
            return
        current_value = float(
            sum((valuation.total_calculated_value for valuation in valuations), start=money(0))
        )
    else:
        current_value = float(
            sum((portfolio_cost(portfolio) for portfolio in scoped_portfolios), start=money(0))
        )
    if current_value <= 0:
        st.info("A projection requires a selected portfolio with a positive current value.")
        return

    st.caption(
        f"Starting value for {scope}: {dollars(current_value)}"
        + (" (cost basis because live market data is unavailable)." if closes.empty else ".")
    )
    st.session_state.setdefault("forecast_preset", "Baseline")
    st.session_state.setdefault("forecast_annual_return", 7.0)
    st.session_state.setdefault("forecast_annual_volatility", 12.0)
    st.session_state.setdefault("forecast_monthly_contribution", 0.0)
    st.session_state.setdefault("forecast_horizon", 10)

    first_row = st.columns(4)
    first_row[0].selectbox(
        "Scenario preset",
        list(FORECAST_PRESETS),
        key="forecast_preset",
        on_change=_apply_forecast_preset,
    )
    horizon_years = first_row[1].select_slider(
        "Forecast horizon (years)", options=list(range(1, 11)), key="forecast_horizon"
    )
    annual_return = first_row[2].number_input(
        "Expected annual return (%)",
        min_value=-99.0,
        max_value=50.0,
        step=0.25,
        key="forecast_annual_return",
    )
    annual_volatility = first_row[3].number_input(
        "Annual volatility (%)",
        min_value=0.0,
        max_value=100.0,
        step=0.25,
        key="forecast_annual_volatility",
    )
    second_row = st.columns([1.2, 1.0, 2.0])
    monthly_contribution = second_row[0].number_input(
        "Monthly contribution ($)", min_value=0.0, step=100.0, key="forecast_monthly_contribution"
    )
    include_target = second_row[1].checkbox("Set a target value", key="forecast_include_target")
    target_value = (
        second_row[2].number_input(
            "Target portfolio value ($)", min_value=1.0, step=1_000.0, key="forecast_target_value"
        )
        if include_target
        else None
    )

    result = cached_projection(
        current_value=current_value,
        annual_return=annual_return / 100,
        annual_volatility=annual_volatility / 100,
        monthly_contribution=monthly_contribution,
        horizon_years=horizon_years,
        target_value=target_value,
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
        st.metric(
            f"Chance of reaching {dollars(target_value)} in {horizon_years} years",
            f"{result.target_probability:.0%}",
        )
    st.warning(
        "Planning illustration only—not a prediction or financial advice. Outcomes are based on "
        "your editable return, volatility, and contribution assumptions; past performance does "
        "not guarantee future results. The range widens substantially over longer horizons."
    )


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
    st.markdown("### Edit or delete selected transaction")
    st.caption(
        f"Transaction #{transaction.id} belongs to **{portfolio_name}**. "
        "Any edit is applied to that portfolio."
    )
    if transaction.source != "manual":
        st.warning(
            f"This transaction originated from `{transaction.source}`. Editing or deleting it "
            "manually overrides that source and may make the internal portfolio differ from the "
            "brokerage statement or Alpaca account. The original transaction will be retained in "
            "the override audit history."
        )
    else:
        st.warning(
            "Editing or deleting this transaction recalculates the portfolio's cash and holdings. "
            "The original transaction will be retained in the override audit history."
        )
    with st.form(f"edit_transaction_{transaction.id}"):
        first_row = st.columns(3)
        edit_date = first_row[0].text_input(
            "Transaction date (M/D/YY)", value=format_short_date(transaction.transaction_date)
        )
        edit_kind = first_row[1].selectbox(
            "Transaction type",
            EDITABLE_KINDS,
            index=EDITABLE_KINDS.index(transaction.kind),
            format_func=kind_label,
        )
        edit_symbol = first_row[2].text_input("Symbol", value=transaction.symbol or "")
        edit_action = st.text_input("Action", value=transaction.action, max_chars=80)
        edit_description = st.text_input(
            "Description", value=transaction.description, max_chars=500
        )
        trade_row = st.columns(3)
        edit_quantity = trade_row[0].number_input(
            "Shares",
            min_value=0.0,
            value=float(transaction.quantity or 0),
            format="%.2f",
        )
        edit_price = trade_row[1].number_input(
            "Price per share",
            min_value=0.0,
            value=float(transaction.price or 0),
            format="%.6f",
        )
        edit_fees = trade_row[2].number_input(
            "Fees / commission",
            min_value=0.0,
            value=float(transaction.fees or 0),
            format="%.4f",
        )
        edit_cash = st.number_input(
            "Cash change",
            value=float(transaction.cash_delta),
            step=1.0,
            format="%.4f",
            help="Buy and sell cash changes are recalculated from shares, price, and fees.",
        )
        update_phrase = st.text_input(f'Type "UPDATE {transaction.id}" to confirm')
        updated = st.form_submit_button("Save transaction changes")
    if updated:
        try:
            draft = transaction_draft(
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
            with session_scope() as session:
                update_transaction(
                    session,
                    transaction.portfolio_id,
                    transaction.id,
                    draft,
                    confirmation=update_phrase,
                )
            cached_closes.clear()
            st.session_state.flash = f"Transaction #{transaction.id} updated."
            st.rerun()
        except Exception as exc:
            st.info(f"Transaction was not updated: {exc}")

    with st.form(f"delete_transaction_{transaction.id}"):
        delete_phrase = st.text_input(f'Type "DELETE {transaction.id}" to confirm')
        deleted = st.form_submit_button("Delete transaction", type="secondary")
    if deleted:
        try:
            with session_scope() as session:
                delete_transaction(
                    session,
                    transaction.portfolio_id,
                    transaction.id,
                    confirmation=delete_phrase,
                )
            cached_closes.clear()
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
            "Price": st.column_config.NumberColumn(format="$%,.4f"),
            "Fees": st.column_config.NumberColumn(format="$%,.2f"),
            "Cash change": st.column_config.NumberColumn(format="$%,.2f"),
        },
    )
    import_columns = st.columns([1, 1])
    if import_columns[0].button("Import new transactions", key=f"import_{portfolio.id}"):
        try:
            with session_scope() as session:
                added, duplicates = import_statement(session, portfolio.id, parsed_statement)
            cached_closes.clear()
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
                cached_closes.clear()
                st.session_state.flash = f"Portfolio rebuilt from {rebuilt} transaction(s)."
                st.rerun()
            except Exception as exc:
                st.info(f"Portfolio was not rebuilt: {exc}")


def render_transactions(
    portfolios: list[Portfolio],
    names_by_id: dict[int, str],
    custom_start: date,
    custom_end: date,
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

        totals = (
            frame.groupby("Type", as_index=False)
            .agg(Transactions=("Cash change", "size"), Total=("Cash change", "sum"))
            .sort_values("Type")
        )
        totals_columns = st.columns([2, 1])
        totals_columns[0].dataframe(
            totals,
            hide_index=True,
            width="stretch",
            column_config={"Total": st.column_config.NumberColumn(format="$%,.2f")},
        )
        totals_columns[1].metric("Grand total", dollars(frame["Cash change"].sum()))

        quantity_rows = frame.dropna(subset=["Symbol", "Quantity"])
        if not quantity_rows.empty:
            quantity_totals = (
                quantity_rows.groupby(["Symbol", "Type"], as_index=False)
                .agg(
                    Transactions=("Quantity", "size"),
                    **{"Total quantity": ("Quantity", "sum")},
                )
                .sort_values(["Symbol", "Type"])
            )
            st.markdown("#### Quantity totals by symbol")
            st.dataframe(
                quantity_totals,
                hide_index=True,
                width="stretch",
                column_config={"Total quantity": st.column_config.NumberColumn(format="%.2f")},
            )

        st.caption("Select any row to edit or delete it. Click a column header to sort.")
        table_scope = "_".join(
            [
                *[str(portfolio.id) for portfolio in portfolios],
                *selected_types,
                custom_start.isoformat(),
                custom_end.isoformat(),
            ]
        )
        table_event = st.dataframe(
            frame,
            hide_index=True,
            width="stretch",
            column_order=[
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
            column_config={
                "Date": st.column_config.DateColumn(format="M/D/YY"),
                "Quantity": st.column_config.NumberColumn(format="%.2f"),
                "Price": st.column_config.NumberColumn(format="$%,.4f"),
                "Fees": st.column_config.NumberColumn(format="$%,.2f"),
                "Cash change": st.column_config.NumberColumn(format="$%,.2f"),
            },
            key=f"manage_transaction_table_{table_scope}",
            on_select="rerun",
            selection_mode="single-row",
        )
        selected_rows = table_event.selection.rows
        if selected_rows:
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
        st.session_state[key] = normalize_symbol(value)
        st.session_state.pop(error_key, None)
    except ValueError as exc:
        st.session_state[error_key] = str(exc)


def _calculated_trade_cash(
    kind: str, quantity_value: float, price_value: float, fees_value: float
) -> float:
    quantity_value = max(quantity_value, 0.0)
    price_value = max(price_value, 0.0)
    fees_value = max(fees_value, 0.0)
    notional = quantity_value * price_value
    return -(notional + fees_value) if kind == "buy" else notional - fees_value


def render_add_transaction(
    portfolios: list[Portfolio], selected_portfolios: list[Portfolio]
) -> None:
    with st.expander("Add transaction", expanded=False):
        if not portfolios:
            st.caption("Create a portfolio before recording a transaction.")
            return
        first_row = st.columns(2)
        transaction_date_text = first_row[0].text_input(
            "Transaction date (M/D/YY)",
            value=format_short_date(date.today()),
            key="add_transaction_date",
        )
        kind = first_row[1].selectbox(
            "Transaction type",
            MANUAL_KINDS,
            format_func=kind_label,
            key="add_transaction_kind",
        )
        target = _target_portfolio_selector(
            "Target portfolio",
            "transaction_target",
            portfolios,
            selected_portfolios,
        )
        if target is None:
            st.caption("Select a portfolio before recording a transaction.")
            return
        symbol = ""
        manual_quantity = 0.0
        manual_price = 0.0
        manual_fees = 0.0
        symbol_error = ""
        if kind in TRADE_KINDS:
            symbol = st.text_input(
                "Symbol",
                max_chars=16,
                key="add_transaction_symbol",
                on_change=_validate_manual_trade_symbol,
            )
            symbol_error = st.session_state.get("add_transaction_symbol_error", "")
            if symbol_error:
                st.error(symbol_error)
            trade_row = st.columns(3)
            manual_quantity = trade_row[0].number_input(
                "Shares",
                min_value=0.0,
                value=0.0,
                format="%.2f",
                key="add_transaction_quantity",
            )
            manual_price = trade_row[1].number_input(
                "Price per share",
                min_value=0.0,
                value=0.0,
                format="%.6f",
                key="add_transaction_price",
            )
            manual_fees = trade_row[2].number_input(
                "Fees / commission",
                min_value=0.0,
                value=0.0,
                format="%.4f",
                key="add_transaction_fees",
            )
            calculated_cash = _calculated_trade_cash(
                kind, manual_quantity, manual_price, manual_fees
            )
            st.session_state.add_transaction_calculated_cash = calculated_cash
            cash_delta = st.number_input(
                "Cash change",
                key="add_transaction_calculated_cash",
                format="%.4f",
                disabled=True,
                help="Buy and sell cash changes are calculated from shares, price, and fees.",
            )
        else:
            if kind in SYMBOL_CASH_KINDS:
                symbol = st.text_input(
                    "Symbol (optional)", max_chars=16, key="add_transaction_symbol"
                )
            cash_delta = st.number_input(
                "Cash change",
                value=0.0,
                step=1.0,
                format="%.4f",
                key="add_transaction_cash",
            )
        description = st.text_input("Description", max_chars=500, key="add_transaction_description")
        recorded = st.button(
            "Record transaction",
            key="record_manual_transaction",
            disabled=bool(symbol_error),
        )
        if recorded:
            try:
                draft = transaction_draft(
                    transaction_date_text,
                    kind,
                    symbol,
                    description,
                    manual_quantity,
                    manual_price,
                    manual_fees,
                    cash_delta,
                )
                with session_scope() as session:
                    record_transaction(session, target.id, draft)
                cached_closes.clear()
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
                renamed = st.form_submit_button("Save portfolio name")
            if renamed:
                try:
                    with session_scope() as session:
                        rename_portfolio(session, target.id, renamed_name)
                    st.session_state.flash = "Portfolio renamed."
                    st.rerun()
                except Exception as exc:
                    st.info(f"Portfolio was not renamed: {exc}")

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
                        cached_closes.clear()
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
) -> None:
    names_by_id = {portfolio.id: portfolio.name for portfolio in action_portfolios}
    render_transactions(reporting_portfolios, names_by_id, custom_start, custom_end)
    render_add_transaction(action_portfolios, reporting_portfolios)
    render_csv_section(action_portfolios, reporting_portfolios)
    render_portfolio_actions(action_portfolios)


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


def render_trade_admin(portfolios: list[Portfolio]) -> None:
    settings = get_settings()
    with st.expander("Assigned order ticket", expanded=True):
        if not portfolios:
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
        action_columns = st.columns([1, 1, 3])
        if action_columns[0].button(
            "Sync fills",
            width="stretch",
            disabled=not settings.alpaca_configured,
        ):
            try:
                with session_scope() as session:
                    changed = sync_allocations(session)
                cached_closes.clear()
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


def main() -> None:
    init_database()
    with session_scope() as session:
        seed_database(session)
    all_portfolios = load_portfolios()
    reporting_portfolios = [
        portfolio for portfolio in all_portfolios if portfolio_has_data(portfolio)
    ]
    owner = authenticate_owner()
    render_header()
    selected_portfolios, master_start, master_end = render_master_controls(reporting_portfolios)
    if flash := st.session_state.pop("flash", None):
        st.info(flash)
    closes, data_note = get_prices(selected_portfolios, master_start - timedelta(days=7))
    labels = ["Overview", "Compare", "Forecast"]
    if owner:
        labels.extend(["Manage", "Trade", "Architecture"])
    if st.session_state.get("active_page") not in {None, *labels}:
        st.session_state.active_page = labels[0]
    tabs = st.tabs(labels, key="active_page", on_change="rerun")
    with tabs[0]:
        render_overview(
            selected_portfolios,
            closes,
            data_note,
            master_start,
            master_end,
        )
    with tabs[1]:
        render_compare(
            selected_portfolios,
            closes,
            master_start,
            master_end,
        )
    with tabs[2]:
        render_forecast(selected_portfolios, closes)
    if owner:
        with tabs[3]:
            render_portfolio_admin(
                selected_portfolios,
                all_portfolios,
                master_start,
                master_end,
            )
        with tabs[4]:
            render_trade_admin(reporting_portfolios)
        with tabs[5]:
            render_architecture()
    else:
        st.caption("Owner controls are password-protected and hidden from public viewers.")


if __name__ == "__main__":
    main()
