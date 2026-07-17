from __future__ import annotations

import hmac
from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chat_alpaca.analytics import (
    combined_series,
    earliest_acquisition,
    latest_values,
    normalized_growth,
    portfolio_series,
    summary_metrics,
)
from chat_alpaca.config import get_settings
from chat_alpaca.db import init_database, session_scope
from chat_alpaca.market_data import get_daily_closes
from chat_alpaca.models import OrderAllocation, Portfolio
from chat_alpaca.portfolio_service import (
    list_ledger,
    list_portfolios,
    portfolio_cost,
    rename_portfolio,
    replace_holdings,
    seed_database,
    set_cash,
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


def dollars(value: object) -> str:
    return f"${float(value):,.2f}"


def quantity(value: object) -> str:
    numeric = float(value)
    return f"{numeric:,.8f}".rstrip("0").rstrip(".")


def load_portfolios() -> list[Portfolio]:
    with session_scope() as session:
        return list_portfolios(session)


def get_prices(
    portfolios: list[Portfolio], start: date, extra: tuple[str, ...] = ()
) -> tuple[pd.DataFrame, str | None]:
    symbols = {lot.symbol for portfolio in portfolios for lot in portfolio.holdings} | set(extra)
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
            if st.button("Lock owner controls", use_container_width=True):
                st.session_state.owner_authenticated = False
                st.rerun()
            return True
        if not settings.admin_password:
            st.caption("Set ADMIN_PASSWORD to enable owner controls.")
            return False
        with st.form("owner_login", border=False):
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Unlock", use_container_width=True)
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
    st.caption(
        "Five portfolios · benchmarks · Alpaca paper orders"
    )


def render_portfolio_cards(portfolios: list[Portfolio], closes: pd.DataFrame) -> None:
    columns = st.columns(5)
    for column, portfolio in zip(columns, portfolios, strict=False):
        if closes.empty:
            market_value = portfolio_cost(portfolio) - Decimal(portfolio.cash)
            total = portfolio_cost(portfolio)
            value_label = "cost basis + cash"
        else:
            market_value, total = latest_values(portfolio, closes)
            value_label = "latest market value"
        unique_symbols = len({lot.symbol for lot in portfolio.holdings})
        with column:
            st.markdown(
                f"""
                <div class="portfolio-card">
                  <div class="eyebrow">{portfolio.name}</div>
                  <div class="value">{dollars(total)}</div>
                  <div class="detail">{unique_symbols} symbols · {dollars(portfolio.cash)} cash</div>
                  <div class="detail">{value_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def holdings_frame(portfolios: list[Portfolio], closes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for portfolio in portfolios:
        for lot in sorted(portfolio.holdings, key=lambda item: (item.symbol, item.acquired_on)):
            latest = None
            if lot.symbol in closes and not closes[lot.symbol].dropna().empty:
                latest = float(closes[lot.symbol].dropna().iloc[-1])
            rows.append(
                {
                    "Portfolio": portfolio.name,
                    "Symbol": lot.symbol,
                    "Shares": float(lot.shares),
                    "Acquired": lot.acquired_on,
                    "Cost / share": float(lot.cost_basis),
                    "Cost basis": float(lot.shares * lot.cost_basis),
                    "Latest price": latest,
                    "Market value": latest * float(lot.shares) if latest is not None else None,
                }
            )
    return pd.DataFrame(rows)


def render_overview(
    portfolios: list[Portfolio], closes: pd.DataFrame, data_note: str | None
) -> None:
    render_portfolio_cards(portfolios, closes)
    if data_note:
        st.info(f"Live market values are unavailable, so cost basis is shown. {data_note}")
    st.markdown("### Exact holdings")
    frame = holdings_frame(portfolios, closes)
    if frame.empty:
        st.caption("No holdings yet.")
    else:
        st.dataframe(
            frame,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Shares": st.column_config.NumberColumn(format="%.8f"),
                "Cost / share": st.column_config.NumberColumn(format="$%.5f"),
                "Cost basis": st.column_config.NumberColumn(format="$%.2f"),
                "Latest price": st.column_config.NumberColumn(format="$%.2f"),
                "Market value": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
    st.markdown("### Cash positions")
    st.dataframe(
        pd.DataFrame(
            {
                "Portfolio": [item.name for item in portfolios],
                "Cash": [float(item.cash) for item in portfolios],
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={"Cash": st.column_config.NumberColumn(format="$%.2f")},
    )


def render_compare(portfolios: list[Portfolio]) -> None:
    fallback_start = date.today() - timedelta(days=365)
    earliest = earliest_acquisition(portfolios, fallback_start)
    controls = st.columns([1.1, 1.4, 2.2])
    with controls[0]:
        start = st.date_input("Start date", value=earliest, max_value=date.today())
    with controls[1]:
        selected_benchmarks = st.multiselect("Benchmark ETFs", BENCHMARKS, default=list(BENCHMARKS))
    with controls[2]:
        extra_text = st.text_input("Additional stocks or ETFs", placeholder="AAPL, MSFT, VTI")
    extras = tuple(
        dict.fromkeys(symbol.strip().upper() for symbol in extra_text.split(",") if symbol.strip())
    )
    portfolio_options = ["All portfolios", *[item.name for item in portfolios]]
    chosen_portfolios = st.multiselect(
        "Portfolio series", portfolio_options, default=portfolio_options
    )
    all_symbols = tuple(
        sorted(
            {lot.symbol for portfolio in portfolios for lot in portfolio.holdings}
            | set(selected_benchmarks)
            | set(extras)
        )
    )
    try:
        closes = cached_closes(all_symbols, start, date.today())
    except Exception as exc:
        st.info(f"Comparison data is unavailable. Configure rotated Alpaca credentials. {exc}")
        return
    if closes.empty:
        st.info("No market data was returned for this comparison.")
        return
    per_portfolio = {
        portfolio.name: portfolio_series(portfolio, closes) for portfolio in portfolios
    }
    series: list[pd.Series] = []
    if "All portfolios" in chosen_portfolios:
        series.append(combined_series(per_portfolio.values()))
    series.extend(
        per_portfolio[name]
        for name in chosen_portfolios
        if name != "All portfolios" and name in per_portfolio
    )
    for symbol in [*selected_benchmarks, *extras]:
        if symbol in closes:
            stock_series = closes[symbol].copy()
            stock_series.name = symbol
            series.append(stock_series)
    normalized = [normalized_growth(item) for item in series]
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
                line={"width": 2.3, "color": PLOT_COLORS[index % len(PLOT_COLORS)]},
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
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    rows = []
    for item in normalized:
        metrics = {key: value * 100 for key, value in summary_metrics(item).items()}
        rows.append({"Series": item.name, **metrics})
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            key: st.column_config.NumberColumn(format="%.2f%%")
            for key in ("Total return", "Annualized return", "Volatility", "Max drawdown")
        },
    )
    st.caption("All series are rebased to $100. Metrics use adjusted daily closes from Alpaca.")


def editor_rows(portfolio: Portfolio) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": lot.symbol,
                "Shares": float(lot.shares),
                "Acquired": lot.acquired_on,
                "Cost / share": float(lot.cost_basis),
            }
            for lot in sorted(portfolio.holdings, key=lambda item: (item.symbol, item.acquired_on))
        ],
        columns=["Symbol", "Shares", "Acquired", "Cost / share"],
    )


def render_portfolio_admin(portfolios: list[Portfolio]) -> None:
    st.markdown("### Portfolio editor")
    selected_name = st.selectbox("Portfolio", [item.name for item in portfolios])
    portfolio = next(item for item in portfolios if item.name == selected_name)
    with st.form(f"edit_portfolio_{portfolio.id}"):
        name = st.text_input("Portfolio name", value=portfolio.name, max_chars=80)
        cash = st.number_input(
            "Cash position", value=float(portfolio.cash), step=100.0, format="%.2f"
        )
        edited = st.data_editor(
            editor_rows(portfolio),
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_config={
                "Symbol": st.column_config.TextColumn(required=True, max_chars=16),
                "Shares": st.column_config.NumberColumn(
                    required=True, min_value=-1_000_000.0, format="%.8f"
                ),
                "Acquired": st.column_config.DateColumn(required=True, max_value=date.today()),
                "Cost / share": st.column_config.NumberColumn(
                    required=True, min_value=0.0, format="$%.5f"
                ),
            },
        )
        saved = st.form_submit_button("Save portfolio", use_container_width=True)
    if saved:
        try:
            with session_scope() as session:
                rename_portfolio(session, portfolio.id, name)
                set_cash(session, portfolio.id, cash)
                replace_holdings(session, portfolio.id, edited.to_dict("records"))
            cached_closes.clear()
            st.session_state.flash = "Portfolio saved automatically."
            st.rerun()
        except Exception as exc:
            st.info(f"Portfolio was not saved: {exc}")
    st.caption(
        "Up to 25 unique symbols are allowed. Negative shares represent an internally tracked short position."
    )
    with session_scope() as session:
        entries = list_ledger(session)
    if entries:
        names = {item.id: item.name for item in portfolios}
        st.markdown("### Internal cash and trade ledger")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Portfolio": names.get(entry.portfolio_id, str(entry.portfolio_id)),
                        "Type": entry.kind,
                        "Symbol": entry.symbol,
                        "Quantity": float(entry.quantity) if entry.quantity is not None else None,
                        "Price": float(entry.price) if entry.price is not None else None,
                        "Cash change": float(entry.cash_delta),
                        "Note": entry.note,
                        "Recorded": entry.created_at,
                    }
                    for entry in entries
                ]
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Quantity": st.column_config.NumberColumn(format="%.8f"),
                "Price": st.column_config.NumberColumn(format="$%.4f"),
                "Cash change": st.column_config.NumberColumn(format="$%.2f"),
            },
        )


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
    st.markdown("### Assigned order ticket")
    if not settings.alpaca_configured:
        st.info("Add rotated Alpaca credentials to enable paper order submission.")
    mode_text = "Paper execution" if settings.paper else "Live execution"
    st.caption(f"{mode_text}. Every order must be assigned to one internal portfolio.")
    with st.form("trade_ticket"):
        columns = st.columns(3)
        portfolio_name = columns[0].selectbox("Portfolio", [item.name for item in portfolios])
        symbol = columns[1].text_input("Symbol", max_chars=16)
        side = columns[2].selectbox("Side", ["Buy", "Sell"])
        columns = st.columns(3)
        order_type = columns[0].selectbox("Order type", ["Market", "Limit"])
        qty = columns[1].number_input("Shares", min_value=0.00000001, value=1.0, format="%.8f")
        limit_price = columns[2].number_input(
            "Limit price", min_value=0.0, value=0.0, format="%.4f"
        )
        confirmed = st.checkbox("I reviewed this order and its assigned portfolio.")
        submitted = st.form_submit_button(
            "Submit assigned order",
            use_container_width=True,
            disabled=not settings.alpaca_configured,
        )
    if submitted:
        if not confirmed:
            st.info("Review and confirm the order before submitting it.")
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
                st.session_state.flash = f"Order submitted and assigned: {order.alpaca_order_id}"
                st.rerun()
            except Exception as exc:
                st.info(f"Order was not submitted: {exc}")
    st.markdown("### Alpaca account")
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
        "Sync fills", use_container_width=True, disabled=not settings.alpaca_configured
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
        "Cancel selected order", disabled=not cancel_id or not settings.alpaca_configured
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
            use_container_width=True,
            column_config={
                "Requested": st.column_config.NumberColumn(format="%.8f"),
                "Filled": st.column_config.NumberColumn(format="%.8f"),
                "Avg fill": st.column_config.NumberColumn(format="$%.4f"),
            },
        )
    else:
        st.caption("No assigned orders have been submitted yet.")


def render_architecture() -> None:
    st.markdown("### Extension architecture")
    st.caption(
        "These capabilities are designed into the service boundaries but intentionally inactive in the first release."
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
        use_container_width=True,
    )


def main() -> None:
    init_database()
    with session_scope() as session:
        seed_database(session)
    portfolios = load_portfolios()
    owner = authenticate_owner()
    render_header()
    if flash := st.session_state.pop("flash", None):
        st.info(flash)
    overview_start = earliest_acquisition(portfolios, date.today() - timedelta(days=365))
    closes, data_note = get_prices(portfolios, overview_start)
    labels = ["Overview", "Compare"]
    if owner:
        labels.extend(["Manage", "Trade", "Architecture"])
    tabs = st.tabs(labels)
    with tabs[0]:
        render_overview(portfolios, closes, data_note)
    with tabs[1]:
        render_compare(portfolios)
    if owner:
        with tabs[2]:
            render_portfolio_admin(portfolios)
        with tabs[3]:
            render_trade_admin(portfolios)
        with tabs[4]:
            render_architecture()
    else:
        st.caption("Owner controls are password-protected and hidden from public viewers.")


if __name__ == "__main__":
    main()
