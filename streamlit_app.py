from __future__ import annotations

import hmac
from datetime import date, timedelta

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
    MANUAL_KINDS,
    TransactionDraft,
    create_portfolio,
    delete_portfolio,
    import_statement,
    list_portfolios,
    list_transactions,
    money,
    normalize_symbol,
    parse_statement_csv,
    portfolio_cost,
    rebuild_portfolio_from_csv,
    record_transaction,
    rename_portfolio,
    seed_database,
    shares,
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
    st.caption("portfolios · benchmarks · Alpaca paper orders")


def render_portfolio_cards(portfolios: list[Portfolio], closes: pd.DataFrame) -> None:
    for start in range(0, len(portfolios), 5):
        columns = st.columns(5)
        for column, portfolio in zip(columns, portfolios[start : start + 5], strict=False):
            if closes.empty:
                total = portfolio_cost(portfolio)
                value_label = "cost basis + cash"
            else:
                _, total = latest_values(portfolio, closes)
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
    if not portfolios:
        st.caption("No portfolios with holdings yet.")
        return
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
                "Shares": st.column_config.NumberColumn(format="%.1f"),
                "Acquired": st.column_config.DateColumn(format="M/D/YY"),
                "Cost / share": st.column_config.NumberColumn(format="$%,.2f"),
                "Cost basis": st.column_config.NumberColumn(format="$%,.2f"),
                "Latest price": st.column_config.NumberColumn(format="$%,.2f"),
                "Market value": st.column_config.NumberColumn(format="$%,.2f"),
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
        column_config={"Cash": st.column_config.NumberColumn(format="$%,.2f")},
    )


def render_compare(portfolios: list[Portfolio]) -> None:
    if not portfolios:
        st.caption("No portfolios with holdings are available to compare.")
        return
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


def render_portfolio_admin(portfolios: list[Portfolio]) -> None:
    st.markdown("### Portfolios")
    with st.form("create_portfolio"):
        name = st.text_input("New portfolio name", value="Another One!", max_chars=80)
        created = st.form_submit_button("Add portfolio")
    if created:
        try:
            with session_scope() as session:
                portfolio = create_portfolio(session, name)
            st.session_state.manage_portfolio_id = portfolio.id
            st.session_state.flash = (
                f"Portfolio '{portfolio.name}' added. Record or import holdings to display it."
            )
            st.rerun()
        except Exception as exc:
            st.info(f"Portfolio was not added: {exc}")

    selected_id = st.session_state.get("manage_portfolio_id")
    selectable = [item for item in portfolios if item.holdings or item.id == selected_id]
    if not selectable:
        st.caption(
            "No portfolios with holdings yet. Add a portfolio, then record or import its first holding."
        )
        return

    st.markdown("### Portfolio transactions")
    selected_name = st.selectbox("Portfolio", [item.name for item in selectable])
    portfolio = next(item for item in selectable if item.name == selected_name)
    with st.form(f"rename_portfolio_{portfolio.id}"):
        name = st.text_input("Portfolio name", value=portfolio.name, max_chars=80)
        renamed = st.form_submit_button("Rename portfolio")
    if renamed:
        try:
            with session_scope() as session:
                rename_portfolio(session, portfolio.id, name)
            st.session_state.flash = "Portfolio renamed."
            st.rerun()
        except Exception as exc:
            st.info(f"Portfolio was not renamed: {exc}")

    deletion_phrase = st.text_input(
        "Type DELETE to permanently remove this portfolio and its holdings",
        key=f"delete_phrase_{portfolio.id}",
    )
    if st.button("Delete portfolio", key=f"delete_{portfolio.id}", type="secondary"):
        if deletion_phrase != "DELETE":
            st.info("Type DELETE before permanently removing this portfolio.")
        else:
            try:
                with session_scope() as session:
                    delete_portfolio(session, portfolio.id)
                if st.session_state.get("manage_portfolio_id") == portfolio.id:
                    del st.session_state.manage_portfolio_id
                cached_closes.clear()
                st.session_state.flash = "Portfolio and its holdings were permanently deleted."
                st.rerun()
            except Exception as exc:
                st.info(f"Portfolio was not deleted: {exc}")

    st.markdown("### Enter transaction")
    with st.form(f"manual_transaction_{portfolio.id}"):
        first_row = st.columns(3)
        transaction_date = first_row[0].date_input("Transaction date", value=date.today())
        kind = first_row[1].selectbox(
            "Transaction type",
            MANUAL_KINDS,
            format_func=lambda item: item.replace("_", " ").title(),
        )
        symbol = first_row[2].text_input("Symbol", max_chars=16)
        description = st.text_input("Description", max_chars=500)
        trade_row = st.columns(3)
        manual_quantity = trade_row[0].number_input(
            "Shares", min_value=0.0, value=0.0, format="%.8f"
        )
        manual_price = trade_row[1].number_input(
            "Price per share", min_value=0.0, value=0.0, format="%.6f"
        )
        manual_fees = trade_row[2].number_input(
            "Fees / commission", min_value=0.0, value=0.0, format="%.4f"
        )
        cash_delta = st.number_input(
            "Cash change",
            value=0.0,
            step=1.0,
            format="%.4f",
            disabled=kind in {"buy", "sell"},
        )
        recorded = st.form_submit_button("Record transaction")
    if recorded:
        try:
            parsed_quantity = shares(manual_quantity) if kind in {"buy", "sell"} else None
            parsed_price = money(manual_price) if kind in {"buy", "sell"} else None
            parsed_fees = money(manual_fees) if manual_fees else None
            if kind == "buy":
                assert parsed_quantity is not None and parsed_price is not None
                parsed_cash_delta = -(parsed_quantity * parsed_price + (parsed_fees or 0))
            elif kind == "sell":
                assert parsed_quantity is not None and parsed_price is not None
                parsed_cash_delta = parsed_quantity * parsed_price - (parsed_fees or 0)
            else:
                parsed_cash_delta = money(cash_delta)
            draft = TransactionDraft(
                transaction_date=transaction_date,
                action=kind.replace("_", " ").title(),
                kind=kind,
                symbol=normalize_symbol(symbol) if symbol.strip() else None,
                description=description.strip(),
                quantity=parsed_quantity,
                price=parsed_price,
                fees=parsed_fees,
                cash_delta=parsed_cash_delta,
            )
            with session_scope() as session:
                record_transaction(session, portfolio.id, draft)
            cached_closes.clear()
            st.session_state.flash = "Transaction recorded."
            st.rerun()
        except Exception as exc:
            st.info(f"Transaction was not recorded: {exc}")

    st.markdown("### Import statement")
    uploaded_statement = st.file_uploader(
        "Brokerage CSV", type=["csv"], key=f"statement_{portfolio.id}"
    )
    if uploaded_statement is not None:
        statement_content = uploaded_statement.getvalue()
        parsed_statement = parse_statement_csv(statement_content)
        if parsed_statement.errors:
            st.dataframe(pd.DataFrame({"Import issues": parsed_statement.errors}), hide_index=True)
        else:
            preview = pd.DataFrame(
                [
                    {
                        "Date": item.transaction_date,
                        "Action": item.action,
                        "Category": item.kind.replace("_", " ").title(),
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
                use_container_width=True,
                column_config={
                    "Date": st.column_config.DateColumn(format="M/D/YY"),
                    "Quantity": st.column_config.NumberColumn(format="%.8f"),
                    "Price": st.column_config.NumberColumn(format="$%,.4f"),
                    "Fees": st.column_config.NumberColumn(format="$%,.2f"),
                    "Cash change": st.column_config.NumberColumn(format="$%,.2f"),
                },
            )
            import_columns = st.columns([1, 1])
            if import_columns[0].button("Import new transactions", key=f"import_{portfolio.id}"):
                try:
                    with session_scope() as session:
                        added, duplicates = import_statement(
                            session, portfolio.id, parsed_statement
                        )
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
            if import_columns[1].button(
                "Rebuild portfolio from statement", key=f"rebuild_{portfolio.id}"
            ):
                if rebuild_phrase != "REBUILD":
                    st.info("Type REBUILD before replacing this portfolio.")
                else:
                    try:
                        with session_scope() as session:
                            rebuilt = rebuild_portfolio_from_csv(
                                session, portfolio.id, statement_content
                            )
                        cached_closes.clear()
                        st.session_state.flash = f"Portfolio rebuilt from {rebuilt} transaction(s)."
                        st.rerun()
                    except Exception as exc:
                        st.info(f"Portfolio was not rebuilt: {exc}")

    with session_scope() as session:
        transactions = list_transactions(session, portfolio.id)
    if transactions:
        st.markdown("### Posted transactions")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Date": entry.transaction_date,
                        "Action": entry.action,
                        "Type": entry.kind.replace("_", " ").title(),
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
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Date": st.column_config.DateColumn(format="M/D/YY"),
                "Quantity": st.column_config.NumberColumn(format="%.8f"),
                "Price": st.column_config.NumberColumn(format="$%,.4f"),
                "Fees": st.column_config.NumberColumn(format="$%,.2f"),
                "Cash change": st.column_config.NumberColumn(format="$%,.2f"),
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
                "Requested": st.column_config.NumberColumn(format="%.1f"),
                "Filled": st.column_config.NumberColumn(format="%.1f"),
                "Avg fill": st.column_config.NumberColumn(format="$%,.2f"),
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
    all_portfolios = load_portfolios()
    portfolios = [portfolio for portfolio in all_portfolios if portfolio.holdings]
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
            render_portfolio_admin(all_portfolios)
        with tabs[3]:
            render_trade_admin(portfolios)
        with tabs[4]:
            render_architecture()
    else:
        st.caption("Owner controls are password-protected and hidden from public viewers.")


if __name__ == "__main__":
    main()
