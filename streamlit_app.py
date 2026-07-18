from __future__ import annotations

import hmac
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chat_alpaca.analytics import (
    combined_series,
    consolidated_holdings,
    earliest_acquisition,
    latest_values,
    normalized_growth,
    portfolio_gain_loss,
    portfolio_series,
    summary_metrics,
    total_portfolio_value,
)
from chat_alpaca.config import get_settings
from chat_alpaca.db import init_database, session_scope
from chat_alpaca.market_data import get_daily_closes
from chat_alpaca.models import OrderAllocation, Portfolio, PortfolioTransaction
from chat_alpaca.portfolio_service import (
    MANUAL_KINDS,
    TransactionDraft,
    create_portfolio,
    delete_portfolio,
    delete_transaction,
    dividend_totals,
    format_short_date,
    import_statement,
    list_portfolios,
    list_transactions_for_portfolios,
    money,
    normalize_symbol,
    parse_short_date,
    parse_statement_csv,
    portfolio_cost,
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
EDITABLE_KINDS = (*MANUAL_KINDS, "opening_position")
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


def dollars(value: object) -> str:
    return f"${float(value):,.2f}"


def quantity(value: object) -> str:
    numeric = float(value)
    return f"{numeric:,.8f}".rstrip("0").rstrip(".")


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
    st.caption("portfolios · benchmarks · Alpaca orders")


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


def _summed_metric(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) if available else None


def _metric_dollars(value: float | None) -> str:
    return dollars(value) if value is not None else "—"


def render_performance_summary(
    portfolios: list[Portfolio],
    closes: pd.DataFrame,
    custom_start: date,
    custom_end: date,
    key_prefix: str,
) -> None:
    st.markdown("### Portfolio value and gain/loss")
    if closes.empty:
        total_value = sum((portfolio_cost(portfolio) for portfolio in portfolios), start=money(0))
        rows = []
    else:
        total_value = total_portfolio_value(portfolios, closes)
        rows = [
            {
                "Portfolio": portfolio.name,
                **portfolio_gain_loss(portfolio, closes, custom_start, custom_end).__dict__,
            }
            for portfolio in portfolios
        ]

    metrics = st.columns(4)
    metrics[0].metric("Total selected value", dollars(total_value))
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
    with st.expander("Per-portfolio gain/loss", expanded=False):
        st.dataframe(
            performance,
            hide_index=True,
            use_container_width=True,
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
    st.markdown("### Exact holdings")
    summary, detail = consolidated_holdings(portfolios, closes, custom_start, custom_end)
    if summary.empty:
        st.caption("No holdings yet.")
        return
    money_columns = (
        "Average cost / share",
        "Total cost basis",
        "Latest price",
        "Market value",
        "All-time gain/loss",
        "Daily gain/loss",
        "Custom gain/loss",
    )
    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Shares": st.column_config.NumberColumn(format="%.8f"),
            **{column: st.column_config.NumberColumn(format="$%,.2f") for column in money_columns},
        },
    )
    with st.expander("Per-portfolio / lot breakdown", expanded=False):
        st.dataframe(
            detail,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Shares": st.column_config.NumberColumn(format="%.8f"),
                "Acquired": st.column_config.DateColumn(format="M/D/YY"),
                **{
                    column: st.column_config.NumberColumn(format="$%,.2f")
                    for column in (
                        "Cost / share",
                        "Cost basis",
                        "Latest price",
                        "Market value",
                        "All-time gain/loss",
                        "Daily gain/loss",
                        "Custom gain/loss",
                    )
                },
            },
        )


def render_dividend_totals(portfolios: list[Portfolio], key_prefix: str) -> None:
    today = date.today()
    st.markdown("### Dividend totals")
    date_columns = st.columns(2)
    custom_start = date_columns[0].date_input(
        "Custom dividend start",
        value=date(today.year, 1, 1),
        max_value=today,
        format="MM/DD/YYYY",
        key=f"{key_prefix}_dividend_start",
    )
    custom_end = date_columns[1].date_input(
        "Custom dividend end",
        value=today,
        min_value=custom_start,
        max_value=today,
        format="MM/DD/YYYY",
        key=f"{key_prefix}_dividend_end",
    )
    with session_scope() as session:
        totals = dividend_totals(
            session,
            [portfolio.id for portfolio in portfolios],
            custom_start,
            custom_end,
            as_of=today,
        )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Year to date", dollars(totals.year_to_date))
    metric_columns[1].metric("Trailing 365 days", dollars(totals.trailing_365_days))
    metric_columns[2].metric("Custom range", dollars(totals.custom_range))


def render_overview(
    portfolios: list[Portfolio], closes: pd.DataFrame, data_note: str | None
) -> None:
    if not portfolios:
        st.caption("No portfolios yet.")
        return
    default_names = [portfolio.name for portfolio in portfolios]
    selected_names = st.multiselect(
        "Overview portfolios",
        [portfolio.name for portfolio in portfolios],
        default=default_names or [portfolio.name for portfolio in portfolios],
        key="overview_portfolios",
    )
    selected = [portfolio for portfolio in portfolios if portfolio.name in selected_names]
    if not selected:
        st.caption("Select one or more portfolios to show overview data.")
        return
    today = date.today()
    earliest = earliest_acquisition(selected, today - timedelta(days=365))
    range_columns = st.columns(2)
    custom_start = range_columns[0].date_input(
        "Custom gain/loss start",
        value=max(date(today.year, 1, 1), earliest),
        min_value=earliest,
        max_value=today,
        format="MM/DD/YYYY",
        key="overview_gain_loss_start",
    )
    custom_end = range_columns[1].date_input(
        "Custom gain/loss end",
        value=today,
        min_value=custom_start,
        max_value=today,
        format="MM/DD/YYYY",
        key="overview_gain_loss_end",
    )
    render_performance_summary(selected, closes, custom_start, custom_end, "overview")
    render_portfolio_cards(selected, closes)
    if data_note:
        st.info(f"Live market values are unavailable, so cost basis is shown. {data_note}")
    render_dividend_totals(selected, "overview")
    render_consolidated_holdings(selected, closes, custom_start, custom_end)
    st.markdown("### Cash positions")
    st.dataframe(
        pd.DataFrame(
            {
                "Portfolio": [item.name for item in selected],
                "Cash": [float(item.cash) for item in selected],
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={"Cash": st.column_config.NumberColumn(format="$%,.2f")},
    )


def render_compare(portfolios: list[Portfolio]) -> None:
    if not portfolios:
        st.caption("No portfolios are available to compare.")
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
    selected_for_totals = (
        portfolios
        if "All portfolios" in chosen_portfolios
        else [portfolio for portfolio in portfolios if portfolio.name in chosen_portfolios]
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
    fetch_start = min(start, earliest) - timedelta(days=7)
    try:
        closes = cached_closes(all_symbols, fetch_start, date.today())
    except Exception as exc:
        if selected_for_totals:
            render_performance_summary(
                selected_for_totals, pd.DataFrame(), start, date.today(), "compare"
            )
        st.info(f"Comparison data is unavailable. Configure rotated Alpaca credentials. {exc}")
        return
    if closes.empty:
        if selected_for_totals:
            render_performance_summary(selected_for_totals, closes, start, date.today(), "compare")
        st.info("No market data was returned for this comparison.")
        return
    if selected_for_totals:
        render_performance_summary(selected_for_totals, closes, start, date.today(), "compare")
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
    series = [item[item.index.date >= start] for item in series]
    for symbol in [*selected_benchmarks, *extras]:
        if symbol in closes:
            stock_series = closes.loc[closes.index.date >= start, symbol].copy()
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
            format="%.8f",
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
    st.markdown("### Brokerage CSV import")
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


def render_portfolio_admin(portfolios: list[Portfolio]) -> None:
    if not portfolios:
        st.caption("Create a portfolio below to begin.")
    names_by_id = {portfolio.id: portfolio.name for portfolio in portfolios}
    managed_names = st.multiselect(
        "Manage portfolios",
        [portfolio.name for portfolio in portfolios],
        default=[portfolio.name for portfolio in portfolios],
        key="manage_portfolios",
        help="This selection controls dividend totals and the combined transaction table.",
    )
    managed = [portfolio for portfolio in portfolios if portfolio.name in managed_names]
    managed_scope = "_".join(str(portfolio.id) for portfolio in managed) or "none"
    render_dividend_totals(managed, "manage")

    st.markdown("### Add transaction")
    target_name = st.selectbox(
        "Target portfolio",
        [portfolio.name for portfolio in portfolios],
        key="transaction_target",
    )
    target = next(portfolio for portfolio in portfolios if portfolio.name == target_name)
    with st.form(f"manual_transaction_{target.id}"):
        first_row = st.columns(3)
        transaction_date_text = first_row[0].text_input(
            "Transaction date (M/D/YY)", value=format_short_date(date.today())
        )
        kind = first_row[1].selectbox("Transaction type", MANUAL_KINDS, format_func=kind_label)
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
            help="Buy and sell cash changes are calculated automatically.",
        )
        recorded = st.form_submit_button("Record transaction")
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

    render_csv_import(target)

    st.markdown("### Transactions")
    with session_scope() as session:
        transactions = list_transactions_for_portfolios(
            session, [portfolio.id for portfolio in managed]
        )
    if not transactions:
        st.caption("No transactions match the selected portfolios.")
    else:
        filter_columns = st.columns([1, 1.35])
        type_options = sorted({transaction.kind for transaction in transactions})
        selected_types = filter_columns[0].multiselect(
            "Transaction type filter",
            type_options,
            default=type_options,
            format_func=kind_label,
            key=f"manage_transaction_types_{managed_scope}",
        )
        first_date = min(transaction.transaction_date for transaction in transactions)
        last_date = max(transaction.transaction_date for transaction in transactions)
        selected_range = filter_columns[1].date_input(
            "Transaction date range",
            value=(first_date, last_date),
            min_value=first_date,
            max_value=last_date,
            format="MM/DD/YYYY",
            key=f"manage_transaction_dates_{managed_scope}",
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            range_start, range_end = selected_range
        else:
            range_start = range_end = selected_range
        filtered = [
            transaction
            for transaction in transactions
            if transaction.kind in selected_types
            and range_start <= transaction.transaction_date <= range_end
        ]
        frame = transaction_frame(filtered, names_by_id)
        if frame.empty:
            st.caption("No transactions match the type and date filters.")
        else:
            totals = (
                frame.groupby("Type", as_index=False)
                .agg(Transactions=("Cash change", "size"), Total=("Cash change", "sum"))
                .sort_values("Type")
            )
            totals_columns = st.columns([2, 1])
            totals_columns[0].dataframe(
                totals,
                hide_index=True,
                use_container_width=True,
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
                    use_container_width=True,
                    column_config={"Total quantity": st.column_config.NumberColumn(format="%.8f")},
                )
            st.caption("Select any row to edit or delete it. Click a column header to sort.")
            table_scope = "_".join(
                [
                    managed_scope,
                    *selected_types,
                    range_start.isoformat(),
                    range_end.isoformat(),
                ]
            )
            table_event = st.dataframe(
                frame,
                hide_index=True,
                use_container_width=True,
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
                    "Quantity": st.column_config.NumberColumn(format="%.8f"),
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

    st.divider()
    st.markdown("### Rename portfolio")
    rename_target_name = st.selectbox(
        "Portfolio to rename",
        [portfolio.name for portfolio in portfolios],
        key="rename_target",
    )
    rename_target = next(
        portfolio for portfolio in portfolios if portfolio.name == rename_target_name
    )
    with st.form(f"rename_portfolio_{rename_target.id}"):
        renamed_name = st.text_input("Portfolio name", value=rename_target.name, max_chars=80)
        renamed = st.form_submit_button("Rename portfolio")
    if renamed:
        try:
            with session_scope() as session:
                rename_portfolio(session, rename_target.id, renamed_name)
            st.session_state.flash = "Portfolio renamed."
            st.rerun()
        except Exception as exc:
            st.info(f"Portfolio was not renamed: {exc}")

    st.markdown("### New portfolio")
    with st.form("create_portfolio"):
        new_name = st.text_input("New portfolio name", value="Another One!", max_chars=80)
        created = st.form_submit_button("Add portfolio")
    if created:
        try:
            with session_scope() as session:
                portfolio = create_portfolio(session, new_name)
            st.session_state.flash = (
                f"Portfolio '{portfolio.name}' added. Record or import its first transaction."
            )
            st.rerun()
        except Exception as exc:
            st.info(f"Portfolio was not added: {exc}")

    st.markdown("### Delete portfolio")
    delete_target_name = st.selectbox(
        "Portfolio to delete",
        [portfolio.name for portfolio in portfolios],
        key="delete_target",
    )
    delete_target = next(
        portfolio for portfolio in portfolios if portfolio.name == delete_target_name
    )
    deletion_phrase = st.text_input(
        "Type DELETE to permanently remove this portfolio and all of its data",
        key=f"delete_phrase_{delete_target.id}",
    )
    if st.button("Delete portfolio", key=f"delete_{delete_target.id}", type="secondary"):
        if deletion_phrase != "DELETE":
            st.info("Type DELETE before permanently removing this portfolio.")
        else:
            try:
                with session_scope() as session:
                    delete_portfolio(session, delete_target.id)
                cached_closes.clear()
                st.session_state.flash = "Portfolio and all of its data were permanently deleted."
                st.rerun()
            except Exception as exc:
                st.info(f"Portfolio was not deleted: {exc}")


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
    overview_start = earliest_acquisition(
        all_portfolios, date.today() - timedelta(days=365)
    ) - timedelta(days=7)
    closes, data_note = get_prices(all_portfolios, overview_start)
    labels = ["Overview", "Compare"]
    if owner:
        labels.extend(["Manage", "Trade", "Architecture"])
    tabs = st.tabs(labels)
    with tabs[0]:
        render_overview(all_portfolios, closes, data_note)
    with tabs[1]:
        render_compare(all_portfolios)
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
