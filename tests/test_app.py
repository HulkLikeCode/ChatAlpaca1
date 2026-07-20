from __future__ import annotations

from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

from chat_alpaca.config import get_settings

SELECT_PORTFOLIO_OPTION = "__select_portfolio__"


def viewer_app(page: str = "Overview") -> AppTest:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30)
    app.session_state["access_role"] = "user"
    app.session_state["active_page"] = page
    return app.run()


def test_unauthenticated_app_only_renders_login(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_PASSWORD", "admin-test-password")
    monkeypatch.setenv("USER_PASSWORD", "user-test-password")
    get_settings.cache_clear()
    try:
        app = AppTest.from_file("streamlit_app.py", default_timeout=30)
        app.session_state["active_page"] = "Manage"
        app.run()
    finally:
        get_settings.cache_clear()

    assert not app.exception
    assert [item.label for item in app.text_input] == ["Password"]
    assert not app.tabs


@pytest.mark.parametrize(
    ("password", "expected_role", "has_admin_controls"),
    [
        ("admin-test-password", "admin", True),
        ("user-test-password", "user", False),
    ],
)
def test_password_selects_the_expected_permission_role(
    monkeypatch: pytest.MonkeyPatch,
    password: str,
    expected_role: str,
    has_admin_controls: bool,
) -> None:
    monkeypatch.setenv("ADMIN_PASSWORD", "admin-test-password")
    monkeypatch.setenv("USER_PASSWORD", "user-test-password")
    get_settings.cache_clear()
    try:
        app = AppTest.from_file("streamlit_app.py", default_timeout=30)
        app.session_state["active_page"] = "Manage"
        app.run()
        app.text_input[0].set_value(password)
        next(item for item in app.button if item.label == "Sign in").click().run()
    finally:
        get_settings.cache_clear()

    assert app.session_state["access_role"] == expected_role
    assert ("transaction_target" in [item.key for item in app.selectbox]) is has_admin_controls
    trade_app = AppTest.from_file("streamlit_app.py", default_timeout=30)
    trade_app.session_state["access_role"] = expected_role
    trade_app.session_state["active_page"] = "Trade"
    trade_app.run()
    assert (
        "Submit assigned order" in [item.label for item in trade_app.button]
    ) is has_admin_controls


def test_read_only_app_renders_overview_and_lazy_navigation() -> None:
    app = viewer_app()

    assert not app.exception
    assert not any("portfolios · benchmarks · Alpaca orders" in item.value for item in app.markdown)
    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Monitor",
        "Compare",
        "Forecast",
        "Hypothetical",
        "Manage",
        "Trade",
        "Architecture",
    ]
    assert any("KCs Traditional IRA" in text.value for text in app.markdown)
    assert [item.label for item in app.metric].count("Selected Totals") == 1
    assert not app.checkbox
    assert any(item.label.startswith("Portfolios · Applied:") for item in app.multiselect)
    portfolio_selector = next(
        item for item in app.multiselect if item.label.startswith("Portfolios · Applied:")
    )
    assert portfolio_selector.value == ["__all_portfolios__"]
    assert [item.label for item in app.date_input] == ["Custom Start", "Custom End"]
    assert app.date_input[0].value == date(2026, 5, 15)
    assert "Exact holdings" in [item.label for item in app.expander]
    assert "By portfolio / lot" in app.radio[0].options
    gain_loss = next(
        item.value
        for item in app.dataframe
        if list(item.value.columns)
        == [
            "Portfolio",
            "Cash",
            "All-time gain/loss",
            "Daily gain/loss",
            "Custom gain/loss",
            "Annualized Alpha",
            "Beta",
            "Observations",
        ]
    )
    assert gain_loss["Cash"].sum() > 0
    assert "Cash positions" not in [item.label for item in app.expander]
    assert not any("latest market value" in item.value for item in app.markdown)

    portfolio_grid = next(
        item.value for item in app.markdown if item.value.startswith('<div class="portfolio-grid">')
    )
    assert portfolio_grid.count('<div class="portfolio-card">') > 1
    assert "CDT Div" in portfolio_grid
    assert portfolio_grid.index("Cash:") < portfolio_grid.index("CDT Div:")
    assert "symbols" not in portfolio_grid
    assert "\n" not in portfolio_grid
    assert not app.get("file_uploader")
    assert "Record transaction" not in [item.label for item in app.button]
    assert "Submit assigned order" not in [item.label for item in app.button]


def test_comparison_defaults_to_spy_benchmark() -> None:
    app = viewer_app("Compare")

    benchmark_selector = next(item for item in app.multiselect if item.label == "Benchmark ETFs")
    assert benchmark_selector.value == ["SPY"]


def test_monitor_consolidates_movers_and_simplifies_freshness() -> None:
    app = viewer_app("Monitor")

    movers = next(
        item.value
        for item in app.dataframe
        if "Daily change" in item.value and "Contribution" in item.value
    )
    assert "Portfolio" not in movers
    assert movers["Symbol"].is_unique
    assert set(movers["Freshness"]) <= {"Fresh", "Stale"}
    assert any("stale-symbol-alert" in item.value for item in app.markdown)


def test_forecast_target_and_methodology_render_together() -> None:
    app = viewer_app("Forecast")

    target = next(item for item in app.number_input if item.label == "Target portfolio value ($)")
    assert target.disabled
    next(item for item in app.checkbox if item.label == "Set a target value").check().run()

    target = next(item for item in app.number_input if item.label == "Target portfolio value ($)")
    assert not target.disabled
    assert any(item.label.startswith("Chance of reaching") for item in app.metric)
    assert any("10,000 reproducible monthly" in item.value for item in app.caption)


def test_incomplete_data_warnings_are_presented_without_credentials() -> None:
    app = viewer_app("Forecast")

    assert any("explicitly uses cost basis plus cash" in item.value for item in app.warning)
    assert any("Market-price coverage unavailable" in item.value for item in app.caption)


def test_phase_2_owner_manage_controls_render() -> None:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30)
    app.session_state["owner_authenticated"] = True
    app.session_state["active_page"] = "Manage"
    app.run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Monitor",
        "Compare",
        "Forecast",
        "Hypothetical",
        "Manage",
        "Trade",
        "Architecture",
    ]
    assert sum(item.label.startswith("Portfolios · Applied:") for item in app.multiselect) == 1
    assert "Transaction type filter" in [item.label for item in app.multiselect]
    assert "Portfolio" in [item.label for item in app.selectbox]
    assert "Import target portfolio" in [item.label for item in app.selectbox]
    assert "Portfolio action" in [item.label for item in app.selectbox]
    target_selector = next(item for item in app.selectbox if item.key == "transaction_target")
    csv_selector = next(item for item in app.selectbox if item.label == "Import target portfolio")
    assert target_selector.value == SELECT_PORTFOLIO_OPTION
    assert csv_selector.value == SELECT_PORTFOLIO_OPTION
    assert "Date (M/D/YY)" in [item.label for item in app.text_input]
    assert not app.metric
    assert [item.label for item in app.date_input] == ["Custom Start", "Custom End"]
    manage_sections = [item.label for item in app.expander]
    assert manage_sections.index("Transactions") < manage_sections.index("Add transaction")
    assert manage_sections.index("Add transaction") < manage_sections.index("Brokerage CSV")
    assert manage_sections.index("Brokerage CSV") < manage_sections.index(
        "Portfolio administration"
    )
    assert [item.label for item in app.expander if item.proto.expanded] == [
        "Transactions",
    ]
    assert not app.get("download_button")
    assert not any("Quantity totals by symbol" in item.value for item in app.markdown)


def test_manage_targets_follow_single_master_portfolio_and_blank_for_multiple() -> None:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30)
    app.session_state["owner_authenticated"] = True
    app.session_state["active_page"] = "Manage"
    app.run()

    master_selector = next(
        item for item in app.multiselect if item.label.startswith("Portfolios · Applied:")
    )
    apply_button = next(item for item in app.button if item.label == "Apply")
    master_selector.set_value([1])
    apply_button.click().run()

    target_selector = next(item for item in app.selectbox if item.key == "transaction_target")
    csv_selector = next(item for item in app.selectbox if item.label == "Import target portfolio")
    assert target_selector.value == 1
    assert csv_selector.value == 1
    assert "Date (M/D/YY)" in [item.label for item in app.text_input]
    assert [item.label for item in app.get("download_button")] == [
        "Download Brokerage CSV template"
    ]

    master_selector = next(
        item for item in app.multiselect if item.label.startswith("Portfolios · Applied:")
    )
    apply_button = next(item for item in app.button if item.label == "Apply")
    master_selector.set_value([1, 2])
    apply_button.click().run()

    target_selector = next(item for item in app.selectbox if item.key == "transaction_target")
    csv_selector = next(item for item in app.selectbox if item.label == "Import target portfolio")
    assert target_selector.value == SELECT_PORTFOLIO_OPTION
    assert csv_selector.value == SELECT_PORTFOLIO_OPTION


def test_add_transaction_customizes_fields_and_validates_trade_symbols() -> None:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30)
    app.session_state["owner_authenticated"] = True
    app.session_state["active_page"] = "Manage"
    app.run()

    master_selector = next(
        item for item in app.multiselect if item.label.startswith("Portfolios · Applied:")
    )
    apply_button = next(item for item in app.button if item.label == "Apply")
    master_selector.set_value([1])
    apply_button.click().run()

    symbol = next(item for item in app.text_input if item.key == "add_transaction_symbol")
    symbol.set_value("BAD SYMBOL").run()
    record = next(item for item in app.button if item.key == "record_manual_transaction")
    assert record.disabled
    assert any("Invalid stock or ETF symbol" in item.value for item in app.error)

    symbol = next(item for item in app.text_input if item.key == "add_transaction_symbol")
    symbol.set_value("aapl").run()
    assert (
        next(item for item in app.text_input if item.key == "add_transaction_symbol").value
        == "AAPL"
    )

    next(item for item in app.number_input if item.key == "add_transaction_quantity").set_value(2.0)
    next(item for item in app.number_input if item.key == "add_transaction_price").set_value(10.0)
    next(item for item in app.number_input if item.key == "add_transaction_fees").set_value(
        1.0
    ).run()
    calculated_cash = next(
        item for item in app.number_input if item.key == "add_transaction_calculated_cash"
    )
    assert calculated_cash.value == -21.0
    assert calculated_cash.disabled

    kind = next(item for item in app.selectbox if item.key == "add_transaction_kind")
    kind.set_value("cash_adjustment").run()
    assert not any(item.key == "add_transaction_symbol" for item in app.text_input)
    assert not any(item.key == "add_transaction_quantity" for item in app.number_input)
    assert not any(item.key == "add_transaction_price" for item in app.number_input)
    assert not any(item.key == "add_transaction_fees" for item in app.number_input)
    cash_change = next(item for item in app.number_input if item.key == "add_transaction_cash")
    assert not cash_change.disabled


def test_master_filters_apply_portfolios_and_dates_together() -> None:
    app = viewer_app()
    first_portfolio_id = 1

    app.multiselect[0].set_value([first_portfolio_id])
    app.date_input[0].set_value(date(2026, 6, 1))
    app.date_input[1].set_value(date(2026, 6, 30))
    app.button[0].click().run()

    assert not app.exception
    assert app.session_state["master_portfolio_ids"] == [first_portfolio_id]
    assert app.session_state["master_start_date"] == date(2026, 6, 1)
    assert app.session_state["master_end_date"] == date(2026, 6, 30)
    portfolio_grid = next(
        item.value for item in app.markdown if item.value.startswith('<div class="portfolio-grid">')
    )
    assert portfolio_grid.count('<div class="portfolio-card">') == 1


def test_exact_holdings_summary_and_detail_column_order() -> None:
    app = viewer_app()
    expected_summary = [
        "Symbol",
        "Avg/share",
        "Current",
        "Cost basis",
        "Value",
        "All time",
        "Day",
        "Custom",
        "Annualized Alpha",
        "Beta",
        "Alpha/Beta observations",
        "Shares",
        "Portfolios",
    ]
    summary = next(
        item.value for item in app.dataframe if list(item.value.columns) == expected_summary
    )
    assert list(summary.columns) == expected_summary

    app.radio[0].set_value("By portfolio / lot").run()
    expected_detail = [
        "Symbol",
        "Avg/share",
        "Current",
        "Cost basis",
        "Value",
        "All time",
        "Day",
        "Custom",
        "Annualized Alpha",
        "Beta",
        "Alpha/Beta observations",
        "Shares",
        "Portfolio",
        "Acquired",
    ]
    detail = next(
        item.value for item in app.dataframe if list(item.value.columns) == expected_detail
    )
    assert list(detail.columns) == expected_detail
