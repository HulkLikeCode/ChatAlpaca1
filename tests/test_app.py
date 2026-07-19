from __future__ import annotations

from datetime import date

from streamlit.testing.v1 import AppTest


def test_public_app_renders_without_credentials() -> None:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "KC's Retirement Dough, Let's GO!!!"
    assert [tab.label for tab in app.tabs] == ["Overview", "Compare", "Forecast"]
    assert any("KCs Traditional IRA" in text.value for text in app.markdown)
    assert [item.label for item in app.metric].count("Total selected value") == 2
    assert [item.label for item in app.checkbox] == ["Set a target value"]
    assert "Portfolios" in [item.label for item in app.multiselect]
    portfolio_selector = next(item for item in app.multiselect if item.label == "Portfolios")
    assert portfolio_selector.value == ["__all_portfolios__"]
    assert [item.label for item in app.date_input] == ["Custom Start", "Custom End"]
    assert app.date_input[0].value == date(2026, 5, 15)
    assert "Exact holdings" in [item.label for item in app.expander]
    assert "Projection scope" in [item.label for item in app.selectbox]
    assert "Forecast horizon (years)" in [item.label for item in app.select_slider]
    assert "By portfolio / lot" in app.radio[0].options
    cash_table = next(
        item.value for item in app.dataframe if list(item.value.columns) == ["Portfolio", "Cash"]
    )
    assert cash_table.iloc[-1]["Portfolio"] == "Total"
    assert not any("latest market value" in item.value for item in app.markdown)

    portfolio_grid = next(
        item.value for item in app.markdown if item.value.startswith('<div class="portfolio-grid">')
    )
    assert portfolio_grid.count('<div class="portfolio-card">') > 1
    assert "\n" not in portfolio_grid


def test_comparison_defaults_to_spy_benchmark() -> None:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30).run()

    benchmark_selector = next(item for item in app.multiselect if item.label == "Benchmark ETFs")
    assert benchmark_selector.value == ["SPY"]


def test_phase_2_owner_manage_controls_render() -> None:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30)
    app.session_state["owner_authenticated"] = True
    app.run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Compare",
        "Forecast",
        "Manage",
        "Trade",
        "Architecture",
    ]
    assert [item.label for item in app.multiselect].count("Portfolios") == 1
    assert "Transaction type filter" in [item.label for item in app.multiselect]
    assert "Target portfolio" in [item.label for item in app.selectbox]
    assert "Import target portfolio" in [item.label for item in app.selectbox]
    assert "Portfolio action" in [item.label for item in app.selectbox]
    assert "Transaction date (M/D/YY)" in [item.label for item in app.text_input]
    assert [item.label for item in app.metric].count("Selected-range income") == 1
    assert [item.label for item in app.metric].count("YTD through end date") == 1
    assert [item.label for item in app.metric].count("Trailing 365 through end date") == 1
    assert [item.label for item in app.metric].count("Normalized quarterly average") == 1
    assert "Portfolio income" in [item.label for item in app.expander]
    assert [item.label for item in app.date_input] == ["Custom Start", "Custom End"]
    manage_sections = [item.label for item in app.expander]
    assert manage_sections.index("Transactions") < manage_sections.index("Add transaction")
    assert manage_sections.index("Add transaction") < manage_sections.index("Brokerage CSV")
    assert manage_sections.index("Brokerage CSV") < manage_sections.index(
        "Portfolio administration"
    )
    assert [item.label for item in app.expander if item.proto.expanded] == [
        "Portfolio value and gain/loss",
        "Portfolio value and gain/loss",
        "Transactions",
        "Assigned order ticket",
        "Extension architecture",
    ]
    assert [item.label for item in app.get("download_button")] == [
        "Download Brokerage CSV template"
    ]
    assert any("Quantity totals by symbol" in item.value for item in app.markdown)


def test_master_filters_apply_portfolios_and_dates_together() -> None:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30).run()
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
    app = AppTest.from_file("streamlit_app.py", default_timeout=30).run()
    expected_summary = [
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
        "Shares",
        "Portfolio",
        "Acquired",
    ]
    detail = next(
        item.value for item in app.dataframe if list(item.value.columns) == expected_detail
    )
    assert list(detail.columns) == expected_detail
