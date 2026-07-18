from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_public_app_renders_without_credentials() -> None:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "KC's Retirement Dough, Let's GO!!!"
    assert [tab.label for tab in app.tabs] == ["Overview", "Compare"]
    assert any("KCs Traditional IRA" in text.value for text in app.markdown)
    assert [item.label for item in app.metric].count("Total selected value") == 2
    assert "Custom gain/loss start" in [item.label for item in app.date_input]
    assert "Per-portfolio / lot breakdown" in [item.label for item in app.expander]


def test_phase_2_owner_manage_controls_render() -> None:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30)
    app.session_state["owner_authenticated"] = True
    app.run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Compare",
        "Manage",
        "Trade",
        "Architecture",
    ]
    assert "Overview portfolios" in [item.label for item in app.multiselect]
    assert "Manage portfolios" in [item.label for item in app.multiselect]
    assert "Transaction type filter" in [item.label for item in app.multiselect]
    assert "Target portfolio" in [item.label for item in app.selectbox]
    assert "Transaction date (M/D/YY)" in [item.label for item in app.text_input]
    assert [item.label for item in app.metric].count("Year to date") == 2
    assert [item.label for item in app.metric].count("Trailing 365 days") == 2
    assert [item.label for item in app.metric].count("Custom range") == 2
    assert [item.label for item in app.get("download_button")] == [
        "Download Brokerage CSV template"
    ]
    assert any("Quantity totals by symbol" in item.value for item in app.markdown)
