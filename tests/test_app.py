from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_public_app_renders_without_credentials() -> None:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "KC's Retirement Dough, Let's GO!!!"
    assert [tab.label for tab in app.tabs] == ["Overview", "Compare"]
    assert any("KCs Traditional IRA" in text.value for text in app.markdown)
