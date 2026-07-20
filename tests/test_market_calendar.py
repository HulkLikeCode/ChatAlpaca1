from datetime import date, datetime
from zoneinfo import ZoneInfo

from chat_alpaca.market_calendar import market_session_index


def test_market_sessions_exclude_holidays_and_in_progress_date() -> None:
    sessions = market_session_index(
        date(2026, 5, 22),
        date(2026, 7, 20),
        now=datetime(2026, 7, 20, 10, tzinfo=ZoneInfo("America/New_York")),
    )
    dates = set(sessions.date)

    assert date(2026, 5, 25) not in dates  # Memorial Day
    assert date(2026, 6, 19) not in dates  # Juneteenth
    assert date(2026, 7, 3) not in dates  # observed Independence Day
    assert date(2026, 7, 20) not in dates  # no completed daily close yet
    assert date(2026, 7, 17) in dates


def test_market_sessions_include_known_completed_trading_days() -> None:
    sessions = market_session_index(
        date(2026, 4, 2),
        date(2026, 4, 6),
        now=datetime(2026, 4, 7, 9, tzinfo=ZoneInfo("America/New_York")),
    )

    assert list(sessions.date) == [date(2026, 4, 2), date(2026, 4, 6)]


def test_session_is_available_after_daily_bar_delay() -> None:
    sessions = market_session_index(
        date(2026, 7, 20),
        date(2026, 7, 20),
        now=datetime(2026, 7, 20, 16, 16, tzinfo=ZoneInfo("America/New_York")),
    )

    assert list(sessions.date) == [date(2026, 7, 20)]
