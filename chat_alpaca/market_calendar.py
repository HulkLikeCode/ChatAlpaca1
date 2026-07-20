from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)

NEW_YORK = ZoneInfo("America/New_York")
DAILY_BAR_READY_TIME = time(16, 15)


class UsEquityHolidayCalendar(AbstractHolidayCalendar):
    """Regular full-day NYSE/Nasdaq closures used for daily-bar coverage."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            start_date="2022-01-01",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def completed_session_cutoff(now: datetime | None = None) -> date:
    """Return a conservative cutoff that never requires an in-progress day's close."""
    timestamp = now or datetime.now(NEW_YORK)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=NEW_YORK)
    else:
        timestamp = timestamp.astimezone(NEW_YORK)
    if timestamp.time() >= DAILY_BAR_READY_TIME:
        return timestamp.date()
    return timestamp.date() - timedelta(days=1)


def market_session_index(
    start: date,
    end: date,
    *,
    now: datetime | None = None,
    completed_only: bool = True,
) -> pd.DatetimeIndex:
    """Return regular US-equity sessions, excluding holidays and unfinished dates."""
    effective_end = min(end, completed_session_cutoff(now)) if completed_only else end
    if start > effective_end:
        return pd.DatetimeIndex([])
    weekdays = pd.bdate_range(start, effective_end).normalize()
    holidays = UsEquityHolidayCalendar().holidays(start=start, end=effective_end).normalize()
    return weekdays.difference(holidays)
