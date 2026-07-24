from datetime import date, datetime, timezone

import pandas as pd
import pytest

from chat_alpaca.presentation import (
    assumption_comparison_frame,
    date_preset_range,
    format_correlation,
    format_one_decimal_percent,
    format_relative_age,
    market_context_display_frame,
    matching_date_preset,
    monte_carlo_hover_text,
    nearest_hundred,
    retirement_date_for_horizon,
    sorted_hover_text,
)


def test_date_presets_are_calendar_aware_at_january_month_end() -> None:
    today = date(2024, 1, 31)

    assert date_preset_range("5D", today) == (date(2024, 1, 27), today)
    assert date_preset_range("YTD", today) == (date(2024, 1, 1), today)
    assert date_preset_range("1M", today) == (date(2023, 12, 31), today)
    assert date_preset_range("6M", today) == (date(2023, 7, 31), today)


def test_date_presets_normalize_february_and_leap_year_boundaries() -> None:
    leap_day = date(2024, 2, 29)

    assert date_preset_range("1M", date(2024, 3, 31)) == (
        date(2024, 2, 29),
        date(2024, 3, 31),
    )
    assert date_preset_range("1Y", leap_day) == (date(2023, 2, 28), leap_day)
    assert date_preset_range("5Y", leap_day) == (date(2019, 2, 28), leap_day)
    assert matching_date_preset(date(2023, 2, 28), leap_day, leap_day) == "1Y"


def test_relative_age_uses_units_and_clamps_future_clock_skew() -> None:
    now = datetime(2026, 7, 24, 12, tzinfo=timezone.utc)

    assert format_relative_age(now, now=now) == "0 seconds ago"
    assert format_relative_age(datetime(2026, 7, 24, 11, 59, 28), now=now) == "32 seconds ago"
    assert format_relative_age(datetime(2026, 7, 24, 11, 43), now=now) == "17 minutes ago"
    assert format_relative_age(datetime(2026, 7, 23, 21), now=now) == "15 hours ago"
    assert format_relative_age(datetime(2026, 7, 22, 12), now=now) == "2 days ago"
    assert format_relative_age(datetime(2026, 7, 24, 12, 1), now=now) == "0 seconds ago"
    assert format_relative_age(None, now=now) == "Unavailable"


def test_display_formatters_keep_percent_and_correlation_semantics_separate() -> None:
    assert format_one_decimal_percent(0.037) == "3.7%"
    assert format_one_decimal_percent(-0.012) == "-1.2%"
    assert format_correlation(0.84) == "0.8"
    assert format_correlation(-0.26) == "-0.3"
    assert "%" not in format_correlation(0.84)


def test_sorted_hover_text_orders_each_date_and_omits_missing_values() -> None:
    index = pd.to_datetime(["2026-07-22", "2026-07-23"])
    first = pd.Series([100.0, 110.0], index=index, name="First")
    second = pd.Series([120.0, float("nan")], index=index, name="Second")

    text = sorted_hover_text((first, second))

    assert text[0].index("Second: 120.00") < text[0].index("First: 100.00")
    assert "Second" not in text[1]
    assert "First: 110.00" in text[1]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (599_849, 599_800),
        (599_850, 599_900),
        (49, 0),
        (50, 100),
        (149, 100),
        (150, 200),
        (-149, -100),
        (-150, -200),
    ],
)
def test_nearest_hundred_uses_half_up_display_rounding(value: float, expected: float) -> None:
    assert nearest_hundred(value) == expected


def test_retirement_date_horizon_is_bounded_and_leap_safe() -> None:
    assert retirement_date_for_horizon(date(2024, 2, 29), 1) == date(2025, 2, 28)
    assert retirement_date_for_horizon(date(2024, 2, 29), 40) == date(2064, 2, 29)
    with pytest.raises(ValueError, match="between 1 and 40"):
        retirement_date_for_horizon(date(2024, 2, 29), 0)
    with pytest.raises(ValueError, match="between 1 and 40"):
        retirement_date_for_horizon(date(2024, 2, 29), 41)


def test_monte_carlo_tooltip_sorts_actual_values_with_rank_tie_break() -> None:
    dates = pd.to_datetime(["2026-07-22", "2026-07-23"])
    values = pd.DataFrame(
        {
            "P5": [500, 100],
            "P25": [300, 400],
            "P50": [400, 300],
            "P75": [200, 400],
            "P95": [100, 200],
        }
    )

    text = monte_carlo_hover_text(dates, values)

    assert text[0].index("5th percentile: $500") < text[0].index("Median scenario: $400")
    assert text[0].index("25th percentile: $300") < text[0].index("75th percentile: $200")
    assert text[1].index("75th percentile: $400") < text[1].index("25th percentile: $400")


def test_market_context_display_order_and_scaling_preserve_raw_correlation() -> None:
    raw = pd.DataFrame(
        [
            {
                "Symbol": "SPY",
                "Name": "S&P 500",
                "Daily return": 0.037,
                "1M return": -0.012,
                "3M return": 0.04,
                "12M return": 0.10,
                "Trend": "above",
                "Drawdown from available-window peak": -0.02,
                "Realized volatility": 0.16,
                "21-session SPY correlation": 0.84,
                "Correlation observations": "21/21",
            }
        ]
    )

    result = market_context_display_frame(raw)

    return_end = result.columns.get_loc("12M return")
    assert list(result.columns[return_end + 1 : return_end + 4]) == [
        "Realized volatility",
        "21-session SPY correlation",
        "Trend",
    ]
    assert result.loc[0, "Daily return"] == pytest.approx(3.7)
    assert result.loc[0, "Realized volatility"] == pytest.approx(16.0)
    assert result.loc[0, "21-session SPY correlation"] == pytest.approx(0.84)


def test_assumption_comparison_formats_raw_positive_negative_zero_and_text_deltas() -> None:
    values = {
        "market_decline": -0.30,
        "expected_return": 0.05,
        "contribution_amount": 1250.0,
        "spending": 125.0,
        "interruption_months": 12,
        "holding_symbol": "AAPL",
    }
    defaults = {
        "market_decline": -0.20,
        "expected_return": 0.07,
        "contribution_amount": 1500.0,
        "spending": 100.0,
        "interruption_months": 12,
        "holding_symbol": None,
    }

    result = assumption_comparison_frame(values, defaults).set_index("Assumption")

    assert list(result.reset_index().columns) == [
        "Assumption",
        "Value",
        "Default Value",
        "Delta",
    ]
    assert result.loc["Market Decline", "Delta"] == "-10.0%"
    assert result.loc["Expected Return", "Delta"] == "-2.0%"
    assert result.loc["Contribution Amount", "Delta"] == "-$250"
    assert result.loc["Spending", "Delta"] == "$25"
    assert result.loc["Interruption Months", "Delta"] == "0"
    assert result.loc["Holding Symbol", "Delta"] == "Changed"
