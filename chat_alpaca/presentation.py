from __future__ import annotations

import calendar
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from numbers import Real

import pandas as pd

DATE_PRESET_LABELS = ("5D", "1M", "6M", "YTD", "1Y", "5Y")
PERCENT_ASSUMPTIONS = {
    "market_decline",
    "holding_decline",
    "sector_decline",
    "dividend_reduction",
    "inflation",
    "inflation_increase",
    "expected_return",
    "low_return",
}
CURRENCY_ASSUMPTIONS = {"contribution_amount", "spending"}
MARKET_CONTEXT_PERCENT_COLUMNS = (
    "Daily return",
    "1M return",
    "3M return",
    "12M return",
    "Drawdown from available-window peak",
    "Realized volatility",
)


def nearest_hundred(value: object) -> float | None:
    """Round a display value to $100 using explicit conventional half-up behavior."""
    if value is None or pd.isna(value):
        return None
    numeric = Decimal(str(value))
    if not numeric.is_finite():
        return None
    return float(numeric.quantize(Decimal("1E2"), rounding=ROUND_HALF_UP))


def retirement_date_for_horizon(as_of: date, horizon_years: int) -> date:
    """Return the same calendar date after the horizon, clamping leap day safely."""
    if horizon_years < 1 or horizon_years > 40:
        raise ValueError("Scenario horizon must be between 1 and 40 years.")
    target_year = as_of.year + horizon_years
    return as_of.replace(
        year=target_year,
        day=min(as_of.day, calendar.monthrange(target_year, as_of.month)[1]),
    )


def _shift_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def date_preset_range(label: str, today: date) -> tuple[date, date]:
    """Resolve one inclusive, calendar-aware master date preset."""
    if label == "5D":
        start = today.fromordinal(today.toordinal() - 4)
    elif label == "1M":
        start = _shift_months(today, -1)
    elif label == "6M":
        start = _shift_months(today, -6)
    elif label == "YTD":
        start = date(today.year, 1, 1)
    elif label == "1Y":
        start = _shift_months(today, -12)
    elif label == "5Y":
        start = _shift_months(today, -60)
    else:
        raise ValueError(f"Unknown date preset: {label}")
    return start, today


def matching_date_preset(start: date, end: date, today: date) -> str | None:
    return next(
        (label for label in DATE_PRESET_LABELS if date_preset_range(label, today) == (start, end)),
        None,
    )


def format_relative_age(observed_at: datetime | None, *, now: datetime | None = None) -> str:
    """Format an authoritative observation timestamp without refreshing its source."""
    if observed_at is None:
        return "Unavailable"
    observed = (
        observed_at.replace(tzinfo=timezone.utc)
        if observed_at.tzinfo is None
        else observed_at.astimezone(timezone.utc)
    )
    reference = now or datetime.now(timezone.utc)
    reference = (
        reference.replace(tzinfo=timezone.utc)
        if reference.tzinfo is None
        else reference.astimezone(timezone.utc)
    )
    seconds = max(0, int((reference - observed).total_seconds()))
    if seconds < 60:
        value, unit = seconds, "second"
    elif seconds < 3_600:
        value, unit = seconds // 60, "minute"
    elif seconds < 86_400:
        value, unit = seconds // 3_600, "hour"
    else:
        value, unit = seconds // 86_400, "day"
    return f"{value} {unit}{'' if value == 1 else 's'} ago"


def format_one_decimal_percent(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1%}"


def format_correlation(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1f}"


def market_context_display_frame(context: pd.DataFrame) -> pd.DataFrame:
    """Scale display percentages and apply the V.15 Monitor column order."""
    result = context.copy()
    available_percentages = [
        column for column in MARKET_CONTEXT_PERCENT_COLUMNS if column in result
    ]
    result[available_percentages] *= 100
    if "12M return" not in result:
        return result
    through_returns = list(result.columns[: result.columns.get_loc("12M return") + 1])
    promoted = [
        column
        for column in ("Realized volatility", "21-session SPY correlation", "Trend")
        if column in result
    ]
    remaining = [column for column in result.columns if column not in {*through_returns, *promoted}]
    return result[[*through_returns, *promoted, *remaining]]


def sorted_hover_text(
    series: Sequence[pd.Series],
    *,
    value_format: str = ",.2f",
) -> list[str]:
    """Build per-date hover text sorted by that date's finite values."""
    if not series:
        return []
    index = series[0].index
    for item in series[1:]:
        index = index.union(item.index)
    rows: list[str] = []
    for timestamp in index:
        values = []
        for item in series:
            value = item.get(timestamp)
            if value is None or not isinstance(value, Real) or not math.isfinite(float(value)):
                continue
            values.append((str(item.name), float(value)))
        values.sort(key=lambda item: (-item[1], item[0]))
        rendered = "<br>".join(f"{name}: {format(value, value_format)}" for name, value in values)
        rows.append(f"{pd.Timestamp(timestamp):%b %d, %Y}<br>{rendered}")
    return rows


def monte_carlo_hover_text(
    dates: Sequence[object],
    percentiles: pd.DataFrame,
) -> list[str]:
    """Build one descending, rank-stable currency tooltip for each Monte Carlo date."""
    rank = {"P95": 95, "P75": 75, "P50": 50, "P25": 25, "P5": 5}
    labels = {
        "P95": "95th percentile",
        "P75": "75th percentile",
        "P50": "Median scenario",
        "P25": "25th percentile",
        "P5": "5th percentile",
    }
    rows = []
    for position, timestamp in enumerate(dates):
        values = [
            (column, float(percentiles[column].iloc[position]))
            for column in rank
            if column in percentiles
            and position < len(percentiles[column])
            and pd.notna(percentiles[column].iloc[position])
        ]
        values.sort(key=lambda item: (-item[1], -rank[item[0]]))
        rendered = "<br>".join(f"{labels[name]}: ${value:,.0f}" for name, value in values)
        rows.append(f"{pd.Timestamp(timestamp):%b %d, %Y}<br>{rendered}")
    return rows


@dataclass(frozen=True)
class AssumptionComparison:
    assumption: str
    value: object
    default_value: object
    delta: object
    unit: str


def assumption_comparisons(
    values: Mapping[str, object], defaults: Mapping[str, object]
) -> tuple[AssumptionComparison, ...]:
    """Compare raw assumptions before applying unit-aware display formatting."""
    comparisons = []
    for name, value in values.items():
        default = defaults.get(name)
        is_numeric = (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and isinstance(default, Real)
            and not isinstance(default, bool)
        )
        if is_numeric:
            delta: object = float(value) - float(default)
        else:
            delta = "Unchanged" if value == default else "Changed"
        unit = (
            "percent"
            if name in PERCENT_ASSUMPTIONS
            else "currency"
            if name in CURRENCY_ASSUMPTIONS
            else "number"
            if is_numeric
            else "text"
        )
        comparisons.append(
            AssumptionComparison(
                name.replace("_", " ").title(),
                value,
                default,
                delta,
                unit,
            )
        )
    return tuple(comparisons)


def format_assumption_value(value: object, unit: str) -> str:
    if value is None:
        return "None"
    if unit == "percent" and isinstance(value, Real):
        return f"{float(value):.1%}"
    if unit == "currency" and isinstance(value, Real):
        numeric = float(value)
        return f"{'-' if numeric < 0 else ''}${abs(numeric):,.0f}"
    if unit == "number" and isinstance(value, Real):
        return f"{float(value):,.2f}".rstrip("0").rstrip(".")
    return str(value)


def assumption_comparison_frame(
    values: Mapping[str, object], defaults: Mapping[str, object]
) -> pd.DataFrame:
    rows = []
    for item in assumption_comparisons(values, defaults):
        rows.append(
            {
                "Assumption": item.assumption,
                "Value": format_assumption_value(item.value, item.unit),
                "Default Value": format_assumption_value(item.default_value, item.unit),
                "Delta": format_assumption_value(item.delta, item.unit)
                if isinstance(item.delta, Real) and not isinstance(item.delta, bool)
                else item.delta,
            }
        )
    return pd.DataFrame(rows, columns=["Assumption", "Value", "Default Value", "Delta"])
