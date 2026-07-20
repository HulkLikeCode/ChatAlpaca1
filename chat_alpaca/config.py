from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _value(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        import streamlit as st

        return str(st.secrets.get(name, default))
    except Exception:
        return default


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(_value(name, str(default)))
    except ValueError:
        return default
    return min(max(parsed, minimum), maximum)


def _database_url(value: str) -> str:
    """Use the PostgreSQL driver installed by this application.

    Hosted database providers commonly supply URLs for ``psycopg2`` or without
    a driver name.  This project installs psycopg 3 (``psycopg[binary]``), so
    normalize every PostgreSQL URL spelling before SQLAlchemy loads a dialect.
    """
    value = value.strip()
    scheme, separator, remainder = value.partition("://")
    if separator and (scheme == "postgres" or scheme.startswith("postgresql")):
        return f"postgresql+psycopg://{remainder}"
    return value


@dataclass(frozen=True)
class Settings:
    database_url: str
    admin_password: str
    user_password: str
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_data_feed: str
    trading_mode: str
    allow_live_trading: bool
    realtime_stream_cap: int = 30
    realtime_regular_seconds: int = 45
    realtime_off_hours_seconds: int = 180
    realtime_calls_per_minute: int = 180

    @property
    def alpaca_configured(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    @property
    def paper(self) -> bool:
        return self.trading_mode != "live"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    mode = _value("TRADING_MODE", "paper").strip().lower()
    if mode not in {"paper", "live"}:
        mode = "paper"
    regular_seconds = _bounded_int("REALTIME_REGULAR_SECONDS", 45, 30, 60)
    off_hours_seconds = max(
        _bounded_int("REALTIME_OFF_HOURS_SECONDS", 180, 60, 900), regular_seconds + 30
    )
    return Settings(
        database_url=_database_url(_value("DATABASE_URL", "sqlite:///data/chat_alpaca.db")),
        admin_password=_value("ADMIN_PASSWORD"),
        user_password=_value("USER_PASSWORD"),
        alpaca_api_key=_value("ALPACA_API_KEY"),
        alpaca_secret_key=_value("ALPACA_SECRET_KEY"),
        alpaca_data_feed=_value("ALPACA_DATA_FEED", "iex").lower(),
        trading_mode=mode,
        allow_live_trading=_as_bool(_value("ALLOW_LIVE_TRADING", "false")),
        realtime_stream_cap=_bounded_int("REALTIME_STREAM_CAP", 30, 1, 30),
        realtime_regular_seconds=regular_seconds,
        realtime_off_hours_seconds=off_hours_seconds,
        realtime_calls_per_minute=_bounded_int("REALTIME_CALLS_PER_MINUTE", 180, 1, 200),
    )
