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


def _database_url(value: str) -> str:
    if value.startswith("postgres://"):
        return value.replace("postgres://", "postgresql+psycopg://", 1)
    if value.startswith("postgresql://"):
        return value.replace("postgresql://", "postgresql+psycopg://", 1)
    return value


@dataclass(frozen=True)
class Settings:
    database_url: str
    admin_password: str
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_data_feed: str
    trading_mode: str
    allow_live_trading: bool

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
    return Settings(
        database_url=_database_url(_value("DATABASE_URL", "sqlite:///data/chat_alpaca.db")),
        admin_password=_value("ADMIN_PASSWORD"),
        alpaca_api_key=_value("ALPACA_API_KEY"),
        alpaca_secret_key=_value("ALPACA_SECRET_KEY"),
        alpaca_data_feed=_value("ALPACA_DATA_FEED", "iex").lower(),
        trading_mode=mode,
        allow_live_trading=_as_bool(_value("ALLOW_LIVE_TRADING", "false")),
    )
