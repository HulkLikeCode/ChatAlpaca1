from __future__ import annotations

from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from chat_alpaca.db import get_engine
from chat_alpaca.migrations import upgrade_database
from chat_alpaca.models import Portfolio
from chat_alpaca.portfolio_service import list_portfolios, seed_database


def bootstrap_database(engine: Engine | None = None) -> None:
    """Upgrade schema before running seed and durable data-migration logic."""
    target_engine = engine or get_engine()
    upgrade_database(target_engine)
    factory = sessionmaker(bind=target_engine, expire_on_commit=False)
    with factory.begin() as session:
        seed_database(session)


def initialize_application(engine: Engine | None = None) -> list[Portfolio]:
    """Initialize persistence and return detached application state for the UI."""
    target_engine = engine or get_engine()
    bootstrap_database(target_engine)
    factory = sessionmaker(bind=target_engine, expire_on_commit=False)
    with factory() as session:
        return list_portfolios(session)
