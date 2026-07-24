from __future__ import annotations

from threading import Lock
from weakref import WeakKeyDictionary, WeakSet

from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from chat_alpaca.db import get_engine
from chat_alpaca.migrations import upgrade_database
from chat_alpaca.models import Portfolio
from chat_alpaca.portfolio_service import list_portfolios, seed_database

_bootstrap_registry_lock = Lock()
_bootstrap_locks: WeakKeyDictionary[Engine, Lock] = WeakKeyDictionary()
_bootstrapped_engines: WeakSet[Engine] = WeakSet()


def bootstrap_database_once(engine: Engine | None = None) -> None:
    """Bootstrap an engine once per process, retrying after any failure."""
    target_engine = engine or get_engine()
    with _bootstrap_registry_lock:
        if target_engine in _bootstrapped_engines:
            return
        engine_lock = _bootstrap_locks.setdefault(target_engine, Lock())

    with engine_lock:
        with _bootstrap_registry_lock:
            if target_engine in _bootstrapped_engines:
                return
        upgrade_database(target_engine)
        factory = sessionmaker(bind=target_engine, expire_on_commit=False)
        with factory.begin() as session:
            seed_database(session)
        with _bootstrap_registry_lock:
            _bootstrapped_engines.add(target_engine)


def bootstrap_database(engine: Engine | None = None) -> None:
    """Backward-compatible entry point for process-lifetime database bootstrap."""
    bootstrap_database_once(engine)


def _reset_bootstrap_state_for_tests() -> None:
    """Forget successful bootstraps; call only when no bootstrap is active."""
    with _bootstrap_registry_lock:
        _bootstrapped_engines.clear()


def initialize_application(engine: Engine | None = None) -> list[Portfolio]:
    """Initialize persistence and return detached application state for the UI."""
    target_engine = engine or get_engine()
    bootstrap_database_once(target_engine)
    factory = sessionmaker(bind=target_engine, expire_on_commit=False)
    with factory() as session:
        return list_portfolios(session)
