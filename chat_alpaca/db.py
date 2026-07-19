from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from chat_alpaca.config import get_settings


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection: object, _connection_record: object) -> None:
    """Enable SQLite's opt-in foreign-key enforcement on every connection."""
    if dbapi_connection.__class__.__module__ == "sqlite3":
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = get_settings().database_url
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite:///"):
        db_path = Path(url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connect_args["check_same_thread"] = False
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
