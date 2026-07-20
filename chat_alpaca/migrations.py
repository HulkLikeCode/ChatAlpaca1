from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Engine, UniqueConstraint, inspect
from sqlalchemy.engine import Connection
from sqlalchemy.sql.schema import Column, Table
from sqlalchemy.sql.sqltypes import Numeric, String

from chat_alpaca.db import get_engine
from chat_alpaca.models import Base

BASELINE_REVISION = "20260719_0001"
CURRENT_REVISION = "20260719_0003"
ALEMBIC_TABLE = "alembic_version"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_TABLES = {
    "data_migrations",
    "holding_lots",
    "ledger_entries",
    "order_allocations",
    "portfolio_transactions",
    "portfolios",
    "transaction_overrides",
}
PHASE_3_TABLES = BASELINE_TABLES | {
    "instruments",
    "symbol_aliases",
    "market_datasets",
    "daily_bars",
    "proxy_assignments",
}


class SchemaAdoptionError(RuntimeError):
    """Raised when an unversioned database is unsafe to adopt."""


def _alembic_config(connection: Connection | None = None) -> Config:
    config = Config(PROJECT_ROOT / "alembic.ini")
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def _type_signature(column_type: Any) -> tuple[object, ...]:
    affinity = column_type._type_affinity.__name__
    if isinstance(column_type, String):
        return affinity, column_type.length
    if isinstance(column_type, Numeric):
        return affinity, column_type.precision, column_type.scale
    return (affinity,)


def _expected_columns(table: Table) -> dict[str, Column[Any]]:
    return {column.name: column for column in table.columns}


def _validate_columns(
    connection: Connection, table: Table, *, excluded_expected: set[str] | None = None
) -> list[str]:
    inspector = inspect(connection)
    actual = {column["name"]: column for column in inspector.get_columns(table.name)}
    expected = {
        name: column
        for name, column in _expected_columns(table).items()
        if name not in (excluded_expected or set())
    }
    issues: list[str] = []
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing:
        issues.append(f"{table.name}: missing columns {', '.join(missing)}")
    if unexpected:
        issues.append(f"{table.name}: unexpected columns {', '.join(unexpected)}")
    for name in sorted(set(expected) & set(actual)):
        expected_column = expected[name]
        actual_column = actual[name]
        if _type_signature(expected_column.type) != _type_signature(actual_column["type"]):
            issues.append(
                f"{table.name}.{name}: expected type {expected_column.type}, "
                f"found {actual_column['type']}"
            )
        if expected_column.nullable != actual_column["nullable"]:
            issues.append(
                f"{table.name}.{name}: expected nullable={expected_column.nullable}, "
                f"found nullable={actual_column['nullable']}"
            )
    expected_pk = tuple(column.name for column in table.primary_key.columns)
    actual_pk = tuple(inspector.get_pk_constraint(table.name).get("constrained_columns") or ())
    if expected_pk != actual_pk:
        issues.append(f"{table.name}: expected primary key {expected_pk}, found {actual_pk}")
    return issues


def _foreign_keys(table: Table) -> set[tuple[tuple[str, ...], str, tuple[str, ...], str | None]]:
    return {
        (
            tuple(constraint.column_keys),
            constraint.referred_table.name,
            tuple(element.column.name for element in constraint.elements),
            constraint.ondelete,
        )
        for constraint in table.foreign_key_constraints
    }


def _validate_constraints(connection: Connection, table: Table) -> list[str]:
    inspector = inspect(connection)
    issues: list[str] = []
    expected_fks = _foreign_keys(table)
    actual_fks = {
        (
            tuple(item.get("constrained_columns") or ()),
            item["referred_table"],
            tuple(item.get("referred_columns") or ()),
            (item.get("options") or {}).get("ondelete"),
        )
        for item in inspector.get_foreign_keys(table.name)
    }
    if expected_fks != actual_fks:
        issues.append(f"{table.name}: foreign-key definitions differ")

    expected_unique = {
        tuple(constraint.columns.keys())
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    actual_unique = {
        tuple(item.get("column_names") or ())
        for item in inspector.get_unique_constraints(table.name)
    }
    if expected_unique != actual_unique:
        issues.append(f"{table.name}: unique constraints differ")

    expected_indexes = {
        (index.name, tuple(column.name for column in index.columns), index.unique)
        for index in table.indexes
    }
    actual_indexes = {
        (item["name"], tuple(item.get("column_names") or ()), item["unique"])
        for item in inspector.get_indexes(table.name)
        if not item.get("duplicates_constraint")
    }
    if expected_indexes != actual_indexes:
        issues.append(f"{table.name}: indexes differ")
    return issues


def validate_schema_for_adoption(connection: Connection) -> str:
    """Validate structure only; never inspect application row values."""
    inspector = inspect(connection)
    actual_tables = set(inspector.get_table_names()) - {ALEMBIC_TABLE}
    current_tables = set(Base.metadata.tables)
    if actual_tables == BASELINE_TABLES:
        expected_tables = BASELINE_TABLES
        adoption_revision = BASELINE_REVISION
    elif actual_tables == PHASE_3_TABLES:
        expected_tables = PHASE_3_TABLES
        adoption_revision = "20260719_0002"
    else:
        expected_tables = current_tables
        adoption_revision = CURRENT_REVISION
    issues: list[str] = []
    missing = sorted(expected_tables - actual_tables)
    unexpected = sorted(actual_tables - expected_tables)
    if missing:
        issues.append(f"missing tables: {', '.join(missing)}")
    if unexpected:
        issues.append(f"unexpected tables: {', '.join(unexpected)}")
    for table_name in sorted(expected_tables & actual_tables):
        table = Base.metadata.tables[table_name]
        excluded = (
            {"account_type"}
            if adoption_revision in {BASELINE_REVISION, "20260719_0002"}
            and table_name == "portfolios"
            else set()
        )
        issues.extend(_validate_columns(connection, table, excluded_expected=excluded))
        issues.extend(_validate_constraints(connection, table))
    if issues:
        details = "\n- ".join(issues)
        raise SchemaAdoptionError(
            "The existing database does not match the Phase 2 baseline or current schema and "
            "was not stamped. "
            "Back up the database, then reconcile its schema before retrying. Differences:\n"
            f"- {details}"
        )
    return adoption_revision


def upgrade_database(engine: Engine | None = None) -> None:
    """Upgrade a fresh/versioned database or safely adopt a matching legacy database."""
    target_engine = engine or get_engine()
    with target_engine.begin() as connection:
        tables = set(inspect(connection).get_table_names())
        config = _alembic_config(connection)
        current_revision = MigrationContext.configure(connection).get_current_revision()
        application_tables = tables - {ALEMBIC_TABLE}
        if current_revision is None and application_tables:
            adoption_revision = validate_schema_for_adoption(connection)
            command.stamp(config, adoption_revision)
        command.upgrade(config, "head")


def downgrade_database(revision: str = "base", engine: Engine | None = None) -> None:
    target_engine = engine or get_engine()
    with target_engine.begin() as connection:
        command.downgrade(_alembic_config(connection), revision)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChatAlpaca database migrations safely.")
    parser.add_argument("action", choices=("upgrade", "downgrade"))
    parser.add_argument("revision", nargs="?", default=None)
    args = parser.parse_args()
    if args.action == "upgrade":
        upgrade_database()
    else:
        downgrade_database(args.revision or "base")


if __name__ == "__main__":
    main()
