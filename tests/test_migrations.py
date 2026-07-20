from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from chat_alpaca.bootstrap import bootstrap_database
from chat_alpaca.migrations import (
    BASELINE_REVISION,
    CURRENT_REVISION,
    PROJECT_ROOT,
    SchemaAdoptionError,
    upgrade_database,
)
from chat_alpaca.models import Base, DataMigration, HoldingLot, Portfolio, PortfolioTransaction
from chat_alpaca.portfolio_service import (
    PHASE_1_DATE_CORRECTION_KEY,
    PHASE_1_MIGRATION_KEY,
)

EXPECTED_TABLES = {
    "alembic_version",
    "data_migrations",
    "holding_lots",
    "ledger_entries",
    "order_allocations",
    "portfolio_transactions",
    "portfolios",
    "transaction_overrides",
    "instruments",
    "symbol_aliases",
    "market_datasets",
    "daily_bars",
    "proxy_assignments",
}


def _revision(engine: sa.Engine) -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def test_empty_sqlite_database_upgrades_to_current_schema(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")

    upgrade_database(engine)

    assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES
    assert _revision(engine) == CURRENT_REVISION


def test_populated_pre_alembic_database_is_adopted_without_data_loss(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    Base.metadata.create_all(engine)
    applied_at = datetime(2026, 7, 18, 12, 30, tzinfo=timezone.utc)
    with Session(engine) as session:
        portfolio = Portfolio(name="Existing portfolio", cash=Decimal("123.4567"))
        session.add(portfolio)
        session.flush()
        session.add_all(
            [
                DataMigration(key=PHASE_1_MIGRATION_KEY, applied_at=applied_at),
                DataMigration(key=PHASE_1_DATE_CORRECTION_KEY, applied_at=applied_at),
                PortfolioTransaction(
                    portfolio_id=portfolio.id,
                    transaction_date=datetime(2026, 7, 1).date(),
                    kind="cash_adjustment",
                    action="Cash Adjustment",
                    description="Existing ledger value",
                    cash_delta=Decimal("123.4567"),
                    source="legacy",
                ),
            ]
        )
        session.commit()

    upgrade_database(engine)

    with Session(engine) as session:
        adopted = session.scalar(select(Portfolio))
        transaction = session.scalar(select(PortfolioTransaction))
        markers = list(session.scalars(select(DataMigration).order_by(DataMigration.key)))
        assert adopted is not None
        assert adopted.name == "Existing portfolio"
        assert adopted.cash == Decimal("123.4567")
        assert transaction is not None
        assert transaction.description == "Existing ledger value"
        assert transaction.cash_delta == Decimal("123.4567")
        assert [marker.key for marker in markers] == sorted(
            [PHASE_1_MIGRATION_KEY, PHASE_1_DATE_CORRECTION_KEY]
        )
        assert all(marker.applied_at == applied_at.replace(tzinfo=None) for marker in markers)
    assert _revision(engine) == CURRENT_REVISION


def test_phase_two_legacy_schema_is_adopted_then_upgraded(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'phase-two-legacy.db'}")
    with engine.begin() as connection:
        config = Config(PROJECT_ROOT / "alembic.ini")
        config.attributes["connection"] = connection
        command.upgrade(config, BASELINE_REVISION)
        connection.execute(text("DROP TABLE alembic_version"))

    upgrade_database(engine)

    assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES
    assert _revision(engine) == CURRENT_REVISION


def test_migration_execution_is_repeatable(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'repeat.db'}")

    upgrade_database(engine)
    upgrade_database(engine)

    with engine.connect() as connection:
        versions = connection.scalar(text("SELECT count(*) FROM alembic_version"))
    assert versions == 1
    assert _revision(engine) == CURRENT_REVISION


def test_incompatible_existing_schema_is_rejected_without_stamping(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'incompatible.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE portfolios ADD COLUMN incompatible_value TEXT"))

    with pytest.raises(
        SchemaAdoptionError,
        match=r"portfolios: unexpected columns incompatible_value",
    ):
        upgrade_database(engine)

    assert "alembic_version" not in inspect(engine).get_table_names()


def test_application_bootstrap_migrates_before_seeding(tmp_path, monkeypatch) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'startup-order.db'}")
    observed: list[tuple[set[str], str | None]] = []

    def observe_seed(session: Session) -> None:
        observed.append((set(inspect(session.connection()).get_table_names()), _revision(engine)))

    monkeypatch.setattr("chat_alpaca.bootstrap.seed_database", observe_seed)

    bootstrap_database(engine)

    assert observed == [(EXPECTED_TABLES, CURRENT_REVISION)]


def test_application_bootstrap_seeds_after_schema_upgrade(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'startup.db'}")

    bootstrap_database(engine)

    with Session(engine) as session:
        names = list(session.scalars(select(Portfolio.name).order_by(Portfolio.id)))
        markers = set(session.scalars(select(DataMigration.key)))
    assert names == [
        "KCs Traditional IRA",
        "KCs Roth IRA",
        "KC and Papa",
        "Portfolio 4",
        "Portfolio 5",
    ]
    assert {PHASE_1_MIGRATION_KEY, PHASE_1_DATE_CORRECTION_KEY} <= markers


def test_sqlite_foreign_keys_remain_enforced_after_migration(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'foreign-keys.db'}")
    upgrade_database(engine)

    with Session(engine) as session:
        session.add(
            HoldingLot(
                portfolio_id=999999,
                symbol="ABC",
                shares=Decimal("1"),
                acquired_on=datetime(2026, 1, 1).date(),
                cost_basis=Decimal("10"),
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()


def test_baseline_generates_postgresql_compatible_sql() -> None:
    output = StringIO()
    config = Config(PROJECT_ROOT / "alembic.ini", output_buffer=output)
    config.set_main_option(
        "sqlalchemy.url", "postgresql+psycopg://migration_test:unused@localhost/migration_test"
    )

    command.upgrade(config, "head", sql=True)

    sql = output.getvalue()
    assert "CREATE TABLE portfolios" in sql
    assert "CREATE TABLE portfolio_transactions" in sql
    assert "SERIAL NOT NULL" in sql
    assert f"'{BASELINE_REVISION}'" in sql
    assert f"'{CURRENT_REVISION}'" in sql
