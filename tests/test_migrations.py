from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

from chat_alpaca.bootstrap import (
    _reset_bootstrap_state_for_tests,
    bootstrap_database,
    bootstrap_database_once,
    initialize_application,
)
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
    create_portfolio,
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
    "portfolio_benchmark_components",
    "security_metadata",
    "etf_sector_weights",
    "forecast_runs",
    "forecast_run_datasets",
    "model_validations",
    "hypothetical_scenarios",
}


@pytest.fixture(autouse=True)
def reset_bootstrap_state() -> None:
    _reset_bootstrap_state_for_tests()
    yield
    _reset_bootstrap_state_for_tests()


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


def test_phase_six_account_type_migration_is_conservative(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'account-types.db'}")
    with engine.begin() as connection:
        config = Config(PROJECT_ROOT / "alembic.ini")
        config.attributes["connection"] = connection
        command.upgrade(config, "20260719_0002")
        connection.execute(
            text(
                "INSERT INTO portfolios (name, cash, created_at) VALUES "
                "('Family Traditional IRA', 0, CURRENT_TIMESTAMP), "
                "('Family Roth IRA', 0, CURRENT_TIMESTAMP), "
                "('Ordinary Brokerage', 0, CURRENT_TIMESTAMP), "
                "('IRA wording is absent', 0, CURRENT_TIMESTAMP)"
            )
        )
        traditional_id = connection.scalar(
            text("SELECT id FROM portfolios WHERE name = 'Family Traditional IRA'")
        )
        connection.execute(
            text(
                "INSERT INTO holding_lots "
                "(portfolio_id, symbol, shares, acquired_on, cost_basis) "
                "VALUES (:portfolio_id, 'KEEP', 2, '2026-01-01', 10)"
            ),
            {"portfolio_id": traditional_id},
        )

    upgrade_database(engine)

    with engine.connect() as connection:
        rows = dict(
            connection.execute(text("SELECT name, account_type FROM portfolios")).tuples().all()
        )
    assert rows == {
        "Family Traditional IRA": "traditional_ira",
        "Family Roth IRA": "roth_ira",
        "Ordinary Brokerage": "unknown",
        "IRA wording is absent": "unknown",
    }
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM holding_lots")) == 1


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


def test_application_initialization_returns_bootstrapped_state(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'application-state.db'}")

    portfolios = initialize_application(engine)

    assert [portfolio.name for portfolio in portfolios[:3]] == [
        "KCs Traditional IRA",
        "KCs Roth IRA",
        "KC and Papa",
    ]
    assert portfolios[0].holdings


def test_application_bootstraps_once_and_loads_portfolios_every_time(tmp_path, monkeypatch) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'bootstrap-once.db'}")
    calls = {"migration": 0, "seed": 0, "list": 0}
    original_upgrade = upgrade_database

    def observe_upgrade(target_engine: sa.Engine) -> None:
        calls["migration"] += 1
        original_upgrade(target_engine)

    def observe_seed(_session: Session) -> None:
        calls["seed"] += 1

    def observe_list(_session: Session) -> list[Portfolio]:
        calls["list"] += 1
        return []

    monkeypatch.setattr("chat_alpaca.bootstrap.upgrade_database", observe_upgrade)
    monkeypatch.setattr("chat_alpaca.bootstrap.seed_database", observe_seed)
    monkeypatch.setattr("chat_alpaca.bootstrap.list_portfolios", observe_list)

    assert initialize_application(engine) == []
    assert initialize_application(engine) == []
    assert initialize_application(engine) == []

    assert calls == {"migration": 1, "seed": 1, "list": 3}


def test_different_engines_bootstrap_independently(tmp_path, monkeypatch) -> None:
    engines = [
        create_engine(f"sqlite:///{tmp_path / 'first.db'}"),
        create_engine(f"sqlite:///{tmp_path / 'second.db'}"),
    ]
    migrations: list[sa.Engine] = []
    seeds: list[sa.Engine] = []

    monkeypatch.setattr(
        "chat_alpaca.bootstrap.upgrade_database",
        lambda engine: migrations.append(engine),
    )
    monkeypatch.setattr(
        "chat_alpaca.bootstrap.seed_database",
        lambda session: seeds.append(session.get_bind()),
    )

    for engine in engines:
        bootstrap_database_once(engine)
        bootstrap_database_once(engine)

    assert migrations == engines
    assert seeds == engines


def test_failed_migration_is_retried(tmp_path, monkeypatch) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'migration-retry.db'}")
    calls = {"migration": 0, "seed": 0}
    original_upgrade = upgrade_database

    def fail_once(target_engine: sa.Engine) -> None:
        calls["migration"] += 1
        if calls["migration"] == 1:
            raise RuntimeError("migration failed")
        original_upgrade(target_engine)

    def observe_seed(_session: Session) -> None:
        calls["seed"] += 1

    monkeypatch.setattr("chat_alpaca.bootstrap.upgrade_database", fail_once)
    monkeypatch.setattr("chat_alpaca.bootstrap.seed_database", observe_seed)

    with pytest.raises(RuntimeError, match="migration failed"):
        bootstrap_database_once(engine)
    bootstrap_database_once(engine)
    bootstrap_database_once(engine)

    assert calls == {"migration": 2, "seed": 1}


def test_failed_seed_retries_the_incomplete_bootstrap(tmp_path, monkeypatch) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'seed-retry.db'}")
    calls = {"migration": 0, "seed": 0}
    original_upgrade = upgrade_database

    def observe_upgrade(target_engine: sa.Engine) -> None:
        calls["migration"] += 1
        original_upgrade(target_engine)

    def fail_once(_session: Session) -> None:
        calls["seed"] += 1
        if calls["seed"] == 1:
            raise RuntimeError("seed failed")

    monkeypatch.setattr("chat_alpaca.bootstrap.upgrade_database", observe_upgrade)
    monkeypatch.setattr("chat_alpaca.bootstrap.seed_database", fail_once)

    with pytest.raises(RuntimeError, match="seed failed"):
        bootstrap_database_once(engine)
    bootstrap_database_once(engine)
    bootstrap_database_once(engine)

    assert calls == {"migration": 2, "seed": 2}


def test_concurrent_first_calls_bootstrap_once(tmp_path, monkeypatch) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'concurrent.db'}")
    upgrade_database(engine)
    calls = {"migration": 0, "seed": 0, "list": 0}
    calls_lock = threading.Lock()
    original_upgrade = upgrade_database

    def observe_upgrade(target_engine: sa.Engine) -> None:
        with calls_lock:
            calls["migration"] += 1
        time.sleep(0.05)
        original_upgrade(target_engine)

    def observe_seed(_session: Session) -> None:
        with calls_lock:
            calls["seed"] += 1
        time.sleep(0.05)

    def observe_list(_session: Session) -> list[Portfolio]:
        with calls_lock:
            calls["list"] += 1
        return []

    monkeypatch.setattr("chat_alpaca.bootstrap.upgrade_database", observe_upgrade)
    monkeypatch.setattr("chat_alpaca.bootstrap.seed_database", observe_seed)
    monkeypatch.setattr("chat_alpaca.bootstrap.list_portfolios", observe_list)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: initialize_application(engine), range(2)))

    assert results == [[], []]
    assert calls == {"migration": 1, "seed": 1, "list": 2}


def test_reset_allows_the_same_engine_to_bootstrap_again(tmp_path, monkeypatch) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'reset.db'}")
    calls = 0

    def observe_upgrade(_engine: sa.Engine) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr("chat_alpaca.bootstrap.upgrade_database", observe_upgrade)
    monkeypatch.setattr("chat_alpaca.bootstrap.seed_database", lambda _session: None)

    bootstrap_database_once(engine)
    _reset_bootstrap_state_for_tests()
    bootstrap_database_once(engine)

    assert calls == 2


def test_application_initialization_reloads_fresh_portfolio_state(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh-state.db'}")

    initial = initialize_application(engine)
    with Session(engine) as session:
        created = create_portfolio(session, "Freshly added")
        created_id = created.id
        session.commit()
    refreshed = initialize_application(engine)

    assert all(portfolio.id != created_id for portfolio in initial)
    assert any(
        portfolio.id == created_id and portfolio.name == "Freshly added" for portfolio in refreshed
    )


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
