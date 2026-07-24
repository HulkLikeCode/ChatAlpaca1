from pathlib import Path
from subprocess import run

from chat_alpaca.portfolio_service import seed_database

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_private_statement_is_not_tracked_or_a_runtime_seed_dependency() -> None:
    tracked = run(
        ["git", "ls-files", "*.csv"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert "KC and Papa.csv" not in tracked
    assert "KC.csv" not in tracked
    assert "Papa.csv" not in tracked
    source = (PROJECT_ROOT / "chat_alpaca" / "portfolio_service.py").read_text()
    assert ".csv" not in source[: source.index("def parse_statement_csv")]
    assert "read_bytes" not in source[: source.index("def parse_statement_csv")]
    assert seed_database.__doc__ is not None
    assert "non-private" in seed_database.__doc__


def test_runtime_upload_export_database_cache_and_snapshot_paths_are_ignored() -> None:
    paths = [
        "runtime/uploads/statement.csv",
        "runtime/tmp/upload.tmp",
        "runtime/exports/portfolio.csv",
        "runtime/snapshots/portfolio.csv",
        "uploads/statement.csv",
        "exports/portfolio.csv",
        "cache/portfolio.csv",
        "snapshots/portfolio.csv",
        "local.sqlite3",
        "local.db-wal",
    ]
    result = run(
        ["git", "check-ignore", *paths],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert set(result.stdout.splitlines()) == set(paths)
