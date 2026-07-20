import pytest

from chat_alpaca.config import _database_url, get_settings


@pytest.mark.parametrize(
    "url",
    [
        "postgres://user:password@host/database",
        "postgresql://user:password@host/database",
        "postgresql+psycopg://user:password@host/database",
        "postgresql+psycopg2://user:password@host/database",
        "postgresql+pg8000://user:password@host/database",
    ],
)
def test_database_url_uses_installed_psycopg_driver(url: str) -> None:
    assert _database_url(url) == "postgresql+psycopg://user:password@host/database"


def test_database_url_preserves_non_postgresql_urls() -> None:
    assert _database_url("sqlite:///data/chat_alpaca.db") == "sqlite:///data/chat_alpaca.db"


def test_settings_load_separate_admin_and_user_passwords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ADMIN_PASSWORD", "admin-secret")
    monkeypatch.setenv("USER_PASSWORD", "user-secret")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.admin_password == "admin-secret"
        assert settings.user_password == "user-secret"
    finally:
        get_settings.cache_clear()


def test_realtime_settings_are_clamped_to_safe_entitlement_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REALTIME_STREAM_CAP", "100")
    monkeypatch.setenv("REALTIME_REGULAR_SECONDS", "2")
    monkeypatch.setenv("REALTIME_OFF_HOURS_SECONDS", "10")
    monkeypatch.setenv("REALTIME_CALLS_PER_MINUTE", "999")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.realtime_stream_cap == 30
        assert settings.realtime_regular_seconds == 30
        assert settings.realtime_off_hours_seconds == 60
        assert settings.realtime_calls_per_minute == 200
    finally:
        get_settings.cache_clear()
