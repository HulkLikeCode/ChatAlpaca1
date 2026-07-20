"""Add the Phase 3 historical market-data foundation.

Revision ID: 20260719_0002
Revises: 20260719_0001
"""

import sqlalchemy as sa
from alembic import op

revision = "20260719_0002"
down_revision = "20260719_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "instruments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("canonical_symbol", sa.String(length=32), nullable=False),
        sa.Column("asset_type", sa.String(length=24), nullable=False),
        sa.Column("exchange", sa.String(length=32), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("base_currency", sa.String(length=3), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("canonical_symbol"),
    )
    op.create_table(
        "market_datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=48), nullable=False),
        sa.Column("source", sa.String(length=48), nullable=False),
        sa.Column("feed", sa.String(length=32), nullable=True),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("adjustment_method", sa.String(length=32), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("coverage_start", sa.Date(), nullable=True),
        sa.Column("coverage_end", sa.Date(), nullable=True),
        sa.Column("quality_status", sa.String(length=24), nullable=False),
        sa.Column("validation_warnings", sa.Text(), nullable=False),
        sa.Column("override_priority", sa.Integer(), nullable=False),
        sa.Column("imported_file_hash", sa.String(length=64), nullable=True),
        sa.Column("request_metadata", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "coverage_end IS NULL OR coverage_start IS NULL OR coverage_end >= coverage_start",
            name="ck_market_dataset_coverage_dates",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("imported_file_hash", name="uq_market_dataset_file_hash"),
    )
    op.create_index("ix_market_datasets_provider", "market_datasets", ["provider"])
    op.create_index(
        "ix_market_datasets_adjustment_method", "market_datasets", ["adjustment_method"]
    )
    op.create_index(
        "ix_market_datasets_override_priority", "market_datasets", ["override_priority"]
    )
    op.create_table(
        "symbol_aliases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("instrument_id", sa.Integer(), nullable=False),
        sa.Column("alias", sa.String(length=32), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=True),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_from IS NULL OR effective_to >= effective_from",
            name="ck_symbol_alias_dates",
        ),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("alias", "effective_from", name="uq_symbol_alias_effective_from"),
    )
    op.create_index("ix_symbol_aliases_alias", "symbol_aliases", ["alias"])
    op.create_index("ix_symbol_aliases_instrument_id", "symbol_aliases", ["instrument_id"])
    op.create_table(
        "daily_bars",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("instrument_id", sa.Integer(), nullable=False),
        sa.Column("bar_date", sa.Date(), nullable=False),
        sa.Column("open", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("high", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("low", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("close", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("volume", sa.Numeric(precision=24, scale=8), nullable=True),
        sa.Column("trade_count", sa.Integer(), nullable=True),
        sa.Column("vwap", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "open > 0 AND high > 0 AND low > 0 AND close > 0", name="ck_daily_bar_prices"
        ),
        sa.CheckConstraint(
            "high >= open AND high >= close AND high >= low", name="ck_daily_bar_high"
        ),
        sa.CheckConstraint("low <= open AND low <= close AND low <= high", name="ck_daily_bar_low"),
        sa.CheckConstraint("volume IS NULL OR volume >= 0", name="ck_daily_bar_volume"),
        sa.ForeignKeyConstraint(["dataset_id"], ["market_datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "dataset_id", "instrument_id", "bar_date", name="uq_daily_bar_dataset_instrument_date"
        ),
    )
    op.create_index("ix_daily_bars_bar_date", "daily_bars", ["bar_date"])
    op.create_index("ix_daily_bars_dataset_id", "daily_bars", ["dataset_id"])
    op.create_index("ix_daily_bars_instrument_id", "daily_bars", ["instrument_id"])
    op.create_table(
        "proxy_assignments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("target_instrument_id", sa.Integer(), nullable=False),
        sa.Column("proxy_instrument_id", sa.Integer(), nullable=True),
        sa.Column("proxy_series", sa.String(length=80), nullable=True),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("data_sufficiency_rationale", sa.Text(), nullable=False),
        sa.Column("assignment_source", sa.String(length=12), nullable=False),
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_to >= effective_from",
            name="ck_proxy_assignment_dates",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_proxy_assignment_confidence",
        ),
        sa.CheckConstraint(
            "assignment_source IN ('manual', 'automatic')",
            name="ck_proxy_assignment_source",
        ),
        sa.CheckConstraint(
            "(proxy_instrument_id IS NOT NULL AND proxy_series IS NULL) OR "
            "(proxy_instrument_id IS NULL AND proxy_series IS NOT NULL)",
            name="ck_proxy_assignment_target",
        ),
        sa.ForeignKeyConstraint(["proxy_instrument_id"], ["instruments.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["target_instrument_id"], ["instruments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "target_instrument_id", "effective_from", name="uq_proxy_assignment_effective_from"
        ),
    )
    op.create_index(
        "ix_proxy_assignments_proxy_instrument_id", "proxy_assignments", ["proxy_instrument_id"]
    )
    op.create_index(
        "ix_proxy_assignments_target_instrument_id", "proxy_assignments", ["target_instrument_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_proxy_assignments_target_instrument_id", table_name="proxy_assignments")
    op.drop_index("ix_proxy_assignments_proxy_instrument_id", table_name="proxy_assignments")
    op.drop_table("proxy_assignments")
    op.drop_index("ix_daily_bars_instrument_id", table_name="daily_bars")
    op.drop_index("ix_daily_bars_dataset_id", table_name="daily_bars")
    op.drop_index("ix_daily_bars_bar_date", table_name="daily_bars")
    op.drop_table("daily_bars")
    op.drop_index("ix_symbol_aliases_instrument_id", table_name="symbol_aliases")
    op.drop_index("ix_symbol_aliases_alias", table_name="symbol_aliases")
    op.drop_table("symbol_aliases")
    op.drop_index("ix_market_datasets_override_priority", table_name="market_datasets")
    op.drop_index("ix_market_datasets_adjustment_method", table_name="market_datasets")
    op.drop_index("ix_market_datasets_provider", table_name="market_datasets")
    op.drop_table("market_datasets")
    op.drop_table("instruments")
