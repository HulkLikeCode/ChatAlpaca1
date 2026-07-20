"""Add Phase 6 portfolio configuration and classification foundations.

Revision ID: 20260719_0003
Revises: 20260719_0002
"""

import sqlalchemy as sa
from alembic import op

revision = "20260719_0003"
down_revision = "20260719_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "portfolios",
        sa.Column(
            "account_type",
            sa.String(length=24),
            nullable=False,
            server_default="unknown",
        ),
    )
    op.execute(
        sa.text(
            "UPDATE portfolios SET account_type = CASE "
            "WHEN lower(name) LIKE '%roth ira%' THEN 'roth_ira' "
            "WHEN lower(name) LIKE '%traditional ira%' THEN 'traditional_ira' "
            "ELSE 'unknown' END"
        )
    )
    # SQLite cannot add a table-level check without rebuilding the referenced
    # portfolios table. A rebuild can activate ON DELETE cascades while the old
    # table is dropped, so preserve rows and enforce the enum in services/model
    # metadata instead. PostgreSQL can add the check safely in place.
    if op.get_bind().dialect.name != "sqlite":
        op.create_check_constraint(
            "ck_portfolio_account_type",
            "portfolios",
            "account_type IN ('traditional_ira', 'roth_ira', 'taxable', 'unknown')",
        )

    op.create_table(
        "portfolio_benchmark_components",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("portfolio_id", sa.Integer(), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("weight", sa.Numeric(precision=9, scale=8), nullable=False),
        sa.Column("rebalancing_frequency", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("weight > 0 AND weight <= 1", name="ck_benchmark_component_weight"),
        sa.CheckConstraint(
            "rebalancing_frequency IN ('daily', 'monthly', 'quarterly', 'annual', 'none')",
            name="ck_benchmark_rebalancing_frequency",
        ),
        sa.ForeignKeyConstraint(["portfolio_id"], ["portfolios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "portfolio_id", "effective_from", "symbol", name="uq_benchmark_period_symbol"
        ),
    )
    op.create_index(
        "ix_portfolio_benchmark_components_portfolio_id",
        "portfolio_benchmark_components",
        ["portfolio_id"],
    )
    op.create_index(
        "ix_portfolio_benchmark_components_effective_from",
        "portfolio_benchmark_components",
        ["effective_from"],
    )

    op.create_table(
        "security_metadata",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("instrument_id", sa.Integer(), nullable=False),
        sa.Column("security_name", sa.String(length=240), nullable=True),
        sa.Column("asset_type", sa.String(length=24), nullable=True),
        sa.Column("sector", sa.String(length=120), nullable=True),
        sa.Column("industry", sa.String(length=160), nullable=True),
        sa.Column("source", sa.String(length=48), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("quality_status", sa.String(length=24), nullable=False),
        sa.Column("manual_override", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_security_metadata_confidence",
        ),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id", "source", "retrieved_at", name="uq_security_metadata_snapshot"
        ),
    )
    op.create_index("ix_security_metadata_instrument_id", "security_metadata", ["instrument_id"])

    op.create_table(
        "etf_sector_weights",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("instrument_id", sa.Integer(), nullable=False),
        sa.Column("sector", sa.String(length=120), nullable=False),
        sa.Column("weight", sa.Numeric(precision=9, scale=8), nullable=False),
        sa.Column("source", sa.String(length=48), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quality_status", sa.String(length=24), nullable=False),
        sa.CheckConstraint("weight >= 0 AND weight <= 1", name="ck_etf_sector_weight"),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id",
            "effective_date",
            "source",
            "sector",
            name="uq_etf_sector_snapshot_sector",
        ),
    )
    op.create_index("ix_etf_sector_weights_instrument_id", "etf_sector_weights", ["instrument_id"])
    op.create_index(
        "ix_etf_sector_weights_effective_date", "etf_sector_weights", ["effective_date"]
    )


def downgrade() -> None:
    op.drop_index("ix_etf_sector_weights_effective_date", table_name="etf_sector_weights")
    op.drop_index("ix_etf_sector_weights_instrument_id", table_name="etf_sector_weights")
    op.drop_table("etf_sector_weights")
    op.drop_index("ix_security_metadata_instrument_id", table_name="security_metadata")
    op.drop_table("security_metadata")
    op.drop_index(
        "ix_portfolio_benchmark_components_effective_from",
        table_name="portfolio_benchmark_components",
    )
    op.drop_index(
        "ix_portfolio_benchmark_components_portfolio_id",
        table_name="portfolio_benchmark_components",
    )
    op.drop_table("portfolio_benchmark_components")
    if op.get_bind().dialect.name != "sqlite":
        op.drop_constraint("ck_portfolio_account_type", "portfolios", type_="check")
    op.drop_column("portfolios", "account_type")
