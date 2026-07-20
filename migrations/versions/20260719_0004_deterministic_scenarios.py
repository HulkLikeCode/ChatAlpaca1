"""Add Phase 7 deterministic forecast persistence and validation records.

Revision ID: 20260719_0004
Revises: 20260719_0003
"""

import sqlalchemy as sa
from alembic import op

revision = "20260719_0004"
down_revision = "20260719_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forecast_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_type", sa.String(length=48), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("portfolio_scope", sa.Text(), nullable=False),
        sa.Column("ledger_state_hash", sa.String(length=64), nullable=False),
        sa.Column("assumptions", sa.Text(), nullable=False),
        sa.Column("data_coverage", sa.Text(), nullable=False),
        sa.Column("proxy_use", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("validation_status", sa.String(length=16), nullable=False),
        sa.Column("summary_outputs", sa.Text(), nullable=False),
        sa.Column("scenario_bands", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'completed', 'failed')", name="ck_forecast_run_status"
        ),
        sa.CheckConstraint(
            "validation_status IN ('unvalidated', 'in_review', 'validated', 'rejected')",
            name="ck_forecast_run_validation_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_forecast_runs_model_type", "forecast_runs", ["model_type"])
    op.create_index("ix_forecast_runs_created_at", "forecast_runs", ["created_at"])
    op.create_index("ix_forecast_runs_ledger_state_hash", "forecast_runs", ["ledger_state_hash"])

    op.create_table(
        "forecast_run_datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("forecast_run_id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("purpose", sa.String(length=48), nullable=False),
        sa.ForeignKeyConstraint(["forecast_run_id"], ["forecast_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["dataset_id"], ["market_datasets.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("forecast_run_id", "dataset_id", name="uq_forecast_run_dataset"),
    )
    op.create_index(
        "ix_forecast_run_datasets_forecast_run_id",
        "forecast_run_datasets",
        ["forecast_run_id"],
    )
    op.create_index("ix_forecast_run_datasets_dataset_id", "forecast_run_datasets", ["dataset_id"])

    op.create_table(
        "model_validations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_type", sa.String(length=48), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("automated_tests_passed", sa.Boolean(), nullable=False),
        sa.Column("evidence", sa.Text(), nullable=False),
        sa.Column("limitations", sa.Text(), nullable=False),
        sa.Column("reviewer", sa.String(length=120), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('unvalidated', 'in_review', 'validated', 'rejected')",
            name="ck_model_validation_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_type", "model_version", name="uq_model_validation_version"),
    )
    op.create_index("ix_model_validations_model_type", "model_validations", ["model_type"])


def downgrade() -> None:
    op.drop_index("ix_model_validations_model_type", table_name="model_validations")
    op.drop_table("model_validations")
    op.drop_index("ix_forecast_run_datasets_dataset_id", table_name="forecast_run_datasets")
    op.drop_index("ix_forecast_run_datasets_forecast_run_id", table_name="forecast_run_datasets")
    op.drop_table("forecast_run_datasets")
    op.drop_index("ix_forecast_runs_ledger_state_hash", table_name="forecast_runs")
    op.drop_index("ix_forecast_runs_created_at", table_name="forecast_runs")
    op.drop_index("ix_forecast_runs_model_type", table_name="forecast_runs")
    op.drop_table("forecast_runs")
