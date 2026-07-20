"""Add Phase 11 non-executable hypothetical scenario persistence.

Revision ID: 20260720_0005
Revises: 20260719_0004
"""

import sqlalchemy as sa
from alembic import op

revision = "20260720_0005"
down_revision = "20260719_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hypothetical_scenarios",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("creator", sa.String(length=120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("portfolio_scope", sa.Text(), nullable=False),
        sa.Column("baseline_ledger_hash", sa.String(length=64), nullable=False),
        sa.Column("market_data_as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("assumptions", sa.Text(), nullable=False),
        sa.Column("proposed_trades", sa.Text(), nullable=False),
        sa.Column("results", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("creator", "name", name="uq_hypothetical_scenario_creator_name"),
    )
    op.create_index(
        "ix_hypothetical_scenarios_created_at", "hypothetical_scenarios", ["created_at"]
    )
    op.create_index(
        "ix_hypothetical_scenarios_baseline_ledger_hash",
        "hypothetical_scenarios",
        ["baseline_ledger_hash"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_hypothetical_scenarios_baseline_ledger_hash",
        table_name="hypothetical_scenarios",
    )
    op.drop_index("ix_hypothetical_scenarios_created_at", table_name="hypothetical_scenarios")
    op.drop_table("hypothetical_scenarios")
