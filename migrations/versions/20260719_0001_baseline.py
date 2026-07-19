"""Define the pre-Alembic DashApp schema as the baseline.

Revision ID: 20260719_0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "20260719_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "data_migrations",
        sa.Column("key", sa.String(length=120), nullable=False),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_table(
        "portfolios",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("cash", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "holding_lots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("portfolio_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("shares", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("acquired_on", sa.Date(), nullable=False),
        sa.Column("cost_basis", sa.Numeric(precision=18, scale=6), nullable=False),
        sa.ForeignKeyConstraint(["portfolio_id"], ["portfolios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_holding_lots_portfolio_id", "holding_lots", ["portfolio_id"])
    op.create_index("ix_holding_lots_symbol", "holding_lots", ["symbol"])
    op.create_table(
        "ledger_entries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("portfolio_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=True),
        sa.Column("quantity", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("price", sa.Numeric(precision=18, scale=6), nullable=True),
        sa.Column("cash_delta", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("note", sa.String(length=240), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["portfolio_id"], ["portfolios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ledger_entries_portfolio_id", "ledger_entries", ["portfolio_id"])
    op.create_table(
        "order_allocations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("portfolio_id", sa.Integer(), nullable=False),
        sa.Column("alpaca_order_id", sa.String(length=64), nullable=False),
        sa.Column("client_order_id", sa.String(length=48), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("order_type", sa.String(length=16), nullable=False),
        sa.Column("requested_qty", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("limit_price", sa.Numeric(precision=18, scale=6), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("filled_qty", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("filled_avg_price", sa.Numeric(precision=18, scale=6), nullable=True),
        sa.Column("applied_qty", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("applied_notional", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["portfolio_id"], ["portfolios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("alpaca_order_id", name="uq_order_alpaca_id"),
        sa.UniqueConstraint("client_order_id", name="uq_order_client_id"),
    )
    op.create_index("ix_order_allocations_portfolio_id", "order_allocations", ["portfolio_id"])
    op.create_table(
        "portfolio_transactions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("portfolio_id", sa.Integer(), nullable=False),
        sa.Column("transaction_date", sa.Date(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("action", sa.String(length=80), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("price", sa.Numeric(precision=18, scale=6), nullable=True),
        sa.Column("fees", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("cash_delta", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("source", sa.String(length=24), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["portfolio_id"], ["portfolios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "portfolio_id", "fingerprint", name="uq_transaction_portfolio_fingerprint"
        ),
    )
    op.create_index("ix_portfolio_transactions_kind", "portfolio_transactions", ["kind"])
    op.create_index(
        "ix_portfolio_transactions_portfolio_id", "portfolio_transactions", ["portfolio_id"]
    )
    op.create_index(
        "ix_portfolio_transactions_transaction_date",
        "portfolio_transactions",
        ["transaction_date"],
    )
    op.create_table(
        "transaction_overrides",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("portfolio_id", sa.Integer(), nullable=False),
        sa.Column("transaction_id", sa.Integer(), nullable=False),
        sa.Column("operation", sa.String(length=12), nullable=False),
        sa.Column("original_source", sa.String(length=24), nullable=False),
        sa.Column("before_state", sa.Text(), nullable=False),
        sa.Column("after_state", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["portfolio_id"], ["portfolios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_transaction_overrides_portfolio_id", "transaction_overrides", ["portfolio_id"]
    )
    op.create_index(
        "ix_transaction_overrides_transaction_id", "transaction_overrides", ["transaction_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_transaction_overrides_transaction_id", table_name="transaction_overrides")
    op.drop_index("ix_transaction_overrides_portfolio_id", table_name="transaction_overrides")
    op.drop_table("transaction_overrides")
    op.drop_index("ix_portfolio_transactions_transaction_date", table_name="portfolio_transactions")
    op.drop_index("ix_portfolio_transactions_portfolio_id", table_name="portfolio_transactions")
    op.drop_index("ix_portfolio_transactions_kind", table_name="portfolio_transactions")
    op.drop_table("portfolio_transactions")
    op.drop_index("ix_order_allocations_portfolio_id", table_name="order_allocations")
    op.drop_table("order_allocations")
    op.drop_index("ix_ledger_entries_portfolio_id", table_name="ledger_entries")
    op.drop_table("ledger_entries")
    op.drop_index("ix_holding_lots_symbol", table_name="holding_lots")
    op.drop_index("ix_holding_lots_portfolio_id", table_name="holding_lots")
    op.drop_table("holding_lots")
    op.drop_table("portfolios")
    op.drop_table("data_migrations")
