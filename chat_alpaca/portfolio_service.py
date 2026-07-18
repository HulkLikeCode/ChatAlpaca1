from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from chat_alpaca.models import (
    DataMigration,
    HoldingLot,
    LedgerEntry,
    OrderAllocation,
    Portfolio,
    PortfolioTransaction,
    TransactionOverride,
)

MAX_PORTFOLIOS = 20
DEFAULT_STATEMENT_PATH = Path(__file__).resolve().parent.parent / "KC and Papa.csv"
PHASE_1_EFFECTIVE_DATE = date(2026, 5, 15)
PHASE_1_MIGRATION_KEY = "2026-07-17-phase-1-opening-positions-and-cash"
PHASE_1_DATE_CORRECTION_KEY = "2026-07-18-phase-1-effective-date-correction"

CASH_BALANCE_TARGETS = {
    "KCs Traditional IRA": Decimal("59780.15"),
    "KCs Roth IRA": Decimal("10331.93"),
    "KC and Papa": Decimal("77.77"),
}

SEED_PORTFOLIOS = (
    (
        "KCs Traditional IRA",
        (
            ("DCO", "39", date(2026, 5, 15), "145.17948"),
            ("VTV", "547", date(2026, 5, 15), "207.0364"),
        ),
    ),
    (
        "KCs Roth IRA",
        (("ONEQ", "918", date(2026, 5, 15), "104.06513"),),
    ),
    ("KC and Papa", ()),
    ("Portfolio 4", ()),
    ("Portfolio 5", ()),
)

ACTION_KINDS = {
    "buy": "buy",
    "sell": "sell",
    "cash dividend": "dividend",
    "qualified dividend": "dividend",
    "non-qualified div": "dividend",
    "pr yr cash div": "dividend",
    "credit interest": "interest",
    "moneylink transfer": "transfer",
    "promotional award": "award",
    "adr mgmt fee": "fee",
    "foreign tax paid": "tax",
}
TRADE_KINDS = {"buy", "sell"}
MANUAL_KINDS = (
    "buy",
    "sell",
    "dividend",
    "interest",
    "transfer",
    "award",
    "fee",
    "tax",
    "cash_adjustment",
)
POSITION_KINDS = {*TRADE_KINDS, "opening_position"}


@dataclass(frozen=True)
class TransactionDraft:
    transaction_date: date
    action: str
    kind: str
    symbol: str | None
    description: str
    quantity: Decimal | None
    price: Decimal | None
    fees: Decimal | None
    cash_delta: Decimal
    fingerprint: str | None = None


@dataclass(frozen=True)
class StatementParseResult:
    transactions: list[TransactionDraft]
    errors: list[str]


@dataclass(frozen=True)
class DividendTotals:
    year_to_date: Decimal
    trailing_365_days: Decimal
    custom_range: Decimal


def format_short_date(value: date) -> str:
    """Format a date with the compact transaction convention used by the UI."""
    return f"{value.month}/{value.day}/{value.strftime('%y')}"


def parse_short_date(value: str) -> date:
    """Parse an M/D/YY transaction date and return a useful validation error."""
    try:
        return datetime.strptime(value.strip(), "%m/%d/%y").date()
    except ValueError as exc:
        raise ValueError("Enter the date as M/D/YY, for example 7/17/26.") from exc


def money(value: object) -> Decimal:
    try:
        return Decimal(str(value)).quantize(Decimal("0.0001"))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Enter a valid dollar amount.") from exc


def _optional_money(value: object) -> Decimal | None:
    text = str(value or "").strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    normalized = text.strip("()").replace("$", "").replace(",", "")
    try:
        parsed = Decimal(normalized)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid dollar amount: {value!s}") from exc
    if negative:
        parsed = -parsed
    return parsed.quantize(Decimal("0.0001"))


def _optional_quantity(value: object) -> Decimal | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return Decimal(text.replace(",", "")).quantize(Decimal("0.00000001"))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid share quantity: {value!s}") from exc


def shares(value: object) -> Decimal:
    try:
        parsed = Decimal(str(value).replace(",", "")).quantize(Decimal("0.00000001"))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Enter a valid share quantity.") from exc
    if parsed <= 0:
        raise ValueError("Shares must be greater than zero.")
    return parsed


def position_shares(value: object) -> Decimal:
    try:
        parsed = Decimal(str(value).replace(",", "")).quantize(Decimal("0.00000001"))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Enter a valid share quantity.") from exc
    if parsed == 0:
        raise ValueError("Position shares cannot be zero.")
    return parsed


def normalize_symbol(value: object) -> str:
    symbol = str(value).strip().upper()
    if (
        not symbol
        or len(symbol) > 16
        or not all(character.isalnum() or character in {".", "-"} for character in symbol)
    ):
        raise ValueError(f"Invalid stock or ETF symbol: {value!s}")
    return symbol


def _parse_statement_date(value: str) -> date:
    primary_date = value.split(" as of ", maxsplit=1)[0].strip()
    try:
        return date.fromisoformat(primary_date)
    except ValueError:
        try:
            return date.strptime(primary_date, "%m/%d/%Y")
        except AttributeError:
            from datetime import datetime

            return datetime.strptime(primary_date, "%m/%d/%Y").date()
        except ValueError as exc:
            raise ValueError(f"Invalid transaction date: {value}") from exc


def _statement_fingerprint(row: dict[str, str], occurrence: int) -> str:
    fields = (
        "Date",
        "Action",
        "Symbol",
        "Description",
        "Quantity",
        "Price",
        "Fees & Comm",
        "Amount",
    )
    canonical = "|".join(row.get(field, "").strip() for field in fields)
    return hashlib.sha256(f"{canonical}|{occurrence}".encode()).hexdigest()


def parse_statement_csv(content: bytes | str) -> StatementParseResult:
    text = content.decode("utf-8-sig") if isinstance(content, bytes) else content
    reader = csv.DictReader(StringIO(text))
    required = {
        "Date",
        "Action",
        "Symbol",
        "Description",
        "Quantity",
        "Price",
        "Fees & Comm",
        "Amount",
    }
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        missing = ", ".join(sorted(required - set(reader.fieldnames or [])))
        return StatementParseResult([], [f"Missing required columns: {missing}"])

    transactions: list[TransactionDraft] = []
    errors: list[str] = []
    occurrences: dict[str, int] = {}
    for line_number, row in enumerate(reader, start=2):
        try:
            action = (row.get("Action") or "").strip()
            if not action:
                raise ValueError("Action is required.")
            kind = ACTION_KINDS.get(action.lower(), "cash_adjustment")
            raw_key = "|".join(str(row.get(key, "")).strip() for key in sorted(required))
            occurrences[raw_key] = occurrences.get(raw_key, 0) + 1
            symbol_value = (row.get("Symbol") or "").strip()
            symbol = normalize_symbol(symbol_value) if symbol_value else None
            quantity = _optional_quantity(row.get("Quantity"))
            price = _optional_money(row.get("Price"))
            fees = _optional_money(row.get("Fees & Comm"))
            cash_delta = _optional_money(row.get("Amount"))
            if cash_delta is None:
                raise ValueError("Amount is required.")
            draft = TransactionDraft(
                transaction_date=_parse_statement_date(row["Date"]),
                action=action,
                kind=kind,
                symbol=symbol,
                description=(row.get("Description") or "").strip(),
                quantity=quantity,
                price=price,
                fees=fees,
                cash_delta=cash_delta,
                fingerprint=_statement_fingerprint(row, occurrences[raw_key]),
            )
            _validate_transaction(draft)
            transactions.append(draft)
        except ValueError as exc:
            errors.append(f"Row {line_number}: {exc}")
    return StatementParseResult(transactions, errors)


def _validate_transaction(draft: TransactionDraft) -> None:
    if draft.kind not in {*MANUAL_KINDS, "opening_position"}:
        raise ValueError(f"Unsupported transaction category: {draft.kind}")
    if draft.kind in POSITION_KINDS:
        if draft.symbol is None:
            raise ValueError("Position transactions require a symbol.")
        if draft.quantity is None or draft.quantity <= 0:
            raise ValueError("Position transactions require a positive quantity.")
        if draft.price is None or draft.price <= 0:
            raise ValueError("Position transactions require a positive price.")
        if draft.kind == "buy" and draft.cash_delta >= 0:
            raise ValueError("Buy transactions must reduce cash.")
        if draft.kind == "sell" and draft.cash_delta <= 0:
            raise ValueError("Sell transactions must increase cash.")
        if draft.kind == "opening_position" and draft.cash_delta != 0:
            raise ValueError("Opening positions must be cash-neutral.")


def seed_database(session: Session) -> None:
    count = session.scalar(select(func.count()).select_from(Portfolio)) or 0
    if not count:
        for name, _initial_holdings in SEED_PORTFOLIOS:
            session.add(Portfolio(name=name, cash=Decimal("0")))
        session.flush()
        statement_portfolio = session.scalar(
            select(Portfolio).where(Portfolio.name == "KC and Papa")
        )
        if statement_portfolio is not None and DEFAULT_STATEMENT_PATH.exists():
            rebuild_portfolio_from_csv(
                session,
                statement_portfolio.id,
                DEFAULT_STATEMENT_PATH.read_bytes(),
                source="seed_csv",
            )
    _apply_phase_1_data_migration(session)
    _apply_phase_1_date_correction(session)


def _migration_fingerprint(label: str) -> str:
    return hashlib.sha256(f"{PHASE_1_MIGRATION_KEY}:{label}".encode()).hexdigest()


def _apply_phase_1_data_migration(session: Session) -> None:
    """Convert legacy seed state once, including already-populated databases."""
    if session.get(DataMigration, PHASE_1_MIGRATION_KEY) is not None:
        return

    portfolios = {
        portfolio.name: portfolio
        for portfolio in session.scalars(
            select(Portfolio).where(Portfolio.name.in_(CASH_BALANCE_TARGETS))
        )
    }
    seed_positions = {name: positions for name, positions in SEED_PORTFOLIOS}
    for portfolio_name in ("KCs Traditional IRA", "KCs Roth IRA"):
        portfolio = portfolios.get(portfolio_name)
        if portfolio is None:
            continue
        for symbol, quantity, acquired_on, cost_basis in seed_positions[portfolio_name]:
            transaction = PortfolioTransaction(
                portfolio_id=portfolio.id,
                transaction_date=acquired_on,
                kind="opening_position",
                action="Opening Position",
                symbol=symbol,
                description="Phase 1 conversion of legacy seeded holding",
                quantity=Decimal(quantity),
                price=Decimal(cost_basis),
                fees=None,
                cash_delta=Decimal("0"),
                source="phase_1_migration",
                fingerprint=_migration_fingerprint(
                    f"opening:{portfolio_name}:{symbol}:{quantity}:{acquired_on.isoformat()}"
                ),
            )
            session.add(transaction)

    session.flush()
    for portfolio_name, target in CASH_BALANCE_TARGETS.items():
        portfolio = portfolios.get(portfolio_name)
        if portfolio is None:
            continue
        transaction_cash = session.scalar(
            select(func.coalesce(func.sum(PortfolioTransaction.cash_delta), 0)).where(
                PortfolioTransaction.portfolio_id == portfolio.id
            )
        )
        delta = money(target - Decimal(transaction_cash or 0))
        session.add(
            PortfolioTransaction(
                portfolio_id=portfolio.id,
                transaction_date=PHASE_1_EFFECTIVE_DATE,
                kind="cash_adjustment",
                action="Cash Adjustment",
                symbol=None,
                description=f"Phase 1 cash balance set to ${target:,.2f}",
                quantity=None,
                price=None,
                fees=None,
                cash_delta=delta,
                source="phase_1_migration",
                fingerprint=_migration_fingerprint(f"cash:{portfolio_name}"),
            )
        )

    session.add(DataMigration(key=PHASE_1_MIGRATION_KEY))
    session.flush()
    for portfolio in portfolios.values():
        replay_portfolio(session, portfolio.id)


def _apply_phase_1_date_correction(session: Session) -> None:
    """Move already-applied Phase 1 cash adjustments to their confirmed effective date."""
    if session.get(DataMigration, PHASE_1_DATE_CORRECTION_KEY) is not None:
        return

    portfolio_ids = list(
        session.scalars(select(Portfolio.id).where(Portfolio.name.in_(CASH_BALANCE_TARGETS)))
    )
    if portfolio_ids:
        adjustments = list(
            session.scalars(
                select(PortfolioTransaction).where(
                    PortfolioTransaction.portfolio_id.in_(portfolio_ids),
                    PortfolioTransaction.kind == "cash_adjustment",
                    PortfolioTransaction.source == "phase_1_migration",
                )
            )
        )
        for adjustment in adjustments:
            adjustment.transaction_date = PHASE_1_EFFECTIVE_DATE

    session.add(DataMigration(key=PHASE_1_DATE_CORRECTION_KEY))
    session.flush()


def list_portfolios(session: Session) -> list[Portfolio]:
    statement = (
        select(Portfolio)
        .options(selectinload(Portfolio.holdings), selectinload(Portfolio.transactions))
        .order_by(Portfolio.id)
    )
    return list(session.scalars(statement).unique())


def create_portfolio(session: Session, name: str) -> Portfolio:
    cleaned = name.strip()
    if not cleaned or len(cleaned) > 80:
        raise ValueError("Portfolio name must contain 1 to 80 characters.")
    count = session.scalar(select(func.count()).select_from(Portfolio)) or 0
    if count >= MAX_PORTFOLIOS:
        raise ValueError(f"A maximum of {MAX_PORTFOLIOS} portfolios is allowed.")
    portfolio = Portfolio(name=cleaned, cash=Decimal("0"))
    session.add(portfolio)
    session.flush()
    return portfolio


def list_ledger(
    session: Session, portfolio_id: int | None = None, limit: int = 200
) -> list[LedgerEntry]:
    statement = select(LedgerEntry)
    if portfolio_id is not None:
        statement = statement.where(LedgerEntry.portfolio_id == portfolio_id)
    statement = statement.order_by(LedgerEntry.created_at.desc()).limit(limit)
    return list(session.scalars(statement))


def list_transactions(
    session: Session, portfolio_id: int, limit: int | None = None
) -> list[PortfolioTransaction]:
    statement = (
        select(PortfolioTransaction)
        .where(PortfolioTransaction.portfolio_id == portfolio_id)
        .order_by(PortfolioTransaction.transaction_date.desc(), PortfolioTransaction.id.desc())
    )
    if limit is not None:
        statement = statement.limit(limit)
    return list(session.scalars(statement))


def list_transactions_for_portfolios(
    session: Session, portfolio_ids: Iterable[int]
) -> list[PortfolioTransaction]:
    selected_ids = list(dict.fromkeys(portfolio_ids))
    if not selected_ids:
        return []
    statement = (
        select(PortfolioTransaction)
        .where(PortfolioTransaction.portfolio_id.in_(selected_ids))
        .order_by(PortfolioTransaction.transaction_date.desc(), PortfolioTransaction.id.desc())
    )
    return list(session.scalars(statement))


def dividend_totals(
    session: Session,
    portfolio_ids: Iterable[int],
    custom_start: date,
    custom_end: date,
    as_of: date | None = None,
) -> DividendTotals:
    """Return inclusive dividend cash totals for the standard presentation periods."""
    selected_ids = list(dict.fromkeys(portfolio_ids))
    if custom_start > custom_end:
        raise ValueError("The custom dividend start date must be on or before the end date.")
    if not selected_ids:
        zero = Decimal("0.0000")
        return DividendTotals(zero, zero, zero)

    effective_date = as_of or date.today()

    def total(start: date, end: date) -> Decimal:
        value = session.scalar(
            select(func.coalesce(func.sum(PortfolioTransaction.cash_delta), 0)).where(
                PortfolioTransaction.portfolio_id.in_(selected_ids),
                PortfolioTransaction.kind == "dividend",
                PortfolioTransaction.transaction_date.between(start, end),
            )
        )
        return money(value or 0)

    return DividendTotals(
        year_to_date=total(date(effective_date.year, 1, 1), effective_date),
        trailing_365_days=total(effective_date - timedelta(days=364), effective_date),
        custom_range=total(custom_start, custom_end),
    )


def get_portfolio(session: Session, portfolio_id: int) -> Portfolio:
    statement = (
        select(Portfolio)
        .where(Portfolio.id == portfolio_id)
        .options(selectinload(Portfolio.holdings))
    )
    portfolio = session.scalar(statement)
    if portfolio is None:
        raise ValueError("Portfolio not found.")
    return portfolio


def rename_portfolio(session: Session, portfolio_id: int, name: str) -> None:
    cleaned = name.strip()
    if not cleaned or len(cleaned) > 80:
        raise ValueError("Portfolio name must contain 1 to 80 characters.")
    portfolio = get_portfolio(session, portfolio_id)
    portfolio.name = cleaned


def delete_portfolio(session: Session, portfolio_id: int) -> None:
    """Permanently remove a portfolio and every record assigned to it."""
    portfolio = get_portfolio(session, portfolio_id)
    for model in (LedgerEntry, OrderAllocation, TransactionOverride):
        session.execute(delete(model).where(model.portfolio_id == portfolio.id))
    session.delete(portfolio)


def _reduce_fifo_or_short(
    portfolio: Portfolio,
    symbol: str,
    quantity: Decimal,
    sale_price: Decimal,
    transaction_date: date,
) -> None:
    lots = [lot for lot in portfolio.holdings if lot.symbol == symbol and Decimal(lot.shares) > 0]
    remaining = quantity
    for lot in lots:
        if remaining <= 0:
            break
        consumed = min(Decimal(lot.shares), remaining)
        lot.shares = Decimal(lot.shares) - consumed
        remaining -= consumed
        if lot.shares == 0:
            portfolio.holdings.remove(lot)
    if remaining > 0:
        portfolio.holdings.append(
            HoldingLot(
                symbol=symbol,
                shares=-remaining,
                acquired_on=transaction_date,
                cost_basis=sale_price,
            )
        )


def _draft_from_transaction(transaction: PortfolioTransaction) -> TransactionDraft:
    return TransactionDraft(
        transaction_date=transaction.transaction_date,
        action=transaction.action,
        kind=transaction.kind,
        symbol=transaction.symbol,
        description=transaction.description,
        quantity=(Decimal(transaction.quantity) if transaction.quantity is not None else None),
        price=Decimal(transaction.price) if transaction.price is not None else None,
        fees=Decimal(transaction.fees) if transaction.fees is not None else None,
        cash_delta=Decimal(transaction.cash_delta),
        fingerprint=transaction.fingerprint,
    )


def replay_portfolio(session: Session, portfolio_id: int) -> None:
    """Rebuild cash, FIFO lots, and ledger entries from canonical transactions."""
    portfolio = get_portfolio(session, portfolio_id)
    portfolio.holdings.clear()
    session.execute(delete(LedgerEntry).where(LedgerEntry.portfolio_id == portfolio.id))
    portfolio.cash = Decimal("0")
    session.flush()

    transactions = list(
        session.scalars(
            select(PortfolioTransaction)
            .where(PortfolioTransaction.portfolio_id == portfolio.id)
            .order_by(PortfolioTransaction.transaction_date, PortfolioTransaction.id)
        )
    )
    for transaction in transactions:
        draft = _draft_from_transaction(transaction)
        _validate_transaction(draft)
        portfolio.cash = Decimal(portfolio.cash) + draft.cash_delta
        if draft.kind in {"buy", "opening_position"}:
            assert draft.symbol is not None and draft.quantity is not None
            assert draft.price is not None
            cost_basis = (
                draft.price
                if draft.kind == "opening_position"
                else abs(draft.cash_delta / draft.quantity)
            )
            portfolio.holdings.append(
                HoldingLot(
                    symbol=draft.symbol,
                    shares=draft.quantity,
                    acquired_on=draft.transaction_date,
                    cost_basis=cost_basis,
                )
            )
        elif draft.kind == "sell":
            assert draft.symbol is not None and draft.quantity is not None
            assert draft.price is not None
            _reduce_fifo_or_short(
                portfolio,
                draft.symbol,
                draft.quantity,
                draft.price,
                draft.transaction_date,
            )
        session.add(
            LedgerEntry(
                portfolio_id=portfolio.id,
                kind=draft.kind,
                symbol=draft.symbol,
                quantity=draft.quantity,
                price=draft.price,
                cash_delta=draft.cash_delta,
                note=draft.description[:240] or draft.action[:240],
                created_at=transaction.created_at,
            )
        )
    session.flush()


def record_transaction(
    session: Session,
    portfolio_id: int,
    draft: TransactionDraft,
    source: str = "manual",
) -> bool:
    """Persist a transaction and replay its portfolio; return False for a duplicate."""
    added = _insert_transaction(session, portfolio_id, draft, source)
    if added:
        replay_portfolio(session, portfolio_id)
    return added


def _insert_transaction(
    session: Session,
    portfolio_id: int,
    draft: TransactionDraft,
    source: str,
) -> bool:
    """Persist a canonical event without rebuilding its derived state."""
    _validate_transaction(draft)
    portfolio = get_portfolio(session, portfolio_id)
    if draft.fingerprint:
        existing = session.scalar(
            select(PortfolioTransaction.id).where(
                PortfolioTransaction.portfolio_id == portfolio.id,
                PortfolioTransaction.fingerprint == draft.fingerprint,
            )
        )
        if existing is not None:
            return False

    transaction = PortfolioTransaction(
        portfolio_id=portfolio.id,
        transaction_date=draft.transaction_date,
        kind=draft.kind,
        action=draft.action[:80],
        symbol=draft.symbol,
        description=draft.description,
        quantity=draft.quantity,
        price=draft.price,
        fees=draft.fees,
        cash_delta=draft.cash_delta,
        source=source[:24],
        fingerprint=draft.fingerprint,
    )
    session.add(transaction)
    session.flush()
    return True


def _confirmed_transaction(
    session: Session,
    portfolio_id: int,
    transaction_id: int,
    operation: str,
    confirmation: str | None,
) -> PortfolioTransaction:
    expected = f"{operation} {transaction_id}"
    if confirmation != expected:
        raise ValueError(f'Type "{expected}" to confirm this transaction {operation.lower()}.')
    transaction = session.scalar(
        select(PortfolioTransaction).where(
            PortfolioTransaction.id == transaction_id,
            PortfolioTransaction.portfolio_id == portfolio_id,
        )
    )
    if transaction is None:
        raise ValueError("Transaction not found in this portfolio.")
    return transaction


def _transaction_snapshot(transaction: PortfolioTransaction) -> str:
    """Serialize the financial fields needed to audit a manual override."""
    values = {
        "id": transaction.id,
        "portfolio_id": transaction.portfolio_id,
        "transaction_date": transaction.transaction_date.isoformat(),
        "kind": transaction.kind,
        "action": transaction.action,
        "symbol": transaction.symbol,
        "description": transaction.description,
        "quantity": str(transaction.quantity) if transaction.quantity is not None else None,
        "price": str(transaction.price) if transaction.price is not None else None,
        "fees": str(transaction.fees) if transaction.fees is not None else None,
        "cash_delta": str(transaction.cash_delta),
        "source": transaction.source,
        "fingerprint": transaction.fingerprint,
        "created_at": transaction.created_at.isoformat(),
    }
    return json.dumps(values, sort_keys=True)


def update_transaction(
    session: Session,
    portfolio_id: int,
    transaction_id: int,
    draft: TransactionDraft,
    confirmation: str | None = None,
) -> PortfolioTransaction:
    """Update one canonical event after an explicit, transaction-specific confirmation."""
    _validate_transaction(draft)
    transaction = _confirmed_transaction(
        session, portfolio_id, transaction_id, "UPDATE", confirmation
    )
    original_source = transaction.source
    before_state = _transaction_snapshot(transaction)
    transaction.transaction_date = draft.transaction_date
    transaction.kind = draft.kind
    transaction.action = draft.action[:80]
    transaction.symbol = draft.symbol
    transaction.description = draft.description
    transaction.quantity = draft.quantity
    transaction.price = draft.price
    transaction.fees = draft.fees
    transaction.cash_delta = draft.cash_delta
    transaction.source = "manual_override"
    session.add(
        TransactionOverride(
            portfolio_id=portfolio_id,
            transaction_id=transaction_id,
            operation="update",
            original_source=original_source,
            before_state=before_state,
            after_state=_transaction_snapshot(transaction),
        )
    )
    session.flush()
    replay_portfolio(session, portfolio_id)
    return transaction


def delete_transaction(
    session: Session,
    portfolio_id: int,
    transaction_id: int,
    confirmation: str | None = None,
) -> None:
    """Delete one canonical event after an explicit, transaction-specific confirmation."""
    transaction = _confirmed_transaction(
        session, portfolio_id, transaction_id, "DELETE", confirmation
    )
    session.add(
        TransactionOverride(
            portfolio_id=portfolio_id,
            transaction_id=transaction_id,
            operation="delete",
            original_source=transaction.source,
            before_state=_transaction_snapshot(transaction),
            after_state=None,
        )
    )
    session.delete(transaction)
    session.flush()
    replay_portfolio(session, portfolio_id)


def import_statement(
    session: Session, portfolio_id: int, parsed: StatementParseResult, source: str = "csv"
) -> tuple[int, int]:
    if parsed.errors:
        raise ValueError("Fix every invalid row before importing this statement.")
    added = 0
    duplicates = 0
    for draft in sorted(parsed.transactions, key=lambda item: item.transaction_date):
        if _insert_transaction(session, portfolio_id, draft, source=source):
            added += 1
        else:
            duplicates += 1
    if added:
        replay_portfolio(session, portfolio_id)
    return added, duplicates


def rebuild_portfolio_from_csv(
    session: Session, portfolio_id: int, content: bytes | str, source: str = "csv_rebuild"
) -> int:
    """Replace a portfolio's cash, lots, ledger, and transactions from one statement."""
    parsed = parse_statement_csv(content)
    if parsed.errors:
        raise ValueError("Statement cannot rebuild the portfolio: " + " ".join(parsed.errors))
    portfolio = get_portfolio(session, portfolio_id)
    session.execute(delete(OrderAllocation).where(OrderAllocation.portfolio_id == portfolio.id))
    session.execute(delete(LedgerEntry).where(LedgerEntry.portfolio_id == portfolio.id))
    session.execute(
        delete(PortfolioTransaction).where(PortfolioTransaction.portfolio_id == portfolio.id)
    )
    portfolio.holdings.clear()
    portfolio.cash = Decimal("0")
    for draft in sorted(parsed.transactions, key=lambda item: item.transaction_date):
        _insert_transaction(session, portfolio.id, draft, source=source)
    replay_portfolio(session, portfolio.id)
    return len(parsed.transactions)


def set_cash(
    session: Session, portfolio_id: int, new_cash: object, note: str = "Manual cash edit"
) -> None:
    portfolio = get_portfolio(session, portfolio_id)
    parsed = money(new_cash)
    delta = parsed - Decimal(portfolio.cash)
    if not delta:
        return
    record_transaction(
        session,
        portfolio_id,
        TransactionDraft(
            date.today(), "Cash adjustment", "cash_adjustment", None, note, None, None, None, delta
        ),
    )


def replace_holdings(
    session: Session, portfolio_id: int, rows: Iterable[dict[str, object]]
) -> None:
    """Legacy opening-balance helper. Transaction entry is the normal owner workflow."""
    portfolio = get_portfolio(session, portfolio_id)
    cleaned: list[tuple[str, Decimal, date, Decimal]] = []
    for row in rows:
        if not str(row.get("Symbol", "")).strip():
            continue
        symbol = normalize_symbol(row["Symbol"])
        quantity = position_shares(row["Shares"])
        acquired = row["Acquired"]
        if not isinstance(acquired, date):
            acquired = date.fromisoformat(str(acquired))
        cost_basis = money(row["Cost / share"])
        if cost_basis < 0:
            raise ValueError("Cost per share cannot be negative.")
        cleaned.append((symbol, quantity, acquired, cost_basis))
    portfolio.holdings.clear()
    for symbol, quantity, acquired, cost_basis in cleaned:
        portfolio.holdings.append(
            HoldingLot(
                symbol=symbol,
                shares=quantity,
                acquired_on=acquired,
                cost_basis=cost_basis,
            )
        )


def portfolio_cost(portfolio: Portfolio) -> Decimal:
    return sum(
        (Decimal(lot.shares) * Decimal(lot.cost_basis) for lot in portfolio.holdings),
        Decimal(portfolio.cash),
    )
