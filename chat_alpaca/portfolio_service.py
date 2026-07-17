from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from chat_alpaca.models import (
    HoldingLot,
    LedgerEntry,
    OrderAllocation,
    Portfolio,
    PortfolioTransaction,
)

MAX_PORTFOLIOS = 20
DEFAULT_STATEMENT_PATH = Path(__file__).resolve().parent.parent / "KC and Papa.csv"

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
    if draft.kind not in {*MANUAL_KINDS, "cash_adjustment"}:
        raise ValueError(f"Unsupported transaction category: {draft.kind}")
    if draft.kind in TRADE_KINDS:
        if draft.symbol is None:
            raise ValueError("Trades require a symbol.")
        if draft.quantity is None or draft.quantity <= 0:
            raise ValueError("Trades require a positive quantity.")
        if draft.price is None or draft.price <= 0:
            raise ValueError("Trades require a positive price.")
        if draft.kind == "buy" and draft.cash_delta >= 0:
            raise ValueError("Buy transactions must reduce cash.")
        if draft.kind == "sell" and draft.cash_delta <= 0:
            raise ValueError("Sell transactions must increase cash.")


def seed_database(session: Session) -> None:
    count = session.scalar(select(func.count()).select_from(Portfolio)) or 0
    if count:
        return
    for name, initial_holdings in SEED_PORTFOLIOS:
        portfolio = Portfolio(name=name, cash=Decimal("0"))
        session.add(portfolio)
        session.flush()
        for symbol, quantity, acquired_on, cost_basis in initial_holdings:
            session.add(
                HoldingLot(
                    portfolio_id=portfolio.id,
                    symbol=symbol,
                    shares=Decimal(quantity),
                    acquired_on=acquired_on,
                    cost_basis=Decimal(cost_basis),
                )
            )
    statement_portfolio = session.scalar(select(Portfolio).where(Portfolio.name == "KC and Papa"))
    if statement_portfolio is not None and DEFAULT_STATEMENT_PATH.exists():
        rebuild_portfolio_from_csv(
            session, statement_portfolio.id, DEFAULT_STATEMENT_PATH.read_bytes(), source="seed_csv"
        )


def list_portfolios(session: Session) -> list[Portfolio]:
    statement = select(Portfolio).options(selectinload(Portfolio.holdings)).order_by(Portfolio.id)
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
    session: Session, portfolio_id: int, limit: int = 300
) -> list[PortfolioTransaction]:
    statement = (
        select(PortfolioTransaction)
        .where(PortfolioTransaction.portfolio_id == portfolio_id)
        .order_by(PortfolioTransaction.transaction_date.desc(), PortfolioTransaction.id.desc())
        .limit(limit)
    )
    return list(session.scalars(statement))


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
    for model in (LedgerEntry, OrderAllocation):
        session.execute(delete(model).where(model.portfolio_id == portfolio.id))
    session.delete(portfolio)


def _reduce_fifo_or_short(
    session: Session,
    portfolio_id: int,
    symbol: str,
    quantity: Decimal,
    sale_price: Decimal,
    transaction_date: date,
) -> None:
    lots = list(
        session.scalars(
            select(HoldingLot)
            .where(
                HoldingLot.portfolio_id == portfolio_id,
                HoldingLot.symbol == symbol,
                HoldingLot.shares > 0,
            )
            .order_by(HoldingLot.acquired_on, HoldingLot.id)
        )
    )
    remaining = quantity
    for lot in lots:
        if remaining <= 0:
            break
        consumed = min(Decimal(lot.shares), remaining)
        lot.shares = Decimal(lot.shares) - consumed
        remaining -= consumed
        if lot.shares == 0:
            lot.portfolio.holdings.remove(lot)
    if remaining > 0:
        portfolio = get_portfolio(session, portfolio_id)
        portfolio.holdings.append(
            HoldingLot(
                symbol=symbol,
                shares=-remaining,
                acquired_on=transaction_date,
                cost_basis=sale_price,
            )
        )


def record_transaction(
    session: Session,
    portfolio_id: int,
    draft: TransactionDraft,
    source: str = "manual",
) -> bool:
    """Persist and apply an immutable transaction. Returns False for an import duplicate."""
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
    portfolio.cash = Decimal(portfolio.cash) + draft.cash_delta
    if draft.kind == "buy":
        assert draft.symbol is not None and draft.quantity is not None
        portfolio.holdings.append(
            HoldingLot(
                symbol=draft.symbol,
                shares=draft.quantity,
                acquired_on=draft.transaction_date,
                cost_basis=abs(draft.cash_delta / draft.quantity),
            )
        )
    elif draft.kind == "sell":
        assert draft.symbol is not None and draft.quantity is not None and draft.price is not None
        _reduce_fifo_or_short(
            session,
            portfolio.id,
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
        )
    )
    return True


def import_statement(
    session: Session, portfolio_id: int, parsed: StatementParseResult, source: str = "csv"
) -> tuple[int, int]:
    if parsed.errors:
        raise ValueError("Fix every invalid row before importing this statement.")
    added = 0
    duplicates = 0
    for draft in sorted(parsed.transactions, key=lambda item: item.transaction_date):
        if record_transaction(session, portfolio_id, draft, source=source):
            added += 1
        else:
            duplicates += 1
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
        record_transaction(session, portfolio.id, draft, source=source)
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
