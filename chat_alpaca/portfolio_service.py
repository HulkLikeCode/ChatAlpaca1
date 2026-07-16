from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from decimal import Decimal, InvalidOperation

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from chat_alpaca.models import HoldingLot, LedgerEntry, Portfolio

MAX_PORTFOLIOS = 5
MAX_SYMBOLS_PER_PORTFOLIO = 25

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
    ("Portfolio 3", ()),
    ("Portfolio 4", ()),
    ("Portfolio 5", ()),
)


def money(value: object) -> Decimal:
    try:
        return Decimal(str(value)).quantize(Decimal("0.0001"))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Enter a valid dollar amount.") from exc


def shares(value: object) -> Decimal:
    try:
        parsed = Decimal(str(value)).quantize(Decimal("0.00000001"))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Enter a valid share quantity.") from exc
    if parsed <= 0:
        raise ValueError("Shares must be greater than zero.")
    return parsed


def position_shares(value: object) -> Decimal:
    try:
        parsed = Decimal(str(value)).quantize(Decimal("0.00000001"))
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


def list_portfolios(session: Session) -> list[Portfolio]:
    statement = select(Portfolio).options(selectinload(Portfolio.holdings)).order_by(Portfolio.id)
    return list(session.scalars(statement).unique())


def list_ledger(session: Session, limit: int = 200) -> list[LedgerEntry]:
    statement = select(LedgerEntry).order_by(LedgerEntry.created_at.desc()).limit(limit)
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


def set_cash(
    session: Session, portfolio_id: int, new_cash: object, note: str = "Manual cash edit"
) -> None:
    portfolio = get_portfolio(session, portfolio_id)
    parsed = money(new_cash)
    delta = parsed - Decimal(portfolio.cash)
    if not delta:
        return
    portfolio.cash = parsed
    session.add(
        LedgerEntry(
            portfolio_id=portfolio.id,
            kind="cash_adjustment",
            cash_delta=delta,
            note=note[:240],
        )
    )


def replace_holdings(
    session: Session, portfolio_id: int, rows: Iterable[dict[str, object]]
) -> None:
    portfolio = get_portfolio(session, portfolio_id)
    cleaned: list[tuple[str, Decimal, date, Decimal]] = []
    symbols: set[str] = set()
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
        symbols.add(symbol)
    if len(symbols) > MAX_SYMBOLS_PER_PORTFOLIO:
        raise ValueError(f"A portfolio can contain at most {MAX_SYMBOLS_PER_PORTFOLIO} symbols.")
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
