from __future__ import annotations

from dataclasses import dataclass

from chat_alpaca.portfolio_service import (
    TransactionDraft,
    money,
    normalize_symbol,
    parse_short_date,
    shares,
)

POSITION_KINDS = {"buy", "sell", "opening_position"}


@dataclass(frozen=True)
class TransactionCommand:
    """UI-independent input for constructing a canonical transaction draft."""

    transaction_date_text: str
    kind: str
    symbol: str = ""
    description: str = ""
    quantity: float = 0.0
    price: float = 0.0
    fees: float = 0.0
    cash_delta: float = 0.0
    action: str | None = None


def transaction_kind_label(value: str) -> str:
    return value.replace("_", " ").title()


def validate_transaction_symbol(symbol: str) -> str:
    """Normalize a required trade symbol and raise a user-safe validation error."""
    return normalize_symbol(symbol)


def calculated_trade_cash(kind: str, quantity: float, price: float, fees: float) -> float:
    """Return the cash effect used by buy and sell command previews."""
    if kind not in {"buy", "sell"}:
        raise ValueError("Trade cash can only be calculated for a buy or sell.")
    normalized_quantity = max(quantity, 0.0)
    normalized_price = max(price, 0.0)
    normalized_fees = max(fees, 0.0)
    notional = normalized_quantity * normalized_price
    return -(notional + normalized_fees) if kind == "buy" else notional - normalized_fees


def build_transaction_draft(command: TransactionCommand) -> TransactionDraft:
    """Validate UI input and construct the ledger service command."""
    position_kind = command.kind in POSITION_KINDS
    parsed_quantity = shares(command.quantity) if position_kind else None
    parsed_price = money(command.price) if position_kind else None
    parsed_fees = money(command.fees) if command.fees else None
    if command.kind == "buy":
        assert parsed_quantity is not None and parsed_price is not None
        parsed_cash_delta = -(parsed_quantity * parsed_price + (parsed_fees or 0))
    elif command.kind == "sell":
        assert parsed_quantity is not None and parsed_price is not None
        parsed_cash_delta = parsed_quantity * parsed_price - (parsed_fees or 0)
    elif command.kind == "opening_position":
        parsed_cash_delta = money(0)
    else:
        parsed_cash_delta = money(command.cash_delta)

    normalized_symbol = normalize_symbol(command.symbol) if command.symbol.strip() else None
    if command.kind in {"buy", "sell", "opening_position"} and normalized_symbol is None:
        raise ValueError(f"A symbol is required for {transaction_kind_label(command.kind)}.")

    return TransactionDraft(
        transaction_date=parse_short_date(command.transaction_date_text),
        action=(command.action or transaction_kind_label(command.kind)).strip(),
        kind=command.kind,
        symbol=normalized_symbol,
        description=command.description.strip(),
        quantity=parsed_quantity,
        price=parsed_price,
        fees=parsed_fees,
        cash_delta=parsed_cash_delta,
    )
