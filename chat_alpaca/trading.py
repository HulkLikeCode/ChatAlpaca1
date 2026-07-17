from __future__ import annotations

from datetime import date
from decimal import Decimal
from functools import lru_cache
from uuid import uuid4

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_alpaca.config import get_settings
from chat_alpaca.models import OrderAllocation
from chat_alpaca.portfolio_service import (
    TransactionDraft,
    get_portfolio,
    money,
    normalize_symbol,
    record_transaction,
    shares,
)


class TradingUnavailable(RuntimeError):
    pass


def _value(item: object) -> str:
    return str(getattr(item, "value", item))


@lru_cache(maxsize=1)
def get_trading_client() -> TradingClient:
    settings = get_settings()
    if not settings.alpaca_configured:
        raise TradingUnavailable("Alpaca trading credentials are not configured.")
    if not settings.paper and not settings.allow_live_trading:
        raise TradingUnavailable(
            "Live execution is locked. Set ALLOW_LIVE_TRADING=true only after a separate safety review."
        )
    return TradingClient(
        settings.alpaca_api_key,
        settings.alpaca_secret_key,
        paper=settings.paper,
    )


def submit_allocated_order(
    session: Session,
    portfolio_id: int,
    symbol: str,
    side: str,
    quantity: object,
    order_type: str,
    limit_price: object | None = None,
) -> OrderAllocation:
    get_portfolio(session, portfolio_id)
    normalized_symbol = normalize_symbol(symbol)
    parsed_qty = shares(quantity)
    normalized_side = side.lower()
    if normalized_side not in {"buy", "sell"}:
        raise ValueError("Order side must be buy or sell.")
    normalized_type = order_type.lower()
    if normalized_type not in {"market", "limit"}:
        raise ValueError("Order type must be market or limit.")
    client_order_id = f"chatp-{portfolio_id}-{uuid4().hex[:24]}"
    request_args = {
        "symbol": normalized_symbol,
        "qty": float(parsed_qty),
        "side": OrderSide.BUY if normalized_side == "buy" else OrderSide.SELL,
        "time_in_force": TimeInForce.DAY,
        "client_order_id": client_order_id,
    }
    parsed_limit: Decimal | None = None
    if normalized_type == "limit":
        parsed_limit = money(limit_price)
        if parsed_limit <= 0:
            raise ValueError("Limit price must be greater than zero.")
        request = LimitOrderRequest(**request_args, limit_price=float(parsed_limit))
    else:
        request = MarketOrderRequest(**request_args)
    order = get_trading_client().submit_order(request)
    allocation = OrderAllocation(
        portfolio_id=portfolio_id,
        alpaca_order_id=str(order.id),
        client_order_id=client_order_id,
        symbol=normalized_symbol,
        side=normalized_side,
        order_type=normalized_type,
        requested_qty=parsed_qty,
        limit_price=parsed_limit,
        status=_value(order.status),
        filled_qty=Decimal(str(order.filled_qty or 0)),
        filled_avg_price=(Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None),
    )
    session.add(allocation)
    session.flush()
    return allocation


def list_allocations(session: Session, limit: int = 100) -> list[OrderAllocation]:
    statement = select(OrderAllocation).order_by(OrderAllocation.submitted_at.desc()).limit(limit)
    return list(session.scalars(statement))


def sync_allocations(session: Session) -> int:
    allocations = list_allocations(session)
    changed = 0
    for allocation in allocations:
        order = get_trading_client().get_order_by_id(allocation.alpaca_order_id)
        status = _value(order.status)
        filled_qty = Decimal(str(order.filled_qty or 0))
        average = Decimal(str(order.filled_avg_price or 0))
        total_notional = (filled_qty * average).quantize(Decimal("0.0001"))
        incremental_qty = filled_qty - Decimal(allocation.applied_qty)
        incremental_notional = total_notional - Decimal(allocation.applied_notional)
        if incremental_qty > 0:
            _apply_fill(session, allocation, incremental_qty, incremental_notional)
            allocation.applied_qty = filled_qty
            allocation.applied_notional = total_notional
            changed += 1
        allocation.status = status
        allocation.filled_qty = filled_qty
        allocation.filled_avg_price = average if filled_qty else None
    return changed


def _apply_fill(
    session: Session,
    allocation: OrderAllocation,
    incremental_qty: Decimal,
    incremental_notional: Decimal,
) -> None:
    get_portfolio(session, allocation.portfolio_id)
    effective_price = incremental_notional / incremental_qty
    kind = allocation.side
    cash_delta = -incremental_notional if kind == "buy" else incremental_notional
    record_transaction(
        session,
        allocation.portfolio_id,
        TransactionDraft(
            transaction_date=date.today(),
            action=kind.title(),
            kind=kind,
            symbol=allocation.symbol,
            description=f"Alpaca order {allocation.alpaca_order_id}",
            quantity=incremental_qty,
            price=effective_price,
            fees=None,
            cash_delta=cash_delta,
        ),
        source="alpaca",
    )


def cancel_order(alpaca_order_id: str) -> None:
    get_trading_client().cancel_order_by_id(alpaca_order_id)
