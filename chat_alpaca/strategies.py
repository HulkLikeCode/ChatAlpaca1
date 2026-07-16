from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class AssetKind(str, Enum):
    STOCK = "stock"
    ETF = "etf"
    OPTION = "option"


class PositionIntent(str, Enum):
    OPEN_LONG = "open_long"
    CLOSE_LONG = "close_long"
    OPEN_SHORT = "open_short"
    CLOSE_SHORT = "close_short"


@dataclass(frozen=True)
class StrategySignal:
    portfolio_id: int
    symbol: str
    intent: PositionIntent
    quantity: float
    asset_kind: AssetKind = AssetKind.STOCK


class Strategy(Protocol):
    """Extension point for scheduled strategies in a later release."""

    name: str

    def evaluate(self) -> list[StrategySignal]: ...
