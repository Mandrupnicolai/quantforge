"""Core domain models for QuantForge.

This module defines the foundational value objects and entities that flow
through every layer of the library.  All models are immutable by default
and validated via Pydantic v2 so that invalid state is impossible to construct.

Design principles:
    * Prefer value objects (frozen dataclasses / Pydantic models) over plain dicts.
    * Use ``Decimal`` for monetary values to avoid floating-point drift.
    * Keep models dependency-free — no I/O, no side effects.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AssetClass(str, Enum):
    """Supported asset classes."""

    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    FX = "fx"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    DERIVATIVE = "derivative"


class OrderSide(str, Enum):
    """Direction of an order."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Execution instruction for an order."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Lifecycle state of an order."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SignalDirection(Enum):
    """Trading direction emitted by a strategy."""

    LONG = auto()
    SHORT = auto()
    FLAT = auto()  # Exit / close position


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class Money(BaseModel):
    """An immutable monetary amount with an explicit currency.

    Args:
        amount: The numeric value as a Decimal to prevent floating-point errors.
        currency: ISO-4217 currency code (e.g. ``"USD"``, ``"EUR"``).

    Example:
        >>> price = Money(amount=Decimal("150.25"), currency="USD")
        >>> price.amount
        Decimal('150.25')
    """

    model_config = {"frozen": True}

    amount: Decimal
    currency: Annotated[str, Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")]

    def __add__(self, other: Money) -> Money:
        """Add two monetary amounts; raises if currencies differ."""
        if self.currency != other.currency:
            msg = f"Cannot add {self.currency} and {other.currency}"
            raise ValueError(msg)
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def __mul__(self, scalar: Decimal | float | int) -> Money:
        """Scale a monetary amount by a dimensionless factor."""
        return Money(amount=self.amount * Decimal(str(scalar)), currency=self.currency)

    def __repr__(self) -> str:
        return f"{self.currency} {self.amount:,.4f}"


class OHLCV(BaseModel):
    """A single OHLCV (Open-High-Low-Close-Volume) price bar.

    All price fields use ``Decimal`` for precision.  Volume is stored as an
    integer because fractional share/contract counts are uncommon in practice.

    Args:
        timestamp: The opening timestamp of the bar (UTC).
        open: Opening price.
        high: Highest price during the interval.
        low:  Lowest price during the interval.
        close: Closing price.
        volume: Number of units traded.
        vwap: Optional volume-weighted average price.
    """

    model_config = {"frozen": True}

    timestamp: datetime
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: int = Field(ge=0)
    vwap: Decimal | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_ohlc_consistency(self) -> OHLCV:
        """Ensure high ≥ open, close ≥ low and high ≥ low."""
        if self.high < self.low:
            msg = f"high ({self.high}) must be >= low ({self.low})"
            raise ValueError(msg)
        if self.high < self.open or self.high < self.close:
            msg = "high must be >= both open and close"
            raise ValueError(msg)
        if self.low > self.open or self.low > self.close:
            msg = "low must be <= both open and close"
            raise ValueError(msg)
        return self

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_utc(cls, v: datetime | str) -> datetime:
        """Coerce naive datetimes to UTC."""
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


class Instrument(BaseModel):
    """A tradeable financial instrument.

    An instrument is the canonical identifier for an asset.  It carries the
    minimal metadata needed for order routing, cost calculation, and display.

    Args:
        symbol:       Primary ticker symbol (e.g. ``"AAPL"``).
        isin:         Optional ISIN for cross-exchange deduplication.
        exchange:     Exchange MIC code (e.g. ``"XNAS"``).
        asset_class:  The broad asset class of the instrument.
        currency:     Settlement currency ISO-4217 code.
        description:  Human-readable name (e.g. ``"Apple Inc."``).
        lot_size:     Minimum tradeable quantity (default 1 for equities).
        tick_size:    Minimum price increment.
    """

    model_config = {"frozen": True}

    symbol: str
    isin: str | None = None
    exchange: str = "XNAS"
    asset_class: AssetClass = AssetClass.EQUITY
    currency: Annotated[str, Field(min_length=3, max_length=3)] = "USD"
    description: str | None = None
    lot_size: Decimal = Decimal("1")
    tick_size: Decimal = Decimal("0.01")

    def __str__(self) -> str:
        return f"{self.symbol}:{self.exchange}"


class Signal(BaseModel):
    """A trading signal produced by a strategy.

    Signals are ephemeral — they do not carry position-sizing logic.  The
    backtester and live execution layer translate signals into orders using
    a configurable position-sizing model.

    Args:
        direction:  The directional intent (LONG, SHORT, or FLAT).
        strength:   Optional conviction score in [0, 1].  Used by
                    position-sizing models that support partial allocations.
        metadata:   Arbitrary key-value context (e.g. factor values).
    """

    model_config = {"frozen": True}

    direction: SignalDirection
    strength: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    metadata: dict[str, float | str | bool] = Field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        """Return ``True`` if the signal opens a new position."""
        return self.direction in {SignalDirection.LONG, SignalDirection.SHORT}

    @property
    def is_exit(self) -> bool:
        """Return ``True`` if the signal closes an existing position."""
        return self.direction == SignalDirection.FLAT


class Order(BaseModel):
    """An order submitted to a market or simulated exchange.

    Orders are created by the execution layer from a ``Signal``.  They track
    the full lifecycle from creation through fill or cancellation.

    Args:
        order_id:    Unique order identifier (UUID string).
        instrument:  The instrument to trade.
        side:        BUY or SELL.
        order_type:  MARKET, LIMIT, STOP, or STOP_LIMIT.
        quantity:    Number of units to trade (positive).
        limit_price: Required for LIMIT and STOP_LIMIT orders.
        stop_price:  Required for STOP and STOP_LIMIT orders.
        created_at:  Timestamp when the order was created (UTC).
        status:      Current lifecycle status.
    """

    model_config = {"frozen": False}  # Mutable — status changes during lifetime

    order_id: str
    instrument: Instrument
    side: OrderSide
    order_type: OrderType
    quantity: Decimal = Field(gt=0)
    limit_price: Decimal | None = Field(default=None, gt=0)
    stop_price: Decimal | None = Field(default=None, gt=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Decimal | None = None

    @model_validator(mode="after")
    def _validate_prices_for_order_type(self) -> Order:
        """Ensure the right price fields are populated for each order type."""
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and not self.limit_price:
            msg = f"{self.order_type} orders require a limit_price"
            raise ValueError(msg)
        if self.order_type in {OrderType.STOP, OrderType.STOP_LIMIT} and not self.stop_price:
            msg = f"{self.order_type} orders require a stop_price"
            raise ValueError(msg)
        return self

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` if no further state transitions are possible."""
        return self.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}


class Trade(BaseModel):
    """A completed round-trip trade (entry + exit).

    A ``Trade`` is the unit of performance attribution.  It records the full
    economic outcome of opening and closing a position.

    Args:
        trade_id:     Unique identifier.
        instrument:   The traded instrument.
        entry_date:   Date the position was opened.
        exit_date:    Date the position was closed.
        entry_price:  Average price paid to open.
        exit_price:   Average price received to close.
        quantity:     Number of units (positive = long, negative = short).
        commission:   Total commission paid (both legs).
        slippage:     Total slippage cost (both legs).
    """

    model_config = {"frozen": True}

    trade_id: str
    instrument: Instrument
    entry_date: date
    exit_date: date
    entry_price: Decimal = Field(gt=0)
    exit_price: Decimal = Field(gt=0)
    quantity: Decimal  # Negative for short trades
    commission: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")

    @property
    def pnl(self) -> Decimal:
        """Calculate gross profit and loss for the trade."""
        raw_pnl = (self.exit_price - self.entry_price) * self.quantity
        return raw_pnl - self.commission - self.slippage

    @property
    def pnl_pct(self) -> float:
        """Return the PnL as a percentage of the entry notional."""
        notional = self.entry_price * abs(self.quantity)
        if notional == 0:
            return 0.0
        return float(self.pnl / notional)

    @property
    def is_winner(self) -> bool:
        """Return ``True`` if the trade generated a positive net PnL."""
        return self.pnl > 0

    @property
    def holding_days(self) -> int:
        """Return the number of calendar days the position was held."""
        return (self.exit_date - self.entry_date).days
