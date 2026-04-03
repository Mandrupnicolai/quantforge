"""Portfolio management — tracks positions, capital, and real-time equity.

The ``Portfolio`` is the single source of truth for what the backtester
(or live execution engine) currently holds.  It exposes a minimal, explicit
API so that state mutations are always intentional and auditable.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Iterator

import structlog

from quantforge.core.models import Instrument, Order, OrderSide, OrderStatus, Trade

log = structlog.get_logger(__name__)


class Position:
    """An open position in a single instrument.

    Tracks quantity (signed — negative for shorts), average cost basis, and
    realised PnL as lots are closed against it using FIFO accounting.

    Args:
        instrument: The instrument this position is in.
        quantity:   Initial quantity (positive = long, negative = short).
        avg_price:  Initial average entry price.
    """

    def __init__(
        self,
        instrument: Instrument,
        quantity: Decimal,
        avg_price: Decimal,
    ) -> None:
        self.instrument = instrument
        self.quantity = quantity
        self.avg_price = avg_price
        self.realised_pnl: Decimal = Decimal("0")

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def add(self, quantity: Decimal, price: Decimal) -> None:
        """Increase the position size (same direction as existing).

        Updates the average cost basis using a weighted average.

        Args:
            quantity: Additional units to add (must be same sign as current).
            price:    Execution price for the new units.
        """
        if quantity == 0:
            return
        total_cost = self.avg_price * abs(self.quantity) + price * abs(quantity)
        self.quantity += quantity
        if self.quantity != 0:
            self.avg_price = total_cost / abs(self.quantity)

    def reduce(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Reduce or close the position, returning the realised PnL.

        For a long position (positive quantity), a reduction is a sell.
        For a short position (negative quantity), a reduction is a buy-to-cover.

        Args:
            quantity: Units to close (unsigned; direction inferred from sign of position).
            price:    Execution price at which the units are closed.

        Returns:
            The realised PnL for this partial or full close.
        """
        closed = min(abs(quantity), abs(self.quantity))
        if self.quantity > 0:
            pnl = (price - self.avg_price) * closed
            self.quantity -= closed
        else:
            pnl = (self.avg_price - price) * closed
            self.quantity += closed
        self.realised_pnl += pnl
        return pnl

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    def market_value(self, current_price: Decimal) -> Decimal:
        """Return the current mark-to-market value of the position."""
        return self.quantity * current_price

    def unrealised_pnl(self, current_price: Decimal) -> Decimal:
        """Return the unrealised PnL at the given market price."""
        if self.quantity > 0:
            return (current_price - self.avg_price) * self.quantity
        return (self.avg_price - current_price) * abs(self.quantity)

    @property
    def is_long(self) -> bool:
        """Return ``True`` if the position is long (positive quantity)."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Return ``True`` if the position is short (negative quantity)."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """Return ``True`` if the position has been fully closed."""
        return self.quantity == Decimal("0")

    def __repr__(self) -> str:
        return (
            f"Position({self.instrument.symbol}, qty={self.quantity}, "
            f"avg_px={self.avg_price:.4f})"
        )


class Portfolio:
    """Manages capital allocation, open positions, and trade history.

    This class is the central stateful object.  It is intentionally not
    thread-safe — concurrent access must be coordinated externally.

    Args:
        initial_capital: Starting cash balance in the base currency.
        currency:        ISO-4217 code for the base currency (default ``"USD"``).
        name:            Human-readable portfolio name for reporting.
    """

    def __init__(
        self,
        initial_capital: float | Decimal,
        currency: str = "USD",
        name: str = "Portfolio",
    ) -> None:
        self._initial_capital = Decimal(str(initial_capital))
        self.cash: Decimal = self._initial_capital
        self.currency = currency
        self.name = name
        self._positions: dict[str, Position] = {}
        self._trades: list[Trade] = []
        self._equity_curve: list[tuple[date, Decimal]] = []

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def apply_fill(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal,
        commission: Decimal,
        slippage: Decimal,
        execution_date: date,
    ) -> Trade | None:
        """Apply a fill event to the portfolio and return a completed Trade if any.

        This is the single entry-point for all state mutations that arise from
        order execution.  It handles both opening new positions and reducing /
        closing existing ones.

        Args:
            order:          The order that was filled.
            fill_price:     The actual execution price (after slippage).
            fill_quantity:  The number of units filled.
            commission:     Commission charged for this fill.
            slippage:       Cost attributable to market impact / spread.
            execution_date: The calendar date of execution.

        Returns:
            A completed ``Trade`` if this fill fully closes a position, else ``None``.
        """
        symbol = order.instrument.symbol
        total_cost = fill_price * fill_quantity + commission + slippage

        log.debug(
            "applying_fill",
            symbol=symbol,
            side=order.side.value,
            qty=str(fill_quantity),
            price=str(fill_price),
            commission=str(commission),
        )

        if order.side == OrderSide.BUY:
            self.cash -= total_cost
            if symbol in self._positions and self._positions[symbol].is_short:
                return self._close_position(
                    symbol, fill_quantity, fill_price,
                    commission, slippage, execution_date,
                )
            self._open_or_add(order.instrument, fill_quantity, fill_price)

        else:  # SELL
            self.cash += fill_price * fill_quantity - commission - slippage
            if symbol in self._positions and self._positions[symbol].is_long:
                return self._close_position(
                    symbol, fill_quantity, fill_price,
                    commission, slippage, execution_date,
                )
            self._open_or_add(order.instrument, -fill_quantity, fill_price)

        # Mark order as filled
        order.status = OrderStatus.FILLED
        order.filled_quantity = fill_quantity
        order.average_fill_price = fill_price
        return None

    def _open_or_add(
        self,
        instrument: Instrument,
        quantity: Decimal,
        price: Decimal,
    ) -> None:
        """Open a new position or increase an existing one."""
        symbol = instrument.symbol
        if symbol in self._positions:
            self._positions[symbol].add(quantity, price)
        else:
            self._positions[symbol] = Position(instrument, quantity, price)

    def _close_position(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
        slippage: Decimal,
        execution_date: date,
    ) -> Trade | None:
        """Reduce or fully close a position and record a Trade if closed."""
        position = self._positions[symbol]
        entry_price = position.avg_price
        entry_date_approx = execution_date  # Simplified — production would track entry date

        position.reduce(quantity, price)

        trade = Trade(
            trade_id=str(uuid.uuid4()),
            instrument=position.instrument,
            entry_date=entry_date_approx,
            exit_date=execution_date,
            entry_price=entry_price,
            exit_price=price,
            quantity=quantity if position.is_long else -quantity,
            commission=commission,
            slippage=slippage,
        )
        self._trades.append(trade)

        if position.is_flat:
            del self._positions[symbol]

        return trade

    # ------------------------------------------------------------------
    # Equity accounting
    # ------------------------------------------------------------------

    def snapshot_equity(
        self,
        current_date: date,
        mark_prices: dict[str, Decimal],
    ) -> Decimal:
        """Compute total equity and record a point on the equity curve.

        Equity = cash + sum of all position market values at ``mark_prices``.

        Args:
            current_date: The date of this snapshot (used for the equity curve).
            mark_prices:  Mapping of symbol → current mark price.

        Returns:
            Total portfolio equity as a Decimal.
        """
        position_value = sum(
            pos.market_value(mark_prices[sym])
            for sym, pos in self._positions.items()
            if sym in mark_prices
        )
        equity = self.cash + Decimal(str(position_value))
        self._equity_curve.append((current_date, equity))
        return equity

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def positions(self) -> dict[str, Position]:
        """Return a read-only view of all open positions."""
        return dict(self._positions)

    @property
    def trades(self) -> list[Trade]:
        """Return a copy of the completed trade list."""
        return list(self._trades)

    @property
    def equity_curve(self) -> list[tuple[date, Decimal]]:
        """Return the full equity curve as a list of (date, equity) tuples."""
        return list(self._equity_curve)

    @property
    def initial_capital(self) -> Decimal:
        """Return the starting capital."""
        return self._initial_capital

    def iter_positions(self) -> Iterator[Position]:
        """Iterate over all currently open positions."""
        yield from self._positions.values()

    def __repr__(self) -> str:
        return (
            f"Portfolio(name={self.name!r}, cash={self.cash:.2f}, "
            f"positions={len(self._positions)}, trades={len(self._trades)})"
        )
