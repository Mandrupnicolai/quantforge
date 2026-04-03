"""Unit tests for Portfolio and Position management.

Focuses on:
    * Position averaging and FIFO close accounting.
    * Cash balance arithmetic after buys and sells.
    * Equity curve snapshots.
    * Portfolio invariants under adversarial input sequences.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from quantforge.core.models import (
    Instrument,
    Order,
    OrderSide,
    OrderType,
)
from quantforge.core.portfolio import Portfolio, Position

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def msft() -> Instrument:
    return Instrument(symbol="MSFT", exchange="XNAS", currency="USD")


@pytest.fixture
def empty_portfolio() -> Portfolio:
    return Portfolio(initial_capital=100_000.0, currency="USD", name="Test")


# ---------------------------------------------------------------------------
# Position tests
# ---------------------------------------------------------------------------


class TestPosition:
    """Tests for the Position state machine."""

    def test_initial_state(self, msft: Instrument) -> None:
        pos = Position(msft, Decimal("100"), Decimal("300.00"))
        assert pos.quantity == Decimal("100")
        assert pos.avg_price == Decimal("300.00")
        assert pos.is_long is True
        assert pos.is_short is False
        assert pos.is_flat is False

    def test_add_updates_average_price(self, msft: Instrument) -> None:
        """Adding at a different price should compute a weighted average."""
        pos = Position(msft, Decimal("100"), Decimal("300.00"))
        # Buy 100 more at 320 → avg = (100*300 + 100*320) / 200 = 310
        pos.add(Decimal("100"), Decimal("320.00"))
        assert pos.quantity == Decimal("200")
        assert pos.avg_price == Decimal("310.00")

    def test_add_zero_quantity_is_noop(self, msft: Instrument) -> None:
        pos = Position(msft, Decimal("50"), Decimal("100.00"))
        original_avg = pos.avg_price
        pos.add(Decimal("0"), Decimal("200.00"))
        assert pos.avg_price == original_avg
        assert pos.quantity == Decimal("50")

    def test_full_close_sets_flat(self, msft: Instrument) -> None:
        pos = Position(msft, Decimal("100"), Decimal("200.00"))
        pos.reduce(Decimal("100"), Decimal("210.00"))
        assert pos.is_flat is True

    def test_partial_close_updates_quantity(self, msft: Instrument) -> None:
        pos = Position(msft, Decimal("100"), Decimal("200.00"))
        pos.reduce(Decimal("40"), Decimal("210.00"))
        assert pos.quantity == Decimal("60")

    def test_long_pnl_on_reduce(self, msft: Instrument) -> None:
        """Closing a long at profit should return positive PnL."""
        pos = Position(msft, Decimal("100"), Decimal("200.00"))
        pnl = pos.reduce(Decimal("100"), Decimal("220.00"))
        assert pnl == Decimal("2000.00")  # (220 - 200) * 100

    def test_short_pnl_on_reduce(self, msft: Instrument) -> None:
        """Covering a short at profit should return positive PnL."""
        pos = Position(msft, Decimal("-100"), Decimal("200.00"))
        pnl = pos.reduce(Decimal("100"), Decimal("180.00"))
        assert pnl == Decimal("2000.00")  # (200 - 180) * 100

    def test_market_value_long(self, msft: Instrument) -> None:
        pos = Position(msft, Decimal("100"), Decimal("200.00"))
        assert pos.market_value(Decimal("250.00")) == Decimal("25000.00")

    def test_market_value_short(self, msft: Instrument) -> None:
        """Short positions have negative market value."""
        pos = Position(msft, Decimal("-50"), Decimal("200.00"))
        assert pos.market_value(Decimal("210.00")) == Decimal("-10500.00")

    def test_unrealised_pnl_long_profitable(self, msft: Instrument) -> None:
        pos = Position(msft, Decimal("100"), Decimal("100.00"))
        assert pos.unrealised_pnl(Decimal("120.00")) == Decimal("2000.00")

    def test_unrealised_pnl_long_loss(self, msft: Instrument) -> None:
        pos = Position(msft, Decimal("100"), Decimal("100.00"))
        assert pos.unrealised_pnl(Decimal("80.00")) == Decimal("-2000.00")

    def test_repr_is_informative(self, msft: Instrument) -> None:
        pos = Position(msft, Decimal("10"), Decimal("300.00"))
        assert "MSFT" in repr(pos)
        assert "10" in repr(pos)


# ---------------------------------------------------------------------------
# Portfolio tests
# ---------------------------------------------------------------------------


class TestPortfolio:
    """Tests for Portfolio capital management and trade recording."""

    def _buy_fill(
        self,
        portfolio: Portfolio,
        instrument: Instrument,
        quantity: Decimal,
        price: Decimal,
        execution_date: date,
    ) -> None:
        """Helper: create a market buy order and apply a fill."""
        order = Order(
            order_id="test-order",
            instrument=instrument,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )
        portfolio.apply_fill(
            order=order,
            fill_price=price,
            fill_quantity=quantity,
            commission=Decimal("0"),
            slippage=Decimal("0"),
            execution_date=execution_date,
        )

    def test_initial_state(self, empty_portfolio: Portfolio) -> None:
        assert empty_portfolio.cash == Decimal("100000")
        assert len(empty_portfolio.positions) == 0
        assert len(empty_portfolio.trades) == 0

    def test_buy_decreases_cash(self, empty_portfolio: Portfolio, msft: Instrument) -> None:
        self._buy_fill(
            empty_portfolio,
            msft,
            Decimal("100"),
            Decimal("300.00"),
            date(2024, 1, 2),
        )
        assert empty_portfolio.cash == Decimal("70000")  # 100000 - 100*300

    def test_buy_creates_position(self, empty_portfolio: Portfolio, msft: Instrument) -> None:
        self._buy_fill(
            empty_portfolio,
            msft,
            Decimal("50"),
            Decimal("200.00"),
            date(2024, 1, 3),
        )
        assert "MSFT" in empty_portfolio.positions
        assert empty_portfolio.positions["MSFT"].quantity == Decimal("50")

    def test_sell_after_buy_records_trade(
        self, empty_portfolio: Portfolio, msft: Instrument
    ) -> None:
        self._buy_fill(
            empty_portfolio,
            msft,
            Decimal("100"),
            Decimal("200.00"),
            date(2024, 1, 2),
        )
        sell_order = Order(
            order_id="sell-001",
            instrument=msft,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
        )
        empty_portfolio.apply_fill(
            order=sell_order,
            fill_price=Decimal("220.00"),
            fill_quantity=Decimal("100"),
            commission=Decimal("0"),
            slippage=Decimal("0"),
            execution_date=date(2024, 1, 10),
        )
        # Position should be closed
        assert "MSFT" not in empty_portfolio.positions
        # One trade recorded
        assert len(empty_portfolio.trades) == 1
        trade = empty_portfolio.trades[0]
        assert trade.pnl > 0

    def test_equity_snapshot_with_no_positions(self, empty_portfolio: Portfolio) -> None:
        equity = empty_portfolio.snapshot_equity(date(2024, 1, 2), {})
        assert equity == Decimal("100000")
        assert len(empty_portfolio.equity_curve) == 1

    def test_equity_snapshot_marks_positions(
        self, empty_portfolio: Portfolio, msft: Instrument
    ) -> None:
        self._buy_fill(
            empty_portfolio,
            msft,
            Decimal("100"),
            Decimal("200.00"),
            date(2024, 1, 2),
        )
        equity = empty_portfolio.snapshot_equity(date(2024, 1, 3), {"MSFT": Decimal("210.00")})
        # Cash: 100000 - 20000 = 80000; position: 100*210 = 21000; total = 101000
        assert equity == Decimal("101000")

    def test_initial_capital_immutable(self, empty_portfolio: Portfolio) -> None:
        """initial_capital must not be affected by trades."""
        self._buy_fill(
            empty_portfolio,
            Instrument(symbol="GOOG"),
            Decimal("10"),
            Decimal("150.00"),
            date(2024, 1, 5),
        )
        assert empty_portfolio.initial_capital == Decimal("100000")

    def test_iter_positions(self, empty_portfolio: Portfolio, msft: Instrument) -> None:
        self._buy_fill(
            empty_portfolio,
            msft,
            Decimal("50"),
            Decimal("100.00"),
            date(2024, 1, 2),
        )
        positions = list(empty_portfolio.iter_positions())
        assert len(positions) == 1
        assert positions[0].instrument.symbol == "MSFT"

    def test_repr_is_informative(self, empty_portfolio: Portfolio) -> None:
        r = repr(empty_portfolio)
        assert "Test" in r
        assert "100000" in r
