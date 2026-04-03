"""Unit tests for core domain models.

Tests are organised by class.  Property-based tests use Hypothesis to explore
edge cases that hand-written examples might miss.

Coverage targets:
    * All validation rules on Pydantic models.
    * All computed properties (``pnl``, ``pnl_pct``, ``is_winner``, etc.).
    * Arithmetic on ``Money`` (addition, multiplication, currency guard).
    * ``OHLCV`` high/low consistency invariant.
    * ``Trade`` PnL accounting across long and short positions.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from quantforge.core.models import (
    OHLCV,
    AssetClass,
    Instrument,
    Money,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Signal,
    SignalDirection,
    Trade,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aapl() -> Instrument:
    """Return a canonical Apple Inc. instrument for use across tests."""
    return Instrument(
        symbol="AAPL",
        isin="US0378331005",
        exchange="XNAS",
        asset_class=AssetClass.EQUITY,
        currency="USD",
        description="Apple Inc.",
    )


@pytest.fixture
def sample_ohlcv() -> OHLCV:
    """Return a valid OHLCV bar for AAPL."""
    return OHLCV(
        timestamp=datetime(2024, 1, 15, tzinfo=UTC),
        open=Decimal("185.00"),
        high=Decimal("187.50"),
        low=Decimal("184.00"),
        close=Decimal("186.75"),
        volume=45_123_456,
    )


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------


class TestMoney:
    """Tests for the Money value object."""

    def test_creation_with_decimal_amount(self) -> None:
        m = Money(amount=Decimal("100.00"), currency="USD")
        assert m.amount == Decimal("100.00")
        assert m.currency == "USD"

    def test_addition_same_currency(self) -> None:
        a = Money(amount=Decimal("50.00"), currency="GBP")
        b = Money(amount=Decimal("25.50"), currency="GBP")
        result = a + b
        assert result.amount == Decimal("75.50")
        assert result.currency == "GBP"

    def test_addition_different_currency_raises(self) -> None:
        a = Money(amount=Decimal("100"), currency="USD")
        b = Money(amount=Decimal("100"), currency="EUR")
        with pytest.raises(ValueError, match="Cannot add"):
            _ = a + b

    def test_multiplication_by_scalar(self) -> None:
        m = Money(amount=Decimal("10.00"), currency="USD")
        result = m * 2.5
        assert result.amount == Decimal("25.00")

    def test_multiplication_by_decimal(self) -> None:
        m = Money(amount=Decimal("100.00"), currency="USD")
        result = m * Decimal("0.1")
        assert result.amount == Decimal("10.00")

    def test_repr_includes_currency_and_amount(self) -> None:
        m = Money(amount=Decimal("1234.5678"), currency="EUR")
        assert "EUR" in repr(m)
        assert "1,234.5678" in repr(m)

    def test_immutable_frozen_model(self) -> None:
        m = Money(amount=Decimal("100"), currency="USD")
        with pytest.raises(ValidationError):
            m.amount = Decimal("200")  # type: ignore[misc]

    @given(
        amount_a=st.decimals(min_value=0, max_value=1_000_000, places=4, allow_nan=False),
        amount_b=st.decimals(min_value=0, max_value=1_000_000, places=4, allow_nan=False),
    )
    def test_addition_commutative(self, amount_a: Decimal, amount_b: Decimal) -> None:
        """Money addition should be commutative."""
        a = Money(amount=amount_a, currency="USD")
        b = Money(amount=amount_b, currency="USD")
        assert (a + b).amount == (b + a).amount

    @pytest.mark.parametrize(
        "code",
        ["US", "USDT", "us", "123", "U D"],
    )
    def test_invalid_currency_codes(self, code: str) -> None:
        """Non-ISO-4217 currency codes must be rejected."""
        with pytest.raises(ValidationError):
            Money(amount=Decimal("10"), currency=code)


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------


class TestOHLCV:
    """Tests for the OHLCV price bar model."""

    def test_valid_bar_creates_successfully(self, sample_ohlcv: OHLCV) -> None:
        assert sample_ohlcv.open == Decimal("185.00")
        assert sample_ohlcv.volume == 45_123_456

    def test_high_below_low_raises(self) -> None:
        with pytest.raises(ValidationError, match="high.*must be >= low"):
            OHLCV(
                timestamp=datetime.now(UTC),
                open=Decimal("100"),
                high=Decimal("98"),  # Violates high >= low
                low=Decimal("99"),
                close=Decimal("100"),
                volume=1000,
            )

    def test_close_above_high_raises(self) -> None:
        with pytest.raises(ValidationError):
            OHLCV(
                timestamp=datetime.now(UTC),
                open=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("98"),
                close=Decimal(
                    "110"
                ),  # Close > high ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â impossible
                volume=1000,
            )

    def test_naive_datetime_becomes_utc(self) -> None:
        bar = OHLCV(
            timestamp=datetime(2024, 6, 1),  # Naive
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("95"),
            close=Decimal("105"),
            volume=500,
        )
        assert bar.timestamp.tzinfo == UTC

    def test_zero_volume_is_valid(self, sample_ohlcv: OHLCV) -> None:
        """Auction-only or halted trading days may have zero volume."""
        bar = sample_ohlcv.model_copy(update={"volume": 0})
        assert bar.volume == 0

    def test_negative_volume_raises(self) -> None:
        with pytest.raises(ValidationError):
            OHLCV(
                timestamp=datetime.now(UTC),
                open=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("98"),
                close=Decimal("102"),
                volume=-1,
            )

    @given(
        open_=st.decimals(min_value="0.01", max_value="10000", places=2, allow_nan=False),
        pct=st.floats(
            min_value=0.0, max_value=0.5
        ),  # High/low within ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â±50% of open
    )
    @settings(max_examples=200)
    def test_high_always_gte_low(self, open_: Decimal, pct: float) -> None:
        """Property: for any valid bar, high must always be >= low."""
        spread = open_ * Decimal(str(pct))
        high = open_ + spread
        low = max(open_ - spread, Decimal("0.01"))
        close = open_  # Use open as close for simplicity

        bar = OHLCV(
            timestamp=datetime.now(UTC),
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=1000,
        )
        assert bar.high >= bar.low


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


class TestSignal:
    """Tests for the Signal value object."""

    @pytest.mark.parametrize("direction", list(SignalDirection))
    def test_all_directions_valid(self, direction: SignalDirection) -> None:
        signal = Signal(direction=direction)
        assert signal.direction == direction

    def test_is_entry_for_long(self) -> None:
        assert Signal(direction=SignalDirection.LONG).is_entry is True

    def test_is_entry_for_short(self) -> None:
        assert Signal(direction=SignalDirection.SHORT).is_entry is True

    def test_is_exit_for_flat(self) -> None:
        assert Signal(direction=SignalDirection.FLAT).is_exit is True

    def test_is_entry_false_for_flat(self) -> None:
        assert Signal(direction=SignalDirection.FLAT).is_entry is False

    def test_strength_defaults_to_one(self) -> None:
        signal = Signal(direction=SignalDirection.LONG)
        assert signal.strength == 1.0

    @pytest.mark.parametrize("invalid_strength", [-0.01, 1.01, 2.0, -1.0])
    def test_invalid_strength_raises(self, invalid_strength: float) -> None:
        with pytest.raises(ValidationError):
            Signal(direction=SignalDirection.LONG, strength=invalid_strength)

    def test_metadata_accessible(self) -> None:
        signal = Signal(
            direction=SignalDirection.SHORT,
            strength=0.75,
            metadata={"z_score": -2.3, "regime": "trending"},
        )
        assert signal.metadata["z_score"] == pytest.approx(-2.3)


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------


class TestTrade:
    """Tests for Trade PnL accounting."""

    @pytest.fixture
    def long_trade(self, aapl: Instrument) -> Trade:
        """A profitable long trade: bought at 100, sold at 110, 100 shares."""
        return Trade(
            trade_id="t001",
            instrument=aapl,
            entry_date=date(2024, 1, 2),
            exit_date=date(2024, 1, 10),
            entry_price=Decimal("100.00"),
            exit_price=Decimal("110.00"),
            quantity=Decimal("100"),
            commission=Decimal("2.00"),
            slippage=Decimal("0.50"),
        )

    @pytest.fixture
    def losing_short_trade(self, aapl: Instrument) -> Trade:
        """A losing short trade: shorted at 100, bought back at 105, 50 shares."""
        return Trade(
            trade_id="t002",
            instrument=aapl,
            entry_date=date(2024, 2, 1),
            exit_date=date(2024, 2, 15),
            entry_price=Decimal("100.00"),
            exit_price=Decimal("105.00"),
            quantity=Decimal("-50"),  # Short
            commission=Decimal("1.50"),
            slippage=Decimal("0.25"),
        )

    def test_long_trade_pnl(self, long_trade: Trade) -> None:
        # Gross PnL = (110 - 100) * 100 = 1000
        # Net = 1000 - 2.00 - 0.50 = 997.50
        assert long_trade.pnl == Decimal("997.50")

    def test_long_trade_is_winner(self, long_trade: Trade) -> None:
        assert long_trade.is_winner is True

    def test_long_trade_pnl_pct(self, long_trade: Trade) -> None:
        # Net PnL / entry notional = 997.50 / (100 * 100) = 0.09975
        assert long_trade.pnl_pct == pytest.approx(0.09975, rel=1e-4)

    def test_losing_short_trade_pnl(self, losing_short_trade: Trade) -> None:
        # Gross PnL = (100 - 105) * 50 = -250
        # Net = -250 - 1.50 - 0.25 = -251.75
        assert losing_short_trade.pnl == Decimal("-251.75")

    def test_losing_short_trade_is_not_winner(self, losing_short_trade: Trade) -> None:
        assert losing_short_trade.is_winner is False

    def test_holding_days(self, long_trade: Trade) -> None:
        assert long_trade.holding_days == 8

    def test_zero_cost_trade_pnl(self, aapl: Instrument) -> None:
        """Edge case: zero commission and slippage."""
        t = Trade(
            trade_id="t003",
            instrument=aapl,
            entry_date=date(2024, 1, 1),
            exit_date=date(2024, 1, 2),
            entry_price=Decimal("50.00"),
            exit_price=Decimal("50.00"),
            quantity=Decimal("10"),
        )
        assert t.pnl == Decimal("0")
        assert t.is_winner is False

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        entry=st.decimals(min_value="1", max_value="5000", places=2, allow_nan=False),
        exit_=st.decimals(min_value="1", max_value="5000", places=2, allow_nan=False),
        qty=st.decimals(min_value="1", max_value="10000", places=0, allow_nan=False),
    )
    def test_pnl_sign_consistent_with_is_winner(
        self,
        entry: Decimal,
        exit_: Decimal,
        qty: Decimal,
    ) -> None:
        """Property: is_winner iff pnl > 0 (with zero costs)."""
        t = Trade(
            trade_id="prop",
            instrument=aapl,
            entry_date=date(2024, 1, 1),
            exit_date=date(2024, 1, 2),
            entry_price=entry,
            exit_price=exit_,
            quantity=qty,
        )
        assert t.is_winner == (t.pnl > 0)


# ---------------------------------------------------------------------------
# Order validation
# ---------------------------------------------------------------------------


class TestOrder:
    """Tests for Order lifecycle and validation."""

    def test_market_order_creation(self, aapl: Instrument) -> None:
        order = Order(
            order_id="ord001",
            instrument=aapl,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
        )
        assert order.status == OrderStatus.PENDING
        assert order.is_terminal is False

    def test_limit_order_without_price_raises(self, aapl: Instrument) -> None:
        with pytest.raises(ValidationError, match="require a limit_price"):
            Order(
                order_id="ord002",
                instrument=aapl,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("50"),
                # limit_price intentionally omitted
            )

    def test_stop_order_without_stop_price_raises(self, aapl: Instrument) -> None:
        with pytest.raises(ValidationError, match="require a stop_price"):
            Order(
                order_id="ord003",
                instrument=aapl,
                side=OrderSide.SELL,
                order_type=OrderType.STOP,
                quantity=Decimal("25"),
            )

    @pytest.mark.parametrize(
        "status",
        [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED],
    )
    def test_terminal_statuses(self, aapl: Instrument, status: OrderStatus) -> None:
        order = Order(
            order_id="ord004",
            instrument=aapl,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
        )
        order.status = status
        assert order.is_terminal is True

    def test_pending_is_not_terminal(self, aapl: Instrument) -> None:
        order = Order(
            order_id="ord005",
            instrument=aapl,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
        )
        assert order.is_terminal is False
