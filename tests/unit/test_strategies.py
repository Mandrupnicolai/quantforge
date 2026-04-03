"""Unit tests for built-in trading strategies.

Each strategy test verifies:
    * Signal direction on known price sequences (deterministic examples).
    * No look-ahead bias (signal at bar *t* uses only data up to bar *t-1*).
    * Correct handling of insufficient data (NaN / None signals).
    * Parameter validation in constructors.
    * Property-based invariants via Hypothesis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from quantforge.core.models import Signal, SignalDirection
from quantforge.strategies.base import (
    MeanReversionStrategy,
    MomentumStrategy,
    SMACrossoverStrategy,
    Strategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(close_values: list[float], open_offset: float = 1.0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(close_values)
    closes = pd.Series(close_values, dtype=float)
    return pd.DataFrame(
        {
            "open": closes + open_offset,
            "high": closes + abs(open_offset) * 2,
            "low": closes - abs(open_offset) * 2,
            "close": closes,
            "volume": [100_000] * n,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )


def _count_signals(signals: pd.Series) -> dict[SignalDirection | None, int]:
    """Count signals by direction for assertion."""
    counts: dict[SignalDirection | None, int] = {}
    for s in signals:
        key = s.direction if isinstance(s, Signal) else None
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# SMACrossoverStrategy
# ---------------------------------------------------------------------------


class TestSMACrossoverStrategy:
    """Tests for the SMA crossover strategy."""

    @pytest.fixture
    def default_strategy(self) -> SMACrossoverStrategy:
        return SMACrossoverStrategy(fast_window=3, slow_window=5)

    def test_satisfies_strategy_protocol(self, default_strategy: SMACrossoverStrategy) -> None:
        """Verify duck typing against the Strategy protocol."""
        assert isinstance(default_strategy, Strategy)

    def test_name_includes_windows(self) -> None:
        s = SMACrossoverStrategy(fast_window=10, slow_window=30)
        assert "10" in s.name and "30" in s.name

    def test_fast_gte_slow_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly less than"):
            SMACrossoverStrategy(fast_window=50, slow_window=20)

    def test_fast_equals_slow_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly less than"):
            SMACrossoverStrategy(fast_window=20, slow_window=20)

    def test_fast_window_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            SMACrossoverStrategy(fast_window=1, slow_window=5)

    def test_golden_cross_generates_long_signal(self) -> None:
        """Price trending up should trigger a golden cross (LONG signal)."""
        # Construct a price series that starts flat then trends up sharply,
        # ensuring the fast MA crosses above the slow MA.
        flat = [100.0] * 10
        trending = [100.0 + i * 5 for i in range(20)]
        prices = _make_prices(flat + trending)

        strategy = SMACrossoverStrategy(fast_window=3, slow_window=5)
        signals = strategy.generate_signals(prices)

        counts = _count_signals(signals)
        # There must be at least one LONG signal in the trending period
        assert counts.get(SignalDirection.LONG, 0) >= 1

    def test_no_signals_before_warmup(self, default_strategy: SMACrossoverStrategy) -> None:
        """No signal should fire before the slow MA window is populated."""
        prices = _make_prices([100.0] * 4)  # Fewer bars than slow_window=5
        signals = default_strategy.generate_signals(prices)
        counts = _count_signals(signals)
        assert counts.get(SignalDirection.LONG, 0) == 0
        assert counts.get(SignalDirection.SHORT, 0) == 0

    def test_no_lookahead_bias(self, default_strategy: SMACrossoverStrategy) -> None:
        """Signal at bar t must be identical whether future bars exist or not."""
        full_prices = _make_prices([100.0] * 5 + [120.0] * 10 + [80.0] * 10)
        truncated = full_prices.iloc[:10]

        full_sigs = default_strategy.generate_signals(full_prices)
        trunc_sigs = default_strategy.generate_signals(truncated)

        # Signals for the first 10 bars should be identical in both runs
        for i in range(len(truncated)):
            full_s = full_sigs.iloc[i]
            trunc_s = trunc_sigs.iloc[i]
            full_dir = full_s.direction if isinstance(full_s, Signal) else None
            trunc_dir = trunc_s.direction if isinstance(trunc_s, Signal) else None
            assert full_dir == trunc_dir, f"Look-ahead bias detected at bar {i}"

    def test_output_length_matches_input(self, default_strategy: SMACrossoverStrategy) -> None:
        prices = _make_prices([100.0] * 30)
        signals = default_strategy.generate_signals(prices)
        assert len(signals) == 30

    @given(
        n=st.integers(min_value=30, max_value=100),
        fast=st.integers(min_value=2, max_value=10),
        slow_offset=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50)
    def test_output_length_always_matches_input(self, n: int, fast: int, slow_offset: int) -> None:
        """Property: output Series must have same length as input DataFrame."""
        slow = fast + slow_offset
        strategy = SMACrossoverStrategy(fast_window=fast, slow_window=slow)
        prices = _make_prices([100.0 + i * 0.1 for i in range(n)])
        signals = strategy.generate_signals(prices)
        assert len(signals) == n


# ---------------------------------------------------------------------------
# MomentumStrategy
# ---------------------------------------------------------------------------


class TestMomentumStrategy:
    """Tests for the momentum strategy."""

    @pytest.fixture
    def strategy(self) -> MomentumStrategy:
        return MomentumStrategy(lookback=30, skip=5, min_strength=0.01)

    def test_skip_gte_lookback_raises(self) -> None:
        with pytest.raises(ValueError, match="skip must be less than lookback"):
            MomentumStrategy(lookback=20, skip=20)

    def test_positive_momentum_yields_long(self, strategy: MomentumStrategy) -> None:
        """Strongly rising prices should produce a LONG signal."""
        rising = [100.0 * (1.005**i) for i in range(60)]
        prices = _make_prices(rising)
        signals = strategy.generate_signals(prices)

        last_signal: Signal | None = None
        for s in reversed(signals.tolist()):
            if isinstance(s, Signal):
                last_signal = s
                break

        assert last_signal is not None
        assert last_signal.direction == SignalDirection.LONG

    def test_strength_in_valid_range(self, strategy: MomentumStrategy) -> None:
        """All emitted signal strengths must lie in [0, 1]."""
        prices = _make_prices([100.0 + np.random.randn() for _ in range(60)])
        signals = strategy.generate_signals(prices)
        for s in signals:
            if isinstance(s, Signal):
                assert 0.0 <= s.strength <= 1.0

    def test_name_includes_parameters(self, strategy: MomentumStrategy) -> None:
        assert "30" in strategy.name and "5" in strategy.name


# ---------------------------------------------------------------------------
# MeanReversionStrategy
# ---------------------------------------------------------------------------


class TestMeanReversionStrategy:
    """Tests for the mean-reversion strategy."""

    @pytest.fixture
    def strategy(self) -> MeanReversionStrategy:
        return MeanReversionStrategy(window=10, entry_z=2.0, exit_z=0.5)

    def test_exit_gte_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="exit_z must be less than"):
            MeanReversionStrategy(entry_z=2.0, exit_z=2.5)

    def test_window_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            MeanReversionStrategy(window=2)

    def test_high_zscore_generates_short(self, strategy: MeanReversionStrategy) -> None:
        """A price far above its rolling mean should yield a SHORT signal."""
        base = [100.0] * 20
        spike = [200.0]  # Extreme spike → z >> 2.0
        prices = _make_prices(base + spike)
        signals = strategy.generate_signals(prices)
        last = signals.iloc[-1]
        assert isinstance(last, Signal)
        assert last.direction == SignalDirection.SHORT

    def test_zscore_stored_in_metadata(self, strategy: MeanReversionStrategy) -> None:
        base = [100.0] * 20
        spike = [200.0]
        prices = _make_prices(base + spike)
        signals = strategy.generate_signals(prices)
        last = signals.iloc[-1]
        assert isinstance(last, Signal)
        assert "z_score" in last.metadata

    def test_flat_signal_near_mean(self, strategy: MeanReversionStrategy) -> None:
        """Prices hugging the mean should trigger a FLAT (exit) signal."""
        # First set up a reversal, then return to mean
        prices_data = [100.0] * 15 + [300.0] + [100.0] * 5
        prices = _make_prices(prices_data)
        signals = strategy.generate_signals(prices)

        # After reverting to mean there should be at least one FLAT signal
        counts = _count_signals(signals)
        assert counts.get(SignalDirection.FLAT, 0) >= 1

    def test_name_includes_window(self, strategy: MeanReversionStrategy) -> None:
        assert "10" in strategy.name
