"""Unit tests for built-in trading strategies."""

from __future__ import annotations

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


def _make_prices(close_values: list[float], open_offset: float = 1.0) -> pd.DataFrame:
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


def _get_directions(signals: pd.Series) -> list:
    return [s.direction if isinstance(s, Signal) else None for s in signals]


def _count_direction(signals: pd.Series, direction: SignalDirection) -> int:
    return sum(1 for s in signals if isinstance(s, Signal) and s.direction == direction)


class TestSMACrossoverStrategy:
    @pytest.fixture
    def default_strategy(self) -> SMACrossoverStrategy:
        return SMACrossoverStrategy(fast_window=3, slow_window=5)

    def test_satisfies_strategy_protocol(self, default_strategy: SMACrossoverStrategy) -> None:
        assert isinstance(default_strategy, Strategy)

    def test_name_includes_windows(self) -> None:
        s = SMACrossoverStrategy(fast_window=10, slow_window=30)
        assert "10" in s.name
	assert "30" in s.name

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
        flat = [100.0] * 10
        trending = [100.0 + i * 5 for i in range(20)]
        prices = _make_prices(flat + trending)
        strategy = SMACrossoverStrategy(fast_window=3, slow_window=5)
        signals = strategy.generate_signals(prices)
        assert _count_direction(signals, SignalDirection.LONG) >= 1

    def test_no_signals_before_warmup(self, default_strategy: SMACrossoverStrategy) -> None:
        prices = _make_prices([100.0] * 4)
        signals = default_strategy.generate_signals(prices)
        assert _count_direction(signals, SignalDirection.LONG) == 0
        assert _count_direction(signals, SignalDirection.SHORT) == 0

    def test_no_lookahead_bias(self, default_strategy: SMACrossoverStrategy) -> None:
        full_prices = _make_prices([100.0] * 5 + [120.0] * 10 + [80.0] * 10)
        truncated = full_prices.iloc[:10]
        full_sigs = default_strategy.generate_signals(full_prices)
        trunc_sigs = default_strategy.generate_signals(truncated)
        full_dirs = _get_directions(full_sigs)
        trunc_dirs = _get_directions(trunc_sigs)
        for i in range(len(truncated)):
            assert full_dirs[i] == trunc_dirs[i]

    def test_output_length_matches_input(self, default_strategy: SMACrossoverStrategy) -> None:
        prices = _make_prices([100.0] * 30)
        signals = default_strategy.generate_signals(prices)
        assert len(signals.index) == 30

    @given(
        n=st.integers(min_value=30, max_value=100),
        fast=st.integers(min_value=2, max_value=10),
        slow_offset=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50)
    def test_output_length_always_matches_input(self, n: int, fast: int, slow_offset: int) -> None:
        slow = fast + slow_offset
        strategy = SMACrossoverStrategy(fast_window=fast, slow_window=slow)
        prices = _make_prices([100.0 + i * 0.1 for i in range(n)])
        signals = strategy.generate_signals(prices)
        assert len(signals.index) == n


class TestMomentumStrategy:
    @pytest.fixture
    def strategy(self) -> MomentumStrategy:
        return MomentumStrategy(lookback=30, skip=5, min_strength=0.01)

    def test_skip_gte_lookback_raises(self) -> None:
        with pytest.raises(ValueError, match="skip must be less than lookback"):
            MomentumStrategy(lookback=20, skip=20)

    def test_positive_momentum_yields_long(self, strategy: MomentumStrategy) -> None:
        rising = [100.0 * (1.005**i) for i in range(60)]
        prices = _make_prices(rising)
        signals = strategy.generate_signals(prices)
        assert _count_direction(signals, SignalDirection.LONG) >= 1

    def test_strength_in_valid_range(self, strategy: MomentumStrategy) -> None:
        rng = [100.0 + float(i) * 0.1 for i in range(60)]
        prices = _make_prices(rng)
        signals = strategy.generate_signals(prices)
        for s in signals:
            if isinstance(s, Signal):
                assert 0.0 <= s.strength <= 1.0

    def test_name_includes_parameters(self, strategy: MomentumStrategy) -> None:
        assert "30" in strategy.name
	assert "5" in strategy.name


class TestMeanReversionStrategy:
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
        prices = _make_prices([100.0] * 20 + [300.0])
        signals = strategy.generate_signals(prices)
        last = signals.iloc[-1]
        assert isinstance(last, Signal)
        assert last.direction == SignalDirection.SHORT

    def test_zscore_stored_in_metadata(self, strategy: MeanReversionStrategy) -> None:
        prices = _make_prices([100.0] * 20 + [300.0])
        signals = strategy.generate_signals(prices)
        last = signals.iloc[-1]
        assert isinstance(last, Signal)
        assert "z_score" in last.metadata

    def test_flat_signal_near_mean(self, strategy: MeanReversionStrategy) -> None:
        prices = _make_prices([100.0] * 15 + [300.0] + [100.0] * 5)
        signals = strategy.generate_signals(prices)
        assert _count_direction(signals, SignalDirection.FLAT) >= 1

    def test_name_includes_window(self, strategy: MeanReversionStrategy) -> None:
        assert "10" in strategy.name
