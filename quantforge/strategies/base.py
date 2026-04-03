"""Strategy abstractions and built-in implementations."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from quantforge.core.models import Signal, SignalDirection


@runtime_checkable
class Strategy(Protocol):
    """The interface every QuantForge strategy must satisfy."""

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate a signal for each timestamp in prices."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """A short human-readable identifier used in reports and logs."""
        ...


def _to_signal_series(
    mask_long: pd.Series,
    mask_short: pd.Series,
    mask_flat: pd.Series | None = None,
) -> pd.Series:
    """Convert boolean masks into a Series of Signal objects."""
    result = pd.Series([None] * len(mask_long), index=mask_long.index, dtype=object)
    for i, (_, val) in enumerate(mask_long.items()):
        if val:
            result.iloc[i] = Signal(direction=SignalDirection.LONG)
    for i, (_, val) in enumerate(mask_short.items()):
        if val:
            result.iloc[i] = Signal(direction=SignalDirection.SHORT)
    if mask_flat is not None:
        for i, (_, val) in enumerate(mask_flat.items()):
            if val:
                result.iloc[i] = Signal(direction=SignalDirection.FLAT)
    return result


@dataclass
class SMACrossoverStrategy:
    """Simple moving average crossover strategy.

    Generates a LONG signal when the fast SMA crosses above the slow SMA,
    a SHORT signal on a cross below, and no signal otherwise.

    Args:
        fast_window: Lookback period for the faster SMA.
        slow_window: Lookback period for the slower SMA.
    """

    fast_window: int = 20
    slow_window: int = 50

    def __post_init__(self) -> None:
        if self.fast_window >= self.slow_window:
            msg = (
                f"fast_window ({self.fast_window}) must be strictly less than "
                f"slow_window ({self.slow_window})"
            )
            raise ValueError(msg)
        if self.fast_window < 2:
            msg = "fast_window must be at least 2"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Return a descriptive name including window parameters."""
        return f"SMA({self.fast_window},{self.slow_window})"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Compute SMA crossover signals for each bar.

        Args:
            prices: OHLCV DataFrame with at minimum a close column.

        Returns:
            Series of Signal or None indexed identically to prices.
        """
        close = prices["close"].astype(float)

        fast_ma = close.rolling(window=self.fast_window, min_periods=self.fast_window).mean()
        slow_ma = close.rolling(window=self.slow_window, min_periods=self.slow_window).mean()

        prev_fast = fast_ma.shift(1)
        prev_slow = slow_ma.shift(1)

        golden_cross = (fast_ma > slow_ma) & (prev_fast <= prev_slow)
        death_cross = (fast_ma < slow_ma) & (prev_fast >= prev_slow)

        return _to_signal_series(mask_long=golden_cross, mask_short=death_cross)


@dataclass
class MomentumStrategy:
    """Cross-sectional price momentum strategy.

    Measures the return over a lookback period, skipping the most recent
    skip bars to avoid the short-term reversal effect.

    Args:
        lookback:     Total lookback window in bars (default 252).
        skip:         Bars to skip at the end of the window (default 21).
        min_strength: Minimum absolute return to emit a signal (default 0.02).
    """

    lookback: int = 252
    skip: int = 21
    min_strength: float = 0.02
    _effective_window: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.skip >= self.lookback:
            msg = "skip must be less than lookback"
            raise ValueError(msg)
        self._effective_window = self.lookback - self.skip

    @property
    def name(self) -> str:
        """Return strategy name with parameters."""
        return f"Momentum({self.lookback}-{self.skip})"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate momentum signals.

        Args:
            prices: OHLCV DataFrame with a close column.

        Returns:
            Series of Signal or None with strength scores normalised to 0-1.
        """
        close = prices["close"].astype(float)
        returns = close.pct_change(self._effective_window, fill_method=None).shift(self.skip)

        clipped = returns.clip(-1.0, 1.0)
        strength = (clipped.abs() - self.min_strength).clip(lower=0.0)
        max_s = 1.0 - self.min_strength
        normalised = (strength / max_s).clip(0.0, 1.0)

        def _to_signal(ret: float, s: float) -> Signal | None:
            if np.isnan(ret) or abs(ret) < self.min_strength:
                return None
            direction = SignalDirection.LONG if ret > 0 else SignalDirection.SHORT
            return Signal(direction=direction, strength=float(s))

        combined = pd.DataFrame({"ret": returns, "strength": normalised})
        result = pd.Series([None] * len(combined), index=combined.index, dtype=object)
        for i, (_, row) in enumerate(combined.iterrows()):
            result.iloc[i] = _to_signal(row["ret"], row["strength"])
        return result


@dataclass
class MeanReversionStrategy:
    """Z-score based mean-reversion strategy.

    Enters positions when the z-score exceeds entry_z and exits when
    it reverts to within exit_z of the mean.

    Args:
        window:   Rolling window for mean and std (default 20 bars).
        entry_z:  Z-score threshold to enter a trade (default 2.0).
        exit_z:   Z-score threshold to exit a trade (default 0.5).
    """

    window: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.5

    def __post_init__(self) -> None:
        if self.exit_z >= self.entry_z:
            msg = "exit_z must be less than entry_z"
            raise ValueError(msg)
        if self.window < 3:
            msg = "window must be at least 3"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Return strategy name with parameters."""
        return f"MeanReversion(w={self.window},ez={self.entry_z})"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate mean-reversion signals based on rolling z-scores.

        Args:
            prices: OHLCV DataFrame with a close column.

        Returns:
            Series of Signal or None with z_score in signal metadata.
        """
        close = prices["close"].astype(float)
        rolling_mean = close.rolling(window=self.window).mean()
        rolling_std = close.rolling(window=self.window).std()
        z_score = (close - rolling_mean) / rolling_std.replace(0.0, np.nan)

        def _signal_for(z: float) -> Signal | None:
            if np.isnan(z):
                return None
            meta: dict[str, float | str | bool] = {"z_score": round(z, 4)}
            if z > self.entry_z:
                return Signal(
                    direction=SignalDirection.SHORT,
                    strength=min((z - self.entry_z) / self.entry_z, 1.0),
                    metadata=meta,
                )
            if z < -self.entry_z:
                return Signal(
                    direction=SignalDirection.LONG,
                    strength=min((-z - self.entry_z) / self.entry_z, 1.0),
                    metadata=meta,
                )
            if abs(z) < self.exit_z:
                return Signal(direction=SignalDirection.FLAT, metadata=meta)
            return None

        result = pd.Series([None] * len(z_score), index=z_score.index, dtype=object)
        for i, val in enumerate(z_score):
            result.iloc[i] = _signal_for(float(val))
        return result
