"""Strategy abstractions and built-in implementations.

A ``Strategy`` in QuantForge is any object that implements the ``Strategy``
protocol — duck typing means no inheritance is required.  This keeps the API
open for extension without modification (OCP).

Built-in strategies:
    * ``SMACrossoverStrategy`` — classic fast/slow simple moving average crossover.
    * ``MomentumStrategy``     — cross-sectional 12-1 month momentum.
    * ``MeanReversionStrategy``— z-score-based mean-reversion on rolling windows.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from quantforge.core.models import Signal, SignalDirection

# ---------------------------------------------------------------------------
# Protocol definition (structural subtyping)
# ---------------------------------------------------------------------------


@runtime_checkable
class Strategy(Protocol):
    """The interface every QuantForge strategy must satisfy.

    Any class that implements ``generate_signals`` with this signature is
    automatically a valid ``Strategy`` — no base class required.

    The method must be *pure*: it must not mutate state or perform I/O.
    Strategies that require historical context should cache it in ``__init__``.
    """

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate a signal for each timestamp in ``prices``.

        Args:
            prices: A DataFrame with a DatetimeIndex and at minimum a ``"close"``
                    column.  Additional columns (``"open"``, ``"high"``, ``"low"``,
                    ``"volume"``) may be used by strategies that need them.

        Returns:
            A ``pd.Series`` of ``Signal`` objects with the same index as ``prices``.
            Entries without a signal should be ``None``.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """A short, human-readable identifier used in reports and logs."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_signal_series(
    mask_long: pd.Series,
    mask_short: pd.Series,
    mask_flat: pd.Series | None = None,
) -> pd.Series:
    """Convert boolean masks into a Series of Signal objects.

    Args:
        mask_long:  Boolean Series; True where a LONG signal should fire.
        mask_short: Boolean Series; True where a SHORT signal should fire.
        mask_flat:  Boolean Series; True where a FLAT (exit) signal should fire.

    Returns:
        Series of ``Signal | None`` aligned to the input index.
    """
    result = pd.Series(index=mask_long.index, dtype=object)

    result[mask_long] = Signal(direction=SignalDirection.LONG)
    result[mask_short] = Signal(direction=SignalDirection.SHORT)
    if mask_flat is not None:
        result[mask_flat] = Signal(direction=SignalDirection.FLAT)

    return result


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


@dataclass
class SMACrossoverStrategy:
    """Simple moving average crossover strategy.

    Generates a LONG signal when the fast SMA crosses *above* the slow SMA,
    a SHORT signal on a cross *below*, and no signal otherwise.

    This is the archetypal trend-following strategy — straightforward to
    implement, test, and explain, while still capturing real market dynamics.

    Args:
        fast_window: Lookback period for the faster (more reactive) SMA.
        slow_window: Lookback period for the slower (trend-confirming) SMA.

    Example:
        >>> strategy = SMACrossoverStrategy(fast_window=20, slow_window=50)
        >>> signals = strategy.generate_signals(price_df)
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

        Uses a shift(1) to avoid look-ahead bias: the signal for bar *t* is
        derived from data available at the *close* of bar *t-1*.

        Args:
            prices: OHLCV DataFrame with at minimum a ``"close"`` column.

        Returns:
            Series of ``Signal | None`` indexed identically to ``prices``.
        """
        close = prices["close"].astype(float)

        fast_ma = close.rolling(window=self.fast_window, min_periods=self.fast_window).mean()
        slow_ma = close.rolling(window=self.slow_window, min_periods=self.slow_window).mean()

        # Golden cross / death cross detected on the *previous* bar
        prev_fast = fast_ma.shift(1)
        prev_slow = slow_ma.shift(1)

        golden_cross = (fast_ma > slow_ma) & (prev_fast <= prev_slow)
        death_cross = (fast_ma < slow_ma) & (prev_fast >= prev_slow)

        return _to_signal_series(
            mask_long=golden_cross,
            mask_short=death_cross,
        )


@dataclass
class MomentumStrategy:
    """Cross-sectional price momentum strategy.

    Measures the return over a ``lookback`` period, skipping the most recent
    ``skip`` bars to avoid the well-documented short-term reversal effect.

    A positive momentum score generates a LONG signal; a negative score a
    SHORT signal.  Scores below ``min_strength`` are treated as noise and
    suppressed.

    Args:
        lookback:     Total lookback window in bars (default 252 ≈ 1 year daily).
        skip:         Bars to skip at the end of the window (default 21 ≈ 1 month).
        min_strength: Minimum absolute return to emit a signal (default 0.02).

    References:
        Jegadeesh & Titman (1993). "Returns to Buying Winners and Selling Losers".
        Journal of Finance 48(1): 65-91.
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
        return f"Momentum({self.lookback}-{self.skip})"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate momentum signals.

        Args:
            prices: OHLCV DataFrame with a ``"close"`` column.

        Returns:
            Series of ``Signal | None`` carrying directional bias and a
            strength score normalised to [0, 1].
        """
        close = prices["close"].astype(float)
        returns = close.pct_change(self._effective_window).shift(self.skip)

        # Clip to [-1, 1] before normalising to avoid distortion from extreme events
        clipped = returns.clip(-1.0, 1.0)
        strength = (clipped.abs() - self.min_strength).clip(lower=0.0)
        # Normalise: max possible strength after clipping is (1 - min_strength)
        max_s = 1.0 - self.min_strength
        normalised = (strength / max_s).clip(0.0, 1.0)

        def _to_signal(row: tuple[float, float]) -> Signal | None:
            ret, s = row
            if abs(ret) < self.min_strength or np.isnan(ret):
                return None
            direction = SignalDirection.LONG if ret > 0 else SignalDirection.SHORT
            return Signal(direction=direction, strength=float(s))

        combined = pd.DataFrame({"ret": returns, "strength": normalised})
        return combined.apply(lambda r: _to_signal((r["ret"], r["strength"])), axis=1)


@dataclass
class MeanReversionStrategy:
    """Z-score based mean-reversion strategy.

    Computes a rolling z-score of price relative to its rolling mean and
    standard deviation.  Positions are entered when the z-score exceeds
    ``entry_z`` and exited when it reverts to within ``exit_z`` of the mean.

    Args:
        window:   Rolling window for mean and std calculation (default 20 bars).
        entry_z:  Z-score threshold to enter a trade (default ±2.0).
        exit_z:   Z-score threshold to exit a trade (default ±0.5).

    Signals:
        * z > +entry_z → SHORT (overextended to the upside)
        * z < -entry_z → LONG (overextended to the downside)
        * |z| < exit_z  → FLAT (mean has been restored)
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
        return f"MeanReversion(w={self.window},ez={self.entry_z})"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate mean-reversion signals based on rolling z-scores.

        Args:
            prices: OHLCV DataFrame with a ``"close"`` column.

        Returns:
            Series of ``Signal | None`` with the z-score stored in
            ``signal.metadata["z_score"]`` for diagnostic use.
        """
        close = prices["close"].astype(float)
        rolling_mean = close.rolling(window=self.window).mean()
        rolling_std = close.rolling(window=self.window).std()

        # Avoid division by zero during flat markets
        z_score = (close - rolling_mean) / rolling_std.replace(0.0, np.nan)

        def _signal_for(z: float) -> Signal | None:
            if np.isnan(z):
                return None
            meta = {"z_score": round(z, 4)}
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

        return z_score.map(_signal_for)
