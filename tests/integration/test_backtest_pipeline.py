"""Integration tests for the full backtesting pipeline.

These tests exercise the complete path from strategy → backtester → result,
using synthetic price data designed to trigger known behaviours.

Marked as ``integration`` so they can be excluded from fast unit-test runs:
    pytest -m "not integration"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from quantforge.backtest.engine import Backtester, BacktestMetrics, CostModel
from quantforge.core.portfolio import Portfolio
from quantforge.strategies.base import (
    MeanReversionStrategy,
    MomentumStrategy,
    SMACrossoverStrategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_trending(
    n: int = 500,
    drift: float = 0.0005,
    volatility: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a positive drift."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, volatility, n)
    close = 100.0 * np.cumprod(1 + log_returns)
    open_ = close * (1 + rng.uniform(-0.002, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1_000_000},
        index=pd.date_range("2020-01-02", periods=n, freq="B"),
    )


def _synthetic_mean_reverting(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Generate synthetic OHLCV data that oscillates around 100."""
    rng = np.random.default_rng(seed)
    price = 100.0
    closes = []
    for _ in range(n):
        price += rng.normal(0, 2.0) - 0.1 * (price - 100)
        closes.append(max(price, 1.0))
    close = np.array(closes)
    open_ = close * (1 + rng.uniform(-0.002, 0.002, n))
    high = np.maximum(open_, close) + rng.uniform(0, 1, n)
    low = np.minimum(open_, close) - rng.uniform(0, 1, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 500_000},
        index=pd.date_range("2020-01-02", periods=n, freq="B"),
    )


# ---------------------------------------------------------------------------
# Full pipeline smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBacktesterSmokeTests:
    """Smoke tests: verify the pipeline completes without error."""

    @pytest.mark.parametrize(
        "strategy_cls, kwargs",
        [
            (SMACrossoverStrategy, {"fast_window": 20, "slow_window": 50}),
            (MomentumStrategy, {"lookback": 60, "skip": 10, "min_strength": 0.01}),
            (MeanReversionStrategy, {"window": 20, "entry_z": 1.5, "exit_z": 0.3}),
        ],
    )
    def test_all_built_in_strategies_complete(self, strategy_cls: type, kwargs: dict) -> None:
        prices = _synthetic_trending(n=300)
        portfolio = Portfolio(initial_capital=100_000.0)
        strategy = strategy_cls(**kwargs)
        bt = Backtester(portfolio=portfolio, strategy=strategy)
        result = bt.run(prices)

        assert result is not None
        assert len(result.equity_curve) > 0

    def test_result_has_all_metric_fields(self) -> None:
        prices = _synthetic_trending(n=200)
        portfolio = Portfolio(initial_capital=50_000.0)
        bt = Backtester(portfolio=portfolio, strategy=SMACrossoverStrategy())
        result = bt.run(prices)
        m = result.metrics

        assert isinstance(m, BacktestMetrics)
        # All numeric fields should be finite
        assert np.isfinite(m.total_return)
        assert np.isfinite(m.sharpe_ratio)
        assert np.isfinite(m.max_drawdown)
        assert np.isfinite(m.win_rate)

    def test_equity_never_negative(self) -> None:
        """Portfolio equity should never go below zero (no leverage)."""
        prices = _synthetic_trending(n=250)
        portfolio = Portfolio(initial_capital=10_000.0)
        bt = Backtester(portfolio=portfolio, strategy=SMACrossoverStrategy(10, 30))
        result = bt.run(prices)

        assert (result.equity_curve["equity"] >= 0).all()

    def test_returns_series_has_no_inf(self) -> None:
        prices = _synthetic_trending(n=200)
        bt = Backtester(
            portfolio=Portfolio(initial_capital=100_000.0),
            strategy=SMACrossoverStrategy(),
        )
        result = bt.run(prices)
        returns = result.equity_curve["returns"].dropna()
        assert not np.isinf(returns).any()


# ---------------------------------------------------------------------------
# Cost model integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCostModelIntegration:
    """Verify that non-zero costs reduce returns relative to zero-cost baseline."""

    def test_costs_reduce_returns(self) -> None:
        prices = _synthetic_trending(n=200, drift=0.001)  # Strong trend
        strategy = SMACrossoverStrategy(fast_window=10, slow_window=30)

        free_bt = Backtester(
            portfolio=Portfolio(100_000.0),
            strategy=strategy,
            cost_model=CostModel(commission_bps=0.0, slippage_bps=0.0, min_commission=0.0),
        )
        costly_bt = Backtester(
            portfolio=Portfolio(100_000.0),
            strategy=strategy,
            cost_model=CostModel(commission_bps=20.0, slippage_bps=20.0),
        )

        free_result = free_bt.run(prices)
        costly_result = costly_bt.run(prices)

        assert free_result.metrics.total_return >= costly_result.metrics.total_return


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_backtest_is_deterministic() -> None:
    """Running the same backtest twice must produce identical results."""
    prices = _synthetic_trending(seed=99)
    strategy = SMACrossoverStrategy(fast_window=15, slow_window=40)

    r1 = Backtester(Portfolio(100_000.0), strategy).run(prices)
    r2 = Backtester(Portfolio(100_000.0), strategy).run(prices)

    assert r1.metrics.total_return == pytest.approx(r2.metrics.total_return, rel=1e-9)
    assert r1.metrics.sharpe_ratio == pytest.approx(r2.metrics.sharpe_ratio, rel=1e-9)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEdgeCases:
    """Boundary conditions for the backtesting engine."""

    def test_empty_prices_raises(self) -> None:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with pytest.raises(ValueError, match="empty"):
            Backtester(Portfolio(100_000.0), SMACrossoverStrategy()).run(empty)

    def test_missing_close_column_raises(self) -> None:
        prices = pd.DataFrame({"open": [100.0, 101.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            Backtester(Portfolio(100_000.0), SMACrossoverStrategy()).run(prices)

    def test_unsorted_index_raises(self) -> None:
        prices = _synthetic_trending(n=50).iloc[::-1]  # Reverse → unsorted
        with pytest.raises(ValueError, match="sorted"):
            Backtester(Portfolio(100_000.0), SMACrossoverStrategy()).run(prices)

    def test_single_bar_does_not_crash(self) -> None:
        """A one-bar price series should not crash; it just produces no trades."""
        prices = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1, freq="D"),
        )
        result = Backtester(Portfolio(10_000.0), SMACrossoverStrategy()).run(prices)
        assert result.metrics.total_trades == 0
