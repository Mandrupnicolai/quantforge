"""Backtesting engine — simulates strategy execution on historical data.

The ``Backtester`` class orchestrates the full simulation loop:

    1. Iterate over OHLCV bars in chronological order.
    2. Ask the strategy for a signal at each bar.
    3. If a signal is present, size the trade via the configured sizer.
    4. Execute the order through the cost model (slippage + commission).
    5. Update portfolio state and snapshot the equity curve.
    6. Return a ``BacktestResult`` with the complete performance record.

Design decisions:
    * **Vectorised signal generation** — signals for all bars are computed in one
      pass *before* the simulation loop.  This leverages NumPy's vectorised ops
      and avoids quadratic look-ahead in rolling calculations.
    * **Bar-close execution** — orders are filled at the *next* bar's open to
      simulate realistic execution (no same-bar fills at the signal price).
    * **No implicit state** — the backtester does not mutate the ``Portfolio``
      passed to it; it works on a deep copy so the caller's object is unchanged.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import structlog

from quantforge.core.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Signal,
    SignalDirection,
)
from quantforge.core.portfolio import Portfolio

if TYPE_CHECKING:
    from quantforge.strategies.base import Strategy

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Configures the transaction cost assumptions for a backtest.

    Args:
        commission_per_share:  Fixed cost per unit traded (e.g. £0.005/share).
        commission_bps:        Variable commission as basis points of notional.
        slippage_bps:          One-way market impact / spread estimate in bps.
        min_commission:        Floor on commission per order (default £1.00).
    """

    commission_per_share: float = 0.0
    commission_bps: float = 5.0       # 0.05 % — realistic for retail brokers
    slippage_bps: float = 5.0         # 0.05 % one-way
    min_commission: float = 1.0

    def compute(
        self,
        quantity: Decimal,
        price: Decimal,
    ) -> tuple[Decimal, Decimal]:
        """Return (commission, slippage) for a given order.

        Args:
            quantity: Number of units traded.
            price:    Execution price before slippage.

        Returns:
            A tuple of ``(commission, slippage)`` as Decimals.
        """
        notional = float(quantity * price)
        commission = max(
            self.min_commission,
            float(quantity) * self.commission_per_share
            + notional * self.commission_bps / 10_000,
        )
        slippage = notional * self.slippage_bps / 10_000
        return Decimal(str(round(commission, 6))), Decimal(str(round(slippage, 6)))


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedFractionSizer:
    """Sizes each trade as a fixed fraction of current portfolio equity.

    This is the Kelly-adjacent approach: risk a constant proportion of
    available capital on each trade, so position sizes scale as the
    portfolio grows or shrinks.

    Args:
        fraction: Proportion of equity to commit per trade (default 0.02 = 2 %).
    """

    fraction: float = 0.02

    def __post_init__(self) -> None:
        if not 0 < self.fraction <= 1.0:
            msg = "fraction must be in (0, 1]"
            raise ValueError(msg)

    def compute_quantity(
        self,
        equity: Decimal,
        price: Decimal,
        signal: Signal,
    ) -> Decimal:
        """Return the number of shares/units to trade.

        Args:
            equity:  Current portfolio equity.
            price:   Execution price estimate.
            signal:  The strategy signal (strength scales the allocation).

        Returns:
            Integer number of shares as a Decimal.
        """
        notional = float(equity) * self.fraction * signal.strength
        raw_qty = notional / float(price)
        return Decimal(str(max(1, int(raw_qty))))  # Minimum 1 unit


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """The complete output of a backtest run.

    Attributes:
        strategy_name: Name of the strategy that was tested.
        portfolio:     Final portfolio state after simulation.
        equity_curve:  DataFrame indexed by date with an ``"equity"`` column.
        trades:        List of all completed trades.
        metrics:       Computed performance metrics (lazy — see ``BacktestMetrics``).
    """

    strategy_name: str
    portfolio: Portfolio
    equity_curve: pd.DataFrame
    metrics: BacktestMetrics = field(default_factory=lambda: BacktestMetrics())  # Computed post-init

    @property
    def total_return(self) -> float:
        """Total return as a fraction (e.g. 0.34 = +34 %)."""
        return self.metrics.total_return

    def __repr__(self) -> str:
        return (
            f"BacktestResult(strategy={self.strategy_name!r}, "
            f"total_return={self.total_return:.2%}, "
            f"trades={len(self.portfolio.trades)})"
        )


@dataclass
class BacktestMetrics:
    """Computed performance statistics for a completed backtest.

    All risk/return metrics assume daily bars unless otherwise noted.
    """

    total_return: float = 0.0
    annualised_return: float = 0.0
    annualised_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # Calendar days
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    var_95: float = 0.0   # 1-day 95 % Value at Risk
    cvar_95: float = 0.0  # 1-day 95 % Conditional VaR (Expected Shortfall)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------


class Backtester:
    """Simulates a strategy over historical price data.

    Args:
        portfolio:  The portfolio to simulate into (a deep copy is made internally).
        strategy:   Any object satisfying the ``Strategy`` protocol.
        cost_model: Transaction cost assumptions.
        sizer:      Position-sizing model.
        risk_free:  Annual risk-free rate for Sharpe/Sortino calculation (default 0.05).

    Example:
        >>> bt = Backtester(portfolio=Portfolio(100_000), strategy=SMACrossoverStrategy())
        >>> result = bt.run(prices_df)
        >>> print(result.metrics.sharpe_ratio)
    """

    def __init__(
        self,
        portfolio: Portfolio,
        strategy: Strategy,
        cost_model: CostModel | None = None,
        sizer: FixedFractionSizer | None = None,
        risk_free: float = 0.05,
    ) -> None:
        self._portfolio_template = portfolio
        self.strategy = strategy
        self.cost_model = cost_model or CostModel()
        self.sizer = sizer or FixedFractionSizer()
        self.risk_free = risk_free

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        """Execute the full backtest simulation.

        Args:
            prices: OHLCV DataFrame with a DatetimeIndex (or date index) and
                    at minimum ``"open"`` and ``"close"`` columns.

        Returns:
            A populated ``BacktestResult`` with equity curve and metrics.

        Raises:
            ValueError: If ``prices`` is empty or missing required columns.
        """
        self._validate_prices(prices)
        portfolio = copy.deepcopy(self._portfolio_template)

        log.info(
            "backtest_start",
            strategy=self.strategy.name,
            bars=len(prices),
            start=str(prices.index[0]),
            end=str(prices.index[-1]),
        )

        # Generate all signals in one vectorised pass (no look-ahead possible
        # because we only *use* signal[t] when processing bar t+1).
        signals: pd.Series = self.strategy.generate_signals(prices)

        pending_orders: list[Order] = []

        for i, (timestamp, row) in enumerate(prices.iterrows()):
            bar_date: date = (
                timestamp.date() if hasattr(timestamp, "date") else timestamp
            )
            open_price = Decimal(str(row["open"]))
            close_price = Decimal(str(row["close"]))
            mark_prices = {"default": close_price}  # Simplified: single-asset

            # --- Execute any pending orders at today's open ---
            for order in list(pending_orders):
                self._execute_order(order, open_price, bar_date, portfolio)
                pending_orders.remove(order)

            # --- Process signal from the *previous* bar (avoid look-ahead) ---
            if i > 0:
                prev_signal: Signal | None = signals.iloc[i - 1]
                if prev_signal is not None:
                    order = self._signal_to_order(
                        prev_signal,
                        price_estimate=close_price,
                        portfolio=portfolio,
                        instrument_symbol="default",
                    )
                    if order is not None:
                        pending_orders.append(order)

            # --- Snapshot equity at close ---
            symbol = "default"
            if symbol in portfolio.positions:
                mark = {symbol: close_price}
            else:
                mark = {}
            portfolio.snapshot_equity(bar_date, mark)  # type: ignore[arg-type]

        equity_df = self._build_equity_df(portfolio, prices)
        metrics = self._compute_metrics(equity_df, portfolio)

        result = BacktestResult(
            strategy_name=self.strategy.name,
            portfolio=portfolio,
            equity_curve=equity_df,
            metrics=metrics,
        )

        log.info(
            "backtest_complete",
            strategy=self.strategy.name,
            total_return=f"{metrics.total_return:.2%}",
            sharpe=f"{metrics.sharpe_ratio:.2f}",
            max_drawdown=f"{metrics.max_drawdown:.2%}",
            trades=metrics.total_trades,
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_prices(self, prices: pd.DataFrame) -> None:
        """Raise ValueError if prices DataFrame is unsuitable."""
        if prices.empty:
            msg = "prices DataFrame is empty"
            raise ValueError(msg)
        required = {"open", "close"}
        missing = required - set(prices.columns)
        if missing:
            msg = f"prices is missing required columns: {missing}"
            raise ValueError(msg)
        if not prices.index.is_monotonic_increasing:
            msg = "prices index must be sorted in ascending chronological order"
            raise ValueError(msg)

    def _signal_to_order(
        self,
        signal: Signal,
        price_estimate: Decimal,
        portfolio: Portfolio,
        instrument_symbol: str,
    ) -> Order | None:
        """Convert a strategy signal into an Order, or None if sizing fails."""
        from quantforge.core.models import Instrument  # Local import avoids circular

        if signal.direction == SignalDirection.FLAT:
            # For exits, close the whole position
            pos = portfolio.positions.get(instrument_symbol)
            if pos is None or pos.is_flat:
                return None
            quantity = abs(pos.quantity)
            side = OrderSide.SELL if pos.is_long else OrderSide.BUY
        else:
            # Compute target equity snapshot
            equity = portfolio.cash  # Simplified — ignores unrealised PnL
            quantity = self.sizer.compute_quantity(equity, price_estimate, signal)
            if quantity <= 0:
                return None
            side = OrderSide.BUY if signal.direction == SignalDirection.LONG else OrderSide.SELL

        instrument = Instrument(symbol=instrument_symbol)
        return Order(
            order_id=str(uuid.uuid4()),
            instrument=instrument,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )

    def _execute_order(
        self,
        order: Order,
        open_price: Decimal,
        execution_date: date,
        portfolio: Portfolio,
    ) -> None:
        """Fill a market order at the open price plus slippage."""
        commission, slippage = self.cost_model.compute(order.quantity, open_price)
        fill_price = open_price  # Slippage is accounted for separately in cost model

        portfolio.apply_fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=slippage,
            execution_date=execution_date,
        )

    def _build_equity_df(
        self,
        portfolio: Portfolio,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assemble a clean equity curve DataFrame from portfolio snapshots."""
        if not portfolio.equity_curve:
            # Fallback: flat equity equal to initial capital
            equity_vals = [float(portfolio.initial_capital)] * len(prices)
            df = pd.DataFrame(
                {"equity": equity_vals},
                index=prices.index,
            )
        else:
            dates, equities = zip(*portfolio.equity_curve, strict=True)
            df = pd.DataFrame(
                {"equity": [float(e) for e in equities]},
                index=pd.DatetimeIndex(dates),
            )
        df["returns"] = df["equity"].pct_change()
        return df

    def _compute_metrics(
        self,
        equity_df: pd.DataFrame,
        portfolio: Portfolio,
    ) -> BacktestMetrics:
        """Derive all performance statistics from the equity curve and trade list."""
        returns = equity_df["returns"].dropna()
        equity = equity_df["equity"]

        initial = float(portfolio.initial_capital)
        final = equity.iloc[-1] if len(equity) > 0 else initial
        total_return = (final - initial) / initial

        trading_days = len(returns)
        years = trading_days / 252
        ann_return = (1 + total_return) ** (1 / max(years, 1e-9)) - 1
        ann_vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0

        daily_rf = (1 + self.risk_free) ** (1 / 252) - 1
        excess = returns - daily_rf
        sharpe = (
            float(excess.mean() / excess.std() * np.sqrt(252))
            if excess.std() > 0
            else 0.0
        )

        downside = returns[returns < 0]
        sortino = (
            float(excess.mean() / downside.std() * np.sqrt(252))
            if len(downside) > 0 and downside.std() > 0
            else 0.0
        )

        # Drawdown analysis
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = float(drawdown.min())
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        # Drawdown duration
        in_drawdown = drawdown < 0
        dd_durations = in_drawdown.astype(int).groupby((~in_drawdown).cumsum()).sum()
        max_dd_dur = int(dd_durations.max()) if len(dd_durations) > 0 else 0

        # Trade-level metrics
        trades = portfolio.trades
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        win_rate = len(winners) / max(len(trades), 1)
        avg_win = float(np.mean([float(t.pnl) for t in winners])) if winners else 0.0
        avg_loss = float(np.mean([float(t.pnl) for t in losers])) if losers else 0.0
        gross_profit = sum(float(t.pnl) for t in winners)
        gross_loss = abs(sum(float(t.pnl) for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Value at Risk (historical simulation)
        var_95 = float(np.percentile(returns, 5)) if len(returns) >= 20 else 0.0
        cvar_95 = float(returns[returns <= var_95].mean()) if len(returns) >= 20 else 0.0

        return BacktestMetrics(
            total_return=total_return,
            annualised_return=ann_return,
            annualised_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_dur,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=len(trades),
            var_95=var_95,
            cvar_95=cvar_95,
        )
