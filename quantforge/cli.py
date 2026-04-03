"""Command-line interface for QuantForge.

Provides a ``quantforge`` command with sub-commands for running backtests,
validating configuration, and displaying version information.

Usage::

    quantforge --help
    quantforge backtest --strategy sma --fast 20 --slow 50 --capital 100000
    quantforge version
"""

from __future__ import annotations

import sys

import click
from rich.console import Console

console = Console()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
    help="Logging verbosity level.",
)
@click.option(
    "--json-logs",
    is_flag=True,
    default=False,
    help="Emit structured JSON logs (for production / CI environments).",
)
def main(log_level: str, json_logs: bool) -> None:
    """QuantForge — algorithmic trading and backtesting toolkit."""
    from quantforge.utils.logging import configure_logging

    configure_logging(level=log_level, json_output=json_logs)


@main.command()
def version() -> None:
    """Display the installed QuantForge version."""
    from quantforge import __version__

    console.print(f"QuantForge [bold blue]v{__version__}[/bold blue]")


@main.command()
@click.option(
    "--strategy",
    "strategy_name",
    type=click.Choice(["sma", "momentum", "mean-reversion"], case_sensitive=False),
    default="sma",
    show_default=True,
    help="Built-in strategy to backtest.",
)
@click.option("--fast", default=20, show_default=True, help="Fast SMA window (sma strategy).")
@click.option("--slow", default=50, show_default=True, help="Slow SMA window (sma strategy).")
@click.option("--capital", default=100_000.0, show_default=True, help="Starting capital.")
@click.option("--bars", default=500, show_default=True, help="Number of synthetic price bars.")
@click.option(
    "--seed",
    default=42,
    show_default=True,
    help="Random seed for synthetic data generation.",
)
def backtest(
    strategy_name: str,
    fast: int,
    slow: int,
    capital: float,
    bars: int,
    seed: int,
) -> None:
    """Run a backtest on synthetic price data and print a tear-sheet.

    This command is intended for quick demonstrations and smoke-testing.
    For production use, feed your own OHLCV DataFrame into ``Backtester.run()``.
    """
    import numpy as np
    import pandas as pd

    from quantforge import Backtester, Portfolio
    from quantforge.reporting.tearsheet import print_tearsheet
    from quantforge.strategies import (
        MeanReversionStrategy,
        MomentumStrategy,
        SMACrossoverStrategy,
    )

    # Generate synthetic price data
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0003, 0.012, bars)
    close = 100.0 * np.cumprod(1 + log_returns)
    open_ = close * (1 + rng.uniform(-0.003, 0.003, bars))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.007, bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.007, bars))
    prices = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1_000_000},
        index=pd.date_range("2020-01-02", periods=bars, freq="B"),
    )

    # Select strategy
    strategy_map = {
        "sma": lambda: SMACrossoverStrategy(fast_window=fast, slow_window=slow),
        "momentum": lambda: MomentumStrategy(lookback=min(bars // 2, 252), skip=21),
        "mean-reversion": lambda: MeanReversionStrategy(),
    }
    strategy = strategy_map[strategy_name.lower()]()

    console.print(
        f"\nRunning [bold]{strategy.name}[/bold] on {bars} bars of synthetic data…"
    )

    result = Backtester(
        portfolio=Portfolio(initial_capital=capital),
        strategy=strategy,
    ).run(prices)

    print_tearsheet(result)

    # Exit with non-zero code if the strategy lost money (useful in CI)
    if result.metrics.total_return < -0.5:
        console.print("[red]Warning: strategy lost more than 50% of capital.[/red]")
        sys.exit(1)
