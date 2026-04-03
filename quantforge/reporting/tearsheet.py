"""Performance reporting and tear-sheet generation.

Produces human-readable, colour-coded performance summaries in the terminal
using the ``rich`` library.  All public functions accept a ``BacktestResult``
and write to stdout (or a provided console).

Example::

    from quantforge.reporting import print_tearsheet
    print_tearsheet(result)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from quantforge.backtest.engine import BacktestResult

_console = Console()


def print_tearsheet(result: BacktestResult, console: Console | None = None) -> None:
    """Print a full performance tear-sheet to the terminal.

    Displays:
        * Strategy name and date range.
        * Return metrics (total, annualised, volatility, Sharpe, Sortino, Calmar).
        * Risk metrics (max drawdown, VaR, CVaR).
        * Trade-level statistics (win rate, profit factor, avg win/loss).

    Args:
        result:  A completed ``BacktestResult`` from ``Backtester.run()``.
        console: Optional Rich ``Console`` instance; defaults to stdout.
    """
    c = console or _console
    m = result.metrics

    # ---------------------------------------------------------------------------
    # Return metrics table
    # ---------------------------------------------------------------------------
    returns_table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        show_edge=False,
        padding=(0, 2),
    )
    returns_table.add_column("Metric", style="dim", no_wrap=True)
    returns_table.add_column("Value", justify="right")

    def _pct(v: float, decimals: int = 2) -> str:
        colour = "green" if v >= 0 else "red"
        return f"[{colour}]{v * 100:+.{decimals}f}%[/{colour}]"

    def _float(v: float, decimals: int = 2) -> str:
        colour = "green" if v >= 0 else "red"
        return f"[{colour}]{v:+.{decimals}f}[/{colour}]"

    returns_table.add_row("Total return", _pct(m.total_return))
    returns_table.add_row("Annualised return", _pct(m.annualised_return))
    returns_table.add_row("Annualised volatility", _pct(m.annualised_volatility))
    returns_table.add_row("Sharpe ratio", _float(m.sharpe_ratio))
    returns_table.add_row("Sortino ratio", _float(m.sortino_ratio))
    returns_table.add_row("Calmar ratio", _float(m.calmar_ratio))

    # ---------------------------------------------------------------------------
    # Risk metrics table
    # ---------------------------------------------------------------------------
    risk_table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        show_edge=False,
        padding=(0, 2),
    )
    risk_table.add_column("Metric", style="dim", no_wrap=True)
    risk_table.add_column("Value", justify="right")

    risk_table.add_row("Max drawdown", _pct(m.max_drawdown))
    risk_table.add_row("Max drawdown duration", f"{m.max_drawdown_duration} bars")
    risk_table.add_row("VaR (95%, 1-day)", _pct(m.var_95))
    risk_table.add_row("CVaR (95%, 1-day)", _pct(m.cvar_95))

    # ---------------------------------------------------------------------------
    # Trade statistics table
    # ---------------------------------------------------------------------------
    trades_table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        show_edge=False,
        padding=(0, 2),
    )
    trades_table.add_column("Metric", style="dim", no_wrap=True)
    trades_table.add_column("Value", justify="right")

    trades_table.add_row("Total trades", str(m.total_trades))
    trades_table.add_row("Win rate", f"{m.win_rate * 100:.1f}%")
    trades_table.add_row("Profit factor", f"{m.profit_factor:.2f}")
    trades_table.add_row("Avg winning trade", _float(m.avg_win))
    trades_table.add_row("Avg losing trade", _float(m.avg_loss))

    # ---------------------------------------------------------------------------
    # Render
    # ---------------------------------------------------------------------------
    c.print()
    c.print(
        Panel(
            f"[bold]{result.strategy_name}[/bold]",
            subtitle="QuantForge Performance Report",
            border_style="blue",
        )
    )
    c.print("\n[bold]Returns[/bold]")
    c.print(returns_table)
    c.print("[bold]Risk[/bold]")
    c.print(risk_table)
    c.print("[bold]Trades[/bold]")
    c.print(trades_table)
    c.print()
