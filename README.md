# QuantForge

[![CI](https://github.com/yourusername/quantforge/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/quantforge/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yourusername/quantforge/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/quantforge)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional-grade algorithmic trading and financial data analysis library built in Python.
QuantForge provides a clean, extensible framework for building, testing, and evaluating quantitative
trading strategies with rigorous statistical validation.

## ✨ Features

- **Pluggable strategy architecture** — define strategies via a clean `Strategy` protocol; swap them at runtime
- **Vectorised backtesting engine** — NumPy-accelerated simulation with realistic cost modelling (slippage, commissions, shorting fees)
- **Rich risk analytics** — Sharpe, Sortino, Calmar, max drawdown, VaR, CVaR, and more out of the box
- **Async data pipeline** — concurrent market data fetching with automatic retry and exponential back-off
- **Property-based testing** — Hypothesis-driven invariant tests ensure correctness across thousands of edge cases
- **Fully typed** — 100 % mypy-strict; every public API carries complete type annotations
- **Structured logging** — machine-readable JSON logs via `structlog` for production observability

## 🏗️ Architecture

```
quantforge/
├── core/           # Domain models: Instrument, Portfolio, Position, Trade
├── data/           # Market data abstractions and provider implementations
├── strategies/     # Strategy protocol + built-in implementations (SMA, momentum, mean-reversion)
├── backtest/       # Simulation engine, order matching, cost models
├── reporting/      # Performance analytics, tear-sheet generation
└── utils/          # Logging, configuration, financial math helpers
```

## 🚀 Quick Start

```bash
pip install quantforge
```

```python
from quantforge import Backtester, Portfolio
from quantforge.data import YahooDataProvider
from quantforge.strategies import SMACrossoverStrategy

# Fetch 3 years of daily OHLCV data
provider = YahooDataProvider()
prices = await provider.fetch("AAPL", start="2021-01-01", end="2023-12-31")

# Configure and run a backtest
strategy = SMACrossoverStrategy(fast_window=20, slow_window=50)
portfolio = Portfolio(initial_capital=100_000.0, currency="USD")

result = Backtester(portfolio=portfolio, strategy=strategy).run(prices)

# Print a performance summary
print(result.metrics)
# PerformanceMetrics(
#   total_return=0.342,
#   sharpe_ratio=1.47,
#   max_drawdown=-0.183,
#   win_rate=0.54,
#   ...
# )
```

## 📦 Installation

**Requires Python ≥ 3.11.**

```bash
# From PyPI (stable)
pip install quantforge

# With development extras
pip install "quantforge[dev]"

# From source
git clone https://github.com/yourusername/quantforge.git
cd quantforge
pip install -e ".[dev]"
pre-commit install
```

## 🧪 Running Tests

```bash
# Full test suite with coverage
pytest

# Unit tests only (fast)
pytest -m unit

# Integration tests (requires network)
pytest -m integration

# Run benchmarks
pytest --benchmark-only
```

## 🔧 Development

This project follows strict coding standards enforced automatically:

| Tool | Purpose |
|------|---------|
| `ruff` | Linting + formatting (replaces flake8, black, isort) |
| `mypy --strict` | Static type checking |
| `pytest + hypothesis` | Unit, integration, and property-based testing |
| `pre-commit` | Git hooks that enforce quality gates locally |
| GitHub Actions | Full CI on every push and pull request |

```bash
# Format and lint
ruff format .
ruff check . --fix

# Type check
mypy quantforge

# All quality checks at once
pre-commit run --all-files
```

## 📊 Strategy Development

Implement the `Strategy` protocol to create a custom strategy in minutes:

```python
from quantforge.core.models import Signal, SignalDirection
from quantforge.strategies.base import Strategy
import pandas as pd

class MyMomentumStrategy(Strategy):
    """A simple 12-1 month momentum strategy."""

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        self.lookback = lookback
        self.skip = skip

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Return a Series of Signal objects indexed by timestamp."""
        momentum = prices["close"].pct_change(self.lookback - self.skip)
        return momentum.map(
            lambda r: Signal(
                direction=SignalDirection.LONG if r > 0 else SignalDirection.SHORT,
                strength=abs(r),
            )
        )
```

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
