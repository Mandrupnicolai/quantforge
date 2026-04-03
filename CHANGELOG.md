# Changelog

All notable changes to QuantForge are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- `YahooDataProvider` — async OHLCV fetcher backed by `yfinance`.
- `AlphaVantageProvider` — premium real-time and historical data.
- Walk-forward optimisation for robust parameter selection.
- Monte Carlo simulation for confidence intervals on backtest results.
- Sphinx documentation with ReadTheDocs hosting.
- Docker image for containerised execution.

---

## [0.1.0] — 2024-01-15

### Added
- Core domain models: `Instrument`, `OHLCV`, `Money`, `Signal`, `Order`, `Trade`.
- `Portfolio` class with position management, FIFO P&L accounting, and equity
  curve snapshots.
- `Strategy` protocol for structural subtyping — no inheritance required.
- Three built-in strategies:
  - `SMACrossoverStrategy` — fast/slow SMA golden/death cross.
  - `MomentumStrategy` — 12-1 month cross-sectional momentum.
  - `MeanReversionStrategy` — rolling z-score entry/exit bands.
- `Backtester` with bar-close signal generation and next-open execution to
  eliminate look-ahead bias.
- `CostModel` — configurable commission (per-share + bps) and slippage (bps).
- `FixedFractionSizer` — Kelly-inspired position sizing.
- `BacktestMetrics` — Sharpe, Sortino, Calmar, max drawdown (+ duration),
  VaR 95%, CVaR 95%, win rate, profit factor.
- Rich terminal tear-sheet via `quantforge.reporting.tearsheet`.
- CLI with `quantforge backtest` and `quantforge version` commands.
- Structured logging pipeline via `structlog` (JSON + dev console renderers).
- Financial math helpers: annualised return, Sharpe, Sortino, max drawdown,
  rolling z-score, VaR, CVaR, information ratio.
- 100 % mypy-strict type annotations across all public APIs.
- Full test suite: unit tests, Hypothesis property-based tests, integration
  tests for the complete backtest pipeline.
- GitHub Actions CI: lint, type-check, test on Ubuntu/macOS/Windows,
  security audit, benchmarks, and PyPI publish workflow.
- `pre-commit` hooks: Ruff, mypy, and general file hygiene.

[Unreleased]: https://github.com/yourusername/quantforge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/quantforge/releases/tag/v0.1.0
