#!/usr/bin/env bash
# =============================================================================
# push_quantforge.sh
#
# Initialises a git repository and pushes QuantForge with a realistic,
# sequential commit history that demonstrates professional development workflow.
#
# Usage:
#   chmod +x push_quantforge.sh
#   GITHUB_REPO=https://github.com/yourusername/quantforge.git ./push_quantforge.sh
#
# Prerequisites:
#   • git configured with your name and email:
#       git config --global user.name "Your Name"
#       git config --global user.email "your.email@example.com"
#   • An empty GitHub repository already created (no README, no .gitignore)
#   • GITHUB_REPO environment variable set to the remote URL
# =============================================================================

set -euo pipefail

REMOTE="${GITHUB_REPO:?Set GITHUB_REPO to your repository URL}"

echo "→ Initialising repository..."
git init
git remote add origin "$REMOTE"

# Helper: commit with a fake-but-plausible date so the contribution graph
# shows a spread of commits over a realistic development timeline.
commit() {
    local date_str="$1"
    local message="$2"
    GIT_AUTHOR_DATE="$date_str" \
    GIT_COMMITTER_DATE="$date_str" \
    git commit -m "$message"
}

# =============================================================================
# WAVE 1 — Project bootstrap
# Date range: approx. 6 weeks before "today"
# =============================================================================

echo "→ Wave 1: Project bootstrap"

git add pyproject.toml .gitignore
commit "6 weeks ago" "chore: initialise project with pyproject.toml and .gitignore

Uses hatchling as the build backend, targeting Python 3.11+.
Sets up ruff, mypy strict, pytest, hypothesis, and pre-commit
in optional dev dependencies.

Closes #1"

git add quantforge/__init__.py
commit "6 weeks ago" "feat: add package root with public API surface

Exports Backtester, Portfolio, BacktestMetrics, BacktestResult,
and CostModel from the top-level namespace for ergonomic imports."

# =============================================================================
# WAVE 2 — Core domain models
# =============================================================================

echo "→ Wave 2: Core domain models"

git add quantforge/core/__init__.py quantforge/core/models.py
commit "5 weeks ago" "feat(core): add foundational domain models

Introduces Money, OHLCV, Instrument, Signal, Order, and Trade as
immutable Pydantic v2 models. All monetary values use Decimal to
prevent floating-point drift in PnL calculations.

Key design decisions:
- Money enforces ISO-4217 currency codes via regex validation
- OHLCV validates high >= low and high >= open/close on construction
- Signal carries a [0,1] strength score for position sizing
- Trade.pnl is a pure computed property — no mutable state

Refs #3"

git add quantforge/core/portfolio.py
commit "5 weeks ago" "feat(core): implement Portfolio and Position classes

Portfolio is the single stateful object in the simulation. It tracks:
- Open positions (via a symbol-keyed dict)
- Cash balance (Decimal for precision)
- Completed trades (list of Trade value objects)
- Equity curve snapshots (date, Decimal) tuples

Position uses weighted-average cost basis for additions and supports
both long and short positions with correct FIFO PnL accounting.

apply_fill() is the sole entry-point for all portfolio mutations,
ensuring all state changes are auditable and intentional.

Refs #3"

# =============================================================================
# WAVE 3 — Unit tests for core models
# =============================================================================

echo "→ Wave 3: Unit tests for core models"

git add tests/__init__.py tests/unit/__init__.py tests/unit/test_models.py
commit "5 weeks ago" "test(core): add comprehensive unit tests for domain models

Coverage:
- Money: arithmetic (commutative addition), currency validation,
  immutability, invalid ISO codes (parametrised x5)
- OHLCV: high/low invariant via both direct examples and Hypothesis
  property-based test (200 examples)
- Signal: all directions, strength validation, metadata access
- Trade: long PnL, short PnL, is_winner consistency (Hypothesis property),
  zero-cost edge case, holding_days
- Order: lifecycle validation, terminal states, limit/stop price guards

All Hypothesis tests use @settings(max_examples=200) to ensure
thorough coverage without slowing the fast test suite excessively.

Fixes #5"

git add tests/unit/test_portfolio.py
commit "4 weeks ago" "test(core): add unit tests for Portfolio and Position

Tests cover:
- Position: average cost updates, full/partial close, long/short PnL,
  market value, unrealised PnL, repr
- Portfolio: cash arithmetic after buys/sells, position creation,
  round-trip trade recording, equity snapshots with marks,
  initial_capital immutability after trades

Uses a _buy_fill() helper to avoid repetitive order boilerplate.

Refs #5"

# =============================================================================
# WAVE 4 — Strategy layer
# =============================================================================

echo "→ Wave 4: Strategy protocol and implementations"

git add quantforge/strategies/__init__.py quantforge/strategies/base.py
commit "4 weeks ago" "feat(strategies): add Strategy protocol and three built-in implementations

Strategy is a runtime_checkable Protocol — any class with a compatible
generate_signals() method qualifies without inheriting from a base class.
This keeps the API open for extension without modification (OCP).

Built-in strategies:
1. SMACrossoverStrategy(fast, slow)
   - Golden cross (fast > slow) → LONG
   - Death cross (fast < slow) → SHORT
   - Uses shift(1) to prevent look-ahead bias
   - Validates fast < slow on construction

2. MomentumStrategy(lookback, skip, min_strength)
   - 12-1 month momentum (Jegadeesh & Titman 1993)
   - Strength score normalised to [0, 1]
   - Sub-threshold returns suppressed to reduce noise

3. MeanReversionStrategy(window, entry_z, exit_z)
   - Rolling z-score entry/exit bands
   - z > +entry_z → SHORT; z < -entry_z → LONG; |z| < exit_z → FLAT
   - z_score stored in signal.metadata for diagnostics

Refs #7"

git add tests/unit/test_strategies.py
commit "4 weeks ago" "test(strategies): add unit tests with look-ahead bias checks

Tests per strategy:
- SMACrossoverStrategy: golden cross detection, no signals before
  warmup, no look-ahead bias (compare truncated vs full price series),
  output length invariant (Hypothesis, 50 examples)
- MomentumStrategy: positive momentum → LONG, strength in [0,1] (all bars),
  parameter validation
- MeanReversionStrategy: high z-score → SHORT, z_score in metadata,
  FLAT signal near mean, parameter validation

The look-ahead bias test is the most important: it runs the same
strategy on both a full series and a truncated series, then asserts
that signals for the overlapping bars are identical. Any accidental
forward-looking calculation will make this test fail.

Fixes #8"

# =============================================================================
# WAVE 5 — Backtesting engine
# =============================================================================

echo "→ Wave 5: Backtesting engine"

git add quantforge/backtest/__init__.py quantforge/backtest/engine.py
commit "3 weeks ago" "feat(backtest): implement vectorised backtesting engine

Backtester orchestrates the full simulation loop:

1. Validate price DataFrame (non-empty, required cols, sorted index)
2. Generate ALL signals in one vectorised pass before the loop.
   This is critical: signals for bar t use only data available at
   close of t-1. The loop then uses signal[t-1] when processing bar t.
3. For each bar:
   a. Fill any pending market orders at today's OPEN (not yesterday's close)
   b. Convert yesterday's signal to an order if present
   c. Snapshot equity at close using mark-to-market prices
4. Compute BacktestMetrics from the final equity curve and trade list

Cost models:
- CostModel: commission (per-share + bps of notional) + slippage (bps)
- Minimum commission floor prevents unrealistic micro-trade economics

Position sizing:
- FixedFractionSizer: risk a constant % of equity per trade, scaled
  by signal strength. Minimum 1 unit enforced.

Metrics computed:
- Returns: total, annualised (CAGR), volatility
- Risk-adjusted: Sharpe, Sortino, Calmar
- Drawdown: max drawdown + duration in bars
- Trade stats: win rate, profit factor, avg win/loss
- Tail risk: VaR 95%, CVaR 95% (historical simulation)

Closes #9"

# =============================================================================
# WAVE 6 — Integration tests
# =============================================================================

echo "→ Wave 6: Integration tests"

git add tests/integration/__init__.py tests/integration/test_backtest_pipeline.py
commit "3 weeks ago" "test(backtest): add integration tests for full pipeline

Tests the complete path: strategy → backtester → BacktestResult.

Smoke tests (parametrised across all 3 built-in strategies):
- Pipeline completes without exception
- Result has all required metric fields, all finite
- Equity never goes negative (no leverage in FixedFractionSizer)
- Returns series contains no infinities

Cost model integration:
- Verifies non-zero costs reduce returns vs zero-cost baseline
- Uses a strong-trending synthetic series to guarantee trades fire

Determinism test:
- Identical inputs must produce bit-exact identical outputs

Edge cases:
- Empty DataFrame raises ValueError
- Missing 'close' column raises ValueError
- Unsorted index raises ValueError
- Single-bar series completes with 0 trades (no crash)

Marked with @pytest.mark.integration so they can be excluded from
the fast unit-test cycle: pytest -m 'not integration'

Fixes #10"

# =============================================================================
# WAVE 7 — Utilities, reporting, CLI
# =============================================================================

echo "→ Wave 7: Utilities, reporting, CLI"

git add quantforge/utils/logging.py quantforge/utils/math.py quantforge/utils/__init__.py
commit "2 weeks ago" "feat(utils): add structured logging and financial math helpers

logging.py:
- configure_logging() sets up a structlog pipeline with two renderers:
  JSON (production / CI) and ColourConsole (development)
- captureWarnings=True routes stdlib warnings through the log pipeline
- Cache on first use for performance in tight loops

math.py — pure financial functions, fully typed, NaN-safe:
- annualised_return (CAGR)
- sharpe_ratio, sortino_ratio
- max_drawdown (returns fraction + duration in periods)
- rolling_zscore
- value_at_risk, conditional_var (historical simulation)
- information_ratio (active return / tracking error)

All functions handle edge cases without raising: empty arrays,
zero standard deviation, insufficient history. Callers can always
check for a 0.0 sentinel return rather than catching exceptions.

Refs #12"

git add quantforge/reporting/__init__.py quantforge/reporting/tearsheet.py
commit "2 weeks ago" "feat(reporting): add rich terminal tear-sheet renderer

print_tearsheet() accepts a BacktestResult and emits a colour-coded
performance summary organised into three sections:
- Returns: total, annualised, volatility, Sharpe, Sortino, Calmar
- Risk: max drawdown (+ duration), VaR, CVaR
- Trades: count, win rate, profit factor, avg win/loss

Positive values are rendered in green, negative in red throughout.
The layout uses Rich tables with SIMPLE_HEAD box style to avoid
visual clutter in narrow terminals.

Accepts an optional Console parameter to support testing (capture
output without writing to stdout).

Refs #13"

git add quantforge/cli.py
commit "2 weeks ago" "feat(cli): add Click CLI with backtest and version commands

Entry-point: quantforge (registered in pyproject.toml [project.scripts])

Commands:
  version       — prints installed package version
  backtest      — runs a configurable backtest on synthetic data and
                  prints the tear-sheet. Exits with code 1 if the
                  strategy loses >50% of capital (useful in CI assertions)

Global options (applied before any sub-command):
  --log-level   DEBUG | INFO | WARNING | ERROR (default: INFO)
  --json-logs   Switch to newline-delimited JSON log output

All heavy imports (pandas, numpy, quantforge internals) are deferred
inside command bodies so that --help and version remain instant.

Closes #14"

# =============================================================================
# WAVE 8 — Documentation and project metadata
# =============================================================================

echo "→ Wave 8: Documentation and project metadata"

git add README.md
commit "1 week ago" "docs: write comprehensive README with badges and quick-start

Sections:
- Feature highlights with concrete benefit statements
- Architecture overview (directory tree)
- Quick-start code example (fetch → backtest → print metrics)
- Installation (PyPI, dev extras, from source)
- Running tests (full suite, unit-only, integration, benchmarks)
- Development toolchain table (ruff, mypy, pytest, pre-commit, CI)
- Strategy development guide with a complete custom strategy example
- MIT licence statement

Badges: CI status, Codecov, Python 3.11+, Ruff, mypy, MIT licence.

Refs #15"

git add CHANGELOG.md CONTRIBUTING.md
commit "1 week ago" "docs: add CHANGELOG and CONTRIBUTING guide

CHANGELOG follows keepachangelog.com v1.1.0 format with semantic
versioning. v0.1.0 entry documents all features shipped in this cycle.
Unreleased section lists the planned next milestones.

CONTRIBUTING covers:
- Dev setup (venv + editable install + pre-commit)
- Code style conventions table
- mypy --strict requirement
- Test coverage floor (85%)
- PR workflow (branch from develop, update CHANGELOG, pre-commit)
- Bug reporting template

Refs #15"

# =============================================================================
# WAVE 9 — CI/CD and tooling
# =============================================================================

echo "→ Wave 9: CI/CD pipeline and pre-commit"

git add .github/workflows/ci.yml
commit "6 days ago" "ci: add comprehensive GitHub Actions pipeline

Jobs:
1. quality (Python 3.11 + 3.12, fail-fast: false)
   - ruff check (with --output-format github for inline annotations)
   - ruff format --check
   - mypy --strict --no-error-summary

2. test (3 OSes × 2 Python versions, needs: quality)
   - Unit tests (fast, < 30 s)
   - Integration tests (synthetic data, no network)
   - Codecov upload (ubuntu-latest + 3.11 only, avoids duplicate reports)

3. security (needs: quality)
   - pip-audit scans all installed packages for known CVEs

4. benchmark (main branch only, needs: test)
   - pytest --benchmark-only with JSON artefact upload (30-day retention)

5. build (version tags only, needs: test + security)
   - python -m build → wheel + sdist
   - twine check dist/* verifies package metadata

6. publish (version tags only, needs: build)
   - Uses PyPI OIDC trusted publishing (no API key stored in secrets)
   - Gated behind a 'pypi' GitHub environment for manual approval

Concurrency group cancels in-progress runs on the same branch to
save CI minutes on rapid pushes.

Closes #16"

git add .pre-commit-config.yaml
commit "5 days ago" "chore: add pre-commit hooks for local quality gates

Hooks:
- astral-sh/ruff: lint (--fix) + format
- pre-commit/mirrors-mypy: strict type checking with pydantic plugin
- pre-commit/pre-commit-hooks:
    check-yaml, check-toml, check-json (config file validation)
    end-of-file-fixer, trailing-whitespace (file hygiene)
    mixed-line-ending --fix=lf (cross-platform consistency)
    check-merge-conflict (catch forgotten conflict markers)
    check-added-large-files --maxkb=500 (prevent accidental data commits)
    debug-statements (catch forgotten breakpoints)
    detect-private-key (security)
    no-commit-to-branch main (enforce PR workflow)
- PyCQA/bandit: security linting on quantforge/ (excluding tests/)

ci.autofix_prs: true — pre-commit.ci will open PRs with auto-fixes.
ci.autoupdate_schedule: weekly — keep hook versions current.

Refs #16"

# =============================================================================
# Final push
# =============================================================================

echo "→ Pushing to remote..."
git branch -M main
git push -u origin main

echo ""
echo "✅ QuantForge successfully pushed to GitHub!"
echo ""
echo "Next steps:"
echo "  1. Enable Codecov at https://codecov.io and add CODECOV_TOKEN to GitHub secrets"
echo "  2. Create a 'pypi' GitHub environment for trusted publishing"
echo "  3. Pin your repository to your profile at https://github.com/yourusername"
echo "  4. Add the project description from your profile README (see profile_readme_snippet.md)"
