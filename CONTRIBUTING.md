# Contributing to QuantForge

Thank you for your interest in improving QuantForge.  This document covers
everything you need to make a high-quality contribution.

---

## Development Setup

```bash
git clone https://github.com/yourusername/quantforge.git
cd quantforge

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (runs quality checks before every commit)
pre-commit install
```

---

## Code Style

All formatting and linting is handled automatically by **Ruff**.  You should
never need to think about formatting — just run:

```bash
ruff format .          # Format all Python files
ruff check . --fix     # Auto-fix lint violations where possible
```

The CI pipeline enforces these checks and will fail if they are not satisfied.

### Key conventions

| Convention | Rule |
|-----------|------|
| Docstrings | Google-style (`Args:`, `Returns:`, `Raises:`, `Example:`) |
| Type annotations | Required on every public function and method |
| Imports | Absolute only; no relative imports |
| Constants | `SCREAMING_SNAKE_CASE` |
| Private helpers | Prefix with `_single_underscore` |

---

## Type Checking

```bash
mypy quantforge --strict
```

All code must pass `mypy --strict` with zero errors.  If you need to suppress
a false positive, add a `# type: ignore[<error-code>]` comment with a brief
explanation of why.

---

## Testing

```bash
pytest                             # Full suite (unit + integration + coverage)
pytest -m unit                     # Unit tests only (< 10 s)
pytest -m integration              # Integration tests (network-free synthetic data)
pytest --benchmark-only            # Performance benchmarks
```

### Test coverage

- All new code must maintain the 85 % coverage floor enforced by `pytest-cov`.
- New public functions must have at least one happy-path test and one
  error/edge-case test.
- Prefer **property-based tests** (Hypothesis) over hand-crafted examples for
  numerical invariants.

---

## Submitting a Pull Request

1. **Fork** the repository and create a feature branch from `develop`:
   ```bash
   git checkout develop
   git checkout -b feat/your-feature-name
   ```

2. **Write your code** — follow the conventions above.

3. **Add tests** — every new behaviour must be covered.

4. **Run the full quality suite locally**:
   ```bash
   pre-commit run --all-files
   pytest
   ```

5. **Update `CHANGELOG.md`** — add an entry under `[Unreleased]` describing
   what changed and why.

6. **Open a pull request** against `develop` with a clear description of the
   motivation and approach.

---

## Reporting Bugs

Please open a GitHub Issue with:

- Python version and OS.
- Minimal reproducible example.
- Expected vs actual behaviour.
- Full stack trace (if applicable).

---

## Licence

By contributing, you agree that your contributions will be licensed under the
same MIT licence as the project.
