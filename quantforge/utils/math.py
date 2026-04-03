"""Financial mathematics helpers.

Pure functions with no side effects — safe to use anywhere in the codebase.
All functions are fully typed and handle edge cases (empty arrays, division by
zero, NaN propagation) without raising unexpected exceptions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def annualised_return(total_return: float, years: float) -> float:
    """Compound annual growth rate (CAGR) from a total return over *years*.

    Args:
        total_return: Fractional total return (e.g. 0.34 for +34 %).
        years:        Duration in years (must be > 0).

    Returns:
        CAGR as a fraction.  Returns ``0.0`` if ``years`` is non-positive.

    Example:
        >>> annualised_return(0.34, 2.5)
        0.12597...
    """
    if years <= 0:
        return 0.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def sharpe_ratio(
    returns: npt.NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sharpe ratio for a return series.

    Args:
        returns:          1-D array of periodic returns (e.g. daily).
        risk_free_rate:   Annual risk-free rate (converted internally to per-period).
        periods_per_year: Trading periods per year (252 for daily, 52 for weekly).

    Returns:
        Annualised Sharpe ratio, or ``0.0`` if the return std dev is zero.
    """
    if len(returns) < 2:
        return 0.0
    rfr_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rfr_period
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: npt.NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sortino ratio (penalises only downside deviation).

    Args:
        returns:          1-D array of periodic returns.
        risk_free_rate:   Annual risk-free rate.
        periods_per_year: Trading periods per year.

    Returns:
        Annualised Sortino ratio, or ``0.0`` if there are no negative returns.
    """
    if len(returns) < 2:
        return 0.0
    rfr_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rfr_period
    downside = returns[returns < rfr_period]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: npt.NDArray[np.float64]) -> tuple[float, int]:
    """Maximum peak-to-trough drawdown and its duration in periods.

    Drawdown is measured as a fraction of the preceding peak, so a value of
    ``-0.20`` represents a 20 % decline from the highest prior equity level.

    Args:
        equity: 1-D array of equity or NAV values (must be non-empty).

    Returns:
        A tuple of ``(max_drawdown_fraction, duration_in_periods)`` where
        ``max_drawdown_fraction`` is ≤ 0 and ``duration_in_periods`` is the
        number of periods between the peak and the subsequent trough.

    Raises:
        ValueError: If ``equity`` is empty.
    """
    if len(equity) == 0:
        msg = "equity array must not be empty"
        raise ValueError(msg)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / np.where(peak == 0, 1, peak)
    max_dd = float(drawdown.min())
    worst_idx = int(np.argmin(drawdown))

    # Find the index of the peak that preceded the worst drawdown
    peak_idx = int(np.argmax(equity[: worst_idx + 1]))
    duration = worst_idx - peak_idx
    return max_dd, duration


def rolling_zscore(
    values: npt.NDArray[np.float64],
    window: int,
) -> npt.NDArray[np.float64]:
    """Compute a rolling z-score over a 1-D array.

    The first ``window - 1`` values are set to ``NaN`` because there is
    insufficient history to compute a meaningful statistic.

    Args:
        values: Input 1-D float array.
        window: Rolling window size (must be ≥ 2).

    Returns:
        Array of the same length as ``values`` containing the rolling z-scores.

    Raises:
        ValueError: If ``window < 2`` or ``values`` is shorter than ``window``.
    """
    if window < 2:
        msg = "window must be at least 2"
        raise ValueError(msg)
    n = len(values)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_vals = values[i - window + 1 : i + 1]
        mu = window_vals.mean()
        sigma = window_vals.std()
        result[i] = 0.0 if sigma == 0 else (values[i] - mu) / sigma
    return result


def value_at_risk(
    returns: npt.NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """Historical simulation Value at Risk (VaR).

    Returns the loss threshold such that losses exceed this value with
    probability ``1 - confidence``.  The result is expressed as a negative
    fraction (e.g. ``-0.032`` for a 3.2 % 1-day 95 % VaR).

    Args:
        returns:    1-D array of periodic returns.
        confidence: Confidence level in (0, 1) — default 0.95.

    Returns:
        VaR as a negative fraction, or ``0.0`` if fewer than 20 observations.
    """
    if len(returns) < 20:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(
    returns: npt.NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """Expected Shortfall (CVaR / ES) — the mean loss beyond VaR.

    Args:
        returns:    1-D array of periodic returns.
        confidence: Confidence level in (0, 1).

    Returns:
        CVaR as a negative fraction (worse than VaR), or ``0.0`` if < 20 obs.
    """
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return 0.0
    return float(tail.mean())


def information_ratio(
    portfolio_returns: npt.NDArray[np.float64],
    benchmark_returns: npt.NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Annualised information ratio vs a benchmark.

    IR = annualised active return / annualised tracking error.

    Args:
        portfolio_returns:  1-D array of portfolio periodic returns.
        benchmark_returns:  1-D array of benchmark periodic returns (same length).
        periods_per_year:   Trading periods per year.

    Returns:
        Information ratio, or ``0.0`` if tracking error is zero or arrays differ in length.
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        return 0.0
    active = portfolio_returns - benchmark_returns
    tracking_error = active.std()
    if tracking_error == 0:
        return 0.0
    return float(active.mean() / tracking_error * np.sqrt(periods_per_year))
