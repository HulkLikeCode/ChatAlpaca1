from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERCENTILES = (5, 25, 50, 75, 95)


@dataclass(frozen=True)
class ProjectionResult:
    """Percentile outcomes from an assumption-driven portfolio projection."""

    monthly_percentiles: pd.DataFrame
    annual_percentiles: pd.DataFrame
    target_probability: float | None


def simulate_portfolio_projection(
    current_value: float,
    annual_return: float,
    annual_volatility: float,
    monthly_contribution: float,
    horizon_years: int,
    target_value: float | None = None,
    simulations: int = 10_000,
    seed: int = 20260719,
) -> ProjectionResult:
    """Project monthly portfolio values with a lognormal return model.

    ``annual_return`` is an arithmetic expected annual return. Contributions are
    added at each month-end in today's dollars. Results are planning scenarios,
    not forecasts inferred from the portfolio's short transaction history.
    """
    if current_value <= 0:
        raise ValueError("Current portfolio value must be greater than zero.")
    if annual_return <= -1:
        raise ValueError("Annual return must be greater than -100%.")
    if annual_volatility < 0:
        raise ValueError("Annual volatility cannot be negative.")
    if monthly_contribution < 0:
        raise ValueError("Monthly contribution cannot be negative.")
    if not 1 <= horizon_years <= 10:
        raise ValueError("Forecast horizon must be between 1 and 10 years.")
    if simulations < 1:
        raise ValueError("At least one simulation is required.")
    if target_value is not None and target_value <= 0:
        raise ValueError("Target value must be greater than zero.")

    months = horizon_years * 12
    monthly_step = 1 / 12
    # Calibrate the lognormal process so its expected annual gross return is
    # 1 + annual_return, while volatility controls the spread of outcomes.
    drift = (np.log1p(annual_return) - 0.5 * annual_volatility**2) * monthly_step
    diffusion = annual_volatility * np.sqrt(monthly_step)
    shocks = np.random.default_rng(seed).standard_normal((simulations, months))
    gross_returns = np.exp(drift + diffusion * shocks)

    paths = np.empty((simulations, months + 1), dtype=float)
    paths[:, 0] = current_value
    for month in range(1, months + 1):
        paths[:, month] = paths[:, month - 1] * gross_returns[:, month - 1]
        paths[:, month] += monthly_contribution

    month_index = pd.RangeIndex(0, months + 1, name="Month")
    monthly_percentiles = pd.DataFrame(
        np.percentile(paths, PERCENTILES, axis=0).T,
        columns=[f"P{percentile}" for percentile in PERCENTILES],
        index=month_index,
    )
    annual_percentiles = monthly_percentiles.iloc[::12].copy()
    annual_percentiles.index = pd.Index(range(horizon_years + 1), name="Year")
    target_probability = (
        float(np.mean(paths[:, -1] >= target_value)) if target_value is not None else None
    )
    return ProjectionResult(monthly_percentiles, annual_percentiles, target_probability)
