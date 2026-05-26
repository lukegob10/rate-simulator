from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stochastic_fedfunds.config import SimulationConfig
from stochastic_fedfunds.curves import MonthlyForwardCurve
from stochastic_fedfunds.volatility import MonthlyVolatility


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    forward_curve: MonthlyForwardCurve
    volatility: MonthlyVolatility
    paths: np.ndarray
    deviations: np.ndarray
    shocks: np.ndarray

    def paths_long_frame(self) -> pd.DataFrame:
        path_count, horizon = self.paths.shape
        path_ids = np.repeat(np.arange(1, path_count + 1, dtype=int), horizon)
        months = np.tile(self.forward_curve.months, path_count)
        dates = np.tile(self.forward_curve.dates.strftime("%Y-%m-%d"), path_count)
        forward_rates = np.tile(self.forward_curve.forward_rates, path_count)
        annual_vols = np.tile(self.volatility.annual_normal_vol, path_count)
        rates = self.paths.reshape(-1)

        return pd.DataFrame(
            {
                "path_id": path_ids,
                "month": months,
                "date": dates,
                "fed_funds_rate": rates,
                "fed_funds_rate_percent": rates * 100.0,
                "centerline_forward_rate": forward_rates,
                "centerline_forward_rate_percent": forward_rates * 100.0,
                "annual_normal_vol": annual_vols,
                "annual_normal_vol_bps": annual_vols * 10000.0,
            }
        )

    def paths_wide_frame(self) -> pd.DataFrame:
        columns = [date.strftime("%Y-%m-%d") for date in self.forward_curve.dates]
        frame = pd.DataFrame(self.paths, columns=columns)
        frame.insert(0, "path_id", np.arange(1, self.paths.shape[0] + 1, dtype=int))
        return frame


def _ou_step_std(mean_reversion: float, dt: float) -> tuple[float, float]:
    if abs(mean_reversion) < 1e-12:
        return 1.0, np.sqrt(dt)

    phi = np.exp(-mean_reversion * dt)
    std_scale = np.sqrt((1.0 - np.exp(-2.0 * mean_reversion * dt)) / (2.0 * mean_reversion))
    return phi, std_scale


def simulate_centered_hull_white(
    forward_curve: MonthlyForwardCurve,
    volatility: MonthlyVolatility,
    config: SimulationConfig,
) -> SimulationResult:
    """Simulate monthly Fed Funds paths around the OIS forward centerline."""

    horizon = config.horizon_months
    if len(forward_curve.forward_rates) != horizon:
        raise ValueError("Forward curve horizon does not match config horizon_months")
    if len(volatility.annual_normal_vol) != horizon:
        raise ValueError("Volatility horizon does not match config horizon_months")

    rng = np.random.default_rng(config.random_seed)
    dt = 1.0 / 12.0
    phi, std_scale = _ou_step_std(config.mean_reversion, dt)

    paths = np.zeros((config.num_paths, horizon), dtype=float)
    deviations = np.zeros_like(paths)
    shocks = np.zeros_like(paths)
    initial_deviation = config.initial_rate_shock_bps / 10000.0
    previous_deviation = np.full(config.num_paths, initial_deviation, dtype=float)

    for month_index in range(horizon):
        z = rng.standard_normal(config.num_paths)
        shock = volatility.annual_normal_vol[month_index] * std_scale * z
        deviation = phi * previous_deviation + shock
        rates = forward_curve.forward_rates[month_index] + deviation

        if config.rate_floor is not None:
            rates = np.maximum(rates, config.rate_floor)
            deviation = rates - forward_curve.forward_rates[month_index]

        paths[:, month_index] = rates
        deviations[:, month_index] = deviation
        shocks[:, month_index] = shock
        previous_deviation = deviation

    return SimulationResult(
        config=config,
        forward_curve=forward_curve,
        volatility=volatility,
        paths=paths,
        deviations=deviations,
        shocks=shocks,
    )
