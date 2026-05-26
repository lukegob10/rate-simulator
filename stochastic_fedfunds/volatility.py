from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stochastic_fedfunds.market_data import NormalVolCubeData


@dataclass(frozen=True)
class MonthlyVolatility:
    valuation_date: pd.Timestamp
    months: np.ndarray
    dates: pd.DatetimeIndex
    option_years: np.ndarray
    annual_normal_vol_bps: np.ndarray
    annual_normal_vol: np.ndarray
    source_points: pd.DataFrame
    aggregation: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "month": self.months,
                "date": self.dates,
                "option_years": self.option_years,
                "annual_normal_vol_bps": self.annual_normal_vol_bps,
                "annual_normal_vol": self.annual_normal_vol,
                "aggregation": self.aggregation,
            }
        )


def _aggregate_vols(vol_cube: NormalVolCubeData, aggregation: str) -> pd.DataFrame:
    quotes = vol_cube.quotes.copy()

    if aggregation == "mean_by_option_expiry":
        grouped = (
            quotes.groupby(["option_tenor", "option_years"], as_index=False)
            .agg(
                annual_normal_vol_bps=("normal_vol_bps", "mean"),
                swap_tenor_count=("swap_tenor", "count"),
            )
            .sort_values("option_years")
        )
        return grouped

    if aggregation == "shortest_swap_tenor":
        idx = quotes.groupby("option_years")["swap_years"].idxmin()
        selected = quotes.loc[idx].copy().sort_values("option_years")
        selected = selected.rename(columns={"normal_vol_bps": "annual_normal_vol_bps"})
        selected["swap_tenor_count"] = 1
        return selected[["option_tenor", "option_years", "annual_normal_vol_bps", "swap_tenor_count"]]

    raise ValueError(f"Unsupported vol aggregation: {aggregation!r}")


def build_monthly_volatility(
    vol_cube: NormalVolCubeData,
    horizon_months: int,
    dates: pd.DatetimeIndex,
    aggregation: str = "mean_by_option_expiry",
    multiplier: float = 1.0,
) -> MonthlyVolatility:
    """Build monthly annual normal volatility from the swaption vol cube."""

    source_points = _aggregate_vols(vol_cube, aggregation)
    option_years = source_points["option_years"].to_numpy(dtype=float)
    vols_bps = source_points["annual_normal_vol_bps"].to_numpy(dtype=float)

    months = np.arange(1, horizon_months + 1, dtype=int)
    monthly_option_years = months / 12.0
    monthly_vols_bps = np.interp(
        monthly_option_years,
        option_years,
        vols_bps,
        left=vols_bps[0],
        right=vols_bps[-1],
    )
    monthly_vols = monthly_vols_bps / 10000.0 * multiplier

    return MonthlyVolatility(
        valuation_date=vol_cube.valuation_date,
        months=months,
        dates=dates,
        option_years=monthly_option_years,
        annual_normal_vol_bps=monthly_vols_bps * multiplier,
        annual_normal_vol=monthly_vols,
        source_points=source_points.reset_index(drop=True),
        aggregation=aggregation,
    )
