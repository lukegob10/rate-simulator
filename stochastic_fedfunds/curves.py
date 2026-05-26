from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stochastic_fedfunds.market_data import OISCurveData


@dataclass(frozen=True)
class MonthlyForwardCurve:
    valuation_date: pd.Timestamp
    months: np.ndarray
    dates: pd.DatetimeIndex
    period_start_years: np.ndarray
    period_end_years: np.ndarray
    discount_factors_start: np.ndarray
    discount_factors_end: np.ndarray
    zero_rates_end: np.ndarray
    forward_rates: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "month": self.months,
                "date": self.dates,
                "period_start_years": self.period_start_years,
                "period_end_years": self.period_end_years,
                "discount_factor_start": self.discount_factors_start,
                "discount_factor_end": self.discount_factors_end,
                "zero_rate_end": self.zero_rates_end,
                "forward_rate": self.forward_rates,
                "forward_rate_percent": self.forward_rates * 100.0,
            }
        )


def build_monthly_forward_curve(
    ois_curve: OISCurveData,
    horizon_months: int,
    method: str = "linear_zero_rate",
) -> MonthlyForwardCurve:
    """Convert OIS curve quotes into monthly one-month forward rates."""

    if method != "linear_zero_rate":
        raise ValueError(f"Unsupported curve method: {method!r}")

    tenors = ois_curve.quotes["tenor_years"].to_numpy(dtype=float)
    zero_rates = ois_curve.quotes["par_rate"].to_numpy(dtype=float)

    boundaries = np.arange(horizon_months + 1, dtype=float) / 12.0
    boundary_zero_rates = np.interp(
        boundaries,
        tenors,
        zero_rates,
        left=zero_rates[0],
        right=zero_rates[-1],
    )

    discount_factors = np.ones(horizon_months + 1, dtype=float)
    positive = boundaries > 0.0
    discount_factors[positive] = np.exp(-boundary_zero_rates[positive] * boundaries[positive])

    dt = 1.0 / 12.0
    forward_rates = -np.diff(np.log(discount_factors)) / dt

    months = np.arange(1, horizon_months + 1, dtype=int)
    dates = pd.DatetimeIndex(
        [ois_curve.valuation_date + pd.DateOffset(months=int(month)) for month in months]
    )

    return MonthlyForwardCurve(
        valuation_date=ois_curve.valuation_date,
        months=months,
        dates=dates,
        period_start_years=boundaries[:-1],
        period_end_years=boundaries[1:],
        discount_factors_start=discount_factors[:-1],
        discount_factors_end=discount_factors[1:],
        zero_rates_end=boundary_zero_rates[1:],
        forward_rates=forward_rates,
    )
