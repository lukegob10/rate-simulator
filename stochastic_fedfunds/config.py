from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class SimulationConfig:
    """Configurable assumptions for the Fed Funds path simulation."""

    valuation_date: str = "2025-12-31"
    horizon_months: int = 60
    num_paths: int = 500
    random_seed: int = 20251231

    # Annual mean reversion speed in the centered Hull-White/OU deviation.
    mean_reversion: float = 0.10

    # Instantaneous parallel rate shock applied at valuation time, in basis
    # points. A positive shock starts paths above the OIS forward centerline
    # and then mean-reverts through the stochastic deviation process.
    initial_rate_shock_bps: float = 0.0

    # Curve construction approximation. OIS par quotes are treated as
    # zero-equivalent continuous rates, then linearly interpolated.
    curve_method: str = "linear_zero_rate"
    curve_rate_units: str = "percent"
    tenor_day_count_basis: float = 365.0

    # SOFR ATM swaption normal vol handling.
    vol_aggregation: str = "mean_by_option_expiry"
    vol_units: str = "bps_per_annum"
    short_rate_vol_multiplier: float = 1.0

    # Normal short-rate model can produce negative rates. Leave unset by
    # default to preserve normal dynamics.
    rate_floor: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def assumptions(self) -> dict:
        return {
            **self.to_dict(),
            "model": "one-factor centered Hull-White style short-rate model",
            "model_equation": (
                "x[0] = initial_rate_shock_bps / 10000; "
                "x[t+1] = exp(-a*dt) * x[t] + sigma[t] * "
                "sqrt((1 - exp(-2*a*dt)) / (2*a)) * Z[t]; "
                "r[t+1] = OIS_forward[t+1] + x[t+1]"
            ),
            "curve_assumption": (
                "Fed Funds OIS par quotes are used as zero-equivalent "
                "continuously compounded rates for transparent monthly "
                "forward extraction."
            ),
            "vol_assumption": (
                "USD SOFR ATM normal swaption vols are used as a proxy for "
                "Fed Funds short-rate normal volatility after converting "
                "bps/annum to decimal annual rate volatility."
            ),
            "normal_rate_note": (
                "The default normal model does not floor rates. Set "
                "rate_floor to impose one."
            ),
            "shock_note": (
                "initial_rate_shock_bps is an instantaneous parallel shift "
                "to the short-rate deviation at valuation time. It affects "
                "the first simulated month after one monthly mean-reversion "
                "step and then decays through the OU process."
            ),
        }
