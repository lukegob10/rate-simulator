from pathlib import Path

import numpy as np

from stochastic_fedfunds.config import SimulationConfig
from stochastic_fedfunds.curves import build_monthly_forward_curve
from stochastic_fedfunds.market_data import load_normal_vol_cube, load_ois_curve
from stochastic_fedfunds.model import simulate_centered_hull_white
from stochastic_fedfunds.tenors import tenor_to_years
from stochastic_fedfunds.volatility import build_monthly_volatility


ROOT = Path(__file__).resolve().parents[1]


def test_tenor_to_years():
    assert tenor_to_years("7D") == 7 / 365
    assert tenor_to_years("3M") == 0.25
    assert tenor_to_years("1.5Y") == 1.5


def test_load_market_data_files():
    ois = load_ois_curve(ROOT / "data" / "20251231-fedfunds-ois.csv", valuation_date="2025-12-31")
    vols = load_normal_vol_cube(ROOT / "data" / "20251231-vol-dataset.csv", valuation_date="2025-12-31")

    assert len(ois.quotes) >= 10
    assert len(vols.quotes) >= 20
    assert np.isclose(ois.quotes.iloc[0]["par_rate"], 0.036339)
    assert vols.quotes["normal_vol_bps"].min() > 0


def test_simulation_shape_and_centering():
    config = SimulationConfig(horizon_months=12, num_paths=2000, random_seed=7)
    ois = load_ois_curve(ROOT / "data" / "20251231-fedfunds-ois.csv", valuation_date=config.valuation_date)
    vols = load_normal_vol_cube(ROOT / "data" / "20251231-vol-dataset.csv", valuation_date=config.valuation_date)
    forward = build_monthly_forward_curve(ois, config.horizon_months)
    vol = build_monthly_volatility(vols, config.horizon_months, forward.dates)
    result = simulate_centered_hull_white(forward, vol, config)

    assert result.paths.shape == (2000, 12)
    assert result.deviations.shape == result.paths.shape
    assert result.shocks.shape == result.paths.shape
    assert np.mean(np.abs(result.paths.mean(axis=0) - forward.forward_rates)) < 0.0015


def test_initial_rate_shock_is_applied_as_mean_reverting_deviation():
    base_config = SimulationConfig(horizon_months=6, num_paths=2000, random_seed=11)
    shocked_config = SimulationConfig(
        horizon_months=6,
        num_paths=2000,
        random_seed=11,
        initial_rate_shock_bps=400.0,
    )
    ois = load_ois_curve(ROOT / "data" / "20251231-fedfunds-ois.csv", valuation_date=base_config.valuation_date)
    vols = load_normal_vol_cube(ROOT / "data" / "20251231-vol-dataset.csv", valuation_date=base_config.valuation_date)
    forward = build_monthly_forward_curve(ois, base_config.horizon_months)
    vol = build_monthly_volatility(vols, base_config.horizon_months, forward.dates)

    base = simulate_centered_hull_white(forward, vol, base_config)
    shocked = simulate_centered_hull_white(forward, vol, shocked_config)

    monthly_decay = np.exp(-shocked_config.mean_reversion / 12.0)
    expected_path_shift = 0.04 * monthly_decay ** np.arange(1, shocked_config.horizon_months + 1)
    actual_path_shift = shocked.paths - base.paths

    assert np.allclose(actual_path_shift.mean(axis=0), expected_path_shift)
    assert np.allclose(actual_path_shift.std(axis=0), 0.0)
