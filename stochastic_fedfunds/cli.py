from __future__ import annotations

import argparse
from pathlib import Path

from stochastic_fedfunds.config import SimulationConfig
from stochastic_fedfunds.curves import build_monthly_forward_curve
from stochastic_fedfunds.market_data import load_normal_vol_cube, load_ois_curve
from stochastic_fedfunds.model import simulate_centered_hull_white
from stochastic_fedfunds.reporting import validation_summary, write_outputs
from stochastic_fedfunds.volatility import build_monthly_volatility


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate stochastic monthly Fed Funds paths from local OIS and SOFR vol data."
    )
    parser.add_argument("--ois-file", default="data/20251231-fedfunds-ois.csv")
    parser.add_argument("--vol-file", default="data/20251231-vol-dataset.csv")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--valuation-date", default="2025-12-31")
    parser.add_argument("--horizon-months", type=int, default=60)
    parser.add_argument("--num-paths", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=20251231)
    parser.add_argument("--mean-reversion", type=float, default=0.10)
    parser.add_argument(
        "--initial-rate-shock-bps",
        type=float,
        default=0.0,
        help=(
            "Instantaneous parallel rate shock in basis points applied at "
            "valuation time before monthly mean reversion."
        ),
    )
    parser.add_argument("--short-rate-vol-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--vol-aggregation",
        choices=["mean_by_option_expiry", "shortest_swap_tenor"],
        default="mean_by_option_expiry",
    )
    parser.add_argument("--rate-floor", type=float, default=None)
    parser.add_argument("--curve-method", choices=["linear_zero_rate"], default="linear_zero_rate")
    parser.add_argument("--tenor-day-count-basis", type=float, default=365.0)
    return parser


def run(config: SimulationConfig, ois_file: str | Path, vol_file: str | Path, output_dir: str | Path) -> dict:
    ois_curve = load_ois_curve(
        ois_file,
        valuation_date=config.valuation_date,
        day_count_basis=config.tenor_day_count_basis,
    )
    vol_cube = load_normal_vol_cube(
        vol_file,
        valuation_date=config.valuation_date,
        day_count_basis=config.tenor_day_count_basis,
    )
    forward_curve = build_monthly_forward_curve(
        ois_curve,
        horizon_months=config.horizon_months,
        method=config.curve_method,
    )
    monthly_volatility = build_monthly_volatility(
        vol_cube,
        horizon_months=config.horizon_months,
        dates=forward_curve.dates,
        aggregation=config.vol_aggregation,
        multiplier=config.short_rate_vol_multiplier,
    )
    result = simulate_centered_hull_white(forward_curve, monthly_volatility, config)
    manifest = write_outputs(result, output_dir)
    summary = validation_summary(result)
    return {"result": result, "manifest": manifest, "validation_summary": summary}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SimulationConfig(
        valuation_date=args.valuation_date,
        horizon_months=args.horizon_months,
        num_paths=args.num_paths,
        random_seed=args.random_seed,
        mean_reversion=args.mean_reversion,
        initial_rate_shock_bps=args.initial_rate_shock_bps,
        curve_method=args.curve_method,
        tenor_day_count_basis=args.tenor_day_count_basis,
        vol_aggregation=args.vol_aggregation,
        short_rate_vol_multiplier=args.short_rate_vol_multiplier,
        rate_floor=args.rate_floor,
    )

    run_output = run(config, args.ois_file, args.vol_file, args.output_dir)
    manifest = run_output["manifest"]
    summary = run_output["validation_summary"]

    print("Generated stochastic Fed Funds paths.")
    print(f"Paths CSV: {manifest['paths_long']}")
    print(f"Wide paths CSV: {manifest['paths_wide']}")
    print(f"Validation summary: {manifest['validation_summary']}")
    print(summary.to_string(index=False))
    return 0
