from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stochastic_fedfunds.model import SimulationResult


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def monthly_summary(result: SimulationResult) -> pd.DataFrame:
    paths = result.paths
    quantiles = np.quantile(paths, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], axis=0)
    frame = pd.DataFrame(
        {
            "month": result.forward_curve.months,
            "date": result.forward_curve.dates.strftime("%Y-%m-%d"),
            "centerline_forward_rate": result.forward_curve.forward_rates,
            "mean": paths.mean(axis=0),
            "std": paths.std(axis=0, ddof=1),
            "min": paths.min(axis=0),
            "p01": quantiles[0],
            "p05": quantiles[1],
            "p25": quantiles[2],
            "median": quantiles[3],
            "p75": quantiles[4],
            "p95": quantiles[5],
            "p99": quantiles[6],
            "max": paths.max(axis=0),
            "annual_normal_vol_bps": result.volatility.annual_normal_vol_bps,
        }
    )

    rate_columns = [
        "centerline_forward_rate",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "p99",
        "max",
    ]
    for column in rate_columns:
        frame[f"{column}_percent"] = frame[column] * 100.0

    return frame


def validation_summary(result: SimulationResult) -> pd.DataFrame:
    paths = result.paths
    means = paths.mean(axis=0)
    forward = result.forward_curve.forward_rates
    terminal_rates = paths[:, -1]

    rows = [
        ("valuation_date", result.config.valuation_date),
        ("horizon_months", result.config.horizon_months),
        ("num_paths", result.config.num_paths),
        ("random_seed", result.config.random_seed),
        ("mean_reversion", result.config.mean_reversion),
        ("initial_rate_shock_bps", result.config.initial_rate_shock_bps),
        ("vol_aggregation", result.config.vol_aggregation),
        ("short_rate_vol_multiplier", result.config.short_rate_vol_multiplier),
        ("mean_abs_error_mean_vs_forward_bp", float(np.mean(np.abs(means - forward)) * 10000.0)),
        ("max_abs_error_mean_vs_forward_bp", float(np.max(np.abs(means - forward)) * 10000.0)),
        ("terminal_forward_percent", float(forward[-1] * 100.0)),
        ("terminal_mean_percent", float(terminal_rates.mean() * 100.0)),
        ("terminal_std_percent", float(terminal_rates.std(ddof=1) * 100.0)),
        ("min_path_rate_percent", float(paths.min() * 100.0)),
        ("max_path_rate_percent", float(paths.max() * 100.0)),
        ("negative_rate_fraction", float((paths < 0.0).mean())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _plot_fan_chart(result: SimulationResult, path: Path) -> None:
    dates = result.forward_curve.dates
    paths_percent = result.paths * 100.0
    quantiles = np.quantile(paths_percent, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(dates, paths_percent.T, color="#6b7280", alpha=0.06, linewidth=0.7)
    ax.fill_between(dates, quantiles[0], quantiles[4], color="#93c5fd", alpha=0.35, label="5-95%")
    ax.fill_between(dates, quantiles[1], quantiles[3], color="#2563eb", alpha=0.25, label="25-75%")
    ax.plot(dates, quantiles[2], color="#1d4ed8", linewidth=1.8, label="Median")
    ax.plot(
        dates,
        result.forward_curve.forward_rates * 100.0,
        color="#111827",
        linewidth=2.0,
        linestyle="--",
        label="OIS forward centerline",
    )
    ax.set_title("Simulated Fed Funds Paths")
    ax.set_ylabel("Annualized rate (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_mean_vs_forward(result: SimulationResult, path: Path) -> None:
    dates = result.forward_curve.dates
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, result.forward_curve.forward_rates * 100.0, label="OIS forward", linewidth=2.0)
    ax.plot(dates, result.paths.mean(axis=0) * 100.0, label="Simulation mean", linewidth=1.8)
    ax.set_title("Simulation Mean vs OIS Forward Centerline")
    ax.set_ylabel("Annualized rate (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_vol_term_structure(result: SimulationResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    months = result.volatility.months
    ax.plot(
        months,
        result.volatility.annual_normal_vol_bps,
        color="#047857",
        linewidth=2.0,
    )
    ax.scatter(
        result.volatility.source_points["option_years"].to_numpy() * 12.0,
        result.volatility.source_points["annual_normal_vol_bps"],
        color="#064e3b",
        s=24,
        label="Expiry source points",
    )
    ax.set_title("Monthly Normal Volatility Term Structure")
    ax.set_xlabel("Months from valuation date")
    ax.set_ylabel("Normal vol (bps/annum)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_terminal_histogram(result: SimulationResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(result.paths[:, -1] * 100.0, bins=35, color="#0f766e", alpha=0.75)
    ax.axvline(
        result.forward_curve.forward_rates[-1] * 100.0,
        color="#111827",
        linestyle="--",
        linewidth=2.0,
        label="Terminal OIS forward",
    )
    ax.set_title("Terminal Fed Funds Rate Distribution")
    ax.set_xlabel("Annualized rate (%)")
    ax.set_ylabel("Path count")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_outputs(result: SimulationResult, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    charts_path = output_path / "charts"
    output_path.mkdir(parents=True, exist_ok=True)
    charts_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "paths_long": output_path / "fed_funds_paths.csv",
        "paths_wide": output_path / "fed_funds_paths_wide.csv",
        "monthly_forward_curve": output_path / "monthly_forward_curve.csv",
        "monthly_volatility": output_path / "monthly_volatility.csv",
        "vol_source_points": output_path / "vol_source_points.csv",
        "path_summary": output_path / "path_summary_by_month.csv",
        "validation_summary": output_path / "validation_summary.csv",
        "assumptions": output_path / "assumptions.json",
        "fan_chart": charts_path / "path_fan_chart.png",
        "mean_vs_forward": charts_path / "mean_vs_forward.png",
        "vol_term_structure": charts_path / "vol_term_structure.png",
        "terminal_histogram": charts_path / "terminal_histogram.png",
    }

    result.paths_long_frame().to_csv(manifest["paths_long"], index=False)
    result.paths_wide_frame().to_csv(manifest["paths_wide"], index=False)
    result.forward_curve.to_frame().to_csv(manifest["monthly_forward_curve"], index=False)
    result.volatility.to_frame().to_csv(manifest["monthly_volatility"], index=False)
    result.volatility.source_points.to_csv(manifest["vol_source_points"], index=False)
    monthly_summary(result).to_csv(manifest["path_summary"], index=False)
    validation_summary(result).to_csv(manifest["validation_summary"], index=False)

    with manifest["assumptions"].open("w", encoding="utf-8") as handle:
        json.dump(result.config.assumptions(), handle, indent=2, default=_json_default)

    _plot_fan_chart(result, manifest["fan_chart"])
    _plot_mean_vs_forward(result, manifest["mean_vs_forward"])
    _plot_vol_term_structure(result, manifest["vol_term_structure"])
    _plot_terminal_histogram(result, manifest["terminal_histogram"])

    return manifest
