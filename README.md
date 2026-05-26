# Stochastic Fed Funds Path Simulator

This repo builds monthly stochastic Fed Funds Effective Rate paths from the local market data files:

- `data/20251231-fedfunds-ois.csv`
- `data/20251231-vol-dataset.csv`

Run the default simulation:

```powershell
python -m stochastic_fedfunds
```

Equivalent convenience command:

```powershell
python .\run_simulation.py
```

Default outputs are written under `output/`:

- `fed_funds_paths.csv`: long-form path output
- `fed_funds_paths_wide.csv`: one row per path
- `monthly_forward_curve.csv`: OIS-derived monthly centerline
- `monthly_volatility.csv`: monthly annual normal vol term structure
- `path_summary_by_month.csv`: monthly mean, standard deviation, and quantiles
- `validation_summary.csv`: model run checks
- `assumptions.json`: explicit model and data assumptions
- `charts/*.png`: validation charts

## Model

The implementation uses a one-factor Hull-White style approximation centered on the Fed Funds OIS monthly forward curve:

```text
x[t+1] = exp(-a * dt) * x[t] + sigma[t] * sqrt((1 - exp(-2*a*dt)) / (2*a)) * Z[t]
r[t+1] = f[t+1] + x[t+1]
```

where:

- `f[t]` is the monthly forward rate inferred from the Fed Funds OIS curve.
- `x[t]` is the stochastic deviation from the centerline.
- `a` is annual mean reversion.
- `sigma[t]` is the monthly term structure built from SOFR ATM normal swaption vols.
- `dt = 1/12`.

## Key Assumptions

- OIS par quotes are treated as zero-equivalent continuously compounded rates, then linearly interpolated by tenor.
- Monthly forward rates are extracted from interpolated discount factors.
- SOFR ATM normal swaption vols are used as a proxy for Fed Funds short-rate normal volatility.
- The default vol term structure averages normal vols across swap tenors for each option expiry.
- Normal short-rate dynamics can produce negative rates. Use `--rate-floor` to impose a floor.

## Configurable Parameters

```powershell
python -m stochastic_fedfunds `
  --horizon-months 60 `
  --num-paths 500 `
  --mean-reversion 0.10 `
  --initial-rate-shock-bps 400 `
  --random-seed 20251231 `
  --vol-aggregation mean_by_option_expiry `
  --short-rate-vol-multiplier 1.0
```

`--initial-rate-shock-bps` applies an instantaneous parallel shock to the
short-rate deviation at the valuation date. For example, `400` means paths
start from a +4.00% shock and then mean-revert around the OIS forward curve.

Run tests:

```powershell
pytest
```
