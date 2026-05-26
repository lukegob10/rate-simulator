from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

import pandas as pd

from stochastic_fedfunds.tenors import tenor_to_years


_YYYYMMDD_RE = re.compile(r"(?P<date>\d{8})")
_VOL_ROW_RE = re.compile(
    r"^\s*(?P<option>\d+(?:\.\d+)?[DWMY])\s*x\s*"
    r"(?P<swap>\d+(?:\.\d+)?[DWMY])\s+USD\s+Normal\s+Annual\s+RFR\s+Vol"
    r"\s*\(BPS/ANNUM\)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OISCurveData:
    valuation_date: pd.Timestamp
    quotes: pd.DataFrame
    source_path: Path


@dataclass(frozen=True)
class NormalVolCubeData:
    valuation_date: pd.Timestamp
    quotes: pd.DataFrame
    source_path: Path


def _valuation_date_from_column(column_name: str, fallback: Optional[str]) -> pd.Timestamp:
    if fallback:
        return pd.Timestamp(fallback)

    match = _YYYYMMDD_RE.search(str(column_name))
    if match:
        return pd.to_datetime(match.group("date"), format="%Y%m%d")

    return pd.Timestamp(column_name)


def load_ois_curve(
    path: str | Path,
    valuation_date: Optional[str] = None,
    day_count_basis: float = 365.0,
) -> OISCurveData:
    """Load a two-column Fed Funds OIS curve CSV.

    Expected local schema:
        Date,"20251231, USD Par OIS Rate"
        7D,3.6339
        1M,3.62964
        ...
    """

    source_path = Path(path)
    raw = pd.read_csv(source_path)
    if raw.shape[1] < 2:
        raise ValueError(f"OIS curve file must have at least two columns: {source_path}")

    tenor_col = raw.columns[0]
    rate_col = raw.columns[1]
    quotes = raw[[tenor_col, rate_col]].copy()
    quotes.columns = ["tenor", "par_rate_percent"]
    quotes["par_rate_percent"] = pd.to_numeric(quotes["par_rate_percent"], errors="coerce")
    quotes = quotes.dropna(subset=["tenor", "par_rate_percent"])
    quotes["tenor"] = quotes["tenor"].astype(str).str.strip()
    quotes["tenor_years"] = quotes["tenor"].map(lambda value: tenor_to_years(value, day_count_basis))
    quotes["par_rate"] = quotes["par_rate_percent"] / 100.0
    quotes = quotes.sort_values("tenor_years").reset_index(drop=True)

    if quotes.empty:
        raise ValueError(f"No usable OIS curve rows found in {source_path}")

    val_date = _valuation_date_from_column(rate_col, valuation_date)
    return OISCurveData(valuation_date=val_date, quotes=quotes, source_path=source_path)


def load_normal_vol_cube(
    path: str | Path,
    valuation_date: Optional[str] = None,
    day_count_basis: float = 365.0,
) -> NormalVolCubeData:
    """Load the two-column SOFR ATM normal swaption vol cube CSV."""

    source_path = Path(path)
    raw = pd.read_csv(source_path)
    if raw.shape[1] < 2:
        raise ValueError(f"Vol cube file must have at least two columns: {source_path}")

    desc_col = raw.columns[0]
    value_col = raw.columns[1]
    rows = []
    for _, row in raw[[desc_col, value_col]].iterrows():
        description = str(row[desc_col]).strip()
        match = _VOL_ROW_RE.match(description)
        if not match:
            continue

        normal_vol_bps = pd.to_numeric(row[value_col], errors="coerce")
        if pd.isna(normal_vol_bps):
            continue

        option_tenor = match.group("option").upper()
        swap_tenor = match.group("swap").upper()
        rows.append(
            {
                "option_tenor": option_tenor,
                "swap_tenor": swap_tenor,
                "option_years": tenor_to_years(option_tenor, day_count_basis),
                "swap_years": tenor_to_years(swap_tenor, day_count_basis),
                "normal_vol_bps": float(normal_vol_bps),
                "normal_vol": float(normal_vol_bps) / 10000.0,
                "description": description,
            }
        )

    quotes = pd.DataFrame(rows)
    if quotes.empty:
        raise ValueError(f"No usable normal vol rows found in {source_path}")

    quotes = quotes.sort_values(["option_years", "swap_years"]).reset_index(drop=True)
    val_date = pd.Timestamp(valuation_date) if valuation_date else pd.Timestamp(value_col)
    return NormalVolCubeData(valuation_date=val_date, quotes=quotes, source_path=source_path)
