from __future__ import annotations

import re


_TENOR_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)(?P<unit>[DWMY])\s*$", re.IGNORECASE)


def tenor_to_years(tenor: str, day_count_basis: float = 365.0) -> float:
    """Convert market tenor strings like 7D, 3M, 1.5Y to year fractions."""

    match = _TENOR_RE.match(str(tenor))
    if not match:
        raise ValueError(f"Unsupported tenor format: {tenor!r}")

    value = float(match.group("value"))
    unit = match.group("unit").upper()
    if unit == "D":
        return value / day_count_basis
    if unit == "W":
        return value * 7.0 / day_count_basis
    if unit == "M":
        return value / 12.0
    if unit == "Y":
        return value
    raise ValueError(f"Unsupported tenor unit: {unit!r}")
