"""Analytic Sun direction in the Earth-Centered Inertial (ECI) frame.

Implements the low-precision Sun-position algorithm from the Astronomical
Almanac (Section C, "The Sun"). Valid to about 0.01 degrees between 1950 and
2050, which is more than sufficient for orbit visualization and beta/eclipse
calculations.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timezone

import numpy as np

from orbit_model.constants import AU_KM, JD_J2000, OBLIQUITY_J2000_RAD


# ---------------------------------------------------------------------------
# Julian date helpers
# ---------------------------------------------------------------------------


def julian_date(dt: datetime) -> float:
    """Return the Julian date for a UTC :class:`datetime`."""

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    y, m = dt_utc.year, dt_utc.month
    d = (
        dt_utc.day
        + (dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0) / 24.0
    )
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + B - 1524.5


def _to_datetime(d: date | datetime) -> datetime:
    if isinstance(d, datetime):
        return d
    return datetime.combine(d, time(12, 0, 0), tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Sun position
# ---------------------------------------------------------------------------


def _sun_ecliptic(jd: float) -> tuple[float, float, float]:
    """Return (ecliptic longitude, obliquity, Sun-Earth distance in AU).

    Follows the low-precision Sun formulas from the Astronomical Almanac.
    """

    n = jd - JD_J2000  # days since J2000.0
    # Mean longitude of the Sun, corrected for aberration
    L = math.radians((280.460 + 0.9856474 * n) % 360.0)
    # Mean anomaly
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    # Ecliptic longitude
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2.0 * g)
    # Obliquity of the ecliptic (small secular term)
    eps = OBLIQUITY_J2000_RAD - math.radians(0.0000004) * n
    # Sun-Earth distance in AU
    R_au = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2.0 * g)
    return lam, eps, R_au


def sun_direction_eci(when: date | datetime) -> np.ndarray:
    """Return a unit vector pointing from Earth to the Sun in ECI.

    ``when`` can be a :class:`date` (interpreted as 12:00 UTC on that day) or a
    :class:`datetime` (naive datetimes are treated as UTC).
    """

    jd = julian_date(_to_datetime(when))
    lam, eps, _R = _sun_ecliptic(jd)
    x = math.cos(lam)
    y = math.cos(eps) * math.sin(lam)
    z = math.sin(eps) * math.sin(lam)
    return np.array([x, y, z])


def sun_position_eci_km(when: date | datetime) -> np.ndarray:
    """Return the Sun position vector in ECI, scaled to kilometers."""

    jd = julian_date(_to_datetime(when))
    lam, eps, R_au = _sun_ecliptic(jd)
    r = R_au * AU_KM
    return r * np.array(
        [
            math.cos(lam),
            math.cos(eps) * math.sin(lam),
            math.sin(eps) * math.sin(lam),
        ]
    )


def sun_declination_deg(when: date | datetime) -> float:
    """Return the declination of the Sun in degrees."""

    s = sun_direction_eci(when)
    return math.degrees(math.asin(s[2]))
