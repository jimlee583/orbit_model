"""Beta angle and eclipse duration models (Earth cylindrical shadow)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from orbit_model.constants import R_EARTH
from orbit_model.orbit import (
    OrbitElements,
    orbit_normal_eci,
    orbital_period_s,
    positions_over_orbit,
    raan_rate_rad_s,
    argp_rate_rad_s,
)
from orbit_model.sun import sun_direction_eci


# ---------------------------------------------------------------------------
# Beta angle
# ---------------------------------------------------------------------------


def beta_angle_rad(raan: float, i_rad: float, sun_hat: np.ndarray) -> float:
    """Return the beta angle (sun elevation above the orbit plane), radians.

    ``sun_hat`` must be a unit vector in the same inertial frame as ``raan``.
    """

    h_hat = orbit_normal_eci(raan, i_rad)
    dot = float(np.clip(np.dot(h_hat, sun_hat), -1.0, 1.0))
    return math.asin(dot)


# ---------------------------------------------------------------------------
# Cylindrical-shadow eclipse duration
# ---------------------------------------------------------------------------


def _critical_beta_rad(a_km: float) -> float:
    """Beta above which a circular orbit at radius ``a`` never enters shadow."""

    ratio = min(1.0, R_EARTH / a_km)
    return math.asin(ratio)


def eclipse_fraction_circular(a_km: float, beta_rad: float) -> float:
    """Fraction of a circular orbit spent in Earth's cylindrical shadow.

    Standard closed-form (Vallado, "Fundamentals of Astrodynamics", 3rd ed.,
    eq. 5-4). Returns 0 when the beta angle exceeds the critical value.
    """

    beta_star = _critical_beta_rad(a_km)
    if abs(beta_rad) >= beta_star:
        return 0.0
    num = math.sqrt(a_km * a_km - R_EARTH * R_EARTH)
    denom = a_km * math.cos(beta_rad)
    return math.acos(num / denom) / math.pi


def eclipse_duration_numeric_s(
    elements: OrbitElements,
    sun_hat: np.ndarray,
    num_samples: int = 720,
) -> float:
    """Return time in Earth's cylindrical shadow over one orbit, in seconds.

    Numerically propagates one full orbit using Kepler's equation and evaluates
    the cylindrical-shadow condition at each sample. Works for eccentric
    orbits.
    """

    T = orbital_period_s(elements.a)
    positions = positions_over_orbit(elements, num_points=num_samples)
    # Projections onto sun direction and perpendicular distance from the
    # Earth-sun axis.
    proj = positions @ sun_hat
    perp_sq = np.einsum("ij,ij->i", positions, positions) - proj**2
    in_shadow = (proj < 0.0) & (perp_sq < R_EARTH * R_EARTH)
    dt = T / num_samples
    return float(np.count_nonzero(in_shadow) * dt)


# ---------------------------------------------------------------------------
# Yearly sweep
# ---------------------------------------------------------------------------


@dataclass
class YearlySweep:
    """Container returned by :func:`yearly_sweep`."""

    days: np.ndarray  #: day of year, 0-based
    dates: list[date]
    raan_rad: np.ndarray
    argp_rad: np.ndarray
    beta_rad: np.ndarray
    eclipse_s: np.ndarray
    sun_hat: np.ndarray  # (N, 3) unit vectors

    @property
    def max_eclipse_idx(self) -> int:
        return int(np.argmax(self.eclipse_s))

    @property
    def max_eclipse_s(self) -> float:
        return float(self.eclipse_s[self.max_eclipse_idx])

    @property
    def max_eclipse_date(self) -> date:
        return self.dates[self.max_eclipse_idx]


def yearly_sweep(
    elements: OrbitElements,
    epoch: date,
    num_days: int = 366,
    step_days: int = 1,
    use_numeric_eclipse: bool | None = None,
) -> YearlySweep:
    """Sweep the orbit through ``num_days`` starting at ``epoch``.

    For each day it computes the J2-updated RAAN and argument of perigee, the
    Sun direction, the beta angle, and the eclipse duration in seconds.

    If ``use_numeric_eclipse`` is ``None`` (default), the analytic circular
    formula is used when the orbit is nearly circular (``e < 1e-3``); otherwise
    a numeric integration is used per day.
    """

    if use_numeric_eclipse is None:
        use_numeric_eclipse = elements.e >= 1e-3

    day_indices = np.arange(0, num_days, step_days, dtype=int)
    dates = [epoch + timedelta(days=int(k)) for k in day_indices]

    raan_dot = raan_rate_rad_s(elements.a, elements.e, elements.i)
    argp_dot = argp_rate_rad_s(elements.a, elements.e, elements.i)
    T = orbital_period_s(elements.a)

    raan = np.empty(day_indices.size)
    argp = np.empty(day_indices.size)
    beta = np.empty(day_indices.size)
    ecl = np.empty(day_indices.size)
    sun_vecs = np.empty((day_indices.size, 3))

    for idx, k in enumerate(day_indices):
        dt_s = float(k) * 86_400.0
        raan[idx] = elements.raan + raan_dot * dt_s
        argp[idx] = elements.argp + argp_dot * dt_s
        s = sun_direction_eci(dates[idx])
        sun_vecs[idx] = s
        beta[idx] = beta_angle_rad(raan[idx], elements.i, s)
        if use_numeric_eclipse:
            day_elements = OrbitElements(
                a=elements.a,
                e=elements.e,
                i=elements.i,
                raan=raan[idx],
                argp=argp[idx],
                true_anomaly=elements.true_anomaly,
            )
            ecl[idx] = eclipse_duration_numeric_s(day_elements, s)
        else:
            ecl[idx] = eclipse_fraction_circular(elements.a, beta[idx]) * T

    return YearlySweep(
        days=day_indices,
        dates=dates,
        raan_rad=raan,
        argp_rad=argp,
        beta_rad=beta,
        eclipse_s=ecl,
        sun_hat=sun_vecs,
    )
