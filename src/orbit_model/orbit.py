"""Two-body Keplerian orbit mechanics with a first-order J2 secular model.

All angles are stored in radians and all distances in kilometers. Convenience
functions expose degrees for the user interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from orbit_model.constants import J2, MU_EARTH, R_EARTH, SIDEREAL_YEAR_DAYS


# ---------------------------------------------------------------------------
# Element container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrbitElements:
    """Classical Keplerian orbit elements.

    Angles in radians, distances in kilometers.
    """

    a: float  #: semi-major axis, km
    e: float  #: eccentricity (0 <= e < 1)
    i: float  #: inclination, rad
    raan: float  #: right ascension of the ascending node, rad
    argp: float  #: argument of perigee, rad
    true_anomaly: float = 0.0  #: true anomaly at epoch, rad

    @classmethod
    def from_user_inputs(
        cls,
        perigee_altitude_km: float,
        eccentricity: float,
        inclination_deg: float,
        raan_deg: float,
        argp_deg: float = 0.0,
        true_anomaly_deg: float = 0.0,
    ) -> "OrbitElements":
        """Build an :class:`OrbitElements` from the sidebar inputs.

        The user provides *perigee* altitude (km above the equatorial radius);
        eccentricity and perigee altitude jointly determine the semi-major
        axis.
        """

        if eccentricity < 0.0 or eccentricity >= 1.0:
            raise ValueError("eccentricity must satisfy 0 <= e < 1")
        if perigee_altitude_km <= 0.0:
            raise ValueError("perigee altitude must be positive (km above Earth)")

        a = semi_major_axis_from_perigee(perigee_altitude_km, eccentricity)
        return cls(
            a=a,
            e=eccentricity,
            i=math.radians(inclination_deg),
            raan=math.radians(raan_deg),
            argp=math.radians(argp_deg),
            true_anomaly=math.radians(true_anomaly_deg),
        )


# ---------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------


def semi_major_axis_from_perigee(perigee_altitude_km: float, eccentricity: float) -> float:
    """Return ``a`` in km given the perigee altitude and eccentricity."""

    rp = R_EARTH + perigee_altitude_km
    return rp / (1.0 - eccentricity)


def perigee_altitude(a: float, e: float) -> float:
    """Return the perigee altitude in km above the equatorial radius."""

    return a * (1.0 - e) - R_EARTH


def apogee_altitude(a: float, e: float) -> float:
    """Return the apogee altitude in km above the equatorial radius."""

    return a * (1.0 + e) - R_EARTH


def orbital_period_s(a: float) -> float:
    """Return the two-body Keplerian orbital period in seconds."""

    return 2.0 * math.pi * math.sqrt(a**3 / MU_EARTH)


def mean_motion_rad_s(a: float) -> float:
    """Return the mean motion ``n = sqrt(mu / a^3)`` in rad/s."""

    return math.sqrt(MU_EARTH / a**3)


# ---------------------------------------------------------------------------
# J2 secular rates
# ---------------------------------------------------------------------------


def _factor_J2(a: float, e: float) -> float:
    p = a * (1.0 - e * e)
    n = mean_motion_rad_s(a)
    return n * J2 * (R_EARTH / p) ** 2


def raan_rate_rad_s(a: float, e: float, i_rad: float) -> float:
    """Return the J2 secular nodal precession rate ``dRAAN/dt`` in rad/s.

    Negative for prograde orbits (``i < 90 deg``) and positive for retrograde
    orbits (``i > 90 deg``), which enables the Sun-synchronous condition.
    """

    return -1.5 * _factor_J2(a, e) * math.cos(i_rad)


def argp_rate_rad_s(a: float, e: float, i_rad: float) -> float:
    """Return the J2 secular argument-of-perigee rate in rad/s."""

    return 0.75 * _factor_J2(a, e) * (5.0 * math.cos(i_rad) ** 2 - 1.0)


def sso_inclination_deg(perigee_altitude_km: float, eccentricity: float = 0.0) -> float:
    """Return the inclination (deg) that produces a Sun-synchronous orbit.

    Solves ``dRAAN/dt = 360 deg / sidereal_year`` for the inclination.
    Raises :class:`ValueError` if no real solution exists (orbit too high).
    """

    a = semi_major_axis_from_perigee(perigee_altitude_km, eccentricity)
    target = 2.0 * math.pi / (SIDEREAL_YEAR_DAYS * 86_400.0)  # rad/s
    denom = 1.5 * _factor_J2(a, eccentricity)
    if denom == 0.0:
        raise ValueError("Zero J2 factor; cannot compute SSO inclination")
    cos_i = -target / denom
    if not -1.0 <= cos_i <= 1.0:
        raise ValueError("No SSO inclination exists for these parameters")
    return math.degrees(math.acos(cos_i))


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------


def perifocal_to_eci_matrix(raan: float, i: float, argp: float) -> np.ndarray:
    """Return the 3x3 rotation ``R`` such that ``r_eci = R @ r_pqw``.

    Uses the standard 3-1-3 Euler sequence ``R_z(-raan) R_x(-i) R_z(-argp)``.
    """

    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(i), math.sin(i)
    cw, sw = math.cos(argp), math.sin(argp)

    return np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ]
    )


def orbit_normal_eci(raan: float, i: float) -> np.ndarray:
    """Return the unit angular-momentum vector (orbit normal) in ECI."""

    return np.array(
        [
            math.sin(raan) * math.sin(i),
            -math.cos(raan) * math.sin(i),
            math.cos(i),
        ]
    )


# ---------------------------------------------------------------------------
# Propagation helpers
# ---------------------------------------------------------------------------


def _radius_at_true_anomaly(a: float, e: float, nu: np.ndarray | float) -> np.ndarray | float:
    return a * (1.0 - e * e) / (1.0 + e * np.cos(nu))


def sample_orbit_eci(
    elements: OrbitElements, num_points: int = 361
) -> np.ndarray:
    """Return an ``(N, 3)`` array of ECI positions sampling one full orbit."""

    nu = np.linspace(0.0, 2.0 * math.pi, num_points)
    r = _radius_at_true_anomaly(elements.a, elements.e, nu)
    pqw = np.column_stack((r * np.cos(nu), r * np.sin(nu), np.zeros_like(nu)))
    R = perifocal_to_eci_matrix(elements.raan, elements.i, elements.argp)
    return pqw @ R.T


def _solve_kepler(mean_anomaly: np.ndarray, e: float, tol: float = 1e-12) -> np.ndarray:
    """Vectorized Kepler equation solver via Newton iteration."""

    M = np.asarray(mean_anomaly, dtype=float)
    E = M + e * np.sin(M)  # initial guess
    for _ in range(50):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        delta = f / fp
        E -= delta
        if np.max(np.abs(delta)) < tol:
            break
    return E


def _true_from_eccentric(E: np.ndarray, e: float) -> np.ndarray:
    return 2.0 * np.arctan2(
        math.sqrt(1.0 + e) * np.sin(E / 2.0),
        math.sqrt(1.0 - e) * np.cos(E / 2.0),
    )


def propagate_true_anomaly(
    elements: OrbitElements, times_s: np.ndarray
) -> np.ndarray:
    """Return true anomaly (rad) at each time in ``times_s`` past epoch."""

    n = mean_motion_rad_s(elements.a)
    M = elements.true_anomaly + n * np.asarray(times_s)
    E = _solve_kepler(M, elements.e)
    return _true_from_eccentric(E, elements.e)


def positions_over_orbit(elements: OrbitElements, num_points: int = 720) -> np.ndarray:
    """Return positions at equally spaced *time* samples over one full period."""

    T = orbital_period_s(elements.a)
    t = np.linspace(0.0, T, num_points, endpoint=False)
    nu = propagate_true_anomaly(elements, t)
    r = _radius_at_true_anomaly(elements.a, elements.e, nu)
    pqw = np.column_stack((r * np.cos(nu), r * np.sin(nu), np.zeros_like(nu)))
    R = perifocal_to_eci_matrix(elements.raan, elements.i, elements.argp)
    return pqw @ R.T
