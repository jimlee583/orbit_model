"""Sanity checks against known orbit properties."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from orbit_model import (
    OrbitElements,
    R_EARTH,
    argp_rate_rad_s,
    beta_angle_rad,
    eclipse_fraction_circular,
    orbital_period_s,
    raan_rate_rad_s,
    sample_orbit_eci,
    semi_major_axis_from_perigee,
    sso_inclination_deg,
    sun_declination_deg,
    sun_direction_eci,
)
from orbit_model.eclipse import yearly_sweep
from orbit_model.orbit import perifocal_to_eci_matrix, orbit_normal_eci
from orbit_model.plots import seasonal_plane_picks


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_perifocal_identity_matrix():
    R = perifocal_to_eci_matrix(0.0, 0.0, 0.0)
    assert np.allclose(R, np.eye(3), atol=1e-12)


def test_orbit_normal_is_z_for_equatorial():
    h = orbit_normal_eci(raan=1.234, i=0.0)
    assert np.allclose(h, np.array([0.0, 0.0, 1.0]), atol=1e-12)


def test_orbit_normal_is_unit_length():
    for raan_deg in (0.0, 45.0, 123.0, 359.0):
        for i_deg in (0.0, 30.0, 51.6, 90.0, 98.6, 180.0):
            h = orbit_normal_eci(math.radians(raan_deg), math.radians(i_deg))
            assert abs(np.linalg.norm(h) - 1.0) < 1e-12


def test_sample_orbit_lies_in_plane():
    els = OrbitElements.from_user_inputs(
        perigee_altitude_km=500.0,
        eccentricity=0.0,
        inclination_deg=45.0,
        raan_deg=30.0,
    )
    pts = sample_orbit_eci(els, num_points=181)
    h_hat = orbit_normal_eci(els.raan, els.i)
    residuals = pts @ h_hat
    assert np.max(np.abs(residuals)) < 1e-8


def test_sample_orbit_radius_matches_semi_major_axis_for_circular():
    els = OrbitElements.from_user_inputs(
        perigee_altitude_km=400.0,
        eccentricity=0.0,
        inclination_deg=0.0,
        raan_deg=0.0,
    )
    r = np.linalg.norm(sample_orbit_eci(els, num_points=361), axis=1)
    assert np.allclose(r, els.a, atol=1e-6)


# ---------------------------------------------------------------------------
# Period
# ---------------------------------------------------------------------------


def test_iss_period_close_to_92_min():
    a = semi_major_axis_from_perigee(420.0, 0.0)
    T = orbital_period_s(a)
    assert 92.5 < T / 60.0 < 93.0


def test_geo_period_is_one_sidereal_day():
    # GEO altitude ~ 35786 km, period ~ 23h 56m
    a = 42_164.0
    T = orbital_period_s(a)
    assert abs(T - 86_164.0) < 60.0  # within a minute


# ---------------------------------------------------------------------------
# J2 rates
# ---------------------------------------------------------------------------


def test_iss_raan_drift_about_minus_5_deg_per_day():
    els = OrbitElements.from_user_inputs(420.0, 0.0, 51.6, 0.0)
    d_omega = math.degrees(raan_rate_rad_s(els.a, els.e, els.i)) * 86_400.0
    assert -5.2 < d_omega < -4.8


def test_sso_condition_matches_sidereal_year():
    i_sso = sso_inclination_deg(800.0)
    els = OrbitElements.from_user_inputs(800.0, 0.0, i_sso, 0.0)
    d_omega = math.degrees(raan_rate_rad_s(els.a, els.e, els.i)) * 86_400.0
    assert abs(d_omega - 360.0 / 365.256_363_004) < 1e-4


def test_sso_inclination_is_retrograde_at_800_km():
    i_sso = sso_inclination_deg(800.0)
    assert 98.0 < i_sso < 99.5


def test_argp_rate_zero_at_critical_inclination():
    # 5 cos^2(i) - 1 = 0 -> i = 63.4349 deg
    els = OrbitElements.from_user_inputs(1000.0, 0.1, 63.4349, 0.0)
    rate = argp_rate_rad_s(els.a, els.e, els.i)
    assert abs(rate) < 1e-10


# ---------------------------------------------------------------------------
# Sun position
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "when, expected_dec",
    [
        (date(2025, 3, 20), 0.0),     # vernal equinox
        (date(2025, 6, 21), 23.44),   # summer solstice
        (date(2025, 9, 22), 0.0),     # autumnal equinox
        (date(2025, 12, 21), -23.44), # winter solstice
    ],
)
def test_sun_declination(when, expected_dec):
    dec = sun_declination_deg(when)
    assert abs(dec - expected_dec) < 0.6  # within about half a degree


def test_sun_direction_is_unit_vector():
    v = sun_direction_eci(date(2025, 6, 21))
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Eclipse
# ---------------------------------------------------------------------------


def test_leo_eclipse_zero_beta_is_about_35_minutes():
    a = semi_major_axis_from_perigee(400.0, 0.0)
    T = orbital_period_s(a)
    ecl = eclipse_fraction_circular(a, 0.0) * T
    assert 35.0 * 60.0 < ecl < 37.0 * 60.0


def test_no_eclipse_above_critical_beta():
    a = semi_major_axis_from_perigee(500.0, 0.0)
    beta_star = math.asin(R_EARTH / a)
    assert eclipse_fraction_circular(a, beta_star + math.radians(0.5)) == 0.0


def test_beta_angle_uses_orbit_normal_dot_sun():
    els = OrbitElements.from_user_inputs(500.0, 0.0, 45.0, 0.0)
    h_hat = orbit_normal_eci(els.raan, els.i)
    beta = beta_angle_rad(els.raan, els.i, h_hat)  # sun along orbit normal
    assert abs(beta - math.pi / 2.0) < 1e-12


def test_geo_orbit_has_short_or_zero_eclipse():
    a = 42_164.0
    # At beta = 0 the eclipse fraction is asin(R/a)/pi ~ 4.6 deg -> ~72 min.
    ecl_s = eclipse_fraction_circular(a, 0.0) * orbital_period_s(a)
    assert 60.0 * 60.0 < ecl_s < 75.0 * 60.0


def test_seasonal_planes_are_the_equinoxes_and_solstices():
    els = OrbitElements.from_user_inputs(500.0, 0.0, 51.6, 0.0)
    sweep = yearly_sweep(els, date(2025, 1, 1), num_days=366)
    picks = seasonal_plane_picks(sweep)
    labels = [label for _k, label, _color in picks]
    assert labels == [
        "March equinox",
        "June solstice",
        "September equinox",
        "December solstice",
    ]
    dates = [sweep.dates[k] for k, _label, _color in picks]
    assert dates == [
        date(2025, 3, 20),
        date(2025, 6, 21),
        date(2025, 9, 22),
        date(2025, 12, 21),
    ]
