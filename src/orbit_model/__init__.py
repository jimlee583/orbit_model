"""Orbit Model: pure-Python orbit mechanics helpers used by the Streamlit app."""

from orbit_model.constants import (
    AU_KM,
    J2,
    MU_EARTH,
    R_EARTH,
    R_SUN,
    SIDEREAL_YEAR_DAYS,
)
from orbit_model.orbit import (
    OrbitElements,
    apogee_altitude,
    orbit_normal_eci,
    orbital_period_s,
    perifocal_to_eci_matrix,
    perigee_altitude,
    propagate_true_anomaly,
    raan_rate_rad_s,
    sample_orbit_eci,
    semi_major_axis_from_perigee,
    sso_inclination_deg,
    argp_rate_rad_s,
)
from orbit_model.sun import sun_direction_eci, sun_declination_deg
from orbit_model.eclipse import (
    beta_angle_rad,
    eclipse_fraction_circular,
    eclipse_duration_numeric_s,
    yearly_sweep,
)

__all__ = [
    "AU_KM",
    "J2",
    "MU_EARTH",
    "R_EARTH",
    "R_SUN",
    "SIDEREAL_YEAR_DAYS",
    "OrbitElements",
    "apogee_altitude",
    "argp_rate_rad_s",
    "beta_angle_rad",
    "eclipse_duration_numeric_s",
    "eclipse_fraction_circular",
    "orbit_normal_eci",
    "orbital_period_s",
    "perifocal_to_eci_matrix",
    "perigee_altitude",
    "propagate_true_anomaly",
    "raan_rate_rad_s",
    "sample_orbit_eci",
    "semi_major_axis_from_perigee",
    "sso_inclination_deg",
    "sun_declination_deg",
    "sun_direction_eci",
    "yearly_sweep",
]
