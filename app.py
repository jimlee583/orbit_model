"""Streamlit UI for the orbit visualizer."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import streamlit as st

from orbit_model.constants import R_EARTH, SIDEREAL_YEAR_DAYS
from orbit_model.eclipse import (
    beta_angle_rad,
    eclipse_duration_numeric_s,
    eclipse_fraction_circular,
    yearly_sweep,
)
from orbit_model.orbit import (
    OrbitElements,
    apogee_altitude,
    argp_rate_rad_s,
    orbital_period_s,
    perigee_altitude,
    propagate_true_anomaly,
    raan_rate_rad_s,
    sample_orbit_eci,
    sso_inclination_deg,
    perifocal_to_eci_matrix,
)
from orbit_model.plots import (
    build_beta_figure,
    build_eclipse_figure,
    build_raan_figure,
    build_scene_figure,
)
from orbit_model.sun import sun_direction_eci


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="Orbit visualizer",
    page_icon=":satellite:",
    layout="wide",
)

st.title("Satellite orbit visualizer")
st.caption(
    "Two-body Keplerian orbit with J2 secular precession, cylindrical-shadow "
    "eclipse model, and low-precision analytic Sun. All vectors are in the "
    "Earth-Centered Inertial (ECI) frame."
)


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults = {
        "altitude_km": 500.0,
        "eccentricity": 0.0,
        "inclination_deg": 0.0,
        "raan_deg": 0.0,
        "argp_deg": 0.0,
        "epoch": date(datetime.now(timezone.utc).year, 1, 1),
        "day_of_year": 0,
        "orbit_frac": 0.0,
        "show_yearly": True,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------


with st.sidebar:
    st.header("Orbit parameters")

    st.number_input(
        "Perigee altitude (km)",
        min_value=100.0,
        max_value=45_000.0,
        step=10.0,
        key="altitude_km",
        help="Altitude of perigee above Earth's equatorial radius (6378.137 km).",
    )
    st.number_input(
        "Eccentricity",
        min_value=0.0,
        max_value=0.9,
        step=0.01,
        format="%.3f",
        key="eccentricity",
    )
    st.number_input(
        "Inclination (deg)",
        min_value=0.0,
        max_value=180.0,
        step=0.1,
        key="inclination_deg",
    )
    st.number_input(
        "RAAN at epoch (deg)",
        min_value=0.0,
        max_value=360.0,
        step=1.0,
        key="raan_deg",
    )
    st.number_input(
        "Argument of perigee (deg)",
        min_value=0.0,
        max_value=360.0,
        step=1.0,
        key="argp_deg",
    )

    st.divider()
    st.header("Time")
    st.date_input("Epoch (UTC)", key="epoch")
    st.slider("Day since epoch", min_value=0, max_value=365, key="day_of_year")
    st.slider(
        "Satellite phase (fraction of orbit)",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key="orbit_frac",
    )

    st.divider()
    st.header("View")
    st.checkbox("Overlay equinox and solstice orbit planes", key="show_yearly")

    if st.button("Set inclination for Sun-synchronous"):
        try:
            i_sso = sso_inclination_deg(
                st.session_state["altitude_km"], st.session_state["eccentricity"]
            )
            st.session_state["inclination_deg"] = round(i_sso, 3)
            st.success(f"Inclination set to {i_sso:.3f} deg")
        except ValueError as e:
            st.error(str(e))


# ---------------------------------------------------------------------------
# Build orbit
# ---------------------------------------------------------------------------


try:
    elements = OrbitElements.from_user_inputs(
        perigee_altitude_km=st.session_state["altitude_km"],
        eccentricity=st.session_state["eccentricity"],
        inclination_deg=st.session_state["inclination_deg"],
        raan_deg=st.session_state["raan_deg"],
        argp_deg=st.session_state["argp_deg"],
    )
except ValueError as e:
    st.error(f"Invalid orbit: {e}")
    st.stop()

if apogee_altitude(elements.a, elements.e) < 0:
    st.error("Apogee is below Earth's surface. Reduce eccentricity or raise perigee.")
    st.stop()

# Propagate to the selected day.
day = int(st.session_state["day_of_year"])
epoch = st.session_state["epoch"]
current_date = epoch + timedelta(days=day)
dt_s = day * 86_400.0
raan_dot = raan_rate_rad_s(elements.a, elements.e, elements.i)
argp_dot = argp_rate_rad_s(elements.a, elements.e, elements.i)
day_elements = OrbitElements(
    a=elements.a,
    e=elements.e,
    i=elements.i,
    raan=elements.raan + raan_dot * dt_s,
    argp=elements.argp + argp_dot * dt_s,
    true_anomaly=elements.true_anomaly,
)
T = orbital_period_s(elements.a)
sun_hat = sun_direction_eci(current_date)
beta_today = beta_angle_rad(day_elements.raan, day_elements.i, sun_hat)
if elements.e < 1e-3:
    ecl_today_s = eclipse_fraction_circular(elements.a, beta_today) * T
else:
    ecl_today_s = eclipse_duration_numeric_s(day_elements, sun_hat)

# Satellite position at the requested orbit fraction (time-uniform).
frac = float(st.session_state["orbit_frac"])
nu = float(propagate_true_anomaly(day_elements, np.array([frac * T]))[0])
r_here = day_elements.a * (1.0 - day_elements.e**2) / (1.0 + day_elements.e * math.cos(nu))
pqw = np.array([r_here * math.cos(nu), r_here * math.sin(nu), 0.0])
R = perifocal_to_eci_matrix(day_elements.raan, day_elements.i, day_elements.argp)
sat_position = R @ pqw


# ---------------------------------------------------------------------------
# Yearly sweep (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_yearly(
    a: float,
    e: float,
    i: float,
    raan: float,
    argp: float,
    epoch_iso: str,
    num_days: int = 366,
) -> dict:
    els = OrbitElements(a=a, e=e, i=i, raan=raan, argp=argp)
    sweep = yearly_sweep(els, date.fromisoformat(epoch_iso), num_days=num_days)
    return {
        "days": sweep.days,
        "dates": sweep.dates,
        "raan_rad": sweep.raan_rad,
        "argp_rad": sweep.argp_rad,
        "beta_rad": sweep.beta_rad,
        "eclipse_s": sweep.eclipse_s,
        "sun_hat": sweep.sun_hat,
    }


with st.spinner("Sweeping year..."):
    sweep_dict = _cached_yearly(
        elements.a,
        elements.e,
        elements.i,
        elements.raan,
        elements.argp,
        epoch.isoformat(),
    )

# Rebuild a YearlySweep-like object for the plots module.
from orbit_model.eclipse import YearlySweep  # local import to avoid cycles

sweep = YearlySweep(
    days=sweep_dict["days"],
    dates=sweep_dict["dates"],
    raan_rad=sweep_dict["raan_rad"],
    argp_rad=sweep_dict["argp_rad"],
    beta_rad=sweep_dict["beta_rad"],
    eclipse_s=sweep_dict["eclipse_s"],
    sun_hat=sweep_dict["sun_hat"],
)


# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------


raan_drift_deg_per_day = math.degrees(raan_dot) * 86_400.0
argp_drift_deg_per_day = math.degrees(argp_dot) * 86_400.0
sso_target = 360.0 / SIDEREAL_YEAR_DAYS  # deg/day
sso_flag = abs(raan_drift_deg_per_day - sso_target) < 0.05

m1, m2, m3, m4 = st.columns(4)
m1.metric("Orbital period", f"{T/60:.2f} min", help="Two-body period 2*pi*sqrt(a^3/mu)")
m2.metric(
    "Apogee / perigee",
    f"{apogee_altitude(elements.a, elements.e):.0f} / "
    f"{perigee_altitude(elements.a, elements.e):.0f} km",
)
m3.metric(
    "Max eclipse this year",
    f"{sweep.max_eclipse_s/60:.2f} min",
    help=f"Occurs on {sweep.max_eclipse_date.isoformat()}",
)
m4.metric(
    "RAAN drift",
    f"{raan_drift_deg_per_day:+.4f} deg/day",
    delta="Sun-synchronous" if sso_flag else None,
)

n1, n2, n3, n4 = st.columns(4)
n1.metric("Beta today", f"{math.degrees(beta_today):+.2f} deg")
n2.metric("Eclipse today", f"{ecl_today_s/60:.2f} min")
n3.metric("Argp drift", f"{argp_drift_deg_per_day:+.4f} deg/day")
n4.metric(
    "Semi-major axis",
    f"{elements.a:,.1f} km",
    help=f"{elements.a - R_EARTH:,.1f} km above equator on average",
)


# ---------------------------------------------------------------------------
# Main 3D scene
# ---------------------------------------------------------------------------


st.subheader(f"3D view - {current_date.isoformat()} (day {day} since epoch)")
fig_scene = build_scene_figure(
    day_elements,
    sun_hat=sun_hat,
    sat_position=sat_position,
    yearly=sweep,
    show_yearly_planes=st.session_state["show_yearly"],
)
st.plotly_chart(fig_scene, width="stretch", config={"displaylogo": False})


# ---------------------------------------------------------------------------
# Yearly charts
# ---------------------------------------------------------------------------


c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_beta_figure(sweep), width="stretch")
with c2:
    st.plotly_chart(build_eclipse_figure(sweep), width="stretch")

st.plotly_chart(build_raan_figure(sweep), width="stretch")


with st.expander("How the numbers are computed"):
    st.markdown(
        """
- **Semi-major axis**: `a = (R_earth + h_p) / (1 - e)` where `h_p` is the perigee altitude.
- **Period**: `T = 2*pi*sqrt(a^3 / mu)`.
- **J2 secular RAAN rate**: `dOmega/dt = -1.5 * n * J2 * (R_e / p)^2 * cos(i)` with `p = a(1 - e^2)`.
- **J2 secular argp rate**: `domega/dt = 0.75 * n * J2 * (R_e / p)^2 * (5 cos^2 i - 1)`.
- **Sun direction**: Astronomical Almanac low-precision formula (accurate to about 0.01 deg for 1950-2050).
- **Beta angle**: `beta = asin(h_hat . s_hat)`.
- **Eclipse**: cylindrical Earth shadow. Circular orbits use the closed form `f = acos(sqrt(a^2 - R_e^2) / (a cos beta)) / pi`; eccentric orbits are propagated numerically with Kepler's equation.
- **Sun-synchronous condition**: `dOmega/dt = 360 deg / sidereal_year` (about 0.9856 deg/day).
        """
    )
