# Orbit Model

An interactive satellite-orbit visualizer built with **Streamlit** and **Plotly**,
using **numpy**-only orbit mechanics. Enter a set of orbit parameters (altitude,
inclination, eccentricity, RAAN, argument of perigee) and the app shows:

- A 3D scene in the Earth-Centered Inertial (ECI) frame with Earth, the orbit
  track, the satellite, the Sun direction, and (optionally) 12 monthly orbit
  planes so you can see the J2 nodal-precession sweep the plane over a year.
- Orbital period and apogee/perigee altitudes.
- RAAN and argument-of-perigee drift rates from J2.
- Beta angle vs day of year and eclipse duration vs day of year, with the
  maximum eclipse duration and the day on which it occurs.

## Requirements

- [uv](https://docs.astral.sh/uv/) (>= 0.4)
- Python 3.11+

## Install

```bash
uv sync
```

## Run the app

```bash
uv run streamlit run app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

## Run the tests

```bash
uv run pytest
```

## Layout

```
src/orbit_model/
  constants.py   # physical constants (mu, R_earth, J2, AU, ...)
  orbit.py       # Keplerian + J2 orbit math
  sun.py         # analytic Sun ECI direction
  eclipse.py     # beta angle, cylindrical-shadow eclipse duration
  plots.py       # Plotly figure builders
app.py           # Streamlit UI
tests/           # pytest checks against known orbits
```

## Notes on the physics

- Two-body Keplerian orbit for the shape, plus a first-order **J2** correction
  for RAAN and argument-of-perigee drift (secular rates only).
- Sun position from the Astronomical Almanac low-precision formula (accurate to
  about 0.01 degrees over the years 1950-2050).
- Eclipse model uses Earth's cylindrical shadow. For a circular orbit an
  analytic expression is used; for an eccentric orbit the app numerically
  propagates one orbit per day, evaluates the cylindrical-shadow condition at
  each step, and reports the shadow arc length in seconds.
