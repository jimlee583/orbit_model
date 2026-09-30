"""Plotly figure builders for the Streamlit app."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from orbit_model.constants import OBLIQUITY_J2000_RAD, R_EARTH
from orbit_model.eclipse import YearlySweep
from orbit_model.orbit import (
    OrbitElements,
    orbit_normal_eci,
    sample_orbit_eci,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sphere_mesh(radius: float, n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * math.pi, n)
    v = np.linspace(0.0, math.pi, n // 2)
    uu, vv = np.meshgrid(u, v)
    x = radius * np.cos(uu) * np.sin(vv)
    y = radius * np.sin(uu) * np.sin(vv)
    z = radius * np.cos(vv)
    return x, y, z


def _earth_surface(sun_hat: np.ndarray | None = None) -> go.Surface:
    """Earth sphere shaded by the dot product with the Sun direction."""

    x, y, z = _sphere_mesh(R_EARTH)
    if sun_hat is not None:
        # Cosine of angle between surface normal and sun direction.
        norms = np.sqrt(x**2 + y**2 + z**2)
        cos_theta = (x * sun_hat[0] + y * sun_hat[1] + z * sun_hat[2]) / norms
        surfacecolor = cos_theta
    else:
        surfacecolor = z / R_EARTH
    return go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=surfacecolor,
        colorscale=[
            [0.0, "rgb(6, 12, 40)"],
            [0.5, "rgb(30, 60, 110)"],
            [1.0, "rgb(180, 210, 255)"],
        ],
        showscale=False,
        opacity=1.0,
        lighting=dict(ambient=0.55, diffuse=0.6, specular=0.15, roughness=0.9),
        lightposition=dict(x=0, y=0, z=0),
        name="Earth",
        hoverinfo="skip",
    )


def _equator_trace() -> go.Scatter3d:
    theta = np.linspace(0.0, 2.0 * math.pi, 200)
    x = R_EARTH * np.cos(theta)
    y = R_EARTH * np.sin(theta)
    z = np.zeros_like(theta)
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color="rgba(200, 200, 200, 0.6)", width=1, dash="dot"),
        name="Equator",
        hoverinfo="skip",
        showlegend=False,
    )


def _axis_trace(name: str, direction: np.ndarray, length: float, color: str) -> go.Scatter3d:
    end = direction * length
    return go.Scatter3d(
        x=[0, end[0]],
        y=[0, end[1]],
        z=[0, end[2]],
        mode="lines+text",
        line=dict(color=color, width=3),
        text=["", name],
        textposition="top center",
        name=name,
        showlegend=False,
        hoverinfo="skip",
    )


# Sun ecliptic longitudes of the equinoxes and solstices, with legend colors.
_SEASONS: tuple[tuple[float, str, str], ...] = (
    (0.0, "Vernal equinox", "#7CFFB2"),
    (math.pi / 2.0, "Summer solstice", "#FFD54A"),
    (math.pi, "Autumnal equinox", "#FF8A4C"),
    (3.0 * math.pi / 2.0, "Winter solstice", "#7EB6FF"),
)


def _sun_ecliptic_longitude(sun_hat: np.ndarray) -> np.ndarray:
    """Ecliptic longitude (rad) of ECI Sun unit vectors, shape ``(N, 3)``."""

    eps = OBLIQUITY_J2000_RAD
    y_ecl = math.cos(eps) * sun_hat[:, 1] + math.sin(eps) * sun_hat[:, 2]
    return np.arctan2(y_ecl, sun_hat[:, 0])


def seasonal_plane_picks(
    yearly: YearlySweep,
    max_offset_rad: float = math.radians(2.0),
) -> list[tuple[int, str, str]]:
    """Days in ``yearly`` nearest the equinoxes and solstices.

    Each item is ``(index, label, color)``, ordered by date. A season is
    left out when the sweep never comes within ``max_offset_rad`` of that
    Sun ecliptic longitude.
    """

    lam = _sun_ecliptic_longitude(yearly.sun_hat)
    picks: list[tuple[int, str, str]] = []
    for target, label, color in _SEASONS:
        delta = np.abs((lam - target + math.pi) % (2.0 * math.pi) - math.pi)
        k = int(np.argmin(delta))
        if float(delta[k]) <= max_offset_rad:
            picks.append((k, label, color))
    picks.sort(key=lambda item: item[0])
    return picks


# ---------------------------------------------------------------------------
# 3D scene
# ---------------------------------------------------------------------------


def build_scene_figure(
    elements: OrbitElements,
    sun_hat: np.ndarray,
    sat_position: np.ndarray | None = None,
    yearly: YearlySweep | None = None,
    show_yearly_planes: bool = False,
    show_orbit_normal: bool = True,
) -> go.Figure:
    """Assemble the main 3D ECI figure."""

    fig = go.Figure()
    fig.add_trace(_earth_surface(sun_hat))
    fig.add_trace(_equator_trace())

    # ECI axes (very short, just for orientation).
    axis_len = 1.35 * R_EARTH
    fig.add_trace(_axis_trace("X (vernal eq.)", np.array([1.0, 0.0, 0.0]), axis_len, "rgba(200,80,80,0.7)"))
    fig.add_trace(_axis_trace("Y", np.array([0.0, 1.0, 0.0]), axis_len, "rgba(80,200,80,0.7)"))
    fig.add_trace(_axis_trace("Z (pole)", np.array([0.0, 0.0, 1.0]), axis_len, "rgba(120,120,240,0.7)"))

    # Orbit track for the selected day.
    track = sample_orbit_eci(elements)
    fig.add_trace(
        go.Scatter3d(
            x=track[:, 0],
            y=track[:, 1],
            z=track[:, 2],
            mode="lines",
            line=dict(color="#00c2ff", width=4),
            name="Orbit (selected day)",
        )
    )

    # Equinox and solstice orbit planes.
    if show_yearly_planes and yearly is not None:
        for j, (k, label, color) in enumerate(seasonal_plane_picks(yearly)):
            day_elements = OrbitElements(
                a=elements.a,
                e=elements.e,
                i=elements.i,
                raan=float(yearly.raan_rad[k]),
                argp=float(yearly.argp_rad[k]),
            )
            pts = sample_orbit_eci(day_elements, num_points=181)
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="lines",
                    line=dict(color=color, width=2),
                    opacity=0.7,
                    name=f"{label} ({yearly.dates[k].strftime('%b %d')})",
                    showlegend=True,
                    legendgroup="yearly",
                    legendgrouptitle=dict(text="Equinox & solstice")
                    if j == 0
                    else None,
                )
            )

    # Sun direction arrow.
    sun_len = 2.2 * R_EARTH
    fig.add_trace(
        go.Scatter3d(
            x=[0, sun_hat[0] * sun_len],
            y=[0, sun_hat[1] * sun_len],
            z=[0, sun_hat[2] * sun_len],
            mode="lines+markers+text",
            line=dict(color="#ffd54a", width=6),
            marker=dict(size=[0, 10], color="#ffd54a", symbol="circle"),
            text=["", "Sun"],
            textposition="top center",
            name="Sun direction",
        )
    )

    # Orbit normal.
    if show_orbit_normal:
        h_hat = orbit_normal_eci(elements.raan, elements.i)
        n_len = 1.5 * R_EARTH
        fig.add_trace(
            go.Scatter3d(
                x=[0, h_hat[0] * n_len],
                y=[0, h_hat[1] * n_len],
                z=[0, h_hat[2] * n_len],
                mode="lines+text",
                line=dict(color="#ff7ab6", width=3, dash="dash"),
                text=["", "h"],
                textposition="top center",
                name="Orbit normal",
            )
        )

    # Satellite marker.
    if sat_position is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[sat_position[0]],
                y=[sat_position[1]],
                z=[sat_position[2]],
                mode="markers",
                marker=dict(size=6, color="#ffffff", line=dict(color="#00c2ff", width=1)),
                name="Satellite",
            )
        )

    # Scene styling. Use manually chosen limits so the aspect is 1:1:1 and the
    # Earth stays centered even when the orbit is highly eccentric.
    r_max = max(np.linalg.norm(track, axis=1).max(), 2.5 * R_EARTH)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-r_max, r_max], showbackground=False, title=""),
            yaxis=dict(range=[-r_max, r_max], showbackground=False, title=""),
            zaxis=dict(range=[-r_max, r_max], showbackground=False, title=""),
            aspectmode="cube",
            bgcolor="rgb(6, 8, 22)",
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.1)),
        ),
        paper_bgcolor="rgb(6, 8, 22)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0.35)", font=dict(color="white")),
        showlegend=True,
        uirevision="scene",
    )
    return fig


# ---------------------------------------------------------------------------
# 2D yearly charts
# ---------------------------------------------------------------------------


def build_beta_figure(yearly: YearlySweep) -> go.Figure:
    beta_deg = np.degrees(yearly.beta_rad)
    fig = go.Figure(
        go.Scatter(
            x=yearly.dates,
            y=beta_deg,
            mode="lines",
            line=dict(color="#ffd54a", width=2),
            name="Beta angle",
        )
    )
    fig.update_layout(
        title="Beta angle over the year",
        xaxis_title="Date",
        yaxis_title="Beta (deg)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=40),
        height=280,
    )
    return fig


def build_eclipse_figure(yearly: YearlySweep) -> go.Figure:
    ecl_min = yearly.eclipse_s / 60.0
    fig = go.Figure(
        go.Scatter(
            x=yearly.dates,
            y=ecl_min,
            mode="lines",
            line=dict(color="#00c2ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 194, 255, 0.15)",
            name="Eclipse duration",
        )
    )
    k = yearly.max_eclipse_idx
    fig.add_trace(
        go.Scatter(
            x=[yearly.dates[k]],
            y=[ecl_min[k]],
            mode="markers+text",
            marker=dict(color="#ff5f5f", size=10, symbol="diamond"),
            text=[f"max {ecl_min[k]:.1f} min"],
            textposition="top center",
            name="Max eclipse",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Eclipse duration per orbit over the year",
        xaxis_title="Date",
        yaxis_title="Duration (min)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=40),
        height=280,
    )
    return fig


def build_raan_figure(yearly: YearlySweep) -> go.Figure:
    raan_deg = np.degrees(np.unwrap(yearly.raan_rad))
    fig = go.Figure(
        go.Scatter(
            x=yearly.dates,
            y=raan_deg,
            mode="lines",
            line=dict(color="#ff7ab6", width=2),
            name="RAAN",
        )
    )
    fig.update_layout(
        title="RAAN (J2 secular precession)",
        xaxis_title="Date",
        yaxis_title="RAAN (deg)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=40),
        height=280,
    )
    return fig
