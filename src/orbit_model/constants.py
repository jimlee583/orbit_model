"""Physical constants used throughout the orbit model.

Units: kilometers, seconds, radians unless otherwise noted.
Values follow WGS-84 / EGM-96 conventions where applicable.
"""

from __future__ import annotations

import math

# --- Earth ---------------------------------------------------------------

#: Earth gravitational parameter, km^3 / s^2 (EGM-96).
MU_EARTH: float = 398_600.4418

#: Earth equatorial radius, km (WGS-84).
R_EARTH: float = 6378.137

#: Earth second zonal harmonic (dimensionless).
J2: float = 1.082_626_68e-3

#: Obliquity of the ecliptic at J2000, radians.
OBLIQUITY_J2000_RAD: float = math.radians(23.4392911)

# --- Sun -----------------------------------------------------------------

#: Astronomical unit, km (IAU 2012).
AU_KM: float = 149_597_870.700

#: Sun radius, km.
R_SUN: float = 695_700.0

# --- Time ----------------------------------------------------------------

#: Julian date of the J2000.0 epoch.
JD_J2000: float = 2_451_545.0

#: Length of a sidereal year in mean solar days.
SIDEREAL_YEAR_DAYS: float = 365.256_363_004

#: Length of a mean solar day in seconds.
SECONDS_PER_DAY: float = 86_400.0
