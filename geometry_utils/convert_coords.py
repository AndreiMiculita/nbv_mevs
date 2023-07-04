import numpy as np
from sympy import pi, sin, cos, sqrt, acos, atan2


# credit https://stackoverflow.com/a/43893134/13200217
def as_cartesian(rthetaphi):
    """
    Converts spherical coordinates to cartesian coordinates.
    :param rthetaphi: Spherical coordinates, as a list of [r, theta, phi] in degrees; theta is latitude, phi is longitude
    :return:
    """
    r = rthetaphi[0]
    theta = rthetaphi[1] * pi / 180  # to radian
    phi = rthetaphi[2] * pi / 180
    x = float(r * sin(theta) * cos(phi))
    y = float(r * sin(theta) * sin(phi))
    z = float(r * cos(theta))
    return [x, y, z]


def as_spherical(xyz):
    """
    Converts cartesian coordinates to spherical coordinates. Returns them in radians.
    :param xyz: cartesian coordinates
    :return: spherical coordinates
    """
    # takes list xyz (single coord)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = float(sqrt(x * x + y * y + z * z))
    theta = float(acos(z / r))  # to degrees
    phi = float(atan2(y, x))
    return [r, theta, phi]
