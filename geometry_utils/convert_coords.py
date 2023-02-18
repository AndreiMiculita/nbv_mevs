import numpy as np
from sympy import pi, sin, cos, sqrt, acos, atan2


def as_cartesian(rthetaphi):
    # takes list rthetaphi (single coord)
    r = rthetaphi[0]
    theta = rthetaphi[1] * pi / 180  # to radian
    phi = rthetaphi[2] * pi / 180
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
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
    r = np.float(sqrt(x * x + y * y + z * z))
    theta = np.float(acos(z / r))  # to degrees
    phi = np.float(atan2(y, x))
    return [r, theta, phi]
