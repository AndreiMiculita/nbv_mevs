import math
from sympy import pi, sin, cos, sqrt, acos, atan2
import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import set_axes_equal, show_3d_axes_rgb

# credit: https://stackoverflow.com/a/26127012/13200217
def fibonacci_sphere(samples=1000):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        # Reorder so first points are on the sides
        (x, y, z) = (y, x, z)

        points.append((x, y, z))

    return points


# credit https://stackoverflow.com/a/43893134/13200217
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
    # takes list xyz (single coord)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.float(sqrt(x * x + y * y + z * z))
    theta = np.float(acos(z / r) * 180 / pi)  # to degrees
    phi = np.float(atan2(y, x) * 180 / pi)
    return [r, theta, phi]


if __name__ == "__main__":
    # generate a sphere
    points = fibonacci_sphere(samples=400)

    # convert to numpy array
    points = np.array(points)

    print(points)

    # view the sphere points as interactive 3d scatter plot
    ax = plt.axes(projection="3d")
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    show_3d_axes_rgb(ax)
    plt.show()

    # convert to spherical coordinates as np array
    spherical = np.array(list(map(as_spherical, points)))

    print(spherical)

    # view the spherical coords on a 2d scatter plot
    ax = plt.axes()
    ax.scatter(spherical[:, 2], spherical[:, 1])
    ax.set_ylabel("theta")
    ax.set_xlabel("phi")
    plt.show()

