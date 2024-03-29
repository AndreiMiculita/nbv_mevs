import os

import matplotlib.pyplot as plt
import numpy as np
from sympy import pi

from geometry_utils.convert_coords import as_spherical
from geometry_utils.fibonacci_sphere import fibonacci_sphere
from plotting_utils import set_axes_equal, show_3d_axes_rgb


def main():
    # Create assets folder if it doesn't exist
    if not os.path.exists("../assets/fibonacci_sphere_vertices/"):
        os.makedirs("../assets/fibonacci_sphere_vertices/")

    # generate a sphere of 10 points
    points = fibonacci_sphere(samples=10)

    # convert to numpy array
    points = np.array(points)

    print(points)

    # view the sphere points as interactive 3d scatter plot
    ax = plt.axes(projection="3d")
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], s=100)
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    show_3d_axes_rgb(ax)
    plt.savefig("../assets/fibonacci_sphere_vertices/fibonacci_sphere_points_10.pdf", bbox_inches='tight')

    # generate a sphere of 40 points
    points = fibonacci_sphere(samples=40)

    # convert to numpy array
    points = np.array(points)

    print(points)

    # view the sphere points as interactive 3d scatter plot
    ax = plt.axes(projection="3d")
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], s=100)
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    show_3d_axes_rgb(ax)
    plt.savefig("../assets/fibonacci_sphere_vertices/fibonacci_sphere_points_40.pdf", bbox_inches='tight')

    # convert to spherical coordinates as np array

    spherical = []

    for initial_view in points:
        [r, theta, phi] = as_spherical(initial_view)
        spherical.append(as_spherical(initial_view))

    # Convert to np array
    spherical = np.array(spherical)

    # Decrease theta by pi to get the same rotations as the original
    spherical[:, 1] = spherical[:, 1] - np.pi

    # Increase phi by pi to get the same rotations as the original
    spherical[:, 2] = spherical[:, 2] + np.pi

    # Decrease r to 0 to get the same rotations as the original
    spherical[:, 0] = 0

    # Swap theta and r to get the same rotations as the original
    spherical[:, 0], spherical[:, 1] = spherical[:, 1], spherical[:, 0].copy()

    print(spherical)

    # convert radians to degrees
    spherical[:, 0] = -spherical[:, 0] * 180 / pi
    spherical[:, 2] = spherical[:, 2] * 180 / pi

    # clear the figure
    plt.clf()

    # view the spherical coords on a 2d scatter plot
    ax = plt.axes()
    ax.scatter(spherical[:, 2], spherical[:, 0])
    # Add label to each point with the index
    for i, txt in enumerate(range(len(spherical))):
        ax.annotate(txt, (spherical[i, 2] + 3, spherical[i, 0] + 3))
    ax.set_ylabel("theta")
    ax.set_xlabel("phi")
    # equal axes
    ax.set_aspect('equal', adjustable='box')

    plt.savefig("../assets/fibonacci_sphere_vertices/fibonacci_sphere_spherical_coords_2d.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
