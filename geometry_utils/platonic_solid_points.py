"""
Provides functions that return the coordinates of vertices of Platonic solids,
https://en.wikipedia.org/wiki/Platonic_solid

These can be used to obtain equally spaced camera positions.

For each one, you must give the circumradius.
Circumradius = the radius of the circumscribed sphere (i.e. that touches all vertices).

Circumradii are set to obtain nice coordinates that contain 1, phi and stuff,
but you probably want to set them to something else.
"""
import numpy as np
import matplotlib.pyplot as plt
from visualization.plotting_utils import set_axes_equal, show_3d_axes_rgb

phi = (1 + np.sqrt(5)) / 2  # the golden ratio, appears in some coordinates


def make_tetrahedron_vertices(circumradius=np.sqrt(6) / 2) -> np.array:
    # Based on https://en.wikipedia.org/wiki/Tetrahedron#Coordinates_for_a_regular_tetrahedron
    # Default circumradius leads to edge length 2
    a = 2 / np.sqrt(6)

    vertices = []
    for i in [-1, 1]:
        vertices.append([i * a * circumradius, 0, -1 / np.sqrt(2) * a * circumradius])
        vertices.append([0, i * a * circumradius, 1 / np.sqrt(2) * a * circumradius])

    return np.array(vertices)


def make_cube_vertices(circumradius=np.sqrt(3)) -> np.array:
    # Based on https://en.wikipedia.org/wiki/Cube#Cartesian_coordinates
    # Default circumradius leads to edge length 2
    a = 1 / np.sqrt(3)

    vertices = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                vertices.append([i * a * circumradius, j * a * circumradius, k * a * circumradius])

    return np.array(vertices)


def make_octahedron_vertices(circumradius=1) -> np.array:
    # Based on https://en.wikipedia.org/wiki/Octahedron#Cartesian_coordinates
    # Default circumradius leads to edge length sqrt(2)
    vertices = []
    for i in [-1, 1]:
        vertices.append([i * circumradius, 0, 0])
        vertices.append([0, i * circumradius, 0])
        vertices.append([0, 0, i * circumradius])

    return np.array(vertices)


def make_dodecahedron_vertices(circumradius=np.sqrt(3)) -> np.array:
    # Based on https://stackoverflow.com/a/10462220/13200217
    # also on https://en.wikipedia.org/wiki/Regular_dodecahedron#Cartesian_coordinates
    # Default circumradius leads to edge length 2/phi=sqrt(5)-1
    a = 1 / np.sqrt(3)
    b = a / phi
    c = a * phi

    vertices = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            vertices.append([0, i * c * circumradius, j * b * circumradius])
            vertices.append([i * c * circumradius, j * b * circumradius, 0])
            vertices.append([i * b * circumradius, 0, j * c * circumradius])
            for k in [-1, 1]:
                vertices.append([i * a * circumradius, j * a * circumradius, k * a * circumradius])

    return np.array(vertices)


def make_icosahedron_vertices(circumradius=np.sqrt(phi + 2)) -> np.array:
    # Based on https://en.wikipedia.org/wiki/Regular_icosahedron#Cartesian_coordinates
    # Default circumradius leads to edge length 2
    a = 1 / np.sqrt(phi + 2)
    c = a * phi

    vertices = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            vertices.append([0, i * a * circumradius, j * c * circumradius])
            vertices.append([i * a * circumradius, j * c * circumradius, 0])
            vertices.append([j * c * circumradius, 0, i * a * circumradius])
    return np.array(vertices)


def main():
    # make a 3d plot of each function's output and save them in separate images
    for func in [make_tetrahedron_vertices, make_cube_vertices, make_octahedron_vertices,
                 make_dodecahedron_vertices, make_icosahedron_vertices]:
        vertices = func()
        print(vertices)
        ax = plt.axes(projection='3d')
        ax.scatter3D(*vertices.transpose(), s=100)
        ax.set_box_aspect([1, 1, 1])
        set_axes_equal(ax)
        show_3d_axes_rgb(ax)
        plt.show()
        # Create asset folder if it doesn't exist
        import os
        if not os.path.exists('../assets/platonic_solid_vertices/'):
            os.makedirs('../assets/platonic_solid_vertices/')
        plt.savefig(func.__name__.replace("make_", "../assets/platonic_solid_vertices/") + '.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
