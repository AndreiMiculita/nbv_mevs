"""Given a list of spherical coordinates, build a graph with the nodes, based on the triangulation of the sphere."""
from typing import List

import matplotlib.pyplot as plt
import numpy as np

try:
    import cartopy.crs as ccrs
    import stripy as stripy
except ImportError:
    print("Cartopy or stripy not installed; "
          "if you want to use them, make another environment, as they have conflicting dependencies.")
    exit(1)

from node_weighted_graph import Node
from geometry_utils.convert_coords import as_cartesian


def build_graph_from_spherical_coords_with_delaunay(vertices_spherical_coords: np.ndarray) -> List[Node]:
    """
    Given a 2d array of spherical coordinates (degrees), build a graph with the nodes,
     based on the triangulation of the sphere.
    The shape of the input array is (n, 2) or (n, 3), where n is the number of vertices.
    The first column is the longitude, the second is the latitude, and the third is the (optional) weight.
    :param vertices_spherical_coords:
    :return:
    """

    vertices_thetas: np.ndarray = np.radians(vertices_spherical_coords.T[0])
    vertices_phis: np.ndarray = np.radians(vertices_spherical_coords.T[1])
    # if there is a third column, it is the weight
    if vertices_spherical_coords.shape[1] == 3:
        vertices_weights: np.ndarray = vertices_spherical_coords.T[2]

    # Print them zipped, on new line each
    print(*zip(vertices_thetas, vertices_phis, np.degrees(vertices_thetas), np.degrees(vertices_phis)), sep='\n')

    # Our theta is between 0 and pi, and our phi is between 0 and 2pi;
    # Stripy expects theta to be between -pi/2 and pi/2, and phi to be between 0 and 2pi
    # Thus we need to decrease theta by pi/2
    vertices_thetas_for_stripy = vertices_thetas - np.pi / 2

    spherical_triangulation = stripy.sTriangulation(lats=vertices_thetas_for_stripy, lons=vertices_phis, permute=True)

    print(f'Areas {spherical_triangulation.areas()}')

    # Build the graph
    graph: List[Node] = []
    for i in range(vertices_spherical_coords.shape[0]):
        node = Node(name=f'{i}: {vertices_spherical_coords.T[0][i]}, {vertices_spherical_coords.T[1][i]}',
                    theta=vertices_thetas[i],
                    phi=vertices_phis[i],
                    weight=vertices_weights[i] if vertices_spherical_coords.shape[1] == 3 else None)
        graph.append(node)

    segments = spherical_triangulation.identify_segments()

    for s1, s2 in segments:
        graph[s1].add_neighbor(graph[s2])

    return graph


def build_graph_from_spherical_coords_with_nearest_neighbors(vertices_spherical_coords: np.ndarray,
                                                             threshold: float = 1) -> List[Node]:
    """
    Given a list of spherical coordinates (degrees), build a graph with the nodes,
     based on the nearest neighbors.
    :param vertices_spherical_coords: (n, 2) or (n, 3) array of spherical coordinates (degrees)
    :param threshold: the maximum distance between two nodes to be considered neighbors
    :return: the graph
    """

    vertices_lons: np.ndarray = np.radians(vertices_spherical_coords.T[0])
    vertices_lats: np.ndarray = np.radians(vertices_spherical_coords.T[1])
    # if there is a third column, it is the weight
    if vertices_spherical_coords.shape[1] == 3:
        vertices_weights: np.ndarray = vertices_spherical_coords.T[2]

    # Build the graph
    graph: List[Node] = []
    for i in range(vertices_spherical_coords.shape[0]):
        node = Node(name=f'{i}: {vertices_spherical_coords.T[0][i]}, {vertices_spherical_coords.T[1][i]}',
                    theta=vertices_lons[i],
                    phi=vertices_lats[i],
                    weight=vertices_weights[i] if vertices_spherical_coords.shape[1] == 3 else None)
        graph.append(node)

    # Find the nearest neighbors; note that the angles in node.lat and node.long are in radians
    # and as_cartesian expects degrees
    for i, node in enumerate(graph):
        for j, other_node in enumerate(graph):
            if i != j:
                node_coords = np.array(
                    as_cartesian([1, np.degrees(node.phi), np.degrees(node.theta)])
                )
                other_node_coords = np.array(
                    as_cartesian([1, np.degrees(other_node.phi), np.degrees(other_node.theta)])
                )
                distance = np.linalg.norm(node_coords - other_node_coords)
                print(f'Distance between {node.name} - {other_node.name}: {distance}')
                print(f'Coords of {node.name}: {node_coords}')
                print(f'Coords of {other_node.name}: {other_node_coords}')
                if distance <= threshold:
                    node.add_neighbor(other_node)
                    print(f'Connected: {node.name} - {other_node.name}')

    return graph


def main():
    # Lat, lon
    vertices_coords_degrees = np.array(
        [[102.0, 166.0],
         [102.0, 57.0],
         [106.0, 232.0],
         [109.0, 125.0],
         [109.0, 342.0],
         [114.0, 302.0],
         [115.0, 201.0],
         [116.0, 27.0],
         [125.0, 81.0],
         [129.0, 257.0],
         [134.0, 165.0],
         [139.0, 349.0],
         [142.0, 117.0],
         [148.0, 296.0],
         [148.0, 43.0],
         [149.0, 212.0],
         [15.0, 152.0],
         [16.0, 310.0],
         [177.0, 119.0],
         [32.0, 35.0],
         [38.0, 217.0],
         [40.0, 88.0],
         [45.0, 264.0],
         [47.0, 151.0],
         [47.0, 323.0],
         [58.0, 360.0],
         [62.0, 116.0],
         [64.0, 182.0],
         [64.0, 39.0],
         [67.0, 291.0],
         [69.0, 217.0],
         [73.0, 73.0],
         [78.0, 250.0],
         [84.0, 143.0],
         [87.0, 318.0],
         [88.0, 18.0],
         [90.0, 180.0],
         [90.0, 360.0],
         [92.0, 97.0],
         [98.0, 274.0]])

    vertices_lat = np.radians(vertices_coords_degrees.T[0])
    vertices_lon = np.radians(vertices_coords_degrees.T[1])

    spherical_triangulation = stripy.sTriangulation(lons=vertices_lon, lats=vertices_lat)
    print(spherical_triangulation.areas())
    print(spherical_triangulation.npoints)

    fig = plt.figure(figsize=(20, 10), facecolor="none")

    ax = plt.subplot(121, projection=ccrs.Mollweide(central_longitude=0.0, globe=None))
    ax.set_global()

    lons = np.degrees(spherical_triangulation.lons)
    lats = np.degrees(spherical_triangulation.lats)

    ax.scatter(lons, lats, color="Red",
               marker="o", s=50.0, transform=ccrs.PlateCarree())

    segs = spherical_triangulation.identify_segments()

    for s1, s2 in segs:
        ax.plot([lons[s1], lons[s2]],
                [lats[s1], lats[s2]],
                linewidth=0.5, color="black", transform=ccrs.Geodetic())

    plt.show()


if __name__ == '__main__':
    main()
