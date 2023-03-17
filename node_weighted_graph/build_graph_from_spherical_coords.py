"""Given a list of spherical coordinates, build a graph with the nodes, based on the triangulation of the sphere."""
from typing import List

import stripy as stripy
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from node_weighted_graph import Node


def build_graph_from_spherical_coords(vertices_spherical: np.ndarray) -> List[Node]:
    """
    Given a list of spherical coordinates (degrees), build a graph with the nodes,
     based on the triangulation of the sphere.
    :param vertices_spherical:
    :return:
    """

    vertices_lon: np.ndarray = np.radians(vertices_spherical.T[0])
    vertices_lat: np.ndarray = np.radians(vertices_spherical.T[1])
    vertices_weights: np.ndarray = vertices_spherical.T[2]

    spherical_triangulation = stripy.sTriangulation(lons=vertices_lon, lats=vertices_lat, permute=True)

    # Build the graph
    graph: List[Node] = []
    for i in range(spherical_triangulation.npoints):
        node = Node(name=f'{vertices_spherical.T[0][i]}, {vertices_spherical.T[1][i]}',
                    x=spherical_triangulation.lons[i],
                    y=spherical_triangulation.lats[i],
                    weight=vertices_weights[i])
        graph.append(node)

    segs = spherical_triangulation.identify_segments()

    for s1, s2 in segs:
        graph[s1].add_neighbor(graph[s2])

    return graph


def main():
    vertices_LatLonDeg = np.array(
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

    vertices_lat = np.radians(vertices_LatLonDeg.T[0])
    vertices_lon = np.radians(vertices_LatLonDeg.T[1])

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
