"""
Contains functions for handling graphs of nodes with weights.
"""

from typing import List

import numpy as np

from node_weighted_graph import Node
from node_weighted_graph import create_graph_image_2d
# from node_weighted_graph import build_graph_from_spherical_coords
from node_weighted_graph import find_local_maximum_nodes
from node_weighted_graph.build_graph_from_spherical_coords import build_graph_from_spherical_coords_with_delaunay


def get_example_graph() -> List[Node]:
    # Example graph
    A = Node('A', 1, 0.5, 5)
    B = Node('B', 2, 0, 8)
    C = Node('C', 3, 0, 31)
    D = Node('D', 3, 0.5, 7)
    E = Node('E', 4.5, 0.5, 35)
    F = Node('F', 6, 0.5, 6)
    G = Node('G', 1, 1, 4)
    H = Node('H', 2.5, 0.8, 24)
    I = Node('I', 3, 1.2, 11)
    J = Node('J', 4, 1, 10)

    A.add_neighbor(B)
    A.add_neighbor(C)
    A.add_neighbor(D)
    B.add_neighbor(C)
    B.add_neighbor(E)
    C.add_neighbor(F)
    C.add_neighbor(G)
    D.add_neighbor(G)
    E.add_neighbor(F)
    E.add_neighbor(H)
    E.add_neighbor(J)
    F.add_neighbor(I)
    G.add_neighbor(J)

    return [A, B, C, D, E, F, G, H, I, J]

example_spherical_coords = np.array(
    [
        [87.0, 318.0, 0.961198],
        [106.0, 232.0, 2.421217],
        [102.0, 57.0, 2.322561],
        [78.0, 250.0, 2.959751],
        [47.0, 323.0, 1.478724],
        [92.0, 97.0, 3.342597],
        [148.0, 43.0, 2.309064],
        [109.0, 342.0, 2.073726],
        [90.0, 180.0, 3.252601],
        [69.0, 217.0, 1.791614],
        [64.0, 182.0, 2.895416],
        [125.0, 81.0, 2.141816],
        [58.0, 360.0, 1.662782],
        [129.0, 257.0, 3.284952],
        [38.0, 217.0, 2.833258],
        [102.0, 166.0, 2.094056],
        [32.0, 35.0, 3.661451],
        [98.0, 274.0, 1.618091],
        [47.0, 151.0, 3.757386],
        [73.0, 73.0, 2.681031],
        [62.0, 116.0, 2.292247],
        [177.0, 119.0, 2.961708],
        [40.0, 88.0, 2.648077],
        [116.0, 27.0, 2.926976],
        [139.0, 349.0, 3.462798],
        [15.0, 152.0, 1.836550],
        [114.0, 302.0, 3.789309],
        [45.0, 264.0, 2.595079],
        [84.0, 143.0, 1.224292],
        [109.0, 125.0, 3.046702],
        [64.0, 39.0, 2.016478],
        [149.0, 212.0, 3.525395],
        [16.0, 310.0, 2.715482],
        [115.0, 201.0, 2.041915],
        [148.0, 296.0, 2.549485],
        [90.0, 360.0, 2.662680],
        [142.0, 117.0, 1.972868],
        [134.0, 165.0, 2.680763],
        [67.0, 291.0, 1.825287],
        [88.0, 18.0, 0.941295],
    ])


def main():
    # Divide 3rd column by 6
    example_spherical_coords[:, 2] /= 6

    graph = build_graph_from_spherical_coords_with_delaunay(example_spherical_coords)
    # graph = get_example_graph()

    # Find nodes with higher weights than neighbors
    result = find_local_maximum_nodes(graph)

    # Print the result
    for node in result:
        print(f"Node {node.name} has a higher weight than all its neighbors.")

    print(f'Graph size: {len(graph)}')

    # Generate the graph image
    create_graph_image_2d(graph, result)


if __name__ == '__main__':
    main()
