from typing import List

import numpy as np

from node_weighted_graph.Node import Node
from node_weighted_graph.find_local_maximum_nodes import find_local_maximum_nodes
from node_weighted_graph.generate_graph_image import generate_graph_image
from node_weighted_graph.build_graph_from_spherical_coords import build_graph_from_spherical_coords


def get_example_graph() -> List[Node]:
    # Example graph
    A = Node('A', 5)
    B = Node('B', 8)
    C = Node('C', 31)
    D = Node('D', 7)
    E = Node('E', 44)
    F = Node('F', 6)
    G = Node('G', 4)
    H = Node('H', 24)
    I = Node('I', 11)
    J = Node('J', 10)

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


def main():
    graph = build_graph_from_spherical_coords(example_spherical_coords)

    # Find nodes with higher weights than neighbors
    result = find_local_maximum_nodes(graph)

    # Print the result
    for node in result:
        print(f"Node {node.name} has a higher weight than all its neighbors.")

    print(f'Graph size: {len(graph)}')

    # Generate the graph image
    generate_graph_image(graph)


if __name__ == '__main__':
    main()
