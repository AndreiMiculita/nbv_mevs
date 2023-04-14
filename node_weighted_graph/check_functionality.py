"""
Contains functions for handling graphs of nodes with weights.
"""

from typing import List

import numpy as np

from node_weighted_graph import build_graph_from_spherical_coords
from node_weighted_graph import find_local_maximum_nodes
from node_weighted_graph import generate_graph_image
from node_weighted_graph import Node


def get_example_graph() -> List[Node]:
    # Example graph
    A = Node('A', 1, 1, 5)
    B = Node('B', 2, 0, 8)
    C = Node('C', 3, 0, 31)
    D = Node('D', 4, 1, 7)
    E = Node('E', 5, 1, 44)
    F = Node('F', 6, 1, 6)
    G = Node('G', 1, 2, 4)
    H = Node('H', 2, 2, 24)
    I = Node('I', 3, 2, 11)
    J = Node('J', 4, 2, 10)

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
        [87.0, 318.0, 1.6772159883221216],
        [106.0, 232.0, 1.8792914659519087],
        [102.0, 57.0, 1.693029824628385],
        [78.0, 250.0, 1.995278692158836],
        [47.0, 323.0, 1.8783757322934678],
        [92.0, 97.0, 1.5969404425943785],
        [148.0, 43.0, 1.3677114232545917],
        [109.0, 342.0, 1.2517162110604094],
        [90.0, 180.0, 0.5217527758939396],
        [69.0, 217.0, 1.9111426753644123],
        [64.0, 182.0, 0.9542569504316797],
        [125.0, 81.0, 1.806759881070881],
        [58.0, 360.0, 0.9643909743796494],
        [129.0, 257.0, 1.6900390366170641],
        [38.0, 217.0, 1.7433186565414336],
        [102.0, 166.0, 1.3541976754272547],
        [32.0, 35.0, 1.6413896702171704],
        [98.0, 274.0, 2.2483370292071556],
        [47.0, 151.0, 1.729387908057081],
        [73.0, 73.0, 1.927670686725282],
        [62.0, 116.0, 1.8269451983798175],
        [177.0, 119.0, 0.7600237970925121],
        [40.0, 88.0, 2.2526717721809417],
        [116.0, 27.0, 1.5241130192139298],
        [139.0, 349.0, 1.0161038223193046],
        [15.0, 152.0, 1.3758184042686967],
        [114.0, 302.0, 1.7658814191675363],
        [45.0, 264.0, 2.3674680848123963],
        [84.0, 143.0, 1.5176414244830179],
        [109.0, 125.0, 1.6820808013107496],
        [64.0, 39.0, 1.7889694276585866],
        [149.0, 212.0, 1.1530945737548441],
        [16.0, 310.0, 1.6131717771819725],
        [115.0, 201.0, 1.2797211919689753],
        [148.0, 296.0, 1.346230971133703],
        [90.0, 360.0, 0.5307273507937352],
        [142.0, 117.0, 1.5835134517664597],
        [134.0, 165.0, 1.0594529581594274],
        [67.0, 291.0, 2.1547943330628225],
        [88.0, 18.0, 1.1505704574550892],
    ])


def main():
    graph = build_graph_from_spherical_coords(example_spherical_coords)
    # graph = get_example_graph()

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
