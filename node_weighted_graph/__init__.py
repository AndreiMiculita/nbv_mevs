from node_weighted_graph.Node import Node
from node_weighted_graph.find_local_maximum_nodes import find_local_maximum_nodes
from node_weighted_graph.generate_graph_image import generate_graph_image


def main():
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

    graph = [A, B, C, D, E, F, G, H, I, J]

    # Find nodes with higher weights than neighbors
    result = find_local_maximum_nodes(graph)

    # Print the result
    for node in result:
        print(f"Node {node.name} has a higher weight than all its neighbors.")

    # Generate the graph image
    generate_graph_image(graph)


if __name__ == '__main__':
    main()
