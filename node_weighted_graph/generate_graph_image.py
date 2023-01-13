from typing import List

import networkx as nx
from matplotlib import pyplot as plt

from node_weighted_graph import Node


def generate_graph_image(graph: List[Node]):
    G = nx.Graph()
    node_pos = {}

    # Add nodes with their attributes
    for node in graph:
        G.add_node(node.name, weight=node.weight)
        node_pos[node.name] = (node.x, node.y)
        for neighbor in node.neighbors:
            G.add_edge(node.name, neighbor.name)

    # Set node colors based on their weights
    node_colors = [node.weight for node in graph]

    # Draw the graph with specified node positions
    nx.draw_networkx(G, pos=node_pos, with_labels=False, node_color=node_colors, edgecolors='black', cmap='plasma',
                     node_size=500)

    # Add node weight labels above the nodes
    label_pos = {k: (v[0], v[1] + 0.1) for k, v in node_pos.items()}  # Adjust the y-offset as needed
    labels = nx.get_node_attributes(G, 'weight')
    nx.draw_networkx_labels(G, label_pos, labels=labels)

    # Add node names below the nodes
    name_pos = {k: (v[0], v[1] - 0.1) for k, v in node_pos.items()}  # Adjust the y-offset as needed
    node_names = {node.name: node.name for node in graph}
    nx.draw_networkx_labels(G, pos=name_pos, labels=node_names)

    # Add legend
    plt.colorbar(nx.draw_networkx_nodes(G, pos=node_pos, node_color=node_colors, edgecolors='black', cmap='plasma',
                                        node_size=500))

    # Display the image
    plt.show()
