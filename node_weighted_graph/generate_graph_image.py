from typing import List

import networkx as nx
from matplotlib import pyplot as plt

from node_weighted_graph import Node


def generate_graph_image(graph: List[Node]):
    G = nx.Graph()

    # Add nodes with their attributes
    for node in graph:
        G.add_node(node.name, weight=node.weight)

    # Add edges between neighbors
    for node in graph:
        for neighbor in node.neighbors:
            G.add_edge(node.name, neighbor.name)

    # Generate the graph layout
    pos = nx.spring_layout(G)

    # Set node colors based on their weights
    node_colors = [node.weight for node in graph]

    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_color=node_colors, edgecolors='black', cmap='plasma_r', node_size=500, font_size=10)

    # Add node weight labels above the nodes
    label_pos = {k: (v[0], v[1] + 0.1) for k, v in pos.items()}  # Adjust the y-offset as needed
    labels = nx.get_node_attributes(G, 'weight')
    nx.draw_networkx_labels(G, label_pos, labels=labels)

    # Add node names below the nodes
    name_pos = {k: (v[0], v[1] - 0.1) for k, v in pos.items()}  # Adjust the y-offset as needed
    node_names = {node.name: node.name for node in graph}
    nx.draw_networkx_labels(G, name_pos, labels=node_names)

    # Add legend
    plt.colorbar(nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors='black', cmap='plasma',
                               node_size=500))

    # Display the image
    plt.show()
