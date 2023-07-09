from typing import List

import networkx as nx
from matplotlib import pyplot as plt

from node_weighted_graph import Node


def generate_graph_image_2d(graph: List[Node], identified_maxima = None):
    """
    Generate an image of the graph, with the nodes colored based on their weights.
    It will be a 2D image, with the x-axis being the longitude and the y-axis being the latitude.
    Maxima will be circled in red if they are provided.
    :param graph: the graph to be drawn
    :param identified_maxima: the list of nodes that are identified as maxima
    :return:
    """
    G = nx.Graph()
    node_pos = {}

    # Add nodes with their attributes
    for node in graph:
        G.add_node(node.name, weight=node.weight)
        node_pos[node.name] = (node.lon, node.lat)
        for neighbor in node.neighbors:
            G.add_edge(node.name, neighbor.name)

    for node in G.nodes.items():
        print(node)
        print(type(node))
    # Set node colors based on their weights
    node_colors = [n[1] for n in G.nodes.data('weight')]

    # Draw the graph with specified node positions
    nx.draw_networkx(G, pos=node_pos, with_labels=False, node_color=node_colors, edgecolors='black', cmap='plasma',
                     node_size=500)

    # Add identified maxima, circle them in red
    if identified_maxima is not None:
        for node in identified_maxima:
            nx.draw_networkx_nodes(G, pos=node_pos, nodelist=[node.name], edgecolors='red', node_color='none',
                                   node_size=900)

    # Add node weight labels above the nodes
    label_pos = {k: (v[0], v[1] + 0.12) for k, v in node_pos.items()}  # Adjust the y-offset as needed
    labels = nx.get_node_attributes(G, 'weight')
    nx.draw_networkx_labels(G, label_pos, labels=labels)

    # Add node names below the nodes
    name_pos = {k: (v[0], v[1] - 0.12) for k, v in node_pos.items()}  # Adjust the y-offset as needed
    node_names = {node.name: node.name for node in graph}
    nx.draw_networkx_labels(G, pos=name_pos, labels=node_names)

    # Add legend
    plt.colorbar(nx.draw_networkx_nodes(G, pos=node_pos, node_color=node_colors, edgecolors='black', cmap='plasma',
                                        node_size=500))

    # A bit more space to the top and bottom
    plt.ylim(-0.2, 1.4)

    # Tight layout
    plt.tight_layout()

    # Display the image
    plt.show()
