"""
This script reads from entropy_views_10.graphml
The nodes' coordinates are their id's in the format (theta, phi) in spherical coordinates.
These can be converted to cartesian coordinates, and then plotted in 3D.
The edges are the connections between the nodes.
"""
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import colors

from geometry_utils.convert_coords import as_cartesian
from visualization.plotting_utils import set_axes_equal


def plot_graph_from_file(generate_gif=True):
    # Read the graph from file
    G = nx.read_graphml("../config/entropy_views_40.graphml")

    # Print the nodes and edges
    print(f"G.nodes: {G.nodes}")
    print(f"G.edges: {G.edges}")

    # Convert the nodes' coordinates from spherical to cartesian
    # get the attributes for each node
    for i, node in enumerate(G.nodes(data=True)):
        node_name = node[0]
        theta, phi = float(node[1]['theta']), float(node[1]['phi'])
        node_coords = as_cartesian([1, theta, phi])
        nx.set_node_attributes(G, {node_name: {'name': node_name, 'coords': node_coords}})

    for i, node in enumerate(G.nodes(data=True)):
        print(f"node: {node}, i: {i}")

    # Plot the graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # draw sphere
    # radius = 0.9
    # u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
    # x = radius * np.cos(u) * np.sin(v)
    # y = radius * np.sin(u) * np.sin(v)
    # z = radius * np.cos(v)
    # ax.plot_surface(x, y, z, color='g')

    # Colors for weights, use cmap='jet' for a gradient
    entropies = {
        0: 0.961198,
        1: 2.421217,
        2: 2.322561,
        3: 2.959751,
        4: 1.478724,
        5: 3.342597,
        6: 2.309064,
        7: 2.073726,
        8: 3.252601,
        9: 1.791614,
        10: 2.895416,
        11: 2.141816,
        12: 1.662782,
        13: 3.284952,
        14: 2.833258,
        15: 2.094056,
        16: 3.661451,
        17: 1.618091,
        18: 3.757386,
        19: 2.681031,
        20: 2.292247,
        21: 2.961708,
        22: 2.648077,
        23: 2.926976,
        24: 3.462798,
        25: 1.836550,
        26: 3.789309,
        27: 2.595079,
        28: 1.224292,
        29: 3.046702,
        30: 2.016478,
        31: 3.525395,
        32: 2.715482,
        33: 2.041915,
        34: 2.549485,
        35: 2.662680,
        36: 1.972868,
        37: 2.680763,
        38: 1.825287,
        39: 0.941295,
    }

    cmap = plt.cm.get_cmap('jet', 10)
    normalize = colors.Normalize(vmin=0, vmax=6)

    # Plot the nodes
    node_coords = [G.nodes[node]["coords"] for node in G.nodes]
    node_coords = np.array(node_coords)
    ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], c=list(entropies.values()), cmap=cmap,
               norm=normalize)

    # show the names under the nodes
    for node in G.nodes:
        node_coords = G.nodes[node]["coords"]
        ax.text(node_coords[0], node_coords[1], node_coords[2], G.nodes[node]["name"], color='red')

    # for node in G.nodes:
    #     node_coords = G.nodes[node]["coords"]
    #     ax.scatter(node_coords[0], node_coords[1], node_coords[2])

    # Plot the edges
    for edge in G.edges:
        node1 = edge[0]
        node2 = edge[1]
        node1_coords = G.nodes[node1]["coords"]
        node2_coords = G.nodes[node2]["coords"]
        ax.plot([node1_coords[0], node2_coords[0]], [node1_coords[1], node2_coords[1]],
                [node1_coords[2], node2_coords[2]], color="grey")

    # Set the axes' limits to be equal
    set_axes_equal(ax)

    # We do a 360-degree rotation of the plot, and generate a gif of it
    if generate_gif:
        # create temp dir for frames
        Path("../assets/temp").mkdir(parents=True, exist_ok=True)

        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.001)

            # Save the image
            plt.savefig(f'../assets/temp/entropy_views_{len(G.nodes)}_{angle}.png')

        # Make the gif
        images = []
        for angle in range(0, 360):
            images.append(imageio.imread(f'../assets/temp/entropy_views_{len(G.nodes)}_{angle}.png'))

        imageio.mimsave(f'../assets/entropy_views_{len(G.nodes)}_animation.gif', images, duration=0.01)

    else:
        # Show the plot, square
        plt.show()


if __name__ == "__main__":
    plot_graph_from_file()
