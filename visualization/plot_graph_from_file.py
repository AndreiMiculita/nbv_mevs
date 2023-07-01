"""
This script reads from entropy_views_10.graphml
The nodes' coordinates are their id's in the format (theta, phi) in spherical coordinates.
These can be converted to cartesian coordinates, and then plotted in 3D.
The edges are the connections between the nodes.
"""

import networkx as nx
import matplotlib.pyplot as plt
from geometry_utils.convert_coords import as_cartesian
from visualization.plotting_utils import set_axes_equal, show_3d_axes_rgb
import numpy as np
import imageio


def plot_graph_from_file(generate_gif=False):
    # Read the graph from file
    G = nx.read_graphml("../config/entropy_views_10_best.graphml")

    # Print the nodes and edges
    print(f"G.nodes: {G.nodes}")
    print(f"G.edges: {G.edges}")

    # Convert the nodes' coordinates from spherical to cartesian
    for node in G.nodes:
        node_name = node.split(":")[0]
        node_coords = node.split(":")[1].split(",")
        [theta, phi] = [float(coord) for coord in node_coords]
        node_coords = as_cartesian([1, theta, phi])
        G.nodes[node]["name"] = node_name
        G.nodes[node]["coords"] = node_coords

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

    # Plot the nodes
    node_coords = [G.nodes[node]["coords"] for node in G.nodes]
    node_coords = np.array(node_coords)
    ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2])

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
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.001)

            # Save the image
            plt.savefig(f"../assets/entropy_views_100_{angle}.png")

        # Make the gif
        images = []
        for angle in range(0, 360):
            images.append(imageio.imread(f"../assets/entropy_views_100_{angle}.png"))

        imageio.mimsave('../assets/entropy_views_100.gif', images, fps=20)

    else:
        # Show the plot
        plt.show()


if __name__ == "__main__":
    plot_graph_from_file()
