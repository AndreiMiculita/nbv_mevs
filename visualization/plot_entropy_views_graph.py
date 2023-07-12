import networkx as nx

import numpy as np
from matplotlib import pyplot as plt

from geometry_utils.convert_coords import as_cartesian, as_spherical
from geometry_utils.fibonacci_sphere import fibonacci_sphere
from node_weighted_graph.build_graph_from_spherical_coords import *
from visualization.plotting_utils import set_axes_equal, show_3d_axes_rgb


# credit: https://stackoverflow.com/a/26127012/13200217


def get_fibonacci_sphere_coords(n=10):
    some_fibonacci_sphere = fibonacci_sphere(samples=n)

    some_fibonacci_sphere = np.array(some_fibonacci_sphere)

    # They're cartesian coordinates, so we need to convert them to spherical
    for i in range(len(some_fibonacci_sphere)):
        some_fibonacci_sphere[i] = as_spherical(some_fibonacci_sphere[i])

    # Format is (radius, theta, phi)
    print(some_fibonacci_sphere)

    # Discard 1st column (radius)
    some_fibonacci_sphere = some_fibonacci_sphere[:, 1:]

    # Decrease thetas by pi, negate thetas, and increase phis by pi
    some_fibonacci_sphere[:, 0] -= np.pi
    some_fibonacci_sphere[:, 0] *= -1
    some_fibonacci_sphere[:, 1] += np.pi

    # Convert to degrees
    some_fibonacci_sphere = np.degrees(some_fibonacci_sphere)

    print(some_fibonacci_sphere)

    return some_fibonacci_sphere


def get_example_entropy_views():
    return np.array([
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


def plot_entropy_views_graph_in_3d(entropy_views: np.ndarray):
    """
    Make a 3D plot of the entropy views graph.
    Use a 3D scatter plot to plot the nodes.
    Use a 3D line plot to plot the edges.
    :param entropy_views:
    :return:
    """

    # The columns of entropy_views are: latitude, longitude, entropy
    graph = build_graph_from_spherical_coords_with_delaunay(entropy_views)

    print(f"graph: {graph}")
    for node in graph:
        print(f"node: {node}, theta: {node.theta}, phi: {node.phi}, weight: {node.weight}")

    # Convert to a networkx graph, so we can save it to a file
    # Note that the theta and phi are converted to degrees to make the file more readable
    G = nx.Graph()
    # Add nodes with their attributes
    for node in graph:
        G.add_node(node.name.split(':')[0], theta=np.degrees(node.theta), phi=np.degrees(node.phi), weight=node.weight)
        for neighbor in node.neighbors:
            G.add_edge(node.name.split(':')[0], neighbor.name.split(':')[0])

    print(f"G.nodes: {G.nodes}")
    print(f"G.edges: {G.edges}")

    for node in G.nodes:
        for attrib in G.nodes[node]:
            print(type(G.nodes[node][attrib]))

    # Save to a file with write_graphml, append number of nodes to filename
    nx.write_graphml(G, f'entropy_views_{entropy_views.shape[0]}.graphml')

    # Convert the coords from spherical to cartesian, assume a unit sphere
    nodes_coords = np.array([as_cartesian([1, np.degrees(node.theta), np.degrees(node.phi)]) for node in graph])

    print(f"nodes_coords: {nodes_coords}")
    print(f"nodes_coords.shape: {nodes_coords.shape}")

    # Color based on weight
    nodes_weights = np.array([node.weight for node in graph])

    print(f"nodes_weights: {nodes_weights}")

    # Plot the nodes as a 3D scatter plot
    ax = plt.axes(projection="3d")
    sc = ax.scatter3D(nodes_coords[:, 0], nodes_coords[:, 1], nodes_coords[:, 2], s=100, c=nodes_weights)

    # Show coords and weights as labels
    for i, node in enumerate(graph):
        ax.text(nodes_coords[i, 0], nodes_coords[i, 1], nodes_coords[i, 2],
                f"  {i}: {np.degrees(node.theta):.2f}, {np.degrees(node.phi):.2f}, {node.weight:.2f}")

    # Plot the edges as a 3D line plot
    for node in graph:
        for neighbor in node.neighbors:
            node_coords = as_cartesian([1, np.degrees(node.theta), np.degrees(node.phi)])
            neighbor_coords = as_cartesian([1, np.degrees(neighbor.theta), np.degrees(neighbor.phi)])
            ax.plot3D([node_coords[0], neighbor_coords[0]], [node_coords[1], neighbor_coords[1]],
                      [node_coords[2], neighbor_coords[2]], 'gray')

    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    show_3d_axes_rgb(ax)

    # Show the plot
    plt.show()


def main():
    # Wider print for np arrays
    np.set_printoptions(suppress=True, linewidth=500)

    # Load the entropy views (normally from csv)
    entropy_views = get_fibonacci_sphere_coords(10)

    # Add 3rd column with entropies (random uniform for now)
    entropy_views = np.hstack((entropy_views, np.random.uniform(0, 1, (entropy_views.shape[0], 1))))

    # Plot the entropy views graph in 3D
    plot_entropy_views_graph_in_3d(entropy_views)


if __name__ == "__main__":
    main()
