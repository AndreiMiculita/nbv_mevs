import copy
import heapq

import numpy as np

from node_weighted_graph.build_graph_from_spherical_coords import build_graph_from_spherical_coords_with_delaunay

example_spherical_coords_40 = np.array(
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

example_spherical_coords_10 = np.array(
    [
        [156.0, 36.0, 1.37922013867174],
        [90.0, 180.0, 0.52175277589393],
        [90.0, 360.0, 0.45548316492884],
        [80.0, 96.0, 1.82280544551979],
        [102.0, 323.0, 1.73852709715075],
        [58.0, 278.0, 2.28815775214070],
        [138.0, 240.0, 1.54942298195957],
        [42.0, 35.0, 1.68800060774538],
        [34.0, 187.0, 0.99259078663013],
        [115.0, 149.0, 1.70055628535647],
    ])


def find_local_maximum_nodes(graph):
    """
    Given a node-weighted graph, find the nodes that have higher weights than all their neighbors.
    :param graph: a list of Node objects
    :return: a set of Node objects
    """

    result = set()

    for node in graph:
        is_higher_weight = all(node.weight > neighbor.weight for neighbor in node.neighbors)
        if is_higher_weight:
            result.add(node)

    return result


def find_local_maximum_nodes_nosort(graph):
    """
    Given a node-weighted graph, find the nodes that have higher weights than all their neighbors.
    :param graph: a list of Node objects
    :return: a set of Node objects
    """

    result = set()

    visited = set()

    for node in graph:
        if node in visited:
            continue
        is_higher_weight = all(node.weight > neighbor.weight for neighbor in node.neighbors)
        if is_higher_weight:
            result.add(node)
            for neighbor in node.neighbors:
                visited.add(neighbor)

    return result


def find_local_maximum_nodes(graph):
    """
    Given a node-weighted graph, find the nodes that have higher weights than all their neighbors.
    :param graph: a list of Node objects
    :return: a set of Node objects
    """

    result = set()

    # sort the nodes by weight
    graph = sorted(graph, key=lambda node: node.weight, reverse=True)

    for node in graph:
        is_higher_weight = all(node.weight > neighbor.weight for neighbor in node.neighbors)
        if is_higher_weight:
            result.add(node)

    return result


if __name__ == '__main__':
    import random
    from node_weighted_graph import Node

    # We do a benchmark test to compare it with the naive approach
    import time

    graph = build_graph_from_spherical_coords_with_delaunay(example_spherical_coords_10)

    print("Graph created")
    print("Starting benchmark")

    times1 = []
    for i in range(100):
        start_time = time.time()
        result1 = find_local_maximum_nodes(copy.deepcopy(graph))
        duration2 = time.time() - start_time
        times1.append(duration2)
        # sort it
        result1 = sorted(result1, key=lambda node: node.weight, reverse=True)

    print(f"Time taken for naive approach:  {np.mean(times1):f} seconds ({np.mean(times1) * 1000:f} ms)")

    times2 = []
    for i in range(100):
        start_time = time.time()
        result2 = find_local_maximum_nodes_nosort(copy.deepcopy(graph))
        duration2 = time.time() - start_time
        times2.append(duration2)
        result2 = sorted(result2, key=lambda node: node.weight, reverse=True)
        # make sure the result is the same
        try:
            assert [node.weight for node in result1] == [node.weight for node in result2]
        except AssertionError:
            print("\nResult and result2 is not the same!")
            print(f"len {len(result1)} result1: {[node.weight for node in result1]}")
            print(f"len {len(result2)} result2: {[node.weight for node in result2]}")

    print(f"Time taken for nosort approach: {np.mean(times2):f}")

    times3 = []
    for i in range(100):
        start_time = time.time()
        result3 = find_local_maximum_nodes(copy.deepcopy(graph))
        duration3 = time.time() - start_time
        times3.append(duration3)
        result3 = sorted(result3, key=lambda node: node.weight, reverse=True)
        # make sure the result is the same
        try:
            assert [node.weight for node in result1] == [node.weight for node in result3]
        except AssertionError:
            print("\nResult and result3 is not the same!")
            print(f"len {len(result1)} result1: {[node.weight for node in result1]}")
            print(f"len {len(result3)} result3: {[node.weight for node in result3]}")

    print(f"Time taken for sorted approach: {np.mean(times2):f}")

    times4 = []
    for i in range(100):
        start_time = time.time()
        result4 = find_local_maximum_nodes_samelist(copy.deepcopy(graph))
        duration4 = time.time() - start_time
        times4.append(duration4)
        result4 = sorted(result4, key=lambda node: node.weight, reverse=True)
        # make sure the result is the same
        try:
            assert [node.weight for node in result1] == [node.weight for node in result4]
        except AssertionError:
            print("\nResult and result4 is not the same!")
            print(f"len {len(result1)} result1: {[node.weight for node in result1]}")
            print(f"len {len(result4)} result4: {[node.weight for node in result4]}")

    print(f"Time taken for samelist       : {np.mean(times4):f}")
