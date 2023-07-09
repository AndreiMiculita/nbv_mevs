import heapq


def find_local_maximum_nodes(graph):
    """
    Given a node-weighted graph, find the nodes that have higher weights than all their neighbors.
    :param graph: a list of Node objects
    :return: a set of Node objects
    """

    result = set()
    priority_queue = []

    # Initialize priority queue with all nodes
    for node in graph:
        heapq.heappush(priority_queue, (node.weight, node))

    while priority_queue:
        current_weight, current_node = heapq.heappop(priority_queue)

        # Check if current node has higher weight than neighbors
        is_higher_weight = all(current_node.weight > neighbor.weight for neighbor in current_node.neighbors)
        if is_higher_weight:
            result.add(current_node)

        # Remove neighbors with lower weights from priority queue
        updated_neighbors = []
        for neighbor in current_node.neighbors:
            if neighbor.weight > current_node.weight:
                updated_neighbors.append(neighbor)
        current_node.neighbors = updated_neighbors

        # Update priority queue with the remaining neighbors
        for neighbor in current_node.neighbors:
            heapq.heappush(priority_queue, (neighbor.weight, neighbor))

    return result
