class Node:
    """
    A node in a graph with a name, a weight and a set of neighbors.
    This is used for creating a weighted vertex graph.
    :param name: the name of the node
    :param theta: angle measured in radians [0 - pi]
    :param phi: angle measured in radians [0 - 2pi]
    :param weight: the weight of the node
    """

    def __init__(self, name: str, theta, phi, weight):
        self.name: str = name
        self.theta = theta
        self.phi = phi
        self.weight = weight
        self.neighbors = set()

    def add_neighbor(self, neighbor: 'Node'):
        self.neighbors.add(neighbor)
        neighbor.neighbors.add(self)
