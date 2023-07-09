class Node:
    """
    A node in a graph with a name, a weight and a set of neighbors.
    This is used for creating a weighted vertex graph.
    :param name: the name of the node
    :param lon: the longitude of the node
    :param lat: the latitude of the node
    :param weight: the weight of the node
    """

    def __init__(self, name: str, lon, lat, weight):
        self.name: str = name
        self.lon = lon
        self.lat = lat
        self.weight = weight
        self.neighbors = set()

    def add_neighbor(self, neighbor: 'Node'):
        self.neighbors.add(neighbor)
        neighbor.neighbors.add(self)
