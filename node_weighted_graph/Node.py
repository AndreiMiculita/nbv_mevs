class Node():
    """
    A node in a graph with a name, a weight and a set of neighbors.
    This is used for creating a weighted vertex graph.
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
