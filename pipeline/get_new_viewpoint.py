import random
from pathlib import Path
from typing import List, Tuple


def get_new_viewpoint_coords(mesh_path: Path, attempted_viewpoints: List[Tuple[float, float]],
                             possible_viewpoints: List[Tuple[float, float]], method: str = "random") \
        -> Tuple[float, float]:
    """
    Chooses a next best viewpoint for recognizing an object from a mesh.
     :param mesh_path: The path to the mesh.
     :param attempted_viewpoints: List of attempted viewpoints, in (theta, phi) format
     :param possible_viewpoints: List of possible viewpoints
     :param method: Method for choosing a new viewpoint. Can be "random", "diff", or "pcd"

    :return: Tuple of floats: (theta, phi)
     theta: angle in radians [0 - pi]
     phi: angle in radians [0 - 2pi]
    """
    if method == "random":
        return random.choice(possible_viewpoints)
    elif method == "diff":
        return with_differentiable_renderer(mesh_path, attempted_viewpoints)
    elif method == "pcd":
        return with_point_cloud_embedding_network(mesh_path, attempted_viewpoints)

    raise ValueError("Invalid method: {}".format(method))


def with_differentiable_renderer(mesh_path: Path, attempted_viewpoints: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Chooses a new viewpoint for a mesh using differentiable rendering.
    :param mesh_path: The path to the mesh.
    :param attempted_viewpoints: List of attempted viewpoints

    :return: Tuple of floats: (theta, phi)
     theta: angle in radians [0 - pi]
     phi: angle in radians [0 - 2pi]
    """
    # Implement the functionality here
    pass


def with_point_cloud_embedding_network(mesh_path: Path, attempted_viewpoints: List[Tuple[float, float]]) -> Tuple[
    float, float]:
    """
    Choose a new viewpoint for a mesh using point cloud embedding network.

    :param mesh_path: The path to the mesh.
    :param attempted_viewpoints: List of attempted viewpoints

    :return: Tuple of floats: (theta, phi)
     theta: angle in radians [0 - pi]
     phi: angle in radians [0 - 2pi]
    """
    # Get capture of last attempted viewpoint
    # Run inference with the pointnet on the point cloud retrieved, output is entropy list
    # Build view graph structure
    # Find local maxima in view graph, sort them by entropy
    # Pick the highest entropy view; if already attempted, pick next one and so on; return it
    # If all the local maxima have been returned, sort all views by entropy,
    # return the highest one that is not a local maximum
    pass
