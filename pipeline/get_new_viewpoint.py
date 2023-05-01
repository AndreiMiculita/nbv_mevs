import random

import numpy as np
from open3d import *


def get_new_viewpoint(mesh, attempted_viewpoints, method="random"):
    """
    Chooses a new viewpoint for a mesh.

    Args:
        mesh : open3d.geometry.TriangleMesh
        attempted_viewpoints : List[float, float]
        method (str): Method for choosing a new viewpoint. Can be "random", "diff", or "pcd"

    Returns a tuple of floats:
        theta (float): Azimuthal angle;
        phi (float): Polar angle
    """
    if method == "random":
        return random.random() * np.pi, random.random() * 2 * np.pi - np.pi
    elif method == "diff":
        return with_differentiable_renderer(mesh, attempted_viewpoints)
    elif method == "pcd":
        return with_point_cloud_embedding_network(mesh, attempted_viewpoints)

    raise ValueError("Invalid method: {}".format(method))


def with_differentiable_renderer(mesh, attempted_viewpoints):
    """
    Chooses a new viewpoint for a mesh using differentiable rendering.

    Args:
          mesh : open3d.geometry.TriangleMesh
          attempted_viewpoints : List[float, float]

    Returns:
        theta (float): Azimuthal angle
        phi (float): Polar angle
    """

    pass


def with_point_cloud_embedding_network(mesh, attempted_viewpoints):
    """
    Choose a new viewpoint for a mesh using point cloud embedding network.

    Args:

    Returns:
        theta (float): Azimuthal angle
        phi (float): Polar angle
    """

    pass
