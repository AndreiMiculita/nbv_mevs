# This file contains functions for rendering a mesh from a given viewpoint and for choosing a new viewpoint for a mesh.
# It can use three different methods for choosing a new viewpoint: random, differentiable rendering, and point cloud embedding network.
from random import random

import numpy as np
import trimesh


def render_mesh(mesh_path, theta, phi):
    """
    Renders a mesh from a given viewpoint.

    Args:
        mesh_path (str): Path to the mesh
        theta (float): Azimuthal angle
        phi (float): Polar angle

    Returns:
        image (np.ndarray): Rendered image
    """
    # Load the mesh
    mesh = trimesh.load(mesh_path)
    # Get the camera parameters
    camera = trimesh.scene.Camera(resolution=(224, 224), fov=(np.pi / 3, np.pi / 3))
    # Get the scene
    scene = trimesh.Scene(mesh, camera=camera)
    # Render the scene from the given viewpoint
    image = scene.render(camera=camera, view=(theta, phi))
    return image


def get_new_viewpoint(mesh_path, attempted_viewpoints, method="random"):
    """
    Chooses a new viewpoint for a mesh.

    Args:
        mesh_path (str): Path to the mesh
        attempted_viewpoints (list): List of attempted viewpoints
        method (str): Method for choosing a new viewpoint. Can be "random", "diff", or "pcd"

    Returns:
        theta (float): Azimuthal angle
        phi (float): Polar angle
    """
    if method == "random":
        theta = random.random() * np.pi
        phi = random.random() * 2 * np.pi - np.pi
    elif method == "diff":
        theta, phi = get_new_viewpoint_differentiable_rendering(mesh)
    elif method == "pcd":
        theta, phi = get_new_viewpoint_point_cloud_embedding_network(mesh)

    return theta, phi

def get_new_viewpoint_differentiable_rendering(mesh):
    """
    Chooses a new viewpoint for a mesh using differentiable rendering.

    Args:
        mesh_path (str): Path to the mesh

    Returns:
        theta (float): Azimuthal angle
        phi (float): Polar angle
    """

    pass

def get_new_viewpoint_point_cloud_embedding_network(mesh):
    """
    Chooses a new viewpoint for a mesh using point cloud embedding network.

    Args:
        mesh_path (str): Path to the mesh

    Returns:
        theta (float): Azimuthal angle
        phi (float): Polar angle
    """

    pass
