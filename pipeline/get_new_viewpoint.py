import copy
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

from node_weighted_graph import Node
from pipeline.get_captures import get_capture


def get_new_viewpoint_coords(mesh_path: Path,
                             attempted_viewpoints: List[Tuple[float, float]],
                             possible_viewpoints_graph: List[Node],
                             method: str = "random",
                             pcd_model_path: Path = None,
                             differentiable_rendering_model: Path = None) \
        -> Tuple[float, float]:
    """
    Chooses a next best viewpoint for recognizing an object from a mesh.
    :param mesh_path: The path to the mesh.
    :param attempted_viewpoints: List of attempted viewpoints, in (theta, phi) format
    :param method: Method for choosing a new viewpoint. Can be "random", "diff", or "pcd"
    :param possible_viewpoints_graph: The possible viewpoints, with neighbors
    :param pcd_model_path: Path to the point cloud embedding network model
    :param differentiable_rendering_model: Path to the differentiable rendering model

    :return: Tuple of floats: (theta, phi)
     theta: angle in radians [0 - pi]
     phi: angle in radians [0 - 2pi]
    """

    if method == "random":
        random_node = random.choice(possible_viewpoints_graph)
        return random_node.theta, random_node.phi
    elif method == "diff":
        return with_differentiable_renderer(mesh_path, attempted_viewpoints, differentiable_rendering_model)
    elif method == "pcd":
        if pcd_model_path is None or not pcd_model_path.exists():
            raise ValueError("pcd_model_path must be specified and must exist for method 'pcd'")
        return with_point_cloud_embedding_network(mesh_path, attempted_viewpoints, pcd_model_path,
                                                  possible_viewpoints_graph)

    raise ValueError("Invalid method: {}".format(method))


def with_point_cloud_embedding_network(mesh_path: Path, attempted_viewpoints: List[Tuple[float, float]],
                                       pcd_model_path: Path, possible_viewpoints_graph: List[Node]) \
        -> Tuple[float, float]:
    """
    Choose a new viewpoint for a mesh using point cloud embedding network.
    Use CUDA if available.
    Returns the viewpoint with the highest entropy that hasn't been attempted yet (prefers local maxima).

    :param mesh_path: The path to the mesh.
    :param attempted_viewpoints: List of attempted viewpoints
    :param pcd_model_path: Path to the point cloud embedding network model
    :param possible_viewpoints_graph: The possible viewpoints, with neighbors

    :return: Tuple of floats: (theta, phi)
     theta: angle in radians [0 - pi]
     phi: angle in radians [0 - 2pi]
    """
    import torch
    from node_weighted_graph.find_local_maximum_nodes import find_local_maximum_nodes

    # Get capture pcd of last attempted viewpoint
    pcd = get_capture(mesh_path, attempted_viewpoints[-1][0], attempted_viewpoints[-1][1], capture_type="pcd")

    # Run inference with the pointnet on the point cloud retrieved, output is entropy list; use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # WARNING: this only works on the same environment that the PointNet model was trained on
    # Load the PointNet model
    try:
        from pointnet2.models import PointNet2EntropySSG
    except ModuleNotFoundError:
        raise ModuleNotFoundError("PointNet model not found. Are you in the correct environment?")
    model = PointNet2EntropySSG.load_from_checkpoint(str(pcd_model_path))
    model.to(device)
    model.eval()

    # Run inference
    point_set = np.asarray(pcd.points, dtype=np.float32)
    # num_points random points from the point set
    pt_idxs = np.arange(0, point_set.shape[0])
    np.random.shuffle(pt_idxs)
    pt_idxs = pt_idxs[: 5000]

    point_set = point_set[pt_idxs, :]

    def pc_normalize(pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

    # Pad point_set with 0s up to 6 columns
    point_set = np.pad(point_set, ((0, 0), (0, 6 - point_set.shape[1])), "constant")

    point_set = torch.from_numpy(point_set).to(device)

    with torch.no_grad():
        entropy = model(point_set.unsqueeze(0)).squeeze(0).cpu().numpy()  # Funny business to get rid of batch dimension

    # Build view graph structure; entropy should have the same length as possible_viewpoints_graph
    if len(entropy) != len(possible_viewpoints_graph):
        raise ValueError("Length of pcd model output does not match length of view graph. Are you sure you're using "
                         "the correct model/view graph?")
    else:
        for i, node in enumerate(possible_viewpoints_graph):
            node.weight = entropy[i]

    # Print the graph, with all information
    for node in possible_viewpoints_graph:
        print(f'Node: {node.name}'
              f'\n\ttheta: {node.theta}'
              f'\n\tphi: {node.phi}'
              f'\n\tweight: {node.weight}'
              f'\n\tneighbors: {[neighbor.name for neighbor in node.neighbors]}')

    # Find local maxima in view graph, sort them by entropy
    local_maxima = list(find_local_maximum_nodes(copy.deepcopy(possible_viewpoints_graph)))
    local_maxima.sort(key=lambda x: x.weight, reverse=True)

    # Pick the highest entropy view; if already attempted, pick next one and so on; return it
    # Note that we're working with floats so best to use np.isclose
    for node in local_maxima:
        attempted = False
        for attempted_viewpoint in attempted_viewpoints:
            if np.isclose(node.theta, attempted_viewpoint[0]) and np.isclose(node.phi, attempted_viewpoint[1]):
                attempted = True
                break
        if not attempted:
            return node.theta, node.phi

    # If all the local maxima have been attempted, sort all views by entropy,
    # and return the first one that hasn't been attempted
    possible_viewpoints_graph.sort(key=lambda x: x.weight, reverse=True)
    for node in possible_viewpoints_graph:
        attempted = False
        for attempted_viewpoint in attempted_viewpoints:
            if np.isclose(node.theta, attempted_viewpoint[0]) and np.isclose(node.phi, attempted_viewpoint[1]):
                attempted = True
                break
        if not attempted:
            return node.theta, node.phi

    # If all the views have been attempted, raise an error
    raise ValueError("All viewpoints have been attempted, still not sure")
