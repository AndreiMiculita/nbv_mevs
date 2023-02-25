import numpy as np
import torch
import open3d as o3d
from open3d.cuda.pybind.geometry import TriangleMesh


def load_off(filename_off, normalization=True):
    """
    Loads an off file and returns its vertices and faces.
    :param filename_off:
    :param normalization:
    :return:
    """
    mesh: TriangleMesh = o3d.io.read_triangle_mesh(filename_off)
    vertices = mesh.vertices
    faces = mesh.triangles
    # Convert to cuda tensors
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32)).cuda()
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).cuda()

    # Normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= torch.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    return vertices, faces
