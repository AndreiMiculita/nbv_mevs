# Takes as input a point cloud and an n and returns the n best views
# of the point cloud.
import glob
import math
import os

import imageio
import open3d as o3d
import torch
import torch.nn as nn
import tqdm

import numpy as np
from skimage.io import imread, imsave
import neural_renderer as nr
import cv2
from pprint import pprint
from inspect import getmembers
from types import FunctionType


# credit: https://stackoverflow.com/a/26127012/13200217
def fibonacci_sphere(samples=1000):

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


class Model(nn.Module):
    def __init__(self, mesh, filename_ref=None, normalization=True, tqdm_loop=None, initial_camera_position=None):
        super(Model, self).__init__()

        if initial_camera_position is None:
            initial_camera_position = [6, 10, -14]
        self.distance_threshold = 6.0
        self.distance_penalty_weight = 3000.0
        self.tqdm_loop = tqdm_loop

        # Load mesh vertices and faces
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

        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image
        image_ref = torch.from_numpy((imread(filename_ref).max(-1) != 0).astype(np.float32))
        self.register_buffer('image_ref', image_ref)

        # camera parameters
        self.camera_position = nn.Parameter(torch.from_numpy(np.array(initial_camera_position, dtype=np.float32)))

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.eye = self.camera_position
        self.renderer = renderer

    def forward(self):
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')

        distance = torch.norm(self.camera_position)
        distance_penalty = torch.relu(self.distance_threshold - distance)

        loss = torch.sum((self.image_ref[None, :] - image) ** 2) * (1 + distance_penalty)

        if self.tqdm_loop is not None:
            self.tqdm_loop.set_description(f'Distance: {distance:.2f}, distance penalty: {distance_penalty:.2f}, loss: {loss:.2f}')

        return loss


def get_best_views(pcd: o3d.geometry.PointCloud, n: int) -> list:

    # Remove the table plane using RANSAC
    # TODO

    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=10)

    # We need a mesh, so we do Poisson surface reconstruction
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Info) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd, depth=4)
    print(mesh)
    # print_attributes(mesh)

    # Visualize the mesh
    # o3d.visualization.draw_geometries([mesh])

    # For this mesh, we need to find the best n views
    best_views = []

    for view_i in range(n):
        loop = tqdm.tqdm(range(1000))

        # Initialize the model
        if best_views:  # if we already have some views, use the opposite of the last view
            initial_camera_position = - best_views[-1].detach().cpu().numpy()
            # normalize and multiply by 16
            initial_camera_position /= np.linalg.norm(initial_camera_position)
            initial_camera_position *= 16
            model = Model(mesh, "../data/gaussian_reference.png", tqdm_loop=loop, initial_camera_position=initial_camera_position)
        else:
            model = Model(mesh, "../data/gaussian_reference.png", tqdm_loop=loop)

        model.cuda()

        # optimizer = chainer.optimizers.Adam(alpha=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for i in loop:
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            optimizer.step()
            images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))

            # print(images.shape)

            # image = (images.detach().cpu().numpy()[0].copy() * 255).astype(np.uint8)

            image = (images.detach().cpu().numpy()[0].transpose(1, 2, 0).copy() * 255).astype(np.uint8)

            # https://www.programcreek.com/python/example/89325/cv2.Sobel
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
            abs_sobel_x = cv2.convertScaleAbs(sobel_x)
            abs_sobel_y = cv2.convertScaleAbs(sobel_y)
            edges = cv2.addWeighted(np.uint8(abs_sobel_x), 0.5, np.uint8(abs_sobel_y), 0.5, 0)
            # edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # print("shapes", image.shape, edges.shape)
            concat = cv2.hconcat([image, edges])
            cv2.putText(concat, f"loss: {loss.item():.2f}", (6, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                        cv2.LINE_AA)
            imsave('/tmp/_tmp_%04d.png' % i, concat)

            if loss.item() < 70:
                break
        make_gif(f'standardized_interface_output{view_i}.mp4')
        best_views.append(model.renderer.eye)

    return best_views


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()

if __name__ == "__main__":
    # Load the point cloud
    pcd = o3d.io.read_point_cloud("/home/andrei/datasets/washington_short_version/Category/banana_Category/banana_object_10.pcd")

    # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])

    # exit()

    # Get the best views
    views = get_best_views(pcd, 3)
    print("The best views are:")
    print("\n".join(["\t%s" % str(v.detach().cpu().numpy()) for v in views]))

