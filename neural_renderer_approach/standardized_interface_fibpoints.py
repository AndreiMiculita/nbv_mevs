# Takes as input a point cloud and an n and returns the n best views
# of the point cloud.
import glob
import math
import os
from typing import Tuple

import imageio
import open3d as o3d
import torch
import torch.nn as nn
import tqdm

import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread, imsave
import neural_renderer as nr
import cv2

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

from pprint import pprint
from inspect import getmembers
from types import FunctionType

from geometry_utils.fibonacci_sphere import fibonacci_sphere


class Model(nn.Module):
    def __init__(self, mesh, filename_ref=None, normalization=True, tqdm_loop=None, initial_camera_position=None):
        super(Model, self).__init__()

        if initial_camera_position is None:
            initial_camera_position = [6, 10, -14]
        self.distance_threshold = 6.0
        self.distance_penalty_weight = 10000.0
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


def get_best_views(mesh, n: int) -> Tuple[list, list, list]:

    # Visualize the mesh
    # o3d.visualization.draw_geometries([mesh])

    # For this mesh, we need to find the best n views
    best_view_coords = []
    best_view_images = []
    losses = []

    initial_views = fibonacci_sphere(n)
    initial_view_radius = 16
    # multiply every element in the list of lists by the initial_view_radius
    initial_views = [[initial_view_radius * i for i in inner] for inner in initial_views]
    print(f"Initial views: {initial_views}")

    for idx, initial_view in enumerate(initial_views):
        loop = tqdm.tqdm(range(1000))

        # Initialize the model
        model = Model(mesh, "../data/references/gaussian_reference.png", tqdm_loop=loop, initial_camera_position=initial_view)

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

        make_gif(f'standardized_interface_output{idx}.mp4')
        best_view_coords.append(model.renderer.eye)
        best_view_images.append(image)
        losses.append(loss.item())

    return best_view_coords, best_view_images, losses


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()


def get_best_views_from_pcd(pcd, n: int) -> Tuple[list, list, list]:
    # We need a mesh, so we do Poisson surface reconstruction
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Info) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd, depth=4)
    print(mesh)
    return get_best_views(mesh, n)


def get_best_views_from_file(objfile: str, n: int) -> Tuple[list, list, list]:
    if objfile.endswith('.obj') or objfile.endswith('.off'):
        mesh = o3d.io.read_triangle_mesh(objfile)
        return get_best_views(mesh, n)
    elif objfile.endswith('.pcd'):
        # Load the point cloud
        pcd = o3d.io.read_point_cloud(objfile)

        # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])

        # Remove the table plane using RANSAC
        # TODO

        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=10)

        return get_best_views_from_pcd(pcd, n)


if __name__ == "__main__":
    # Get the best views
    views, view_imgs, losses = get_best_views_from_file(objfile="/home/andrei/datasets/teapot.obj", n=3)

    print("The best views (with respective losses) are:")
    for view, loss in zip(views, losses):
        print(f"\t{view.detach().cpu().numpy()} with loss {loss}")
    for idx, view_img in enumerate(view_imgs):
        cv2.imwrite(f"view_{idx}.png", view_img)

    # print best views to file
    with open("../data/best_views.txt", "w") as f:
        for view, loss in zip(views, losses):
            f.write(f"{view.detach().cpu().numpy()} with loss {loss}\n")

        print("Removing views that are too close to each other (remove higher loss view)")
        # Remove views that are too close to each other using hierarchical clustering
        # credit: https://datascience.stackexchange.com/a/47635
        X = np.array([v.detach().cpu().numpy() for v in views])
        Z = linkage(X,
                    method='complete',  # dissimilarity metric: max distance across all pairs of
                    # records between two clusters
                    metric='euclidean'
                    )  # you can peek into the Z matrix to see how clusters are
                        # merged at each iteration of the algorithm

        # calculate full dendrogram and visualize it
        plt.figure(figsize=(30, 10))
        dendrogram(Z)
        plt.show()

        max_d = 1
        clusters = fcluster(Z, max_d, criterion='distance')

        # print("Clusters:")
        print(clusters)

        # Only keep lowest loss view per cluster
        views_to_keep = []
        losses_to_keep = []

        for i in set(clusters):
            print(f"Cluster {i}")
            for j in range(len(views)):
                if clusters[j] == i:
                    print(f"\t{str(views[j].detach().cpu().numpy())}")
            views_in_cluster = [views[j] for j in range(len(views)) if clusters[j] == i]
            losses_in_cluster = [losses[j] for j in range(len(views)) if clusters[j] == i]

            views_to_keep.append(views_in_cluster[np.argmin(losses_in_cluster)])
            losses_to_keep.append(np.min(losses_in_cluster))

        print("After removing views that are too close to each other, the best views are:")
        for view, loss in zip(views_to_keep, losses_to_keep):
            print(f"\t{view.detach().cpu().numpy()} with loss {loss}")

        f.write("\n\nCleaned up:\n")
        for view, loss in zip(views_to_keep, losses_to_keep):
            f.write(f"{view.detach().cpu().numpy()} with loss {loss}\n")

