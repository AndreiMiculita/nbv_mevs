# Based on https://github.com/tparisotto/more_fyp/blob/master/generate_view_dataset.py

"""
Generates views regularly positioned on a sphere around the objects.

This implementation renders plain images and depth images of views from
viewpoints regularly distributed on a specified number rings parallel to the x-y plain,
spaced vertically on a sphere around the object.
"""

import argparse
import math
import os
import sys
from time import time

import numpy as np
import open3d as o3d

from geometry_utils.convert_coords import as_spherical
from geometry_utils.fibonacci_sphere import fibonacci_sphere
from utility import normalize3d

parser = argparse.ArgumentParser(description="Generates views regularly positioned on a sphere around the object.")
parser.add_argument("--modelnet10", help="Specify root directory to the ModelNet10 dataset.")
parser.add_argument("--set", help="Subdirectory: 'train' or 'test'.", default='test')
parser.add_argument("--out", help="Select a desired output directory.", default="./view-dataset-test")
parser.add_argument("-v", "--verbose", help="Prints current state of the program while executing.", action='store_true')
parser.add_argument("-x", "--horizontal_split", help="Number of views from a single ring. Each ring is divided in x "
                                                     "splits so each viewpoint is at an angle of multiple of 360/x. "
                                                     "Example: -x=12 --> phi=[0, 30, 60, 90, 120, ... , 330].",
                    default=12,
                    metavar='VALUE',
                    type=int
                    )
parser.add_argument("-y", "--vertical_split", help="Number of horizontal rings. Each ring of viewpoints is an "
                                                   "horizontal section of a sphere, looking at the center at an angle "
                                                   "180/(y+1), skipping 0 and 180 degrees (since the diameter of the "
                                                   "ring would be zero). Example: -y=5 --> theta=[30, 60, 90, 120, "
                                                   "150] ; -y=11 --> theta=[15, 30, 45, ... , 150, 165]. ",
                    default=5,
                    metavar='VALUE',
                    type=int
                    )
args = parser.parse_args()

BASE_DIR = sys.path[0]
OUT_DIR = os.path.join(BASE_DIR, args.out)
DATA_PATH = os.path.join(BASE_DIR, args.modelnet10)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
N_VIEWS_W = args.horizontal_split
N_VIEWS_H = args.vertical_split


class ViewData:
    """ Class to keep track of attributes of the views. """
    obj_label = ''
    obj_index = 1
    view_index = 0
    obj_path = ''
    obj_filename = ''
    phi = 0
    theta = 0


if os.path.exists(os.path.join(BASE_DIR, OUT_DIR)):
    print("[Error] Folder already exists.")
    exit(0)


def nonblocking_custom_capture(tr_mesh, rot_xyz, last_rot):
    """
        Custom function for Open3D to allow non-blocking capturing on a headless server.

        The function renders a triangle mesh file in a 224x224 window capturing a depth-image
        and an RGB image-view from the rot_xyz rotation of the object.
        Stores the images in the ./tmp/ folder.

        :param tr_mesh: open3d.geometry.TriangleMesh object to render.
        :param rot_xyz: (x, y, z)-tuple with rotation values.
        :param last_rot: if the function is being called within a loop of rotations,
         specify the previous rot_xyz to reposition the object to thee original rotation before
         applying rot_xyz
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, visible=False)
    vis.add_geometry(tr_mesh)
    # Rotate back from last rotation
    R_0 = tr_mesh.get_rotation_matrix_from_xyz(last_rot)
    tr_mesh.rotate(np.linalg.inv(R_0), center=tr_mesh.get_center())
    # Then rotate to the next rotation
    R = tr_mesh.get_rotation_matrix_from_xyz(rot_xyz)
    ViewData.theta = -round(np.rad2deg(rot_xyz[0]))
    ViewData.phi = round(np.rad2deg(rot_xyz[2]))
    tr_mesh.rotate(R, center=tr_mesh.get_center())
    vis.update_geometry(tr_mesh)
    vis.poll_events()
    vis.update_renderer()

    # Capture depth image
    vis.capture_depth_point_cloud(get_dataset_path())

    vis.destroy_window()


def get_dataset_path():
    return "{}/{}/{}_{}_theta_{}_phi_{}_vc_{}.pcd".format(OUT_DIR,
                                                          ViewData.obj_label,
                                                          ViewData.obj_label,
                                                          ViewData.obj_index,
                                                          int(ViewData.theta),
                                                          int(ViewData.phi),
                                                          ViewData.view_index)


labels = []
for cur in os.listdir(DATA_PATH):
    if os.path.isdir(os.path.join(DATA_PATH, cur)):
        labels.append(cur)

for label in labels:

    # Create output directory for label
    if not os.path.exists(os.path.join(OUT_DIR, label)):
        os.makedirs(os.path.join(OUT_DIR, label))

    files = os.listdir(os.path.join(DATA_PATH, label, args.set))
    files.sort()
    for filename in files:  # Removes file without .off extension
        if not filename.endswith('off'):
            files.remove(filename)

    for filename in files:
        start = time()
        ViewData.obj_path = os.path.join(DATA_PATH, label, args.set, filename)
        ViewData.obj_filename = filename
        ViewData.obj_index = filename.split(".")[0].split("_")[-1]
        ViewData.obj_label = filename.split(".")[0].replace("_" + ViewData.obj_index, '')
        ViewData.view_index = 0
        if args.verbose:
            print(f"[INFO] Current object: {ViewData.obj_label}_{ViewData.obj_index}")
            print(
                f"[DEBUG] ViewData:\n [objpath: {ViewData.obj_path},\n filename: {ViewData.obj_filename},\n label: {ViewData.obj_label},\n index: {ViewData.obj_index}]")
        mesh = o3d.io.read_triangle_mesh(ViewData.obj_path)
        mesh.vertices = normalize3d(mesh.vertices)
        mesh.compute_vertex_normals()

        initial_views = fibonacci_sphere(10)

        rotations = []

        for initial_view in initial_views:
            [r, theta, phi] = as_spherical(initial_view)
            rotations.append(as_spherical(initial_view))

        # Convert to np array
        rotations = np.array(rotations)

        # Decrease theta by pi to get the same rotations as the original
        rotations[:, 1] = rotations[:, 1] - np.pi

        # Increase phi by pi to get the same rotations as the original
        rotations[:, 2] = rotations[:, 2] + np.pi

        # Decrease r to 0 to get the same rotations as the original
        rotations[:, 0] = 0

        # Swap theta and r to get the same rotations as the original
        rotations[:, 0], rotations[:, 1] = rotations[:, 1], rotations[:, 0].copy()

        # Print min and max of each column
        # print(f"[DEBUG] Min and max of each column:")
        # print(f"[DEBUG] theta: {np.min(rotations[:, 0]):.2f} - {np.max(rotations[:, 0]):.2f}")
        # print(f"[DEBUG] r: {np.min(rotations[:, 1]):.2f} - {np.max(rotations[:, 1]):.2f}")
        # print(f"[DEBUG] phi: {np.min(rotations[:, 2]):.2f} - {np.max(rotations[:, 2]):.2f}")
        #
        # print(f"[INFO] Rotations: {rotations}")

        last_rotation = (0, 0, 0)
        for rot in rotations:
            nonblocking_custom_capture(mesh, rot, last_rotation)
            ViewData.view_index += 1
            if args.verbose:
                print(f"[INFO] Elaborating view {ViewData.view_index}/{N_VIEWS_W * N_VIEWS_H}...")
            last_rotation = rot

        end = time()
        if args.verbose:
            print(f"[INFO] Time to elaborate file {filename}: {end - start}")
