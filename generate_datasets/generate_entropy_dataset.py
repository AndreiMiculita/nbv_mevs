"""
Generates a dataset in CSV format of depth-views entropy values.

For generating a dataset, a set of partial views centred on the object was collected.
The views were generated by rotating the object around the x-axis and the y-axis.
As these views had to be uniformly distributed, at first, a set of points representing the vertices of Platonic solids was used.
The points were then converted to spherical coordinates and the theta and phi values were used to generate the views.
The problem this brought was that it would not allow increasing the number of points arbitrarily.
To solve this, a Fibonacci sphere was used to generate the points.
This is an algorithm that allows the even distribution of an arbitrary number of points on a sphere.
The entropy values are stored in the entropy_dataset.csv file.
"""

import argparse
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
from open3d import *
from skimage.measure import shannon_entropy

from geometry_utils.convert_coords import as_spherical
from geometry_utils.fibonacci_sphere import fibonacci_sphere
from utility import normalize3d

parser = argparse.ArgumentParser(description="Generates a dataset in CSV format of depth-views entropy values.")
parser.add_argument("--modelnet10", help="Specify root directory to the ModelNet10 dataset.", required=True)
parser.add_argument("--out", help="Select a desired output directory.", default="./entropy-dataset4")
parser.add_argument("-v", "--verbose", help="Prints current state of the program while executing.", action='store_true')
parser.add_argument("-n", "--number_of_views", help="Number of views. These are generated using the Fibonacci "
                                                    "sphere algorithm",
                    default=10,
                    metavar='VALUE',
                    type=int
                    )
parser.add_argument("--debug", help="Prints debug statements during runtime.", action='store_true')
args = parser.parse_args()

BASE_DIR = sys.path[0]
OUT_DIR = os.path.join(BASE_DIR, args.out)
DATA_PATH = os.path.join(BASE_DIR, args.modelnet10)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
N_VIEWS = args.number_of_views
VIEW_INDEX = 0
FIRST_OBJECT = 1


def nonblocking_custom_capture(tr_mesh, rot_xyz, last_rot):
    """
    Custom function for Open3D to allow non-blocking capturing on a headless server.

    The function renders a triangle mesh file in a 224x224 window capturing a depth-image
    from the rot_xyz rotation of the object. Stores the image in the ./tmp/ folder.

    :param tr_mesh: open3d.geometry.TriangleMesh object to render.
    :param rot_xyz: (x, y, z)-tuple with rotation values.
    :param last_rot: if the function is being called within a loop of rotations,
     specify the previous rot_xyz to reposition the object to thee original rotation before
     applying rot_xyz
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=224, height=224, visible=False)
    vis.add_geometry(tr_mesh)
    # Rotate back from last rotation
    R_0 = tr_mesh.get_rotation_matrix_from_xyz(last_rot)
    tr_mesh.rotate(np.linalg.inv(R_0), center=tr_mesh.get_center())
    # Then rotate to the next rotation
    R = tr_mesh.get_rotation_matrix_from_xyz(rot_xyz)
    tr_mesh.rotate(R, center=tr_mesh.get_center())
    vis.update_geometry(tr_mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_depth_image(
        "{}/tmp/{}_{}_x_{}_y_{}.png".format(BASE_DIR, label, VIEW_INDEX, -round(np.rad2deg(rot_xyz[0])),
                                            round(np.rad2deg(rot_xyz[2]))), depth_scale=10000)
    vis.destroy_window()
    depth = cv2.imread("{}/tmp/{}_{}_x_{}_y_{}.png".format(BASE_DIR, label, VIEW_INDEX, -round(np.rad2deg(rot_xyz[0])),
                                                           round(np.rad2deg(rot_xyz[2]))))
    result = cv2.normalize(depth, depth, 0, 255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite("{}/tmp/{}_{}_x_{}_y_{}.png".format(BASE_DIR, label, VIEW_INDEX, -round(np.rad2deg(rot_xyz[0])),
                                                    round(np.rad2deg(rot_xyz[2]))), result)


labels = []
for cur in os.listdir(DATA_PATH):
    if os.path.isdir(os.path.join(DATA_PATH, cur)):
        labels.append(cur)

TMP_DIR = os.path.join(BASE_DIR, "tmp")
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)
else:
    os.makedirs(TMP_DIR)

for split_set in ['train', 'test']:
    for label in labels:
        files = os.listdir(os.path.join(DATA_PATH, label, split_set))
        files.sort()
        for file in files:  # Removes file without .off extension
            if not file.endswith('off'):
                files.remove(file)

        for file in files:
            OBJECT_INDEX = file.split('.')[0].split('_')[-1]
            VIEW_INDEX = 0
            filepath = os.path.join(DATA_PATH, label, split_set, file)

            mesh = io.read_triangle_mesh(filepath)
            mesh.vertices = normalize3d(mesh.vertices)
            mesh.compute_vertex_normals()

            if args.debug:
                print(f"[DEBUG] Current Object: {file}")

            initial_views = fibonacci_sphere(N_VIEWS)

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
                VIEW_INDEX = VIEW_INDEX + 1
                last_rotation = rot

            data_label = []
            data_code = []
            data_x = []
            data_y = []
            entropy = []
            data_index = []

            for filename in os.listdir(TMP_DIR):
                if "png" in filename:  # skip auto-generated .DS_Store
                    if "night_stand" in filename:
                        data_label.append("night_stand")
                        data_y.append(float((filename.split("_")[-1]).split(".")[0]))
                        data_x.append(float((filename.split("_")[-3]).split(".")[0]))
                        data_code.append(int((filename.split("_")[2])))
                        data_index.append(int(OBJECT_INDEX))
                        image = plt.imread(os.path.join(TMP_DIR, filename))
                        entropy.append(shannon_entropy(image))
                    else:
                        data_label.append(filename.split("_")[0])
                        data_y.append(float((filename.split("_")[-1]).split(".")[0]))
                        data_x.append(float((filename.split("_")[-3]).split(".")[0]))
                        data_code.append(int((filename.split("_")[1])))
                        data_index.append(int(OBJECT_INDEX))
                        image = plt.imread(os.path.join(TMP_DIR, filename))
                        entropy.append(shannon_entropy(image))

            data = pd.DataFrame({"label": data_label,
                                 "obj_ind": data_index,
                                 "code": data_code,
                                 "rot_x": data_x,
                                 "rot_y": data_y,
                                 "entropy": entropy})
            if FIRST_OBJECT == 1:  # Create the main DataFrame and csv, next ones will be appended
                FIRST_OBJECT = 0
                data.to_csv(os.path.join(OUT_DIR, "entropy_dataset.csv"), index=False)
            else:
                data.to_csv(os.path.join(OUT_DIR, "entropy_dataset.csv"), index=False, mode='a', header=False)

            for im in os.listdir(TMP_DIR):
                os.remove(os.path.join(TMP_DIR, im))

os.rmdir(TMP_DIR)
