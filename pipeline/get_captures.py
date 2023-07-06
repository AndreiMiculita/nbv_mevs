"""
This is a script to get captures of a mesh.
Since we've already captured images/point clouds to create the train/test datasets synthetically,
we can just retrieve them from the dataset.
Point clouds are assumed to be in the following directory structure:
data
    view-dataset
        <class>
            <class>_<object_index>_theta_<theta>_phi_<phi>_vc_<view_index>.pcd
Images are assumed to be grayscale, and in the following directory structure:
data
    image-dataset
        <class>
            <class>_<object_index>_theta_<theta>_phi_<phi>_vc_<view_index>.png
"""

import glob
import os

import cv2
import numpy as np

dataset_path = os.path.join(os.path.dirname(__file__), "../data/image-dataset")


def get_image(mesh_path, theta, phi):
    """
    Get the image of the mesh at the specified theta and phi.
    """

    # Get the class name from the mesh path\
    class_name = os.path.basename(mesh_path).split("_")[0]

    # Get the object index from the mesh path
    object_index = os.path.basename(mesh_path).split("_")[1].split(".")[0]

    # convert from radians to degrees
    theta = round(np.degrees(theta))
    phi = round(np.degrees(phi))

    file_path = os.path.join(
        dataset_path,
        class_name,
        f"{class_name}_{object_index}_theta_{theta}_phi_{phi}_vc_*.png",
    )

    print(f'Retrieving captured image from {file_path}')

    # Get the file that matches the class name, object index, theta, and phi
    try:
        file = glob.glob(file_path)[0]
    except IndexError:
        print("No file found, has the dataset been created? Does the angle exist in the dataset?")
        return None

    # Read the image
    image = cv2.imread(file)

    return image
