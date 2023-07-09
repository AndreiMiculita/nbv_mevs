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

from pathlib import Path

import cv2
import numpy as np

image_dataset_path = Path(__file__).resolve().parent.parent / "data" / "image-dataset"


def get_image(mesh_path: Path, theta: float, phi: float):
    """
    Get the image of the mesh at the specified theta and phi.
    :param mesh_path: The path to the mesh.
    :param theta: angle in radians [0 - pi].
    :param phi: angle in radians [0 - 2pi]
    :return: The image of the mesh at the specified theta and phi.
    """

    # Get the class name from the mesh path
    class_name = mesh_path.stem.split("_")[0]

    # Get the object index from the mesh path
    object_index = mesh_path.stem.split("_")[1]

    # Convert from radians to degrees
    theta = round(np.degrees(theta))
    phi = round(np.degrees(phi))

    file_path = image_dataset_path / class_name / f"{class_name}_{object_index}_theta_{theta}_phi_{phi}_vc_*.png"

    print(f'Retrieving captured image from {file_path}')

    # Get the file that matches the class name, object index, theta, and phi
    files = list(file_path.parent.glob(file_path.name))
    if not files:
        print("No file found. Has the dataset been created? Does the angle exist in the dataset?")
        return None

    # Read the image
    image = cv2.imread(str(files[0]))

    return image
