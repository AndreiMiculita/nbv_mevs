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
from typing import Union

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


def get_capture(mesh_path: Path, theta: float, phi: float, capture_type="depth", nviews=10) \
        -> Union[np.ndarray, o3d.geometry.PointCloud, None]:
    """
    Get the capture of the mesh at the specified theta and phi.
    :param mesh_path: The path to the mesh.
    :param theta: angle in radians [0 - pi]
    :param phi: angle in radians [0 - 2pi]
    :param capture_type: The type of capture to retrieve. Can be "image", "depth" or "pcd"
    :param nviews: The number of views to capture
    :return: The capture of the mesh at the specified theta and phi.
    """

    if nviews == 10:
        image_dataset_path = Path(__file__).resolve().parent.parent / "data" / "image-dataset-mnet10-10views"
        depth_dataset_path = Path(__file__).resolve().parent.parent / "data" / "depth-dataset-mnet10-10views-test"
        point_cloud_dataset_path = Path(__file__).resolve().parent.parent / "data" / "pcd-dataset-mnet10-10views"
    elif nviews == 40:
        image_dataset_path = Path(__file__).resolve().parent.parent / "data" / "image-dataset-mnet10-40views"
        depth_dataset_path = Path(__file__).resolve().parent.parent / "data" / "depth-dataset-mnet10-40views"
        point_cloud_dataset_path = Path(__file__).resolve().parent.parent / "data" / "pcd-dataset-mnet10-40views-test"
    else:
        raise ValueError("Invalid number of views: {}".format(nviews))

    if capture_type == "image":
        dataset_path = image_dataset_path
    elif capture_type == "depth":
        dataset_path = depth_dataset_path
    else:
        dataset_path = point_cloud_dataset_path

    # Get the class name from the mesh path
    class_name = mesh_path.parent.parent.stem

    # Get the object index from the mesh path
    object_index = mesh_path.stem.split("_")[-1]

    # Convert from radians to degrees
    theta = round(np.degrees(theta))
    phi = round(np.degrees(phi))

    file_path = dataset_path / class_name / \
                f"{class_name}_{object_index}_theta_{theta}_phi_{phi}_vc_*." \
                f"{'png' if capture_type == 'image' or capture_type == 'depth' else 'pcd'}"

    print(f'Retrieving captured {capture_type} from {file_path}')

    # Get the file that matches the class name, object index, theta, and phi
    files = list(file_path.parent.glob(file_path.name))
    if not files:
        print(f'No file found. Has the {capture_type} dataset been created? Does the angle exist in the dataset?')
        return None

    if capture_type == "image":
        # Read the image
        return cv2.imread(str(files[0]))
    elif capture_type == "depth":
        # Read the depth image
        image = Image.open(str(files[0]))
        image = np.array(image)
        return image
    else:
        # Read the point cloud
        return o3d.io.read_point_cloud(str(files[0]))
