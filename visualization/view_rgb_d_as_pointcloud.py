# Visualize RGB-D data as pointcloud

import open3d as o3d
import numpy as np
import sys

def custom_draw_geometry_with_key_callback(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        print(opt.background_color)
        print(type(opt.background_color))
        if (opt.background_color == np.asarray([1., 1., 1.])).all():
            opt.background_color = np.asarray([0., 0., 0.])
        elif (opt.background_color == np.asarray([0., 0., 0.])).all():
            opt.background_color = np.asarray([1., 1., 1.])
        return False

    key_to_callback = {ord("K"): change_background_to_black}

    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

def view_pcd(rgb_path, depth_path):
    """
    Given an rgb image and its equivalent depth image, we visualize them as a point cloud
    :param rgb_path:
    :param depth_path:
    :return:
    """

    # Load the RGB image
    rgb = o3d.io.read_image(rgb_path)

    # Load the depth image
    depth = o3d.io.read_image(depth_path)

    # Convert the RGB and depth images into pointcloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.geometry.Image(rgb), depth=o3d.geometry.Image(depth), convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=rgbd,
        intrinsic=o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # Set background to black

    custom_draw_geometry_with_key_callback(pcd)


    print(pcd)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: view_rgb_d_as_pointcloud.py <rgb_path> <depth_path>")
        sys.exit(1)
    rgb_path = sys.argv[1]
    depth_path = sys.argv[2]
    view_pcd(rgb_path, depth_path)