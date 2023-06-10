import numpy as np
import open3d as o3d


def custom_draw_geometry_with_key_callback(pcd):
    """
    This function is a modified version of the function custom_draw_geometry from the Open3D examples.
    It allows us to change the background color of the point cloud by pressing the key 'K'
    :param pcd:
    :return:
    """
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
