import open3d as o3d
import numpy as np

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

# read pcd file
pcd = o3d.io.read_point_cloud("/home/andrei/PycharmProjects/diff_mv_rec/view-dataset1/pcd/bathtub_0005_theta_-57_phi_10313_vc_9.pcd")

print(type(pcd.points))

# get points into a numpy array
points = np.asarray(pcd.points)

print(type(points))
print(points.shape)

# add random noise to the point cloud
points += np.random.uniform(-0.01, 0.01, size=points.shape)

# create a new point cloud with the new points
pcd_noisy = o3d.geometry.PointCloud()
pcd_noisy.points = o3d.utility.Vector3dVector(points)

# save the point cloud
o3d.io.write_point_cloud("/home/andrei/Desktop/airplane_0019_noise.pcd", pcd_noisy)
