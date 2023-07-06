import open3d as o3d
import numpy as np

# read pcd file
pcd = o3d.io.read_point_cloud("../data/view-dataset1/pcd/bathtub_0005_theta_-57_phi_10313_vc_9.pcd")

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
