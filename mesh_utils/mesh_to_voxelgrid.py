import numpy as np
import open3d as o3d

n_voxels = 50
out_name = "chair.npy"

VOXEL_SIZE = float(1 / n_voxels)

mesh = o3d.io.read_triangle_mesh("/home/andrei/datasets/ModelNet10/chair/train/chair_0001.off")
mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
           center=mesh.get_center())
center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
mesh = mesh.translate((-center[0], -center[1], -center[2]))

# (1/voxel_size)^3 will be the size of the input of the network, 0.02 results in 50^3=125000
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(input=mesh, voxel_size=VOXEL_SIZE,
                                                                            min_bound=np.array(
                                                                                [-0.5, -0.5, -0.5]),
                                                                            max_bound=np.array([0.5, 0.5, 0.5]))
voxels = voxel_grid.get_voxels()
grid_size = n_voxels
mask = np.zeros((grid_size, grid_size, grid_size))
for vox in voxels:
    mask[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1
np.save(out_name, mask, allow_pickle=False, fix_imports=False)