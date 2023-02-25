import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

import pymeshfix
import pyvista as pv

pcd = o3d.io.read_point_cloud("/home/andrei/datasets/washington_short_version/Category/banana_Category/banana_object_10.pcd")


# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

pcd.estimate_normals()

pcd.orient_normals_consistent_tangent_plane(k=10)

# We need a mesh, so we do Poisson surface reconstruction
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Info) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd, depth=4)
print(mesh)
# print_attributes(mesh)

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])


# Visualize densities
print('visualize densities')
densities = np.asarray(densities)
density_colors = plt.get_cmap('plasma')(
    (densities - densities.min()) / (densities.max() - densities.min()))
density_colors = density_colors[:, :3]
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = mesh.vertices
density_mesh.triangles = mesh.triangles
density_mesh.triangle_normals = mesh.triangle_normals
density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
o3d.visualization.draw_geometries([density_mesh])

# # Remove points with low density
# print('remove low density vertices')
# vertices_to_remove = densities < np.quantile(densities, 0.7)
# mesh.remove_vertices_by_mask(vertices_to_remove)
# print(mesh)
# o3d.visualization.draw_geometries([mesh])


# pv.set_plot_theme('document')
#
# array = np.asarray(pcd.points)
#
# point_cloud = pv.PolyData(array)
# surf = point_cloud.reconstruct_surface(nbr_sz=20, sample_spacing=0.1)
#
# mf = pymeshfix.MeshFix(surf)
# mf.repair()
# repaired = mf.mesh
#
# pl = pv.Plotter()
# pl.add_mesh(point_cloud, color='k', point_size=10)
# pl.add_mesh(repaired)
# pl.add_title('Reconstructed Surface')
# pl.show()