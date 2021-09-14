import open3d as o3d

mesh_path = "data/ModelNet10/bed/train/bed_0001.off"

mesh = o3d.io.read_triangle_mesh(mesh_path)

vertices = mesh.vertices
faces = mesh.triangles

