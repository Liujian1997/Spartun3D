import open3d as o3d
from open3d.visualization import rendering

w, h = 800, 600
renderer = rendering.OffscreenRenderer(w, h)

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
mesh.compute_vertex_normals()

mat = rendering.MaterialRecord()
mat.shader = "defaultLit"

renderer.scene.add_geometry("axis", mesh, mat)
img = renderer.render_to_image()
o3d.io.write_image("frame.png", img)
print("Saved frame.png")
points, colors, instance_labels = torch.load(
    "/data2/liujian/leo_data/scan_data/3RScan-base/3RScan-ours-align/bf9a3dc3-45a5-2e80-832d-842aa34cc859/pcd-align.pth",
    weights_only=False
)
