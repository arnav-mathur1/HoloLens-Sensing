import os
import numpy as np
import open3d as o3d


# ============================================================
# Files
# ============================================================

BASE_DIR = r"C:\Users\GitlabAdmin\Desktop\Arnav\HoloLensForCV\extra"

DEPTH_FILE = os.path.join(
    BASE_DIR,
    "depth_frame_15.npy"
)

LUT_FILE = os.path.join(
    BASE_DIR,
    "longthrow_lut.npy"
)


# ============================================================
# Settings
# ============================================================

MIN_DEPTH_MM = 300
MAX_DEPTH_MM = 4000

# Smaller = more points/detail
VOXEL_SIZE = 0.01


# ============================================================
# Load saved HoloLens data
# ============================================================

depth = np.load(DEPTH_FILE)
lut = np.load(LUT_FILE)

print("Depth shape:", depth.shape)
print("Depth dtype:", depth.dtype)

print("LUT shape:", lut.shape)
print("LUT dtype:", lut.dtype)


# ============================================================
# Verify expected dimensions
# ============================================================

height, width = depth.shape

if lut.shape != (height, width, 3):
    raise RuntimeError(
        f"LUT shape {lut.shape} does not match "
        f"depth shape {depth.shape}. "
        f"Expected {(height, width, 3)}"
    )


# ============================================================
# Depth -> XYZ
#
# Microsoft-style:
#
#     point = depth * calibrated_unit_ray
#
# depth is millimeters.
# LUT contains a 3D direction for every depth pixel.
# ============================================================

depth_flat = depth.reshape(
    -1,
    1
).astype(np.float32)

lut_flat = lut.reshape(
    -1,
    3
).astype(np.float32)

points = (
    depth_flat *
    lut_flat
)


# ============================================================
# Remove invalid depth
# ============================================================

valid = (
    (depth_flat[:, 0] >= MIN_DEPTH_MM)
    &
    (depth_flat[:, 0] <= MAX_DEPTH_MM)
    &
    np.isfinite(points).all(axis=1)
)

points = points[valid]


# Convert millimeters -> meters
points /= 1000.0


print("Valid points:", len(points))

print(
    "XYZ min:",
    points.min(axis=0)
)

print(
    "XYZ max:",
    points.max(axis=0)
)


# ============================================================
# Create Open3D point cloud
# ============================================================

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    points.astype(np.float64)
)


# ============================================================
# Downsample slightly
#
# This cleans up the view without combining different frames.
# ============================================================

pcd = pcd.voxel_down_sample(
    voxel_size=VOXEL_SIZE
)

print(
    "Points after voxel downsampling:",
    len(pcd.points)
)


# ============================================================
# Color according to depth
#
# This makes the geometry substantially easier to understand.
# ============================================================

xyz = np.asarray(
    pcd.points
)

z = xyz[:, 2]

z_min = z.min()
z_max = z.max()

normalized_depth = (
    (z - z_min)
    /
    max(z_max - z_min, 1e-6)
)

# Simple grayscale depth coloring:
# near = bright
# far = dark
colors = np.stack(
    (
        1.0 - normalized_depth,
        1.0 - normalized_depth,
        1.0 - normalized_depth
    ),
    axis=1
)

pcd.colors = o3d.utility.Vector3dVector(
    colors
)


# ============================================================
# Visualize
# ============================================================

vis = o3d.visualization.Visualizer()

vis.create_window(
    window_name="HoloLens Frame 15 - Single Point Cloud",
    width=1200,
    height=900
)

vis.add_geometry(pcd)

render = vis.get_render_option()

render.point_size = 3.0

# Dark background makes grayscale geometry easier to see
render.background_color = np.array([
    0.05,
    0.05,
    0.05
])

vis.run()

vis.destroy_window()