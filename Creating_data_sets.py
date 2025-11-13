"""
Author: Ahmad Mashayekhi
Project: ML-Based CAD–PCD Alignment for Industrial Manipulators

Description
-----------
This script generates training datasets for PointNetLK/ICP by converting the
synthetic robot poses (stored in Mesh_poses_*.json) into paired point clouds:

    • Template point cloud (robot links + under-base + hex plane + optional tools)
    • Source point cloud (world-frame base, under-base, hex plane)
    • Ground-truth 4×4 transformation (source → template)

It supports:
    • Optional tools on the hex plane
    • Hex-plane partialization (vertical cutting plane)
    • Noise injection
    • World-frame and centered-frame visualization in Open3D

Output
------
An HDF5 dataset saved in ./Input_Output/ containing:
    template_final : (N, P, 3)
    source_final   : (N, S, 3)
    gt_final       : (N, 4, 4)

Run
---
python create_training_dataset.py
"""


"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""
#General imports
import os
import open3d as o3d
# import torch
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import time
# from pathlib import Path

import h5_writer_new as w
# import h5_files.file_reader as r
import json
from scipy.spatial.transform import Rotation as R


current_dir = os.path.dirname(os.path.abspath(__file__))


"""
=============================================================================
--------------------------------CAD FILES DIR--------------------------------
=============================================================================
"""
#  baked STLs and JSON of random poses ===
BAKED_STL_DIR = os.path.join(current_dir, "Input_Output", "baked_stl")
POSES_JSON    = os.path.join(current_dir, "Input_Output", "Mesh_poses_H2017_400IT_angle_30_radius_100.json")

BASE_MESH_NAME = "H2017_0_0"   # template object

"""
=============================================================================
-------------------------------FILE PARAMETERS-------------------------------
=============================================================================
"""
# ---- Tool size controls (bounding-sphere radius) ----
TOOL_RADIUS_MIN = 0.05
TOOL_RADIUS_MAX = 0.20   # <- raise this to allow bigger tools

# --- Targeted partialization for hex plane + tools ---
PARTIAL_HEX_TOOLS = True          # master on/off
HEX_TOOLS_DROP_RATIO = 0.3       # e.g., 0.30 => drop 30% (keep 70%)


# Points per cloud (keep equal for PointNet/LK)
Nmb_points_template   = 1024*3
Nmb_points_all_robot  = Nmb_points_template   # source (3 meshes) → same N

ADD_TOOLS = True
MAX_TOOLS = 3  # keep = 3 to match your randint(0,4)


# ---- Template extras (match source placement) ----
HEX_SIDE_FOR_TEMPLATE = 0.6     # must match source hex plane side
PLANE_Z_OFFSET        = -0.34   # place the plane under the under_base


# --- preview controls ---
VIS_IN_OPEN3D = True
VIS_N_SAMPLES = 1

# NEW: world-frame visualization flags
SHOW_WORLD_AXES = True          # <— set True to preview world axes
SHOW_ONLY_FIRST_WORLD = True    # show world preview only for the first sample
AXIS_SIZE_WORLD = 0.15          # size of axis triads in world viewer
AXIS_SIZE_CENTERED = 0.1        # size of axis triad in centered viewer


mu_noise = 0                                                      #   Mean value for noise
sigma_noise_max = 0.01                                            #   Std deviation value for noise
sigma_noise_bound = 0.05                                          #   Values outside [-0.05,0.05] clipped


# --- Measured surface areas (m² or consistent units) ---
SURFACE_AREAS = {
    "hex_plane": 0.897006,
    "under_base": 0.310816,
    "H2017_0_0": 0.160875,  # Base (0_0)
    "1_0": 0.1731,
    "2_0": 0.1942,
    "2_1": 0.245681,
    "2_2": 0.112061,
    "3_0": 0.104803,
    "4_0": 0.190454,
    "4_1": 0.08242,
    "5_0": 0.076626,
    "6_0": 0.018783,
}


Keep =1                                                        #% of points to keep after partial

Noise = True
Partial = False


"""
=============================================================================
----------------------------------FUNCTIONS----------------------------------
=============================================================================
"""
# ---------- Areas for known robot meshes, plus dynamic tools ----------
def make_axis_frame(T=np.eye(4, dtype=float), size=0.1):
    """
    Create an Open3D coordinate frame (X=red, Y=green, Z=blue) and
    place it with pose T (4x4). Returns a TriangleMesh you can draw.
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(size))
    # Open3D's transform mutates in-place and returns the same object.
    frame.transform(T.copy())
    return frame

_AREA_CACHE = {}  # for any robot mesh not in SURFACE_AREAS
def _T_translate_vec(v: np.ndarray) -> np.ndarray:
    """Return 4x4 homogeneous translation by vector v (shape (3,))."""
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(v, dtype=float)
    return T


def _surface_area_for_mesh_name(mesh_name: str) -> float:
    """Return surface area for a baked STL by mesh name (cached)."""
    if mesh_name in SURFACE_AREAS:
        return float(SURFACE_AREAS[mesh_name])
    if mesh_name in _AREA_CACHE:
        return float(_AREA_CACHE[mesh_name])
    stl = _resolve_baked_stl(mesh_name)
    m = o3d.io.read_triangle_mesh(stl)
    a = float(m.get_surface_area())
    _AREA_CACHE[mesh_name] = a
    return a

def _mesh_bounding_radius(mesh: o3d.geometry.TriangleMesh) -> float:
    V = np.asarray(mesh.vertices)
    if V.size == 0:
        return 1.0
    return float(np.linalg.norm(V, axis=1).max())

def _scale_mesh_to_bounding_radius(mesh: o3d.geometry.TriangleMesh, target_r: float):
    r = _mesh_bounding_radius(mesh)
    s = 1.0 if r <= 0 else (target_r / r)
    mesh.scale(s, center=(0.0, 0.0, 0.0))

def _yaw_rotate_mesh(mesh: o3d.geometry.TriangleMesh):
    ang = np.random.uniform(0.0, 2*np.pi)
    c, s = np.cos(ang), np.sin(ang)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    mesh.rotate(Rz, center=(0.0, 0.0, 0.0))

def _create_pyramid_mesh():
    verts = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],  # base
        [ 0,  0,  1],  # apex
    ], dtype=np.float64)
    tris = np.array([[0,1,2],[0,2,3],[0,1,4],[1,2,4],[2,3,4],[3,0,4]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(tris)
    )
    mesh.compute_triangle_normals()
    return mesh

def _create_semi_sphere_mesh(radius=1.0, resolution=32):
    """Dome (no base disk): crop sphere triangles with all vertices z >= 0."""
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    V = np.asarray(sph.vertices)
    T = np.asarray(sph.triangles)
    mask = np.all(V[T][:,:,2] >= 0.0, axis=1)
    T_keep = T[mask]
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V),
        triangles=o3d.utility.Vector3iVector(T_keep)
    )
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    return mesh

def _make_random_tool_mesh_with_area(base_radius_min: float, base_radius_max: float):
    """Return (mesh, area). Mesh is scaled to random bounding-sphere radius and yaw-rotated."""
    shape = np.random.choice(["sphere", "semi_sphere", "cube", "cuboid", "cone", "cylinder", "pyramid"])
    if shape == "sphere":
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=24)
    elif shape == "semi_sphere":
        mesh = _create_semi_sphere_mesh(radius=1.0, resolution=24)
    elif shape == "cube":
        mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
        mesh.translate([-1.0, -1.0, -1.0])
    elif shape == "cuboid":
        sx, sy, sz = _rand(1.2, 2.0), _rand(0.8, 1.6), _rand(0.6, 1.8)
        mesh = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
        mesh.translate([-sx/2, -sy/2, -sz/2])
    elif shape == "cone":
        mesh = o3d.geometry.TriangleMesh.create_cone(radius=1.0, height=2.0, resolution=28)
        mesh.translate([0.0, 0.0, -1.0])  # center around origin
    elif shape == "cylinder":
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=2.0, resolution=28)
        mesh.translate([0.0, 0.0, -1.0])
    elif shape == "pyramid":
        mesh = _create_pyramid_mesh()
    else:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=24)

    target_r = _rand(base_radius_min, base_radius_max)
    _scale_mesh_to_bounding_radius(mesh, target_r)
    _yaw_rotate_mesh(mesh)
    area = float(mesh.get_surface_area())
    return mesh, area

def _random_vertical_normal() -> np.ndarray:
    """Unit normal for a vertical cutting plane (normal lies in XY)."""
    ang = np.random.uniform(0.0, 2*np.pi)
    return np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float32)

def _cut_hex_tools_plane(hex_pts: np.ndarray, tool_pts_list: list, keep_ratio: float):
    """
    Shared vertical-plane cut over combined hex+tools.
    Keeps approximately keep_ratio of points (by projection quantile).
    Returns: hex_kept, tools_kept_list
    """
    if keep_ratio >= 1.0 or (hex_pts.size == 0 and not tool_pts_list):
        return hex_pts.astype(np.float32), [tp.astype(np.float32) for tp in tool_pts_list]

    n = _random_vertical_normal()
    s_hex   = hex_pts @ n
    s_tools = [tp @ n for tp in tool_pts_list]
    s_all   = s_hex if not s_tools else np.concatenate([s_hex] + s_tools, axis=0)

    tau = float(np.quantile(s_all, keep_ratio))  # keep points with n·p <= tau

    hex_kept = hex_pts[s_hex <= tau]
    tools_kept = [tp[s <= tau] for tp, s in zip(tool_pts_list, s_tools)]
    return hex_kept.astype(np.float32), [t.astype(np.float32) for t in tools_kept]

def _rand(a: float, b: float) -> float:
    return float(np.random.uniform(a, b))

def _sample_hex_xy_one(side_len: float) -> np.ndarray:
    """Sample a single (x,y,0) point uniformly inside the hex."""
    p = _sample_hex_xy(side_len, 1)[0]  # (3,) with z=0
    return p[:2]  # (x,y)

def _uniform_points_from_mesh(mesh: o3d.geometry.TriangleMesh, n: int) -> np.ndarray:
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=n)
    return np.asarray(pcd.points, dtype=np.float32)

def _scale_points_to_radius(pts: np.ndarray, target_radius: float) -> np.ndarray:
    """Scale point set so it fits inside a sphere of radius = target_radius."""
    r = np.linalg.norm(pts, axis=1).max()
    if r <= 0:
        return pts
    s = target_radius / r
    return (pts * s).astype(np.float32)

def _random_yaw_rot(points: np.ndarray) -> np.ndarray:
    """Rotate around Z by random yaw."""
    yaw = _rand(0.0, 2*np.pi)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    return (points @ Rz.T).astype(np.float32)

def _make_random_tool_points(base_radius_min: float, base_radius_max: float, n_points: int) -> np.ndarray:
    """
    Create one random primitive (sphere, semi-sphere, cube, cuboid, cone, cylinder, pyramid)
    centered near origin, scaled so its bounding sphere radius is in [min,max].
    Returns (n_points,3) in its local coordinates.
    """
    # Pick shape
    shape = np.random.choice(
        ["sphere", "semi_sphere", "cube", "cuboid", "cone", "cylinder", "pyramid"]
    )
    # Build a unit-ish mesh around origin
    if shape == "sphere" or shape == "semi_sphere":
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    elif shape == "cube":
        mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)  # roughly radius≈√3
        mesh.translate([-1.0, -1.0, -1.0])
    elif shape == "cuboid":
        # random box proportions
        sx, sy, sz = _rand(1.2, 2.0), _rand(0.8, 1.6), _rand(0.6, 1.8)
        mesh = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
        mesh.translate([-sx/2, -sy/2, -sz/2])
    elif shape == "cone":
        mesh = o3d.geometry.TriangleMesh.create_cone(radius=1.0, height=2.0, resolution=24)
        # Open3D cone base at z=0, tip at +h. Recenter roughly around origin:
        mesh.translate([0.0, 0.0, -1.0])
    elif shape == "cylinder":
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=2.0, resolution=24)
        mesh.translate([0.0, 0.0, -1.0])
    elif shape == "pyramid":
        # square pyramid: base 2x2 at z=-1, apex at z=+1
        # base = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=0.0)  # degenerate; we’ll build manually
        # Build pyramid manually with triangles
        verts = np.array([
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],  # base
            [ 0,  0,  1]  # apex
        ], dtype=np.float64)
        triangles = np.array([
            [0,1,2],[0,2,3],        # base (2 triangles)
            [0,1,4],[1,2,4],[2,3,4],[3,0,4]  # sides
        ], dtype=np.int32)
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(verts),
            triangles=o3d.utility.Vector3iVector(triangles)
        )
        mesh.compute_triangle_normals()
    else:
        # Fallback sphere
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)

    pts = _uniform_points_from_mesh(mesh, n_points)

    # Semi-sphere: keep upper half z>=0 to mimic a dome
    # replace the "semi_sphere" block in _make_random_tool_points with:
    if shape == "semi_sphere":
        pts = pts[pts[:, 2] >= 0.0]
        if pts.shape[0] == 0:
            pts = _uniform_points_from_mesh(mesh, n_points)
            pts = pts[pts[:, 2] >= 0.0]
        idx = np.random.choice(pts.shape[0], size=n_points, replace=(pts.shape[0] < n_points))
        pts = pts[idx]


    # Scale to target bounding-sphere radius
    target_r = _rand(base_radius_min, base_radius_max)
    pts = _scale_points_to_radius(pts, target_r)

    # Random yaw about Z for variety
    pts = _random_yaw_rot(pts)

    return pts.astype(np.float32)

def _place_on_plane_and_translate(pts_local: np.ndarray, xy_on_plane: np.ndarray, plane_z: float) -> np.ndarray:
    """
    Given local points, slide them so their lowest point sits exactly on plane_z,
    then translate to (x,y) on the hex.
    """
    min_z = float(pts_local[:,2].min())
    dz = plane_z - min_z
    pts = pts_local.copy()
    pts[:,2] += dz
    pts[:,0] += xy_on_plane[0]
    pts[:,1] += xy_on_plane[1]
    return pts.astype(np.float32)

def _add_extra_noise_to_tools(tool_pts: np.ndarray, mu: float, sigma_base: float, bound: float) -> np.ndarray:
    """
    Add stronger noise (×2 or ×3 sigma) to the tool points only.
    """
    factor = float(np.random.choice([2.0, 3.0]))
    sigma = sigma_base * factor
    noisy = w.add_Gaussian_Noise(mu, sigma, tool_pts[np.newaxis, :, :], bound)[0]
    return noisy.astype(np.float32)

def invert_SE3(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    t  = T[:3,  3]
    Rt = Rm.T
    inv = np.eye(4, dtype=T.dtype)
    inv[:3, :3] = Rt
    inv[:3,  3] = -Rt @ t
    return inv

def _sample_triangle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int) -> np.ndarray:
    """Uniformly sample n points inside triangle (p0,p1,p2) using sqrt trick."""
    u = np.random.rand(n, 2)
    s = np.sqrt(u[:, 0])
    t = u[:, 1]
    pts = (1 - s)[:, None] * p0 + (s * (1 - t))[:, None] * p1 + (s * t)[:, None] * p2
    return pts.astype(np.float32)

def _sample_hex_xy(side_len: float, n: int) -> np.ndarray:
    """
    Uniformly sample n points inside a regular hexagon in the XY plane (z=0), centered at origin.
    For a regular hexagon, circumradius R == side length a.
    """
    a = float(side_len)
    # 6 vertices on unit circle scaled by a (angles 0,60,...,300 deg), CCW
    verts = []
    for k in range(6):
        theta = np.deg2rad(60.0 * k)
        verts.append(np.array([a * np.cos(theta), a * np.sin(theta), 0.0], dtype=np.float32))
    verts = np.stack(verts, axis=0)  # (6,3)

    # Decompose hexagon into 6 triangles sharing the center (0,0,0): (C, v_k, v_{k+1})
    C = np.zeros(3, dtype=np.float32)
    pts_list = []
    # distribute n as evenly as possible across 6 triangles
    base = n // 6
    rem  = n % 6
    for k in range(6):
        m = base + (1 if k < rem else 0)
        if m <= 0: 
            continue
        p1 = verts[k]
        p2 = verts[(k + 1) % 6]
        tri_pts = _sample_triangle(C, p1, p2, m)
        pts_list.append(tri_pts)
    if not pts_list:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(pts_list, axis=0).astype(np.float32)

def _T_translate(dx: float, dy: float, dz: float) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, 3] = [dx, dy, dz]
    return T

def _pose_to_T(pos_xyz, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    T[:3,  3] = np.asarray(pos_xyz, dtype=float)
    return T

def _load_mesh_points(stl_path: str, n_points: int) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if mesh is None or len(mesh.triangles) == 0:
        raise RuntimeError(f"Failed to load STL or empty: {stl_path}")
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pts = np.asarray(pcd.points, dtype=float)
    if pts.shape[0] > n_points:
        idx = np.random.choice(pts.shape[0], size=n_points, replace=False)
        pts = pts[idx]
    return pts

def _resolve_baked_stl(mesh_name: str) -> str:
    p = os.path.join(BAKED_STL_DIR, mesh_name + ".stl")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"STL not found: {p}")
    return p

def apply_T_to_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4×4 homogeneous transform to an N×3 array of points."""
    R, t = T[:3, :3], T[:3, 3]
    return (pts @ R.T) + t



"""
=============================================================================
----------------------------------EXECUTION----------------------------------
=============================================================================
"""
def main():
    # -------------------- Name / tags --------------------
    json_base = os.path.basename(POSES_JSON)
    stem = json_base.replace("Mesh_poses_", "").replace(".json", "")
    name_train = f"ply_data_train_{stem}"

    add = ""
    if not Noise and not Partial:
        add += "_normal"
    if Noise:
        add += f"_noisy_{sigma_noise_max}"
    if Partial:
        add += f"_partial_{Keep}"
    add += "_template12_source3"
    if PARTIAL_HEX_TOOLS:
        add += f"_planeCut{int(HEX_TOOLS_DROP_RATIO*100)}"
    if ADD_TOOLS:
        add += f"_tools{MAX_TOOLS}"
    name_train += add

    # -------------------- Load JSON --------------------
    with open(POSES_JSON, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    elif isinstance(data, list):
        samples = data
    else:
        raise RuntimeError("JSON format not recognized (expected dict with 'samples' or list).")

    total_template_list = []
    total_source_list   = []
    total_gt_list       = []

    # -------------------- Build pairs --------------------
    for idx, s in enumerate(tqdm(samples, desc="Building pairs from JSON")):
        # --- Resolve base pose (template->world) ---
        T_base_world = None
        bp = s.get("base_pose")
        if isinstance(bp, dict):
            pos  = bp.get("position", [0, 0, 0])
            quat = bp.get("quaternion_xyzw", [0, 0, 0, 1])
            T_base_world = _pose_to_T(pos, quat)

        if T_base_world is None:
            if "base_gt_T_4x4" in s and s["base_gt_T_4x4"] is not None:
                T_base_world = np.asarray(s["base_gt_T_4x4"], dtype=float).reshape(4, 4)
            else:
                base_entry = None
                for e in s.get("meshes", []):
                    if e.get("mesh_name") == BASE_MESH_NAME or e.get("link_name") == "base_0":
                        base_entry = e
                        break
                if base_entry is None:
                    raise RuntimeError("Could not find base mesh/pose for GT in sample.")
                wp   = base_entry.get("world_pose", {})
                pos  = wp.get("position", [0, 0, 0])
                quat = wp.get("quaternion_xyzw", [0, 0, 0, 1])
                T_base_world = _pose_to_T(pos, quat)
        T_base_world = T_base_world.astype(np.float32)
        T_world_to_base = invert_SE3(T_base_world).astype(np.float32)

        # --- Collect robot link entries (into template/base frame) ---
        robot_entries = []
        for e in s.get("meshes", []):
            mesh_name = e.get("mesh_name")
            if not mesh_name:
                continue
            wp = e.get("world_pose", {})
            pos  = wp.get("position", [0, 0, 0])
            quat = wp.get("quaternion_xyzw", [0, 0, 0, 1])
            T_m_world = _pose_to_T(pos, quat).astype(np.float32)
            area_i = _surface_area_for_mesh_name(mesh_name)
            robot_entries.append({
                "mesh_name": mesh_name,
                "T_w2b_m": (T_world_to_base @ T_m_world).astype(np.float32),
                "area": area_i,
            })

        # --- Areas and sampling densities ---
        area_under_base = float(SURFACE_AREAS["under_base"])
        area_hex        = float(SURFACE_AREAS["hex_plane"])

        Tool_Number = int(np.random.randint(0, MAX_TOOLS + 1)) if ADD_TOOLS else 0
        tool_meshes, tool_areas, tool_xy = [], [], []
        if Tool_Number > 0:
            for _ in range(Tool_Number):
                mesh_tool, area_tool = _make_random_tool_mesh_with_area(TOOL_RADIUS_MIN, TOOL_RADIUS_MAX)
                xy = _sample_hex_xy_one(HEX_SIDE_FOR_TEMPLATE)
                tool_meshes.append(mesh_tool)
                tool_areas.append(area_tool)
                tool_xy.append(xy)

        A_nc = area_under_base + sum(ei["area"] for ei in robot_entries)  # not cut
        A_c  = area_hex + sum(tool_areas)                                  # cut by plane
        k_keep = 1.0 - HEX_TOOLS_DROP_RATIO
        rho_final = Nmb_points_template / max(1e-12, (A_nc + A_c))

        MIN_ROBOT_PTS, MIN_UB_PTS, MIN_HEX_PTS, MIN_TOOL_PTS = 64, 64, 128, 64

        for ei in robot_entries:
            ei["n_pts"] = max(MIN_ROBOT_PTS, int(round(rho_final * ei["area"])))
        n_under   = max(MIN_UB_PTS,  int(round(rho_final * area_under_base)))
        n_hex_pre = max(MIN_HEX_PTS, int(round((rho_final / max(1e-12, k_keep)) * area_hex)))
        n_tool_pre = [max(MIN_TOOL_PTS, int(round((rho_final / max(1e-12, k_keep)) * a))) for a in tool_areas]

        # --- TEMPLATE points (in base frame) ---
        template_chunks = []

        # robot links
        for ei in robot_entries:
            stl_path = _resolve_baked_stl(ei["mesh_name"])
            pts_local = _load_mesh_points(stl_path, ei["n_pts"])
            pts_in_T  = apply_T_to_points(pts_local, ei["T_w2b_m"])
            template_chunks.append(pts_in_T.astype(np.float32))

        # under_base
        ub_stl   = _resolve_baked_stl("under_base")
        ub_local = _load_mesh_points(ub_stl, n_under)
        ub_in_T  = apply_T_to_points(ub_local, T_world_to_base @ T_base_world)
        template_chunks.append(ub_in_T.astype(np.float32))

        # hex plane (pre-cut)
        hex_local       = _sample_hex_xy(HEX_SIDE_FOR_TEMPLATE, n_hex_pre)
        T_plane_world   = T_base_world @ _T_translate(0.0, 0.0, PLANE_Z_OFFSET)
        hex_in_T        = apply_T_to_points(hex_local, T_world_to_base @ T_plane_world)

        # tools (pre-cut) on plane in template frame
        plane_z = PLANE_Z_OFFSET
        tool_pts_all = []
        for mtool, npre, xy in zip(tool_meshes, n_tool_pre, tool_xy):
            pts_tool   = _uniform_points_from_mesh(mtool, npre)
            pts_on_pln = _place_on_plane_and_translate(pts_tool, xy, plane_z)
            tool_pts_all.append(pts_on_pln.astype(np.float32))

        # shared plane cut over hex+tools (optional)
        if PARTIAL_HEX_TOOLS:
            keep_ratio = float(np.clip(k_keep, 0.0, 1.0))
            hex_in_T, tool_pts_all = _cut_hex_tools_plane(hex_in_T, tool_pts_all, keep_ratio)

        # extra noise to tools after cut (optional)
        if tool_pts_all:
            tool_pts_all = [tp for tp in tool_pts_all if tp.shape[0] > 0]
            if tool_pts_all:
                tool_pts_all = [
                    _add_extra_noise_to_tools(tp, mu=mu_noise, sigma_base=sigma_noise_max, bound=sigma_noise_bound)
                    for tp in tool_pts_all
                ]

        # merge template chunks
        template_chunks.append(hex_in_T.astype(np.float32))
        if tool_pts_all:
            template_chunks.append(np.concatenate(tool_pts_all, axis=0).astype(np.float32))
        template_pts = np.concatenate(template_chunks, axis=0).astype(np.float32)

        # enforce exact count
        if template_pts.shape[0] > Nmb_points_template:
            sel = np.random.choice(template_pts.shape[0], size=Nmb_points_template, replace=False)
            template_pts = template_pts[sel]
        elif template_pts.shape[0] < Nmb_points_template:
            pad = np.random.choice(template_pts.shape[0], size=Nmb_points_template - template_pts.shape[0], replace=True)
            template_pts = np.concatenate([template_pts, template_pts[pad]], axis=0)
        template_pts = template_pts.astype(np.float32)

        # --- SOURCE points (in world frame): base + under_base + hex plane ---
        union_pts_list = []
        per_comp_points = max(int(np.ceil(Nmb_points_all_robot / 3)), 256)

        base_stl   = _resolve_baked_stl(BASE_MESH_NAME)
        base_local = _load_mesh_points(base_stl, per_comp_points)
        base_world = apply_T_to_points(base_local, T_base_world)
        union_pts_list.append(base_world)

        ub_stl   = _resolve_baked_stl("under_base")
        ub_local = _load_mesh_points(ub_stl, per_comp_points)
        ub_world = apply_T_to_points(ub_local, T_base_world)
        union_pts_list.append(ub_world)

        hex_local = _sample_hex_xy(HEX_SIDE_FOR_TEMPLATE, per_comp_points)
        T_plane   = T_base_world @ _T_translate(0.0, 0.0, PLANE_Z_OFFSET)
        hex_world = apply_T_to_points(hex_local, T_plane)
        union_pts_list.append(hex_world)

        union_pts = np.concatenate(union_pts_list, axis=0).astype(np.float32)

        # enforce exact count
        if union_pts.shape[0] > Nmb_points_all_robot:
            sel = np.random.choice(union_pts.shape[0], size=Nmb_points_all_robot, replace=False)
            union_pts = union_pts[sel]
        elif union_pts.shape[0] < Nmb_points_all_robot:
            pad = np.random.choice(union_pts.shape[0], size=Nmb_points_all_robot - union_pts.shape[0], replace=True)
            union_pts = np.concatenate([union_pts, union_pts[pad]], axis=0)
        union_pts = union_pts.astype(np.float32)

        # optional noise / partial
        if Noise:
            union_pts = w.add_Gaussian_Noise(
                mu_noise, sigma_noise_max, union_pts[np.newaxis, :, :], sigma_noise_bound
            )[0].astype(np.float32)
        if Partial:
            n_target = int(Keep * Nmb_points_all_robot)
            if union_pts.shape[0] > n_target:
                union_pts = np.ascontiguousarray(union_pts, dtype=np.float32)
                union_pts, _ = w.farthest_subsample_points(union_pts, n_target)
                
        # -------------------- WORLD-FRAME PREVIEW (optional) --------------------
        if VIS_IN_OPEN3D and SHOW_WORLD_AXES and (not SHOW_ONLY_FIRST_WORLD or idx == 0):
            geoms_world = []

            # Point clouds in WORLD frame (already computed above)
            pc_base_w = o3d.geometry.PointCloud()
            pc_base_w.points = o3d.utility.Vector3dVector(base_world)
            pc_base_w.paint_uniform_color([0.9, 0.2, 0.2])     # reddish
            geoms_world.append(pc_base_w)

            pc_ub_w = o3d.geometry.PointCloud()
            pc_ub_w.points = o3d.utility.Vector3dVector(ub_world)
            pc_ub_w.paint_uniform_color([0.2, 0.6, 0.9])       # bluish
            geoms_world.append(pc_ub_w)

            pc_hex_w = o3d.geometry.PointCloud()
            pc_hex_w.points = o3d.utility.Vector3dVector(hex_world)
            pc_hex_w.paint_uniform_color([0.2, 0.9, 0.4])      # greenish
            geoms_world.append(pc_hex_w)

            # Axes:
            # 1) World axes at identity:
            geoms_world.append(make_axis_frame(np.eye(4, dtype=float), size=AXIS_SIZE_WORLD))
            # 2) Base-frame axes at T_base_world:
            geoms_world.append(make_axis_frame(T_base_world, size=AXIS_SIZE_WORLD*1.25))
            # 3) Hex-plane axes at T_plane_world (same R as base, z-shifted):
            geoms_world.append(make_axis_frame(T_plane_world, size=AXIS_SIZE_WORLD))
            

            o3d.visualization.draw_geometries(
                geoms_world,
                window_name="WORLD preview: clouds + axes (World=I, Base=T_base_world, Plane=T_plane_world)"
            )

        # -------------------- CENTERING (independent zero-mean) --------------------
        c_T = template_pts.mean(axis=0).astype(np.float32)   # template centroid
        c_S = union_pts.mean(axis=0).astype(np.float32)      # source centroid

        template_pts_centered = (template_pts - c_T).astype(np.float32)
        union_pts_centered    = (union_pts    - c_S).astype(np.float32)

        # GT: world->template (S->T), then to centered frame: T' = T_-cT * T * T_+cS
        T_S_to_T = invert_SE3(T_base_world).astype(np.float32)
        T_minus_cT = _T_translate_vec(-c_T)
        T_plus_cS  = _T_translate_vec(+c_S)
        T_S_to_T_centered = (T_minus_cT @ T_S_to_T @ T_plus_cS).astype(np.float32)

        # Append centered data only
        total_template_list.append(template_pts_centered.tolist())
        total_source_list.append(union_pts_centered.tolist())
        total_gt_list.append(T_S_to_T_centered.tolist())

    # -------------------- FINALIZE ARRAYS --------------------
    a, b, c = shuffle(total_template_list, total_source_list, total_gt_list, random_state=0)

    template_final = np.array(a, dtype=np.float32)   # (N, P, 3)
    source_final   = np.array(b, dtype=np.float32)   # (N, S, 3)
    gt_final       = np.array(c, dtype=np.float32)   # (N, 4, 4)

    # Sanity checks
    assert template_final.ndim == 3 and template_final.shape[2] == 3
    assert source_final.ndim   == 3 and source_final.shape[2]   == 3
    assert gt_final.ndim       == 3 and gt_final.shape[1:]      == (4, 4)
    assert template_final.shape[0] == source_final.shape[0] == gt_final.shape[0]



    # -------------------- PREVIEW (centered) --------------------
    if VIS_IN_OPEN3D and VIS_N_SAMPLES > 0:
        K = min(VIS_N_SAMPLES, template_final.shape[0])
        for i in range(K):
            templ = template_final[i]   # (P,3), centered template
            src   = source_final[i]     # (S,3), centered source (BEFORE movement)
            GT    = gt_final[i]         # S'->T' (centered GT)

            # AFTER movement (apply GT to the centered source)
            src_in_T = apply_T_to_points(src, GT)

            geoms = []

            # Red = template
            pc_blue = o3d.geometry.PointCloud()
            pc_blue.points = o3d.utility.Vector3dVector(templ)
            pc_blue.paint_uniform_color([1, 0, 0])
            geoms.append(pc_blue)

            # Green = source BEFORE movement (centered)
            pc_red = o3d.geometry.PointCloud()
            pc_red.points = o3d.utility.Vector3dVector(src)
            pc_red.paint_uniform_color([0, 0, 1])
            geoms.append(pc_red)

            # Blue = source AFTER movement to template frame
            pc_green = o3d.geometry.PointCloud()
            pc_green.points = o3d.utility.Vector3dVector(src_in_T)
            pc_green.paint_uniform_color([0, 1, 0])
            geoms.append(pc_green)

            geoms.append(make_axis_frame(np.eye(4, dtype=float), size=AXIS_SIZE_CENTERED))


            o3d.visualization.draw_geometries(
                geoms,
                window_name="Source (Blue) + Template (Red) + Source after GT (Green)"           
                )

    # -------------------- SAVE HDF5 --------------------
    w.write_h5(
        name_train,
        template_final,
        source_final,
        gt_final,
        FolderName=os.path.join(current_dir, "Input_Output")
    )



os.chdir(os.path.dirname(os.path.abspath(__file__)))

start_time = time.perf_counter()

np.random.seed(0)
 
if __name__ == '__main__':
    main()
    # Record the end time
    end_time = time.perf_counter()

    # Calculate the total time taken
    total_time = end_time - start_time
    print("=============================================================")
    print(f"Total run time: {total_time} seconds")