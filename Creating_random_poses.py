"""
Author: Ahmad Mashayekhi
Project: ML-Based CAD–PCD Alignment for Industrial Manipulators

Description
-----------
Generates synthetic ground-truth poses for the Doosan H2017 robot using PyBullet.
For each randomly sampled configuration, the script:

  • Samples a random base translation and rotation
  • Samples random joint angles within reduced limits
  • Computes the world-frame pose of every visual mesh in the URDF
  • Stores the base 4×4 transform, joint angles, and mesh poses

The script rewrites the URDF so all <mesh> paths point to the local mesh folder
("Input_Output/Doosan H2017 Meshes") and optionally visualizes samples in PyBullet.

Output
------
A JSON file written to ./Input_Output/ containing:
  • Base transform
  • Joint angles (deg)
  • World poses of all meshes
  • Sampling ranges and metadata

Usage
-----
python create_random_robot_poses.py

Requirements
------------
numpy, scipy, pybullet, tqdm
"""

import os, json, numpy as np
import pybullet as p
import pybullet_data
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# -------------------- CONFIG --------------------
VIS_IN_PYBULLET   = True
VIS_N_SAMPLES     = 10

BASE_Z_FIX        = -0.0936
NUM_SAMPLES       = 20
GT_MAX_ANGLE_DEG  = 30.0     # nominal (before reduction)
GT_MAX_RADIUS     = 1      # nominal (before reduction)
Robot_name        = "H2017"

# ------------------------------------------------

def random_translation_in_sphere(max_radius=0.5):
    v = np.random.normal(size=3)
    v /= (np.linalg.norm(v) + 1e-12)
    r = np.random.uniform(0.0, 1.0) ** (1.0/3.0) * max_radius
    return v * r

def random_axis_angle_quat_xyzw(max_deg=30.0):
    axis = np.random.normal(size=3)
    axis /= (np.linalg.norm(axis) + 1e-12)
    ang_deg = np.random.uniform(0.0, max_deg)
    ang_rad = np.deg2rad(ang_deg)
    half = 0.5 * ang_rad
    s = np.sin(half)
    q = np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(half)], dtype=float)  # xyzw
    q /= (np.linalg.norm(q) + 1e-12)
    return q, ang_deg

def quat_trans_to_T(q_xyzw, t_xyz):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    T[:3,  3] = np.asarray(t_xyz, dtype=float)
    return T

def rpy_to_matrix(rpy):
    return R.from_euler('xyz', rpy).as_matrix()

def transform_pose(base_pos, base_quat, local_pos, local_rpy):
    base_rot  = R.from_quat(base_quat).as_matrix()
    local_rot = rpy_to_matrix(local_rpy)
    world_rot = base_rot @ local_rot
    world_pos = base_rot @ local_pos + base_pos
    world_quat = R.from_matrix(world_rot).as_quat()
    return world_pos, world_quat

def parse_urdf_visual_origins(urdf_path):
    mesh_info = {}
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        link_name = link.attrib['name']
        visuals = link.findall("visual")
        mesh_info[link_name] = []
        for visual in visuals:
            origin_elem = visual.find("origin")
            xyz = tuple(float(v) for v in origin_elem.attrib.get("xyz", "0 0 0").split())
            rpy = tuple(float(v) for v in origin_elem.attrib.get("rpy", "0 0 0").split())
            mesh_basename, mesh_fullpath = None, None
            scale = (1.0, 1.0, 1.0)
            geom = visual.find("geometry")
            if geom is not None:
                mesh_elem = geom.find("mesh")
                if mesh_elem is not None:
                    fn = mesh_elem.attrib.get("filename", "")
                    mesh_basename = os.path.splitext(os.path.basename(fn))[0] if fn else None
                    mesh_fullpath = fn if fn else None
                    if "scale" in mesh_elem.attrib:
                        s = [float(v) for v in mesh_elem.attrib["scale"].split()]
                        if len(s) == 3:
                            scale = tuple(s)
            mesh_info[link_name].append({
                "mesh_name": mesh_basename,
                "mesh_filename": mesh_fullpath,
                "local_xyz": xyz,
                "local_rpy": rpy,
                "scale": scale,
            })
    return mesh_info

def draw_frame(pos, quat, length=0.05, width=5):
    rot = R.from_quat(quat).as_matrix()
    x_axis = pos + rot[:, 0] * length
    y_axis = pos + rot[:, 1] * length
    z_axis = pos + rot[:, 2] * length
    ids = []
    ids.append(p.addUserDebugLine(pos, x_axis, [1, 0, 0], lineWidth=width))
    ids.append(p.addUserDebugLine(pos, y_axis, [0, 1, 0], lineWidth=width))
    ids.append(p.addUserDebugLine(pos, z_axis, [0, 0, 1], lineWidth=width))
    return ids

# --- minimal helper to point meshes to the local mesh folder ---
def rewrite_urdf_mesh_paths(urdf_in: str, mesh_dir: str, urdf_out: str) -> None:
    """Point all <mesh filename="..."> to mesh_dir/<basename> (no folder creation)."""
    tree = ET.parse(urdf_in)
    root = tree.getroot()
    changed = 0
    for mesh in root.findall(".//mesh"):
        fn = mesh.attrib.get("filename", "")
        if not fn:
            continue
        base = os.path.basename(fn)
        mesh.set("filename", os.path.join(mesh_dir, base))
        changed += 1
    if changed == 0:
        raise RuntimeError(f"No <mesh> tags rewritten in: {urdf_in}")
    tree.write(urdf_out)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # --- use meshes folder under Input_Output (next to this script); do NOT create it
    mesh_dir = os.path.join(current_dir, "Input_Output", "Doosan H2017 Meshes")
    if not os.path.isdir(mesh_dir):
        raise FileNotFoundError(f"Mesh folder not found: {mesh_dir}")

    # --- URDF must be exactly this file inside the meshes folder
    urdf_path_in = os.path.join(mesh_dir, "URDF_H2017.urdf")
    if not os.path.isfile(urdf_path_in):
        raise FileNotFoundError(f"URDF not found: {urdf_path_in}")

    # --- make a temporary URDF that points meshes to the meshes folder
    urdf_path_tmp = os.path.join(current_dir, "_URDF_H2017_rewritten.urdf")
    rewrite_urdf_mesh_paths(urdf_path_in, mesh_dir, urdf_path_tmp)

    # Use the rewritten URDF from here on
    urdf_path   = urdf_path_tmp
    mesh_info_template = parse_urdf_visual_origins(urdf_path)

    # PyBullet connection (GUI when previewing)
    p.connect(p.GUI if VIS_IN_PYBULLET else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # Load robot once
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)

    # Per-joint nominal limits (deg)
    limits_deg_nominal = {
        "joint1": (-180.0,   180.0),
        "joint2": (-125.002, 125.002),
        "joint3": (-159.998, 159.998),
        "joint4": (-180.0,   180.0),
        "joint5": (-180.0,   180.0),
        "joint6": (-180.0,   180.0),
    }
    # === NEW: reduced joint limits (scaled toward 0) ===
    limits_deg = {
        j: (lo , hi )
        for j, (lo, hi) in limits_deg_nominal.items()
    }

    # ---------- JSON top-level container ----------
    out_doc = {
        "robot_name": Robot_name,
        "mesh_pathes": mesh_dir,  # kept field; now uses local variable
        "effective_ranges": {
            "gt_max_angle_deg": GT_MAX_ANGLE_DEG,
            "gt_max_radius_m":  GT_MAX_RADIUS,
            "joint_limits_deg": limits_deg,
        },
        "samples": []
    }

    for sample_idx in tqdm(range(1, NUM_SAMPLES + 1), desc="Generating random poses", ncols=100):

        # --- Base GT with reduced ranges
        t_gt = random_translation_in_sphere(GT_MAX_RADIUS)
        q_gt, _ = random_axis_angle_quat_xyzw(GT_MAX_ANGLE_DEG)
        T_gt = quat_trans_to_T(q_gt, t_gt)
        if BASE_Z_FIX != 0.0:
            T_gt[:3, 3] += np.array([0.0, 0.0, BASE_Z_FIX])

        # Absolute reset (no accumulation)
        new_pos  = tuple(t_gt)
        new_quat = q_gt
        p.resetBasePositionAndOrientation(robot_id, new_pos, new_quat)

        # --- Random joints within REDUCED limits
        low_deg, high_deg = [], []
        for j_idx in range(6):
            jname = p.getJointInfo(robot_id, j_idx)[1].decode()
            lo, hi = limits_deg[jname]
            low_deg.append(lo); high_deg.append(hi)
        low_deg  = np.array(low_deg,  dtype=float)
        high_deg = np.array(high_deg, dtype=float)

        joint_angles_deg = np.random.uniform(low_deg, high_deg)
        joint_angles_rad = np.deg2rad(joint_angles_deg)
        for j_idx, angle_rad in enumerate(joint_angles_rad):
            p.resetJointState(robot_id, j_idx, float(angle_rad))

        if VIS_IN_PYBULLET and sample_idx <= VIS_N_SAMPLES:
            p.disconnect()
            cid = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf")
            robot_vis = p.loadURDF(urdf_path, useFixedBase=True)
            p.resetBasePositionAndOrientation(robot_vis, new_pos, new_quat)
            for j_idx, angle_rad in enumerate(joint_angles_rad):
                p.resetJointState(robot_vis, j_idx, float(angle_rad))
            p.resetDebugVisualizerCamera(
                cameraDistance=1.8, cameraYaw=45, cameraPitch=-35, cameraTargetPosition=[0, 0, 0]
            )
            draw_frame(np.array(new_pos), new_quat, length=0.1, width=10)
            print(f"[Preview] Sample {sample_idx}/{VIS_N_SAMPLES} — close the window to continue...")
            while p.isConnected(cid):
                p.stepSimulation()
            p.connect(p.DIRECT)
            robot_id = p.loadURDF(urdf_path, useFixedBase=True)
            p.resetBasePositionAndOrientation(robot_id, new_pos, new_quat)
            for j_idx, angle_rad in enumerate(joint_angles_rad):
                p.resetJointState(robot_id, j_idx, float(angle_rad))

        p.stepSimulation()

        # --- Record sample
        sample_data = {
            "sample_idx": sample_idx,
            "base_gt_T_4x4": T_gt.tolist(),
            "base_pose": {},
            "joint_angles_deg": {},
            "meshes": []
        }

        base_pos_now, base_quat_now = p.getBasePositionAndOrientation(robot_id)
        if BASE_Z_FIX != 0.0:
            base_pos_now = (np.array(base_pos_now) + np.array([0.0, 0.0, BASE_Z_FIX])).tolist()
        sample_data["base_pose"] = {
            "position": [float(x) for x in base_pos_now],
            "quaternion_xyzw": [float(x) for x in base_quat_now]
        }

        for j in range(p.getNumJoints(robot_id)):
            js = p.getJointState(robot_id, j)
            sample_data["joint_angles_deg"][str(j)] = float(np.rad2deg(js[0]))

        for link_idx in range(-1, p.getNumJoints(robot_id)):
            link_name = "base_0" if link_idx == -1 else p.getJointInfo(robot_id, link_idx)[12].decode()
            if link_idx == -1:
                lpos, lquat = p.getBasePositionAndOrientation(robot_id)
            else:
                ls = p.getLinkState(robot_id, link_idx, computeForwardKinematics=True)
                lpos, lquat = ls[4], ls[5]

            if link_name in mesh_info_template:
                for i, item in enumerate(mesh_info_template[link_name]):
                    local_pos = np.array(item["local_xyz"], dtype=float)
                    local_rpy = np.array(item["local_rpy"], dtype=float)
                    world_pos, world_quat = transform_pose(np.array(lpos), lquat, local_pos, local_rpy)
                    if BASE_Z_FIX != 0.0:
                        world_pos = world_pos + np.array([0.0, 0.0, BASE_Z_FIX])
                    mesh_key = item["mesh_name"] or f"{link_name}_mesh{i}"
                    sample_data["meshes"].append({
                        "link_name": link_name,
                        "mesh_name": mesh_key,
                        "world_pose": {
                            "position": [float(x) for x in world_pos],
                            "quaternion_xyzw": [float(x) for x in world_quat]
                        },
                        "local_origin_in_link": {
                            "xyz": [float(x) for x in local_pos],
                            "rpy": [float(x) for x in local_rpy]
                        },
                        "scale_xyz": list(item.get("scale", (1.0, 1.0, 1.0)))
                    })

        out_doc["samples"].append(sample_data)

    p.disconnect()

    # === NEW: include reduced values + factor in filename ===
    OUT_JSON_FILE = (
        f"Mesh_poses_{Robot_name}_{NUM_SAMPLES}IT_"
        f"angle_{int(round(GT_MAX_ANGLE_DEG))}_"
        f"radius_{int(round(GT_MAX_RADIUS*100))}.json"
    )

    out_path = os.path.join(current_dir, "Input_Output", OUT_JSON_FILE)
    with open(out_path, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"\n[✓] Wrote {len(out_doc['samples'])} samples to: {out_path}")
