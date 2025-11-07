import os, numpy as np, open3d as o3d

SRC = "/home/ahmadmashayekhi/datasets/CAD/Doosan.ply"
DST = "/home/ahmadmashayekhi/datasets/CAD/Doosan.stl"   # must match JSON "name"
POISSON_DEPTH = 8
DOWNSAMPLE_IF_OVER = 1_000_000
VOXEL_SIZE = 0.001

def main():
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    print(f"[INFO] Source: {SRC}")

    # Try as mesh first (PLY can be mesh or point cloud)
    mesh = o3d.io.read_triangle_mesh(SRC, enable_post_processing=True)
    is_mesh = (mesh is not None and not mesh.is_empty()
               and len(mesh.vertices) > 0 and len(mesh.triangles) > 0)

    if is_mesh:
        print("[INFO] Detected: MESH PLY → exporting to STL")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        # >>> compute normals before writing
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        ok = o3d.io.write_triangle_mesh(DST, mesh)
        if not ok: raise SystemExit("[ERROR] Failed to write STL.")
        print(f"[OK] Wrote {DST} | V={len(mesh.vertices):,}, F={len(mesh.triangles):,}")
        return

    print("[INFO] Detected: POINT-CLOUD PLY → Poisson meshing")
    pcd = o3d.io.read_point_cloud(SRC)
    if pcd.is_empty():
        raise SystemExit(f"[ERROR] Point cloud is empty: {SRC}")
    print(f"[INFO] Points: {len(pcd.points):,}")

    # Clean non-finite
    pts = np.asarray(pcd.points)
    mask = np.isfinite(pts).all(axis=1)
    if not mask.all():
        pcd = pcd.select_by_index(np.flatnonzero(mask))
        print(f"[INFO] Removed non-finite points; now {len(pcd.points):,} pts")

    # Optional downsample if huge
    if len(pcd.points) > DOWNSAMPLE_IF_OVER:
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)
        print(f"[INFO] Voxel-downsampled to {len(pcd.points):,} pts (voxel={VOXEL_SIZE})")

    # Normals for Poisson
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(50)

    print(f"[INFO] Poisson(depth={POISSON_DEPTH}) …")
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=POISSON_DEPTH)

    # Cleanup + crop + keep largest component
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    labels = np.asarray(mesh.cluster_connected_triangles()[0])
    if labels.size > 0:
        largest = int(np.bincount(labels).argmax())
        tri_mask = labels != largest
        mesh.remove_triangles_by_mask(tri_mask)
        mesh.remove_unreferenced_vertices()

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise SystemExit("[ERROR] Reconstructed mesh is empty after cleanup.")

    # >>> compute normals before writing
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    ok = o3d.io.write_triangle_mesh(DST, mesh)
    if not ok: raise SystemExit("[ERROR] Failed to write STL.")
    print(f"[OK] Wrote {DST} | V={len(mesh.vertices):,}, F={len(mesh.triangles):,}")

if __name__ == "__main__":
    main()
