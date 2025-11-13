# h5_writer_Ahmad.py
import os
import h5py
import numpy as np

# ------------------------
# Path helpers
# ------------------------
def uniquify(path: str) -> str:
    base, ext = os.path.splitext(path)
    i = 1
    out = path
    while os.path.exists(out):
        out = f"{base}{i}{ext}"
        i += 1
    return out

def _script_dir() -> str:
    # Directory of the currently running script (caller)
    # If imported from a different working dir, this still points to the .py file location.
    return os.path.dirname(os.path.abspath(__file__))

def create_DIR(File_Name: str, FolderName: str = "") -> str:
    """
    Build a full path for <File_Name>.hdf5 as follows:
      - If FolderName == "": save in the SAME FOLDER as this module (your script folder).
      - If FolderName is absolute: save under that absolute folder.
      - If FolderName is relative: interpret it relative to the script folder.
    """
    if not FolderName:
        target_dir = _script_dir()
    elif os.path.isabs(FolderName):
        target_dir = FolderName
    else:
        target_dir = os.path.join(_script_dir(), FolderName)

    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"{File_Name}.hdf5")
    return uniquify(path)

# ------------------------
# Core writer API
# ------------------------
def write_h5(File_Name, template, source, gt, FolderName: str = ""):
    """
    Create an HDF5 file with datasets:
      - template:  (B, P, 3)
      - source:    (B, S, 3)
      - transformation:     (B, 4, 4)
      - transformation_all: (B, 4, 4) (extendable if you want later)
    """
    out_path = create_DIR(File_Name, FolderName)

    # Basic validation
    template = np.asarray(template)
    source   = np.asarray(source)
    gt       = np.asarray(gt)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("template", data=template)
        f.create_dataset("source", data=source)
        f.create_dataset("transformation", data=gt)
        f.create_dataset("transformation_all", data=gt, chunks=True, maxshape=(None, 4, 4))

    print("\n:: File saved as:", out_path)
    return out_path

# ------------------------
# Utilities used by your pipeline
# ------------------------
def add_Gaussian_Noise(mu, sigma, orig_cloud, bound):
    """
    orig_cloud: shape [1, N, 3]
    Adds per-point Gaussian noise clipped to [-bound, +bound].
    """
    cloud = np.array(orig_cloud, copy=True)
    N = cloud.shape[1]
    noise = np.clip(np.random.normal(mu, sigma, size=(1, N, 3)), -bound, bound)
    cloud[0, :, 0:3] = cloud[0, :, 0:3] + noise
    return cloud

def farthest_subsample_points(points, k=768):
    """
    Pure NumPy farthest point sampling.
    points: (N, 3) -> returns (k, 3), mask (N,) with 1 at selected indices.
    """
    pts = np.asarray(points, dtype=np.float64)
    N = pts.shape[0]
    k = int(min(max(1, k), N))

    sel_idx = np.empty(k, dtype=np.int64)
    dists = np.full(N, np.inf)

    # start from a random point
    sel = np.random.randint(N)
    for i in range(k):
        sel_idx[i] = sel
        diff = pts - pts[sel]
        dist2 = np.einsum('ij,ij->i', diff, diff)
        dists = np.minimum(dists, dist2)
        sel = int(np.argmax(dists))

    mask = np.zeros(N, dtype=np.float32)
    mask[sel_idx] = 1.0
    return pts[sel_idx, :].astype(points.dtype, copy=False), mask
