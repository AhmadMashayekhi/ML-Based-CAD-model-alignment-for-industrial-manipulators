"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



main train

Trains selected method (RPMNet, PointNetLK, PRNet, ROPNet)

Credits: 
    PointNetLK, RPMNet, ROPNet & PRNet Code by vinits5 as part of the Learning3D library 
    Link: https://github.com/vinits5/learning3d#use-your-own-data

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""
import os

# --- Limit CPU threads (BLAS/OpenMP) to keep temps down ---
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"

# Optional: lower PyTorch CPU threading too
import torch
torch.set_num_threads(2)          # internal CPU ops
torch.set_num_interop_threads(1)  # thread pool for parallelism across ops

# (leave the rest of your imports below)
import sys
import _train.train_PointNetLK as PNLK_train


# === SELECT ONE MODE ===
Train_from_scratch                  = True
Train_from_full_previous_PointNetLK = False
Train_from_backbone_PointNet        = False
# ========================

# --- safety: check only one is True ---
modes = [Train_from_scratch, Train_from_full_previous_PointNetLK, Train_from_backbone_PointNet]
if sum(modes) != 1:
    raise SystemExit("❌ Please set exactly ONE of the 3 mode flags to True.")

# --- paths ---
resume_ckpt      = "/home/ahmadmashayekhi/temp/checkpoints/21_exp_pnlk_continue_from_20/models/model_snap.t7"
backbone_weights = "/home/ahmadmashayekhi/temp/checkpoints/5_exp_pnlk_12_3_small_poses_1500_epoch_backbone_used_from_1/models/best_ptnet_model.t7"

# --- forward args to trainer ---
if Train_from_scratch:
    sys.argv += [
        "--exp_name", "exp_pnlk",
        "--fine_tune_pointnet", "tune"   # or "fixed"
    ]

elif Train_from_full_previous_PointNetLK:
    if not os.path.isfile(resume_ckpt):
        raise FileNotFoundError(f"❌ Resume checkpoint not found:\n    {resume_ckpt}")
    sys.argv += [
        "--exp_name", "exp_pnlk",
        "--resume", resume_ckpt
    ]

elif Train_from_backbone_PointNet:
    if not os.path.isfile(backbone_weights):
        raise FileNotFoundError(f"❌ Backbone weights not found:\n    {backbone_weights}")
    sys.argv += [
        "--exp_name", "exp_pnlk",
        "--transfer_ptnet_weights", backbone_weights,
        "--fine_tune_pointnet", "tune"  # or "fixed"
    ]
print("\n=== Launching PointNetLK Trainer ===")
print("Args:", " ".join(sys.argv[1:]))
print("====================================\n")

PNLK_train.main()
