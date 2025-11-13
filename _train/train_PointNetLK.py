"""
Author: Ahmad Mashayekhi
Project: ML-Based CAD–PCD Alignment for Industrial Manipulators

Description
-----------
Training script for PointNetLK on custom HDF5 registration datasets.
It:

  • Loads train/test HDF5 files from ../Input_Output
  • Builds DataLoaders using UserData('registration', ...)
  • Constructs a PointNet backbone + PointNetLK registration network
  • Optionally loads pretrained PointNet weights or resumes from a checkpoint
  • Monitors thermal limits (CPU/GPU) and frees GPU cache when needed
  • Logs to TensorBoard, saves best/latest checkpoints, and plots live losses

Usage
-----
python _train/train_PointNetLK.py --exp_name exp_pnlk [--resume PATH] [--pretrained PATH]
"""



"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""
#General imports
import h5py
import os
os.environ.setdefault("MPLBACKEND", "Agg")            # Matplotlib: no GUI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen") # Qt: no display
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import argparse
from tensorboardX import SummaryWriter
import numpy as np 
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from GPUtil import showUtilization as gpu_usage
from pathlib import Path



# Numba CUDA is optional — avoid segfaults after driver changes
try:
    from numba import cuda
    _HAS_NUMBA_CUDA = True
except Exception:
    _HAS_NUMBA_CUDA = False

# NVML is optional — guard telemetry
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetTemperature, nvmlShutdown, NVML_TEMPERATURE_GPU
    _NVML_OK = True
except Exception:
    _NVML_OK = False
    NVML_TEMPERATURE_GPU = 0  # placeholder, used only inside guarded blocks


import matplotlib.pyplot as plt
from toolboxes.learning3d.data_utils import RegistrationData, UserData   # Ahmad-
from toolboxes.learning3d.losses import FrobeniusNormLoss, RMSEFeaturesLoss
from toolboxes.learning3d.models import PointNet, PointNetLK
import psutil
import sys
from h5_files.h5_writer import uniquify
import threading, time, open3d as o3d

# === put near your other globals ===
BASE_LR = 1e-4              # starting LR for both LK and features
MIN_LR  = 1e-6              # floor so it never goes to ~0
K_frob = 1.0
K_feat = 0.1                # was 1.0 before

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


BASE_DIR = os.getcwd() #Parent folder -> Thesis
#l3D_DIR = "toolboxes/learning3d/"
l3D_DIR = "temp/"
Visualize_min_error = False  # if True, shows pointclouds after each epoch's best sample

def show_pointclouds_auto_close(geoms, window_name="Open3D", timeout=5.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, visible=True)
    for g in geoms:
        vis.add_geometry(g)

    start = time.time()
    # Manual render loop (non-blocking; closes after `timeout` seconds)
    while time.time() - start < float(timeout):
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)  # small sleep to avoid 100% CPU

    vis.destroy_window()

def invert_SE3(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 transform (SE3) as a numpy array."""
    R = T[:3, :3]
    t = T[:3,  3]
    Ti = np.eye(4, dtype=T.dtype)
    Rt = R.T
    Ti[:3, :3] = Rt
    Ti[:3,  3] = -Rt @ t
    return Ti

def apply_transform_np(points_xyz, T_4x4):
    """points_xyz: (N,3) np,  T_4x4: (4,4) np -> (N,3) np"""
    R = T_4x4[:3, :3]
    t = T_4x4[:3, 3]
    return points_xyz @ R.T + t

def show_pointclouds_o3d(template_np, source_np, estT_source_np, gt_source_np,
                         window_title="Blue=Template, Red=Source, Green=est_T(Source→Template), Yellow=GT(Source→Template)"):
    # Template (Blue)
    pc_t = o3d.geometry.PointCloud()
    pc_t.points = o3d.utility.Vector3dVector(template_np)
    pc_t.paint_uniform_color([0, 0, 1])

    # Source (Red) — original (world) for reference
    pc_s = o3d.geometry.PointCloud()
    pc_s.points = o3d.utility.Vector3dVector(source_np)
    pc_s.paint_uniform_color([1, 0, 0])

    # est_T(Source) (Green) — source moved into template frame by estimated T
    pc_est = o3d.geometry.PointCloud()
    pc_est.points = o3d.utility.Vector3dVector(estT_source_np)
    pc_est.paint_uniform_color([0, 1, 0])

    # GT(Source) (Yellow) — source moved into template frame by GT
    pc_gt = o3d.geometry.PointCloud()
    pc_gt.points = o3d.utility.Vector3dVector(gt_source_np)
    pc_gt.paint_uniform_color([1, 1, 0])

    show_pointclouds_auto_close([pc_t, pc_s, pc_est, pc_gt],
                                window_name=window_title, timeout=5)

def check_temp_and_wait(cpu_limit=85, gpu_limit=85, cool_margin=5, poll_sec=3.0, max_wait_sec=None):
    """Pause if CPU/GPU exceed limits; return (cpu_hot_event, gpu_hot_event)."""
    import time, sys
    def read_temps():
        cpu_cur = None
        try:
            temps = psutil.sensors_temperatures()
            if isinstance(temps, dict):
                if "coretemp" in temps and len(temps["coretemp"]):
                    cpu_cur = max([t.current for t in temps["coretemp"] if hasattr(t, "current")], default=None)
                elif len(temps):
                    cpu_cur = max([max([t.current for t in v if hasattr(t, "current")], default=None) for v in temps.values() if v], default=None)
        except Exception:
            pass

        gpu_cur = None
        if _NVML_OK:
            try:
                nvmlInit()
                handle = nvmlDeviceGetHandleByIndex(0)
                gpu_cur = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                nvmlShutdown()
            except Exception:
                pass

        return cpu_cur, gpu_cur

    cpu_cur, gpu_cur = read_temps()
    cpu_hot = (cpu_cur is not None and cpu_cur >= cpu_limit)
    gpu_hot = (gpu_cur is not None and gpu_cur >= gpu_limit)
    if not (cpu_hot or gpu_hot):
        return False, False

    start = time.time()
    print(f"[THERMAL] Hot detected (CPU={cpu_cur}°C, GPU={gpu_cur}°C). Pausing to cool...")
    while True:
        time.sleep(poll_sec)
        cpu_cur, gpu_cur = read_temps()
        cooled_cpu = (cpu_cur is None) or (cpu_cur <= cpu_limit - cool_margin)
        cooled_gpu = (gpu_cur is None) or (gpu_cur <= gpu_limit - cool_margin)
        if cooled_cpu and cooled_gpu:
            print(f"[THERMAL] Cooled (CPU={cpu_cur}°C, GPU={gpu_cur}°C). Resuming.")
            break
        if max_wait_sec is not None and (time.time() - start) > max_wait_sec:
            print(f"[THERMAL] Stayed hot too long (> {max_wait_sec}s). Exiting.")
            sys.exit(1)

    return cpu_hot, gpu_hot

        
def free_gpu_cache():
    # Only try to touch CUDA if it's actually available
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Numba path is already try/except guarded; keep it
    if _HAS_NUMBA_CUDA:
        try:
            from numba import cuda
            cuda.select_device(0)
            cuda.close()
            cuda.select_device(0)
        except Exception:
            pass


def get_transformations(igt):
    R_ba = igt[:, 0:3, 0:3]                                # Ps = R_ba * Pt
    translation_ba = igt[:, 0:3, 3].unsqueeze(2)           # Ps = Pt + t_ba
    R_ab = R_ba.permute(0, 2, 1)                           # Pt = R_ab * Ps
    translation_ab = -torch.bmm(R_ab, translation_ba)      # Pt = Ps + t_ab
    return R_ab, translation_ab, R_ba, translation_ba

def get_lrs(optimizer):
    return [pg['lr'] for pg in optimizer.param_groups]


def _init_(args):
    if not os.path.exists(l3D_DIR+'checkpoints'):
        os.makedirs(l3D_DIR+'checkpoints') #Creates directory with checkpoints folder if it does not yet exist
    if not os.path.exists(l3D_DIR+'checkpoints/' + args.exp_name):
        os.makedirs(l3D_DIR+'checkpoints/' + args.exp_name) #Same for directory to exp_name folder
    if not os.path.exists(l3D_DIR+'checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs(l3D_DIR+'checkpoints/' + args.exp_name + '/' + 'models') #Same for errors
    if not os.path.exists(l3D_DIR+'checkpoints/' + args.exp_name + '/' + 'errors'):
        os.makedirs(l3D_DIR+'checkpoints/' + args.exp_name + '/' + 'errors') #Same for errors
    # os.system('copy main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    # os.system('copy model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')


class IOStream: #For writing to run file in checkpoints/exp_name/run.log
    def __init__(self, path):
        self.f = open(path, 'a') #open file and append info

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n') #Write text to file run.log
        self.f.flush() #clear internal buffer of the file

    def close(self):
        self.f.close() #close file


def write_error_h5_Ahmad(template, source, gt, est_T, transformed_source, 
                         epoch, split='test', loss_total =None, loss_frob=None, loss_feat=None):

    # Prepare directory
    save_dir = os.path.join(l3D_DIR, 'checkpoints', Exp_Name, 'errors', f'EP{epoch}')
    os.makedirs(save_dir, exist_ok=True)

    # Format loss string for filename
    if loss_total is not None:
        # loss_total = f"{loss_total:.4f}".replace('.', '_')  # safer for filenames
        file_name = f"train_error_total_{loss_total:.4f}_frob_{loss_frob:.4f}_feat_{loss_feat:.4f}.hdf5"
    else:
        file_name = "train_error.hdf5"

    file_path = os.path.join(save_dir, file_name)

    # ✅ Detach all tensors safely before converting to numpy
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('template', data=template.detach().cpu().numpy())
        f.create_dataset('source', data=source.detach().cpu().numpy())
        f.create_dataset('gt', data=gt.detach().cpu().numpy())
        f.create_dataset('est_T', data=est_T.detach().cpu().numpy())
        f.create_dataset('transformed_source', data=transformed_source.detach().cpu().numpy())

        if loss_total is not None:
            f.create_dataset('loss_total', data=np.array([loss_total], dtype=np.float32))

def test_one_epoch(device, model, test_loader, epoch, K_frob, K_feat):
    with torch.no_grad():
        model.eval()
        test_loss_total = 0.0
        test_loss_frob  = 0.0       # <— NEW
        test_acc = 0.0
        count = 0
        cpu_warns = 0
        gpu_warns = 0
        
        for i, data in enumerate(tqdm(test_loader)):
            cpu_event, gpu_event = check_temp_and_wait(cpu_limit=85, gpu_limit=85)
            if cpu_event: cpu_warns += 1
            if gpu_event: gpu_warns += 1

            template, source, igt = data
            template = template.to(device); source = source.to(device); igt = igt.to(device)

            output = model(template, source)
            FN_loss       = FrobeniusNormLoss()(output['est_T'], igt)
            RMSEFeat_loss = RMSEFeaturesLoss()(output['r'])
            loss_val      = K_frob * FN_loss + K_feat * RMSEFeat_loss

            # (Optional) make accuracy Frobenius-only:
            if FN_loss.item() < threshold:
                test_acc += 1

            test_loss_total += loss_val.item()
            test_loss_frob  += FN_loss.item()   # <— NEW
            count += 1

        test_loss_total /= count
        test_loss_frob  /= count               # <— NEW
        test_acc        /= count

    # return the new value at the end
    return test_loss_total, test_acc, cpu_warns, gpu_warns, test_loss_frob


def test(args, model, test_loader, textio):
    ep = max(args.start_epoch, 1)
    test_loss_total, test_accuracy, _, _, test_loss_frob = \
        test_one_epoch(args.device, model, test_loader, ep, K_frob, K_feat)
    textio.cprint('Validation (total): %.6f | Validation Frobenius: %.6f | Acc: %.4f'
                  % (test_loss_total, test_loss_frob, test_accuracy))


def train_one_epoch(device, model, train_loader, optimizer, epoch, K_frob, K_feat):
    model.train()
    train_loss = 0.0
    train_acc = 0.0 
    train_loss_frobenius = 0.0
    train_loss_rmse = 0.0
    count = 0

    # ✅ NEW: per-epoch thermal counters
    cpu_warns = 0
    gpu_warns = 0

    # keep the best (lowest-error) sample of the whole epoch
    epoch_best = {
        'template': None, 'source': None, 'igt': None, 'est_T': None,
        'transformed_source_est': None
    }
    epoch_best_err = float('inf')

    for i, data in enumerate(tqdm(train_loader)):
        # ✅ count thermal events
        cpu_event, gpu_event = check_temp_and_wait(cpu_limit=85, gpu_limit=85)
        if cpu_event: cpu_warns += 1
        if gpu_event: gpu_warns += 1

        template, source, igt = data
        template = template.to(device)
        source   = source.to(device)
        igt      = igt.to(device)

        output = model(template, source)

        FN_loss        = FrobeniusNormLoss()(output['est_T'], igt)
        RMSEFeat_loss  = RMSEFeaturesLoss()(output['r'])
        loss_val       = K_frob * FN_loss + K_feat * RMSEFeat_loss

        # (optional) save hard cases
        if loss_val.item() > Err_Thres * 0 and epoch % 50 == 0:
            write_error_h5_Ahmad(template, source, igt, output['est_T'], output['transformed_source'],
                                 epoch, 'train',
                                 loss_total=loss_val.item(),
                                 loss_frob=FN_loss.item(),
                                 loss_feat=RMSEFeat_loss.item())

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # track best sample (lowest Frobenius error)
        with torch.no_grad():
            diff = (output['est_T'] - igt).reshape(output['est_T'].shape[0], -1)
            per_sample_err = torch.norm(diff, dim=1)
            bmin = torch.argmin(per_sample_err).item()
            bmin_err = per_sample_err[bmin].item()
            if bmin_err < epoch_best_err:
                epoch_best_err = bmin_err
                epoch_best['template'] = template[bmin].detach().cpu()
                epoch_best['source']   = source[bmin].detach().cpu()
                epoch_best['igt']      = igt[bmin].detach().cpu()
                epoch_best['est_T']    = output['est_T'][bmin].detach().cpu()
                epoch_best['transformed_source_est'] = output['transformed_source'][bmin].detach().cpu()

        if loss_val.item() < threshold:
            train_acc += 1

        train_loss           += loss_val.item()
        train_loss_frobenius += FN_loss.item()
        train_loss_rmse      += RMSEFeat_loss.item()
        count += 1

    # averages
    train_loss           /= count
    train_loss_frobenius /= count
    train_loss_rmse      /= count
    train_acc            /= count

    # ✅ return 7 values as expected by your train(...)
    return train_loss, train_acc, train_loss_frobenius, train_loss_rmse, epoch_best, cpu_warns, gpu_warns


def train(args, model, train_loader, test_loader, boardio, textio, checkpoint):
    # --- One Adam over ALL params => LK LR == FEAT LR always
    optimizer = torch.optim.Adam(
        model.parameters(), lr=BASE_LR, weight_decay=1e-4, betas=(0.9, 0.999)
    )
    best_test_frob = float('inf')

    # (optional) resume optimizer + best metric
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'min_loss' in checkpoint:
            best_test_frob = float(checkpoint['min_loss'])


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=0.5, patience=20,
        threshold=5e-3,      # 0.5% improvement required
        threshold_mode='rel',
        cooldown=5, min_lr=MIN_LR, verbose=True
    )

    
    # === Plot setup =========================================================
    plt.ion()
    fig, ax = plt.subplots()
    epochs, train_losses, test_losses = [], [], []
    (line1,) = ax.plot([], [], 'r-', label='Train Loss (total)')
    (line2,) = ax.plot([], [], 'b-', label='Test  Loss (total)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Training Progress')
    ax.set_yscale('log')

    cpu_events_per_epoch = []
    ax2 = ax.twinx()
    (line_cpu,) = ax2.plot([], [], linestyle='--', marker='o', label='CPU heat warnings / epoch')
    ax2.set_ylabel('CPU heat warnings / epoch')
    ax.relim(); ax.autoscale_view(); ax2.relim(); ax2.autoscale_view()
    # ========================================================================

    for epoch in range(args.start_epoch, args.epochs):
        ep = epoch + 1

        # ---- Train/Test -----------------------------------------------------
        train_loss, train_acc, train_loss_frob, train_loss_rmse, epoch_best, tr_cpu_warns, tr_gpu_warns = \
            train_one_epoch(args.device, model, train_loader, optimizer, ep, K_frob, K_feat)

        test_loss_total, test_acc, te_cpu_warns, te_gpu_warns, test_loss_frob = \
            test_one_epoch(args.device, model, test_loader, ep, K_frob, K_feat)

        # ---- LR scheduler (val metric)
        if np.isfinite(test_loss_total):
            scheduler.step(test_loss_total)

        current_lr = optimizer.param_groups[0]['lr']
        boardio.add_scalar('LR/all', current_lr, ep)
        
        # ---- Early stop when LR floored AND patience exhausted
        if current_lr <= (MIN_LR + 1e-12) and scheduler.num_bad_epochs >= scheduler.patience:
            textio.cprint(
                f"[EARLY STOP] Epoch {ep}: LR ≤ {MIN_LR:.1e} and "
                f"no improvement for {scheduler.patience} epochs."
            )
            # save a final snapshot
            snap = {
                'epoch': ep,
                'model': model.state_dict(),
                'min_loss': best_test_frob,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(snap, l3D_DIR + f'checkpoints/{args.exp_name}/models/final_early_stop_snap.t7')
            torch.save(model.state_dict(), l3D_DIR + f'checkpoints/{args.exp_name}/models/final_early_stop_model.t7')
            torch.save(model.feature_model.state_dict(), l3D_DIR + f'checkpoints/{args.exp_name}/models/final_early_stop_ptnet.t7')
            break
        

        # ---- Live plot data
        epochs.append(ep)                                # ✅ add the epoch on x-axis
        train_losses.append(train_loss)
        test_losses.append(test_loss_total)
        cpu_events_per_epoch.append((tr_cpu_warns or 0) + (te_cpu_warns or 0))

        line1.set_data(epochs, train_losses)
        line2.set_data(epochs, test_losses)
        line_cpu.set_data(epochs, cpu_events_per_epoch)
        ax.relim(); ax.autoscale_view()
        ax2.relim(); ax2.autoscale_view()
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper right')
        fig.canvas.draw(); fig.canvas.flush_events()

        # ---- checkpoints (best by Frobenius only)
        snap = {
            'epoch': ep,
            'model': model.state_dict(),
            'min_loss': best_test_frob,
            'optimizer': optimizer.state_dict(),
        }
        if test_loss_frob < best_test_frob:
            best_test_frob = test_loss_frob
            snap['min_loss'] = best_test_frob
            torch.save(snap, l3D_DIR + f'checkpoints/{args.exp_name}/models/best_model_snap.t7')
            torch.save(model.state_dict(), l3D_DIR + f'checkpoints/{args.exp_name}/models/best_model.t7')
            torch.save(model.feature_model.state_dict(), l3D_DIR + f'checkpoints/{args.exp_name}/models/best_ptnet_model.t7')

        # always save latest
        torch.save(snap, l3D_DIR + f'checkpoints/{args.exp_name}/models/model_snap.t7')
        torch.save(model.state_dict(), l3D_DIR + f'checkpoints/{args.exp_name}/models/model.t7')
        torch.save(model.feature_model.state_dict(), l3D_DIR + f'checkpoints/{args.exp_name}/models/ptnet_model.t7')

        # ---- TB logs
        boardio.add_scalar('Loss/Train_Total',    train_loss,      ep)
        boardio.add_scalar('Loss/Train_Frob',     train_loss_frob, ep)
        boardio.add_scalar('Loss/Train_RMSEFeat', train_loss_rmse, ep)
        boardio.add_scalar('Loss/Test_Total',     test_loss_total, ep)
        boardio.add_scalar('Loss/Test_Frob',      test_loss_frob,  ep)
        boardio.add_scalar('Best/Test_Frob',      best_test_frob,  ep)
        boardio.add_scalar('Acc/Train',           train_acc,       ep)
        boardio.add_scalar('Acc/Test',            test_acc,        ep)
        boardio.add_scalar('Lambda/Feat',         K_feat,          ep)

        textio.cprint(
            ('EP %4d | TrainTot: %.3e  TrainFrob: %.3e | '
             'TestTot: %.3e  TestFrob: %.3e | BestTestFrob: %.3e | '
             'Train Acc: %.3e  Test Acc: %.3e | ' %
             (ep, train_loss, train_loss_frob, test_loss_total, test_loss_frob,
              best_test_frob, train_acc, test_acc))
            + f"LR(all): {current_lr:.2e} | K_feat={K_feat:.2e}"
        )

        if Visualize_min_error:
            templ_np = epoch_best['template'].detach().cpu().numpy()
            src_np   = epoch_best['source'].detach().cpu().numpy()
            estT_4x4 = epoch_best['est_T'].detach().cpu().numpy()
            igt_4x4  = epoch_best['igt'].detach().cpu().numpy()
            src_est_in_template = apply_transform_np(src_np, estT_4x4)
            src_gt_in_template  = apply_transform_np(src_np, igt_4x4)
            show_pointclouds_o3d(
                template_np=templ_np, source_np=src_np,
                estT_source_np=src_est_in_template, gt_source_np=src_gt_in_template,
                window_title="Blue=Template, Red=Source, Green=est_T(Source→Template), Yellow=GT(Source→Template)"
            )


def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp_pnlk', metavar='N', help='Name of the experiment')
    parser.add_argument('--dataset_path', type=str, default='Cranfield_Parts',  metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

    # settings for input data
    parser.add_argument('--dataset_type', default='Cranfield_Parts', choices=['modelnet', 'shapenet2'], metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--num_points', default=1024, type=int, metavar='N', help='points in point-cloud (default: 1024)')

    # settings for PointNet
    parser.add_argument('--fine_tune_pointnet', default='tune', type=str, choices=['fixed', 'tune'], help='train pointnet (default: tune)')
    # parser.add_argument('--transfer_ptnet_weights', default='./checkpoints/exp_classifier/models/best_ptnet_model.t7', type=str,
    #                     metavar='PATH', help='path to pointnet features file')
    parser.add_argument('--transfer_ptnet_weights', default='/home/ahmadmashayekhi/temp/checkpoints/exp_pnlk-run_1/best_model_Run_1.t7zzzzzzz',
                        type=str,metavar='PATH', help='path to pointnet features file')
    

    parser.add_argument('--emb_dims', default=1024, type=int,  metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'], help='symmetric function (default: max)')

    # settings for on training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=10, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,    metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--resume', default='', type=str,  metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,  metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args()
    print('==============================================================')

    # 🔍 Check if pretrained PointNet weights are provided and valid
    if args.transfer_ptnet_weights.strip() != '' and os.path.isfile(args.transfer_ptnet_weights):
        print(f"✅ Using pretrained PointNet weights from: {args.transfer_ptnet_weights}")
        use_pretrained_pointnet = True
    else:
        print("⚠️ No valid pretrained PointNet weights provided. PointNet will be randomly initialized.")
        use_pretrained_pointnet = False
    
    print('==============================================================')

    return args



def main():
    # --- Resolve Input_Output folder next to this script ---
    current_dir = Path(__file__).resolve().parent          # .../_train
    project_root = current_dir.parent                      # .../
    input_output_dir = project_root / "Input_Output"       # ../Input_Output
    
    
    # Your chosen filenames (adjust names if needed)
    file_test  = input_output_dir / "ply_data_test_H2017_400IT_angle_30_radius_100_noisy_0.01_template12_source3_planeCut30_tools3.hdf5"
    file_train = input_output_dir / "ply_data_train_H2017_2000IT_angle_30_radius_100_noisy_0.01_template12_source3_planeCut30_tools3.hdf5"

    
    # (rest of your code stays the same up to args = options())
    args = options()
    global l3D_DIR
    l3D_DIR = str(input_output_dir) + "/"   # e.g. ../Input_Output/exp_pnlk/

    # -------------------- CHANGED: choose device early --------------------
    if getattr(args, 'device', None) is None:
        args.device = 'cuda:0'  # default

    if (not torch.cuda.is_available()) or (str(args.device).lower() == 'cpu'):
        print("⚠️ CUDA not available — running on CPU. Tip: use smaller batch size and num_workers=0 for stability.")
        device = torch.device('cpu')
    else:
        print("✅ CUDA available.")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        device = torch.device(args.device)

    args.device = device
    # ---------------------------------------------------------------------

    free_gpu_cache()

    # ------DEFINE GLOBALS------
    global Exp_Name;      Exp_Name = args.exp_name
    global threshold;     threshold = 1
    global Err_Thres;     Err_Thres = 10
    global Epoch_Lim;     Epoch_Lim = 0
    # --------------------------

    # ------SETUP LEARNING------
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # --------------------------

    # ------SETUP WRITING------
    boardio = SummaryWriter(log_dir=l3D_DIR + 'checkpoints/' + args.exp_name)
    _init_(args)
    textio = IOStream(l3D_DIR+'checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))
    # -------------------------

    # ------ Sanity checks for HDF5 paths (early, clear error) ------
    if not file_train.is_file() or not file_test.is_file():
        print(f"\n[HDF5] Looking in: {input_output_dir}")
        if not input_output_dir.exists():
            raise FileNotFoundError(f"Input_Output folder not found: {input_output_dir}")
        # Help by listing available .hdf5 files
        candidates = sorted([p.name for p in input_output_dir.glob('*.hdf5')])
        raise FileNotFoundError(
            "One or both HDF5 files not found:\n"
            f"  Train: {file_train.name}\n"
            f"  Test : {file_test.name}\n\n"
            f"Available in Input_Output:\n  - " + "\n  - ".join(candidates) if candidates else
            "No .hdf5 files found in Input_Output."
        )
    print(f"[HDF5] Train: {file_train}")
    print(f"[HDF5] Test : {file_test}")
    # ---------------------------------------------------------------

    # ------SETUP LOADING------
    # cast to str() because your UserData expects string paths
    trainset = UserData('registration', str(file_train))
    testset  = UserData('registration', str(file_test))

    use_cpu = (str(args.device).lower() == 'cpu') or (not torch.cuda.is_available())
    pin_mem = False if use_cpu else True
    num_wrk = 0 if use_cpu else max(0, int(getattr(args, "workers", 0)))

    if use_cpu:
        args.batch_size = max(1, min(args.batch_size, 8))
        try:
            os.environ.setdefault("OMP_NUM_THREADS", "8")
            os.environ.setdefault("MKL_NUM_THREADS", "8")
            torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        except Exception:
            pass

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=num_wrk, pin_memory=pin_mem,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=num_wrk, pin_memory=pin_mem,
        persistent_workers=False,
    )

    print(f"[DataLoader] device={args.device} | batch_size={args.batch_size} | "
          f"num_workers={num_wrk} | pin_memory={pin_mem}")
    # -------------------------

    print('==============================================================')

    # Create PointNet Model.
    ptnet = PointNet(emb_dims=args.emb_dims, use_bn=True)
    if args.transfer_ptnet_weights and os.path.isfile(args.transfer_ptnet_weights):
        ptnet.load_state_dict(torch.load(args.transfer_ptnet_weights, map_location='cpu'))
        print(f"✅ Loaded PointNet weights: {args.transfer_ptnet_weights}")

    if args.fine_tune_pointnet == 'tune':
        print('✅ Fine-tuning PointNet weights.')
    else:
        print('🔒 Freezing PointNet weights.')
        for param in ptnet.parameters():
            param.requires_grad_(False)

    model = PointNetLK(feature_model=ptnet, p0_zero_mean=True, p1_zero_mean=True).to(args.device)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        print(f"✅ PointNetLK: Loaded model from checkpoint: {args.resume}")
    elif args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
        print(f"✅ PointNetLK: Loaded pretrained weights: {args.pretrained}")
    else:
        print("🚀 PointNetLK: Starting training from scratch (no checkpoint or pretrained model)")

    if args.eval:
        test(args, model, test_loader, textio)
    else:
        train(args, model, train_loader, test_loader, boardio, textio, checkpoint)


if __name__ == '__main__':
    main()