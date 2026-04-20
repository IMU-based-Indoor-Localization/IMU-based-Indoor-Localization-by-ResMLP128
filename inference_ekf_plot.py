import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from networks.model_twolayer import TwoLayerModel
from tracker.imu_tracker import ImuTracker
from utils.dotdict import dotdict

# --- Configuration ---
MODEL_PATH = r"C:\Users\hs091\Documents\GitHub\IMU-based-Indoor-Localization-by-ResMLP128\out_tlio_6ch_128\checkpoints\best.pth"
NORM_MEAN_PATH = r"C:\Users\hs091\Documents\GitHub\IMU-based-Indoor-Localization-by-ResMLP128\out_tlio_6ch_128\norm_mean.npy"
NORM_STD_PATH = r"C:\Users\hs091\Documents\GitHub\IMU-based-Indoor-Localization-by-ResMLP128\out_tlio_6ch_128\norm_std.npy"
DATA_PATH = r"C:\Users\hs091\Documents\GitHub\IMU-based-Indoor-Localization-by-ResMLP128\TLIO_Oxford_Dataset\oxford_handheld_1\imu0_resampled.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_for_ekf(model_path, device):
    model_para = {
        "input_len": 200, "input_channel": 6, "patch_len": 20,
        "feature_dim": 128, "out_dim": 3, "active_func": "GELU",
        "extractor": {"name": "ResMLP", "layer_num": 6, "expansion": 2, "dropout": 0.2},
        "reg": {"name": "SimpleMean", "layer_num": 3, "dropout": 0.2},
        "use_classifier": False,
    }
    model = TwoLayerModel(model_para).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def run_ekf_inference(model, data_path, norm_mean, norm_std):
    raw = np.load(data_path)
    ts      = raw[:, 0]
    gyr_raw = raw[:, 1:4]
    acc_raw = raw[:, 4:7]
    quat_gt = raw[:, 14:18]
    pos_gt  = raw[:, 18:21]
    vel_gt  = raw[:, 21:24]

    net_config = {
        "imu_freq_net": 200,
        "past_time": 0.0,
        "window_time": 1.0,
    }
    update_freq = 10
    
    filter_tuning_cfg = dotdict({
        "sigma_na": np.sqrt(1e-3),
        "sigma_ng": np.sqrt(1e-4),
        "ita_ba": 1e-4,
        "ita_bg": 1e-6,
        "init_attitude_sigma": 10.0 / 180.0 * np.pi,
        "init_yaw_sigma": 1.0 / 180.0 * np.pi,
        "init_vel_sigma": 0.5,
        "init_pos_sigma": 0.1,
        "init_bg_sigma": 0.001,
        "init_ba_sigma": 0.1,
        "mahalanobis_fail_scale": 10.0,
        "g_norm": 9.81,
    })

    tracker = ImuTracker(
        model=model, norm_mean=norm_mean, norm_std=norm_std,
        net_config=net_config, update_freq=update_freq,
        filter_tuning_cfg=filter_tuning_cfg, device=DEVICE
    )

    t0_us = int(ts[0] * 1e-3)
    R0 = R.from_quat(quat_gt[0]).as_matrix()
    v0 = vel_gt[0].reshape(3, 1)
    p0 = pos_gt[0].reshape(3, 1)
    
    if np.any(np.isnan(R0)):
         for i in range(len(ts)):
             R0 = R.from_quat(quat_gt[i]).as_matrix()
             v0 = vel_gt[i].reshape(3, 1)
             p0 = pos_gt[i].reshape(3, 1)
             if not np.any(np.isnan(R0)):
                 t0_us = int(ts[i] * 1e-3)
                 start_idx = i
                 break
    else:
        start_idx = 0

    tracker.init_with_state_at_time(t0_us, R0, v0, p0, gyr_raw[start_idx], acc_raw[start_idx])

    estimated_pos = []
    gt_pos_matched = []
    
    print(f"Running EKF inference for {len(ts)-start_idx} samples...")
    for i in range(start_idx + 1, len(ts)):
        t_us = int(ts[i] * 1e-3)
        tracker.on_imu_measurement(t_us, gyr_raw[i], acc_raw[i])
        estimated_pos.append(tracker.filter.state.s_p.flatten().copy())
        gt_pos_matched.append(pos_gt[i].copy())

    return np.array(gt_pos_matched), np.array(estimated_pos)

def plot_results(gt_pos, ekf_pos):
    origin = gt_pos[0].copy()
    gt = gt_pos - origin
    ekf = ekf_pos - origin
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))
    
    plt.plot(gt[:, 0], gt[:, 1], 'k--', label="Ground Truth (Reference)", alpha=0.6, linewidth=2.5)
    plt.plot(ekf[:, 0], ekf[:, 1], 'r-', label="EKF Estimates (Proposed)", alpha=0.9, linewidth=3.0)
    
    plt.scatter(0, 0, marker="o", color="green", label="Start Point", s=150, edgecolors='black', zorder=5)
    plt.scatter(gt[-1, 0], gt[-1, 1], marker="X", color="black", label="GT End", s=150, edgecolors='white', zorder=5)
    plt.scatter(ekf[-1, 0], ekf[-1, 1], marker="*", color="red", label="EKF End", s=200, edgecolors='black', zorder=5)
    
    plt.title("Indoor Localization: Trajectory Comparison (GT vs. EKF-Net)", fontsize=16, fontweight='bold')
    plt.xlabel("X Position (m)", fontsize=14)
    plt.ylabel("Y Position (m)", fontsize=14)
    plt.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis("equal")
    
    rmse_xy = np.sqrt(np.mean(np.sum((ekf[:, :2] - gt[:, :2])**2, axis=1)))
    stats_text = f"RMSE (XY): {rmse_xy:.3f}m\nSamples: {len(ekf)}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=14, fontweight='bold', verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig("inference_ekf_result.png", dpi=200)
    print(f"High-quality result saved to inference_ekf_result.png")
    print(f"Final RMSE_XY: {rmse_xy:.3f}m")

if __name__ == "__main__":
    model = load_model_for_ekf(MODEL_PATH, DEVICE)
    norm_mean = np.load(NORM_MEAN_PATH).reshape(1, 6)
    norm_std = np.load(NORM_STD_PATH).reshape(1, 6)
    gt_pos, ekf_pos = run_ekf_inference(model, DATA_PATH, norm_mean, norm_std)
    plot_results(gt_pos, ekf_pos)
