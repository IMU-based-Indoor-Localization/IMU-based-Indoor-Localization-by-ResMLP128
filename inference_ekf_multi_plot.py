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
# DATA_ROOT = r"C:\Users\hs091\Documents\GitHub\IMU-based-Indoor-Localization-by-ResMLP128\TLIO_Oxford_Dataset" # For all sequences
DATA_ROOT = r"C:\Users\hs091\Documents\GitHub\IMU-based-Indoor-Localization-by-ResMLP128\TLIO_Oxford_Dataset"

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

def run_single_ekf_inference(model, data_path, norm_mean, norm_std):
    raw = np.load(data_path)
    ts      = raw[:, 0]
    
    # Check for valid timestamps
    if len(np.unique(ts)) < 10:
        print(f"  [Warning] Invalid timestamps detected (all same). Synthesizing 100Hz timestamps.")
        ts = np.arange(len(raw)) * 10000.0 # 100Hz = 10ms = 10000us
    else:
        # Check units: if 1.5e11 (ns), convert to us
        if ts[0] > 1e10:
            ts = ts * 1e-3
            
    gyr_raw = raw[:, 1:4]
    acc_raw = raw[:, 4:7]
    quat_gt = raw[:, 14:18]
    pos_gt  = raw[:, 18:21]
    vel_gt  = raw[:, 21:24]

    net_config = {"imu_freq_net": 200, "past_time": 0.0, "window_time": 1.0}
    update_freq = 10
    
    filter_tuning_cfg = dotdict({
        "sigma_na": np.sqrt(1e-3), "sigma_ng": np.sqrt(1e-4),
        "ita_ba": 1e-4, "ita_bg": 1e-6,
        "init_attitude_sigma": 10.0 / 180.0 * np.pi,
        "init_yaw_sigma": 1.0 / 180.0 * np.pi,
        "init_vel_sigma": 0.5, "init_pos_sigma": 0.1,
        "init_bg_sigma": 0.001, "init_ba_sigma": 0.1,
        "mahalanobis_fail_scale": 10.0, "g_norm": 9.81,
    })

    tracker = ImuTracker(
        model=model, norm_mean=norm_mean, norm_std=norm_std,
        net_config=net_config, update_freq=update_freq,
        filter_tuning_cfg=filter_tuning_cfg, device=DEVICE
    )

    # Initial state
    t0_us = int(ts[0] * 1e-3)
    start_idx = 0
    # Find first non-NaN quat
    if np.any(np.isnan(quat_gt[0])):
        for i in range(len(ts)):
            if not np.any(np.isnan(quat_gt[i])):
                t0_us = int(ts[i] * 1e-3)
                start_idx = i
                break
    
    R0 = R.from_quat(quat_gt[start_idx]).as_matrix()
    v0 = vel_gt[start_idx].reshape(3, 1)
    p0 = pos_gt[start_idx].reshape(3, 1)

    tracker.init_with_state_at_time(t0_us, R0, v0, p0, gyr_raw[start_idx], acc_raw[start_idx])

    estimated_pos = [p0.flatten().copy()]
    gt_matched_pos = [p0.flatten().copy()]
    
    # Run loop
    for i in range(start_idx + 1, len(ts)):
        t_us = int(ts[i])  # Already in us (either from synthesis or from converted ns)
        tracker.on_imu_measurement(t_us, gyr_raw[i], acc_raw[i])
        estimated_pos.append(tracker.filter.state.s_p.flatten().copy())
        gt_matched_pos.append(pos_gt[i].copy())

    return np.array(gt_matched_pos), np.array(estimated_pos)

def plot_multi_ekf(model_path, data_root, norm_mean_path, norm_std_path, max_files=6):
    model = load_model_for_ekf(model_path, DEVICE)
    norm_mean = np.load(norm_mean_path).reshape(1, 6)
    norm_std = np.load(norm_std_path).reshape(1, 6)
    
    npy_files = sorted(list(Path(data_root).rglob("imu0_resampled.npy")))[:max_files]
    if not npy_files:
        print(f"No npy files found in {data_root}")
        return

    n = len(npy_files)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).reshape(-1)
    
    rmse_list = []
    
    for i, npy_path in enumerate(npy_files):
        ax = axes[i]
        print(f"Processing ({i+1}/{n}): {npy_path.parent.name}")
        
        gt_pos, ekf_pos = run_single_ekf_inference(model, npy_path, norm_mean, norm_std)
        
        # Debug info
        print(f"  GT  - min: {np.min(gt_pos, axis=0)}, max: {np.max(gt_pos, axis=0)}")
        print(f"  EKF - min: {np.min(ekf_pos, axis=0)}, max: {np.max(ekf_pos, axis=0)}")
        print(f"  NaNs in EKF: {np.any(np.isnan(ekf_pos))}")

        # Origin normalization
        origin = gt_pos[0].copy()
        gt = gt_pos - origin
        ekf = ekf_pos - origin
        
        rmse_xy = np.sqrt(np.mean(np.sum((ekf[:, :2] - gt[:, :2])**2, axis=1)))
        rmse_list.append(rmse_xy)
        
        ax.plot(gt[:, 0], gt[:, 1], 'k--', label="GT", alpha=0.6, linewidth=1.5)
        ax.plot(ekf[:, 0], ekf[:, 1], 'r-', label="EKF", alpha=0.8, linewidth=1.5)
        ax.scatter(0, 0, marker="o", color="green", s=30, label="Start", zorder=5)
        ax.scatter(gt[-1, 0], gt[-1, 1], marker="x", color="black", s=40, label="GT End", zorder=5)
        ax.scatter(ekf[-1, 0], ekf[-1, 1], marker="^", color="red", s=40, label="EKF End", zorder=5)
        
        ax.set_title(f"{npy_path.parent.name}\nRMSE_XY={rmse_xy:.3f}m", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True)
    fig.suptitle(f"Multi-Sequence EKF Trajectories | Mean RMSE_XY={np.mean(rmse_list):.3f}m", fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("inference_ekf_multi_result.png", dpi=150)
    print(f"\nSaved multi-sequence result to inference_ekf_multi_result.png")
    print(f"Overall Mean RMSE_XY: {np.mean(rmse_list):.3f}m")

if __name__ == "__main__":
    plot_multi_ekf(MODEL_PATH, DATA_ROOT, NORM_MEAN_PATH, NORM_STD_PATH, max_files=4)
