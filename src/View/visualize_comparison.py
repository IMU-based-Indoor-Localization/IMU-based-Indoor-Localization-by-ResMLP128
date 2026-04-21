"""
Network 단독 궤적 / EKF 필터 궤적 / Ground Truth 궤적 3가지를 한 화면에 시각화.

실행 예시 (src/ 폴더에서):
    python View/visualize_comparison.py \
        --data_path  TLIO_Oxford_Dataset/oxford_handbag_1/imu0_resampled.npy \
        --model_path outputs/out_classifier2/checkpoints/best.pth \
        --norm_mean  outputs/out_classifier2/norm_mean.npy \
        --norm_std   outputs/out_classifier2/norm_std.npy
"""

import sys
from pathlib import Path

# ---------- 경로 설정 ----------
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "Network"))
sys.path.insert(0, str(_SRC / "Trans"))
sys.path.insert(0, str(_SRC))           # tracker/, utils/ 검색

import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from dataset import TLIONpySingleDataset
from model_twolayer import TwoLayerModel
from tracker.scekf import ImuMSCKF

G_NORM = 9.81  # m/s²


# ---------------------------------------------------------------------------
# 모델 로딩
# ---------------------------------------------------------------------------
def load_model(model_path: str, device: torch.device,
               window_len: int = 100, patch_len: int = 10) -> TwoLayerModel:
    model_para = {
        "input_len": window_len, "input_channel": 6, "patch_len": patch_len,
        "feature_dim": 128, "out_dim": 3, "active_func": "GELU",
        "extractor": {"name": "ResMLP", "layer_num": 6, "expansion": 2, "dropout": 0.0},
        "reg":       {"name": "PoseCondMean", "layer_num": 3, "dropout": 0.0},
        "classifier": {"num_classes": 7, "layer_num": 2, "dropout": 0.0, "pooling_type": "mean"},
        "use_classifier": True,
    }
    model = TwoLayerModel(model_para).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# npy 파일에서 원시 IMU 복원
# Oxford 포맷: acc(4:7)은 선형가속도(중력 제거), gravity(7:10)는 body 좌표 중력 단위벡터
# raw_acc = linear_acc + gravity_unit * G_NORM
# ---------------------------------------------------------------------------
def load_npy(data_path: str):
    data = np.load(data_path)
    ts_us    = data[:, 0].astype(np.int64)
    gyr      = data[:, 1:4].astype(np.float64)   # [T, 3] rad/s
    acc_lin  = data[:, 4:7].astype(np.float64)   # [T, 3] linear acc
    grav_b   = data[:, 7:10].astype(np.float64)  # [T, 3] gravity unit vec (body)
    quat     = data[:, 14:18].astype(np.float64) # [T, 4] xyzw world←device
    pos_gt   = data[:, 18:21].astype(np.float64) # [T, 3]
    vel_gt   = data[:, 21:24].astype(np.float64) # [T, 3]

    acc_raw = acc_lin + grav_b * G_NORM           # [T, 3] raw accelerometer
    return ts_us, gyr, acc_raw, quat, pos_gt, vel_gt


# ---------------------------------------------------------------------------
# Network 단독 궤적 (dead-reckoning)
# inference_plot.py 와 동일한 로직
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_network(model: TwoLayerModel, data_path: str, device: torch.device,
                norm_mean: np.ndarray, norm_std: np.ndarray,
                window_len: int, stride: int):
    dataset = TLIONpySingleDataset(
        npy_path=data_path, window_len=window_len, stride=stride,
        normalize=True, precomputed_stats=(norm_mean, norm_std),
        is_train=False, fmt="oxford", with_label=False,
    )

    data    = np.load(data_path)
    quat    = data[:, 14:18].astype(np.float32)
    pos_gt  = data[:, 18:21].astype(np.float32)

    preds_local = []
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    for imu, _ in loader:
        y_hat, _, _ = model(imu.to(device))
        preds_local.append(y_hat.cpu().numpy())
    preds_local = np.concatenate(preds_local, axis=0)  # [K, 3]

    anchor_gt, pred_world_steps = [], []
    for k, start in enumerate(dataset.indices):
        end    = start + window_len
        r_s    = R.from_quat(quat[start])
        yaw    = r_s.as_euler("zyx", degrees=False)[0]
        R_yaw  = R.from_euler("z", yaw).as_matrix().astype(np.float32)
        pred_world_steps.append(R_yaw @ preds_local[k])
        anchor_gt.append(pos_gt[end])

    pred_world_steps = np.array(pred_world_steps)
    anchor_gt        = np.array(anchor_gt)

    net_pos = [pos_gt[0].copy()]
    for step in pred_world_steps:
        net_pos.append(net_pos[-1] + step)
    net_pos = np.array(net_pos, dtype=np.float32)

    return pos_gt, net_pos, pred_world_steps, anchor_gt, dataset.indices


# ---------------------------------------------------------------------------
# EKF 실행 (ImuMSCKF 직접 구동)
# ---------------------------------------------------------------------------
def run_ekf(ts_us, gyr, acc_raw, quat, pos_gt, vel_gt,
            pred_world_steps, window_indices,
            log_vars: np.ndarray, window_len: int):
    """
    pred_world_steps : [K, 3]  network가 예측한 world 좌표 변위
    log_vars         : [K, 3]  network가 출력한 log-variance (불확실도)
    window_indices   : dataset.indices (각 윈도우 시작 샘플 인덱스)
    """
    ekf = ImuMSCKF(config=None)

    # GT 초기 상태로 EKF 초기화
    R0  = R.from_quat(quat[0]).as_matrix()
    v0  = vel_gt[0].reshape(3, 1)
    p0  = pos_gt[0].reshape(3, 1)
    ekf.initialize_with_state(
        int(ts_us[0]), R0, v0, p0,
        np.zeros((3, 1)), np.zeros((3, 1)),
    )

    # 윈도우 augmentation 트리거:
    # propagate(i=start_idx+1)에서 t_aug=ts_us[start_idx]로 호출하면
    # 부분 적분(dtd=0)으로 ts_us[start_idx]가 si_timestamps_us에 등록된다.
    # → update()의 t_begin_us=ts_us[start_idx]와 정확히 일치.
    aug_trigger = {idx + 1: (k, idx) for k, idx in enumerate(window_indices)}
    win_end     = {idx + window_len: k for k, idx in enumerate(window_indices)}

    ekf_positions = []
    T = len(ts_us)

    for i in range(1, T):
        t_us_i = int(ts_us[i])

        # 윈도우 시작 직후 첫 propagate에서 윈도우 시작 시각으로 augmentation
        t_aug = None
        if i in aug_trigger:
            _, start_idx = aug_trigger[i]
            t_aug = int(ts_us[start_idx])

        ekf.propagate(
            acc_raw[i].reshape(3, 1),
            gyr[i].reshape(3, 1),
            t_us_i,
            t_augmentation_us=t_aug,
        )

        # 윈도우 끝 → network 측정값으로 update
        if i in win_end:
            k          = win_end[i]
            start_idx  = window_indices[k]
            t_begin_us = int(ts_us[start_idx])
            t_end_us   = t_us_i

            meas = pred_world_steps[k].reshape(3, 1).astype(np.float64)

            # log-var → covariance (world 좌표 회전 포함)
            var_local = np.exp(log_vars[k])
            cov_local = np.diag(var_local.astype(np.float64))
            r_s       = R.from_quat(quat[start_idx])
            yaw       = r_s.as_euler("zyx", degrees=False)[0]
            R_yaw     = R.from_euler("z", yaw).as_matrix()
            meas_cov  = R_yaw @ cov_local @ R_yaw.T

            try:
                ekf.update(meas, meas_cov, t_begin_us, t_end_us)
                # 이전 cloned state 정리 (가장 오래된 것 제거)
                if ekf.state.N > 1:
                    ekf.marginalize(0)
            except Exception:
                pass  # update 실패 시 propagation 결과만 사용

        _, _, p_ekf, _, _ = ekf.get_evolving_state()
        ekf_positions.append(p_ekf.flatten().copy())

    return np.array(ekf_positions, dtype=np.float32)


# ---------------------------------------------------------------------------
# 로그 분산 수집 (EKF 공분산 계산용)
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_log_vars(model: TwoLayerModel, data_path: str, device: torch.device,
                     norm_mean: np.ndarray, norm_std: np.ndarray,
                     window_len: int, stride: int):
    dataset = TLIONpySingleDataset(
        npy_path=data_path, window_len=window_len, stride=stride,
        normalize=True, precomputed_stats=(norm_mean, norm_std),
        is_train=False, fmt="oxford", with_label=False,
    )
    from torch.utils.data import DataLoader
    loader    = DataLoader(dataset, batch_size=256, shuffle=False)
    log_vars  = []
    for imu, _ in loader:
        _, lv, _ = model(imu.to(device))
        log_vars.append(lv.cpu().numpy())
    return np.concatenate(log_vars, axis=0)  # [K, 3]


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def plot_comparison(gt_pos, net_pos, ekf_pos, win_indices, window_len,
                    title: str, save_path: str = None):
    """
    gt_pos   : [T, 3]  GT (100Hz 전체)
    net_pos  : [K+1, 3]  Network dead-reckoning (윈도우 단위)
    ekf_pos  : [T-1, 3]  EKF (100Hz, ts[1]~ts[-1])
    win_indices : 각 윈도우 시작 샘플 인덱스 (길이 K)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    origin = gt_pos[0].copy()

    # --- XY 궤적: 각자 자연스러운 샘플링 그대로 표시 ---
    gt_xy  = gt_pos[:, :2]  - origin[:2]
    net_xy = net_pos[:, :2] - origin[:2]
    ekf_xy = ekf_pos[:, :2] - origin[:2]

    ax = axes[0]
    ax.plot(gt_xy[:,0],  gt_xy[:,1],  lw=2.0, label="GT",     alpha=0.8,  color="C0")
    ax.plot(net_xy[:,0], net_xy[:,1], lw=1.8, label="Network", alpha=0.85, color="C1", linestyle="--")
    ax.plot(ekf_xy[:,0], ekf_xy[:,1], lw=1.8, label="EKF",     alpha=0.85, color="C2", linestyle="-.")
    ax.scatter(0, 0, s=50, marker="o", zorder=5, color="k", label="Start")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(); ax.grid(alpha=0.3)

    # --- 오차: 윈도우 앵커(윈도우 끝 샘플)에서만 비교 ---
    # anchor_idxs[k] = win_indices[k] + window_len (윈도우 끝 샘플 인덱스)
    anchor_idxs = np.array([idx + window_len for idx in win_indices])
    valid = anchor_idxs < len(gt_pos)
    anchor_idxs = anchor_idxs[valid]
    K = len(anchor_idxs)

    gt_anchor  = gt_pos[anchor_idxs, :2]  - origin[:2]   # [K, 2]
    net_anchor = net_pos[1:K+1,     :2]   - origin[:2]   # [K, 2]
    # ekf_pos[i] = state at ts[i+1], so ekf_pos[anchor_idx-1] = state at ts[anchor_idx]
    ekf_anchor_idxs = np.clip(anchor_idxs - 1, 0, len(ekf_pos) - 1)
    ekf_anchor = ekf_pos[ekf_anchor_idxs, :2] - origin[:2]  # [K, 2]

    err_net_xy = np.sqrt(np.sum((net_anchor - gt_anchor)**2, axis=1))
    err_ekf_xy = np.sqrt(np.sum((ekf_anchor - gt_anchor)**2, axis=1))
    t_anchors  = anchor_idxs / 100.0  # 100Hz → 초

    ax = axes[1]
    ax.plot(t_anchors, err_net_xy, lw=1.5, label="Network XY err", color="C1", linestyle="--")
    ax.plot(t_anchors, err_ekf_xy, lw=1.5, label="EKF XY err",     color="C2", linestyle="-.")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("XY Error (m)")
    ax.set_title("Positional Error over Time (window anchors)")
    ax.legend(); ax.grid(alpha=0.3)

    rmse_net = float(np.sqrt(np.mean(err_net_xy**2))) if K > 0 else float("nan")
    rmse_ekf = float(np.sqrt(np.mean(err_ekf_xy**2))) if K > 0 else float("nan")
    fig.suptitle(
        f"{title}\n"
        f"RMSE_XY — Network: {rmse_net:.3f} m | EKF: {rmse_ekf:.3f} m",
        fontsize=12
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"저장: {save_path}")
    plt.show()

    return rmse_net, rmse_ekf


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path",  required=True,
                    help="imu0_resampled.npy 경로")
    ap.add_argument("--model_path", required=True,
                    help="체크포인트 .pth 경로")
    ap.add_argument("--norm_mean",  required=True,
                    help="norm_mean.npy 경로")
    ap.add_argument("--norm_std",   required=True,
                    help="norm_std.npy 경로")
    ap.add_argument("--window_len", type=int, default=100)
    ap.add_argument("--stride",     type=int, default=100)
    ap.add_argument("--save",       type=str, default=None,
                    help="그래프 저장 경로 (없으면 화면 출력만)")
    args = ap.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_mean = np.load(args.norm_mean)
    norm_std  = np.load(args.norm_std)

    print(f"모델 로드: {args.model_path}")
    model = load_model(args.model_path, device, args.window_len)

    print(f"데이터 로드: {args.data_path}")
    ts_us, gyr, acc_raw, quat, pos_gt, vel_gt = load_npy(args.data_path)

    print("Network 추론 중...")
    gt_pos, net_pos, pred_steps, _, win_indices = run_network(
        model, args.data_path, device, norm_mean, norm_std,
        args.window_len, args.stride,
    )
    log_vars = collect_log_vars(
        model, args.data_path, device, norm_mean, norm_std,
        args.window_len, args.stride,
    )

    print("EKF 실행 중...")
    ekf_pos = run_ekf(
        ts_us, gyr, acc_raw, quat, pos_gt, vel_gt,
        pred_steps, win_indices, log_vars, args.window_len,
    )

    title = Path(args.data_path).parent.name
    save_path = args.save or f"comparison_{title}.png"
    rmse_net, rmse_ekf = plot_comparison(
        gt_pos, net_pos, ekf_pos,
        win_indices, args.window_len,
        title, save_path,
    )
    print(f"\nRMSE_XY  Network: {rmse_net:.3f} m  |  EKF: {rmse_ekf:.3f} m")


if __name__ == "__main__":
    main()
