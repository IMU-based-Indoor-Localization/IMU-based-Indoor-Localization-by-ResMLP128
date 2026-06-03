"""
visualize_batch_short.py
------------------------
GT / Network / EKF 세 궤적을 카테고리별로 일괄 시각화.
MAX_SAMPLES 샘플(기본 500 = 5초 @ 100Hz)만 사용하여 짧은 궤적 표시.

실행 (src/ 폴더에서):
    python View/visualize_batch_short.py
    python View/visualize_batch_short.py --max_samples 300 --categories handbag trolley

출력: doc/short_traj/{category}_{seq_name}.png
"""

import sys, os, argparse
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parent.parent          # src/
sys.path.insert(0, str(_SRC / "Network"))
sys.path.insert(0, str(_SRC / "Trans"))
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # View/ (visualize_comparison 임포트용)

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from visualize_comparison import load_model, run_ekf_imutracker

# ── 기본 경로 설정 ──────────────────────────────────────────────────────
_ROOT       = _SRC.parent
MODEL_DIR   = _SRC / "outputs" / "out_classifier2"
DATASET_DIR = _SRC / "TLIO_Oxford_Dataset"
OUTPUT_DIR  = _ROOT / "doc" / "short_traj"

DEFAULT_CATEGORIES = ["handbag", "handheld", "trolley"]
G_NORM = 9.81


# ── 시퀀스 탐색 ──────────────────────────────────────────────────────────
def find_sequence(category: str) -> Path:
    """카테고리 이름이 포함된 첫 번째 시퀀스 디렉터리를 반환."""
    for d in sorted(DATASET_DIR.iterdir()):
        if d.is_dir() and category in d.name and "(NO)" not in d.name:
            npy = d / "imu0_resampled.npy"
            if npy.exists():
                return npy
    return None


# ── Oxford npy 파싱 (배열 직접 수신) ────────────────────────────────────
def parse_oxford(data: np.ndarray):
    """24컬럼 Oxford 배열을 구성 요소로 분해."""
    ts_us   = data[:, 0].astype(np.int64)
    gyr     = data[:, 1:4].astype(np.float64)    # body 프레임 각속도
    acc_lin = data[:, 4:7].astype(np.float64)    # body 프레임 선형가속도 (중력 제거됨)
    grav_b  = data[:, 7:10].astype(np.float64)   # body 프레임 중력 단위벡터
    quat    = data[:, 14:18].astype(np.float64)  # xyzw  world←device
    pos_gt  = data[:, 18:21].astype(np.float64)
    vel_gt  = data[:, 21:24].astype(np.float64)
    acc_raw = acc_lin - grav_b * G_NORM           # specific force (가속도계 raw 값)
    return ts_us, gyr, acc_raw, quat, pos_gt, vel_gt


# ── Network 추론 (in-memory, Oxford 포맷) ────────────────────────────────
@torch.no_grad()
def run_network_from_array(model, data_sliced: np.ndarray,
                           norm_mean: np.ndarray, norm_std: np.ndarray,
                           device: torch.device,
                           window_len: int = 100, stride: int = 100):
    """
    슬라이싱된 Oxford 배열에서 네트워크 dead-reckoning 궤적 계산.

    전처리: body 프레임 → world 프레임 → yaw 정규화 (visualize_comparison.run_network 와 동일)
    반환:
        gt_pos       [T, 3]    GT 위치
        net_pos      [K+1, 3]  누적 예측 위치 (윈도우 단위)
        pred_steps   [K, 3]    각 윈도우 예측 변위
        win_indices  [K]       각 윈도우 시작 샘플 인덱스
    """
    gyr_body = data_sliced[:, 1:4].astype(np.float32)
    acc_body = data_sliced[:, 4:7].astype(np.float32)   # linear acc, body frame
    quat     = data_sliced[:, 14:18].astype(np.float32)
    pos_gt   = data_sliced[:, 18:21].astype(np.float32)
    T = len(acc_body)

    indices = list(range(0, T - window_len, stride))
    pred_world_steps = []

    for start in indices:
        end = start + window_len
        R_all     = R.from_quat(quat[start:end]).as_matrix().astype(np.float32)
        r_start   = R.from_quat(quat[start])
        yaw0      = r_start.as_euler("zyx", degrees=False)[0]
        R_yaw_inv = R.from_euler("z", yaw0).inv().as_matrix().astype(np.float32)

        # body → world → yaw 정규화
        acc_w  = np.einsum("tij,tj->ti", R_all, acc_body[start:end])
        gyr_w  = np.einsum("tij,tj->ti", R_all, gyr_body[start:end])
        acc_ga = (R_yaw_inv @ acc_w.T).T
        gyr_ga = (R_yaw_inv @ gyr_w.T).T
        imu_np = np.concatenate([acc_ga, gyr_ga], axis=1)  # [window_len, 6]

        feat = (imu_np - norm_mean) / norm_std
        x    = torch.from_numpy(feat.T[None]).to(device)   # [1, 6, window_len]
        y_hat, _, _ = model(x)
        pred_local  = y_hat[0].cpu().numpy()                # [3] yaw 정규화 프레임

        R_yaw = R.from_euler("z", yaw0).as_matrix().astype(np.float32)
        pred_world_steps.append(R_yaw @ pred_local)         # world 프레임으로 복원

    pred_world_steps = np.array(pred_world_steps, dtype=np.float32)  # [K, 3]

    net_pos = [pos_gt[0].copy()]
    for step in pred_world_steps:
        net_pos.append(net_pos[-1] + step)

    return pos_gt, np.array(net_pos, dtype=np.float32), pred_world_steps, np.array(indices)


# ── 시각화 ───────────────────────────────────────────────────────────────
def plot_short(gt_pos, net_pos, ekf_pos, win_indices, window_len,
               title: str, save_path: Path):
    """
    gt_pos      [T, 3]    GT (100Hz 전체 슬라이스)
    net_pos     [K+1, 3]  Network dead-reckoning
    ekf_pos     [T-1, 3]  EKF (ts[1]~ts[-1])
    win_indices [K]       윈도우 시작 인덱스
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    origin = gt_pos[0].copy()

    # ── 왼쪽: XY 궤적 ──────────────────────────────────────────────────
    gt_xy  = gt_pos[:, :2]  - origin[:2]
    net_xy = net_pos[:, :2] - origin[:2]
    ekf_xy = ekf_pos[:, :2] - origin[:2]

    ax = axes[0]
    ax.plot(gt_xy[:,0],  gt_xy[:,1],  lw=2.2, label="GT",      alpha=0.9,  color="C0")
    ax.plot(net_xy[:,0], net_xy[:,1], lw=1.8, label="Network", alpha=0.85, color="C1", linestyle="--")
    ax.plot(ekf_xy[:,0], ekf_xy[:,1], lw=1.8, label="EKF",     alpha=0.85, color="C2", linestyle="-.")
    ax.scatter(0, 0, s=70, marker="o", zorder=5, color="k", label="Start")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best"); ax.grid(alpha=0.3)

    # ── 오른쪽: 위치 오차 (윈도우 앵커 기준) ───────────────────────────
    anchor_idxs = np.array([idx + window_len for idx in win_indices])
    valid = anchor_idxs < len(gt_pos)
    anchor_idxs = anchor_idxs[valid]
    K = len(anchor_idxs)

    rmse_net = rmse_ekf = float("nan")
    if K > 0:
        gt_anchor  = gt_pos[anchor_idxs, :2] - origin[:2]
        net_anchor = net_pos[1:K+1, :2]      - origin[:2]
        ekf_anchor_idxs = np.clip(anchor_idxs - 1, 0, len(ekf_pos) - 1)
        ekf_anchor = ekf_pos[ekf_anchor_idxs, :2] - origin[:2]

        err_net = np.sqrt(np.sum((net_anchor - gt_anchor)**2, axis=1))
        err_ekf = np.sqrt(np.sum((ekf_anchor - gt_anchor)**2, axis=1))
        t_ax    = anchor_idxs / 100.0  # 100Hz → 초

        ax2 = axes[1]
        ax2.plot(t_ax, err_net, lw=1.8, label="Network XY err", color="C1", linestyle="--")
        ax2.plot(t_ax, err_ekf, lw=1.8, label="EKF XY err",     color="C2", linestyle="-.")
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("XY Error (m)")
        ax2.set_title("Positional Error (window anchors)")
        ax2.legend(); ax2.grid(alpha=0.3)

        rmse_net = float(np.sqrt(np.mean(err_net**2)))
        rmse_ekf = float(np.sqrt(np.mean(err_ekf**2)))

    fig.suptitle(
        f"{title}\n"
        f"RMSE_XY -- Network: {rmse_net:.3f} m | EKF: {rmse_ekf:.3f} m",
        fontsize=12,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {save_path}")
    return rmse_net, rmse_ekf


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="GT/Network/EKF 짧은 궤적 일괄 시각화")
    ap.add_argument("--max_samples",  type=int, default=500,
                    help="사용할 샘플 수 (기본 500 = 5초 @ 100Hz)")
    ap.add_argument("--categories",   nargs="+", default=DEFAULT_CATEGORIES,
                    help="시각화할 카테고리 목록 (기본: handbag handheld trolley)")
    ap.add_argument("--model_dir",    type=str, default=str(MODEL_DIR),
                    help="모델 디렉터리 (checkpoints/best.pth, norm_*.npy 포함)")
    ap.add_argument("--output_dir",   type=str, default=str(OUTPUT_DIR))
    args = ap.parse_args()

    mdl_dir  = Path(args.model_dir)
    out_dir  = Path(args.output_dir)
    max_samp = args.max_samples

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_mean = np.load(mdl_dir / "norm_mean.npy")
    norm_std  = np.load(mdl_dir / "norm_std.npy")

    print(f"Model: {mdl_dir / 'checkpoints/best.pth'}")
    model = load_model(str(mdl_dir / "checkpoints/best.pth"), device)
    print(f"device: {device}")
    print(f"Max samples: {max_samp} ({max_samp/100:.1f}s @ 100Hz)\n")

    summary = []
    for cat in args.categories:
        npy_path = find_sequence(cat)
        if npy_path is None:
            print(f"[{cat}] sequence not found, skipping\n")
            continue

        seq_name = npy_path.parent.name
        print(f"[{cat}]  {seq_name}")

        data_full = np.load(npy_path)
        data = data_full[:max_samp]
        print(f"  total {len(data_full):,} -> {len(data)} samples")

        ts_us, gyr, acc_raw, quat, pos_gt, vel_gt = parse_oxford(data)

        print("  Running network inference...")
        gt_pos, net_pos, pred_steps, win_indices = run_network_from_array(
            model, data, norm_mean, norm_std, device,
            window_len=100, stride=100,
        )

        print("  Running EKF...")
        ekf_pos = run_ekf_imutracker(
            ts_us, gyr, acc_raw, quat, pos_gt, vel_gt,
            model, norm_mean, norm_std, window_len=100,
        )

        title     = f"{cat} -- {seq_name}  (first {max_samp} samples / {max_samp/100:.0f}s)"
        save_path = out_dir / f"{cat}_{seq_name}.png"
        rmse_net, rmse_ekf = plot_short(
            gt_pos, net_pos, ekf_pos, win_indices, 100, title, save_path
        )
        print(f"  RMSE_XY -- Network: {rmse_net:.3f} m | EKF: {rmse_ekf:.3f} m\n")
        summary.append((cat, seq_name, rmse_net, rmse_ekf))

    # 결과 요약
    print("=" * 60)
    print(f"{'Category':<14} {'Sequence':<28} {'Net':>8} {'EKF':>8}")
    print("-" * 60)
    for cat, seq, rn, re in summary:
        print(f"{cat:<14} {seq:<28} {rn:>7.3f}m {re:>7.3f}m")
    print("=" * 60)
    print(f"\n[Done] Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
