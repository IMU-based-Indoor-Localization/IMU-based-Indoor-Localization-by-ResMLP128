"""
visualize_results.py
--------------------
학습된 ResMLP128 모델의 예측 경로 vs GT 경로 시각화.
Oxford 포즈 카테고리별로 대표 시퀀스 1개씩 플롯.

출력:
  - out_regression/viz/{category}.png
    Left:  x,y 평면 경로 (GT vs Pred)
    Right: z 방향 누적 변위 (GT vs Pred)
"""

import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Trans"))

from model_twolayer import TwoLayerModel

# ─── 설정 ─────────────────────────────────────────────────────────
MODEL_DIR  = Path("out_regression")
OXFORD_DIR = Path("../TLIO_Oxford_Dataset")
OUTPUT_DIR = MODEL_DIR / "viz"
WINDOW_LEN = 100

CATEGORIES = [
    "handbag",
    "handheld",
    "large_scale",
    "multi_devices",
    "multi_users",
    "pocket",
    "running",
    "slow_walking",
    "trolley",
]

COLORS = {
    "gt":   ("#2c7bb6", "o", "-",  1.8),
    "pred": ("#d7191c", "s", "--", 1.8),
}
# ──────────────────────────────────────────────────────────────────


def load_model(model_dir: Path, device):
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    model = TwoLayerModel(cfg["model"]).to(device)
    ck = torch.load(model_dir / "checkpoints/best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    mean = np.load(model_dir / "norm_mean.npy")
    std  = np.load(model_dir / "norm_std.npy")
    return model, mean, std


def window_to_gravity_aligned(acc_body, gyr_body, quat_data, start, end):
    """body-frame IMU 윈도우 → gravity-aligned local frame (acc+gyro) + 시작 yaw 반환."""
    R_all     = R.from_quat(quat_data[start:end]).as_matrix().astype(np.float32)
    r_start   = R.from_quat(quat_data[start])
    yaw0      = r_start.as_euler("zyx", degrees=False)[0]
    R_yaw_inv = R.from_euler("z", yaw0).inv().as_matrix().astype(np.float32)

    acc_w = np.einsum("tij,tj->ti", R_all, acc_body[start:end])
    gyr_w = np.einsum("tij,tj->ti", R_all, gyr_body[start:end])
    acc_ga = (R_yaw_inv @ acc_w.T).T
    gyr_ga = (R_yaw_inv @ gyr_w.T).T
    imu    = np.concatenate([acc_ga, gyr_ga], axis=1).astype(np.float32)
    return imu, yaw0


@torch.no_grad()
def reconstruct_trajectory(npy_path, model, mean, std, device, window_len=100):
    """
    비겹침 윈도우로 변위를 예측하고 누적하여 경로를 재구성합니다.
    GT와 Pred 모두 (0,0,0) 시작 기준.
    """
    data = np.load(npy_path)
    # Oxford 24-col: ts(1) gyr(3) acc(3) grav(3) att(3) lbl(1) quat(4) pos(3) vel(3)
    gyr_body  = data[:, 1:4].astype(np.float32)
    acc_body  = data[:, 4:7].astype(np.float32)
    quat_data = data[:, 14:18].astype(np.float32)
    pos_data  = data[:, 18:21].astype(np.float32)

    T = len(acc_body)
    gt_pts   = [np.zeros(3, dtype=np.float32)]
    pred_pts = [np.zeros(3, dtype=np.float32)]

    for start in range(0, T - window_len, window_len):   # 비겹침 윈도우
        end = start + window_len

        # ── GT 누적 (world frame 변위 그대로)
        gt_delta = pos_data[end] - pos_data[start]
        gt_pts.append(gt_pts[-1] + gt_delta)

        # ── Prediction
        imu_np, yaw0 = window_to_gravity_aligned(acc_body, gyr_body, quat_data, start, end)
        imu_norm = (imu_np - mean) / std
        imu_t    = torch.tensor(imu_norm.T[None], dtype=torch.float32, device=device)

        pred_local, _, _ = model(imu_t)
        pred_local = pred_local[0].cpu().numpy()               # [3] local (yaw 제거)

        # local → world (시작 yaw 복원)
        R_yaw      = R.from_euler("z", yaw0).as_matrix().astype(np.float32)
        pred_world = R_yaw @ pred_local
        pred_pts.append(pred_pts[-1] + pred_world)

    gt_arr   = np.array(gt_pts,   dtype=np.float32)
    pred_arr = np.array(pred_pts, dtype=np.float32)
    return gt_arr, pred_arr


def plot_trajectory(category, seq_name, gt, pred, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#fafafa")

    # ── Left: X-Y 평면 경로 ─────────────────────────────────────
    ax = axes[0]
    ax.plot(gt[:,0],   gt[:,1],
            color=COLORS["gt"][0],   marker=COLORS["gt"][1],
            linestyle=COLORS["gt"][2], linewidth=COLORS["gt"][3],
            markersize=3, label="GT")
    ax.plot(pred[:,0], pred[:,1],
            color=COLORS["pred"][0], marker=COLORS["pred"][1],
            linestyle=COLORS["pred"][2], linewidth=COLORS["pred"][3],
            markersize=3, label="Pred")

    ax.plot(*gt[0, :2],   "k^", markersize=9,  zorder=6, label="Start")
    ax.plot(*gt[-1, :2],  "k*", markersize=11, zorder=6, label="End(GT)")
    ax.plot(*pred[-1, :2],"rx", markersize=9,  zorder=6, label="End(Pred)")

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(f"[{category}]  X–Y Trajectory\n{seq_name}", fontsize=11)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    ax.set_facecolor("#f7f7f7")

    # ── Right: Z 방향 누적 변위 ──────────────────────────────────
    ax2 = axes[1]
    t = np.arange(len(gt))
    ax2.plot(t, gt[:,2],
             color=COLORS["gt"][0],   marker=COLORS["gt"][1],
             linestyle=COLORS["gt"][2], linewidth=COLORS["gt"][3],
             markersize=3, label="GT z")
    ax2.plot(t, pred[:,2],
             color=COLORS["pred"][0], marker=COLORS["pred"][1],
             linestyle=COLORS["pred"][2], linewidth=COLORS["pred"][3],
             markersize=3, label="Pred z")

    ax2.set_xlabel("Window index (×1 sec)", fontsize=11)
    ax2.set_ylabel("Cumulative Z displacement (m)", fontsize=11)
    ax2.set_title(f"[{category}]  Z Direction", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.35)
    ax2.set_facecolor("#f7f7f7")

    # ── RMSE
    err  = gt - pred
    rmse = float(np.sqrt((err ** 2).sum(1).mean()))
    rmse_xy = float(np.sqrt((err[:, :2] ** 2).sum(1).mean()))
    fig.suptitle(
        f"Oxford  [{category}]   RMSE={rmse:.3f} m  (xy={rmse_xy:.3f} m)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    out_path = output_dir / f"{category}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")
    return rmse, rmse_xy


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model, mean, std = load_model(MODEL_DIR, device)
    print(f"모델 로드 완료 (best.pth)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for cat in CATEGORIES:
        # 카테고리에 해당하는 첫 번째 유효 시퀀스 선택
        candidates = sorted([
            d for d in OXFORD_DIR.iterdir()
            if d.is_dir() and cat in d.name and "(NO)" not in d.name
        ])
        if not candidates:
            print(f"[{cat}] 시퀀스 없음, 스킵")
            continue

        npy_path = candidates[0] / "imu0_resampled.npy"
        if not npy_path.exists():
            print(f"[{cat}] npy 없음: {npy_path}")
            continue

        seq_name = candidates[0].name
        print(f"\n[{cat}]  {seq_name}")

        gt, pred = reconstruct_trajectory(npy_path, model, mean, std, device, WINDOW_LEN)
        rmse, rmse_xy = plot_trajectory(cat, seq_name, gt, pred, OUTPUT_DIR)
        results.append((cat, seq_name, rmse, rmse_xy, len(gt)))
        print(f"  RMSE={rmse:.4f}m  xy={rmse_xy:.4f}m  ({len(gt)} 윈도우)")

    # ── 결과 요약
    print("\n" + "=" * 70)
    print(f"{'Category':<18} {'Sequence':<38} {'RMSE':>8} {'RMSE_xy':>9}")
    print("-" * 70)
    for cat, seq, rmse, rmse_xy, n in results:
        print(f"{cat:<18} {seq:<38} {rmse:>7.4f}m {rmse_xy:>8.4f}m")
    print("=" * 70)
    print(f"\n모든 시각화 저장 완료: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
