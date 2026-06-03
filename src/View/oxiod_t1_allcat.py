# -*- coding: utf-8 -*-
"""T1 — positive 일반화: 전 카테고리에서 EKF(드리프트 yaw) vs RotVec-DR(절대 yaw).
§4.4 스택(out_regression + ImuMSCKF, 카테고리별 best meascov_scale) 그대로.
각 카테고리: RotVec-DR(Net) ATE, EKF ATE, 비율, EKF 누적 yaw(말기평균·최대).
→ RotVec가 일관 회복·EKF가 일관 예산(~10°) 초과를 입증.

실행: cd src/View; KMP_DUPLICATE_LIB_OK=TRUE python oxiod_t1_allcat.py
"""
import os, sys, logging
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
logging.disable(logging.WARNING)
from pathlib import Path
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.spatial.transform import Rotation as R
_HERE = Path(__file__).resolve().parent; _SRC = _HERE.parent
for p in (_SRC / "Network", _SRC / "Trans", _SRC, _HERE):
    sys.path.insert(0, str(p))
for fn in ("Malgun Gothic", "맑은 고딕", "Gulim"):
    try:
        font_manager.findfont(fn, fallback_to_default=False); plt.rcParams["font.family"] = fn; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
from visualize_comparison import load_npy, run_network            # type: ignore
from oxiod_ekf_vs_rotvec_v2 import load_reg, ekf_with_yaw, rmse_anchor  # type: ignore

DATA = Path(r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
MODEL_DIR = _SRC / "Network" / "out_regression"
WL = 100
# (cat, seq, state_id, best meascov_scale)  — §4.4 표와 동일
CATS = [("handbag", "oxford_handbag_1", 1, 0.001),
        ("handheld", "oxford_handheld_1", 2, 0.001),
        ("pocket", "oxford_pocket_1", 3, 0.001),
        ("running", "oxford_running_1", 4, 0.01),
        ("slow", "oxford_slow_walking_1", 5, 1.0),
        ("trolley", "oxford_trolley_1", 6, 0.001),
        ("large", "oxford_large_scale_1", 9, 0.1)]
LOG = _SRC.parent / "logs"


def cum_yaw(ekf_yaw, quat):
    """EKF yaw 추정 오차(vs GT). wrap [-180,180] 로 진동 누적 버그 제거.
    반환: 후반 50% mean|err|, max|err| (둘 다 bounded)."""
    sub = np.arange(0, len(ekf_yaw), 5)
    gy = np.array([R.from_quat(quat[min(i + 1, len(quat) - 1)]).as_euler("zyx")[0] for i in sub])
    off = gy[0] - ekf_yaw[sub][0]
    raw = ekf_yaw[sub] + off - gy
    err = np.degrees((raw + np.pi) % (2 * np.pi) - np.pi)        # wrap
    late = np.abs(err[len(err) // 2:])
    return float(np.mean(late)), float(np.max(np.abs(err)))


def main():
    dev = torch.device("cpu"); model, mean, std = load_reg(MODEL_DIR, dev)
    rows = []
    for cat, seq, sid, scale in CATS:
        npy = DATA / seq / "imu0_resampled.npy"
        ts, gyr, acc, quat, pos, vel = load_npy(str(npy))
        _g, net_pos, _p, anchor_gt, win = run_network(model, str(npy), dev, mean, std, WL, WL)
        K = min(len(anchor_gt), len(net_pos) - 1)
        rmse_net = float(np.sqrt(np.mean(np.sum((net_pos[1:K + 1, :2] - anchor_gt[:K, :2]) ** 2, axis=1))))
        ekf_xy, ekf_yaw = ekf_with_yaw(ts, gyr, acc, quat, pos, vel, model, mean, std, sid, scale)
        rmse_ekf = rmse_anchor(ekf_xy, pos, list(win))
        cy, mx = cum_yaw(ekf_yaw, quat)
        rows.append((cat, rmse_net, rmse_ekf, rmse_ekf / rmse_net, cy, mx))
        print(f"{cat:<10} Net {rmse_net:5.2f}  EKF {rmse_ekf:6.2f} ({rmse_ekf/rmse_net:4.1f}x)  "
              f"EKF누적yaw {cy:4.0f}° max {mx:4.0f}°")

    print("\n[종합]")
    ratios = [r[3] for r in rows]
    print(f"  RotVec-DR가 EKF보다 우수: {sum(1 for x in ratios if x>1)}/{len(rows)} 카테고리")
    print(f"  EKF 누적 yaw > 10°: {sum(1 for r in rows if r[4]>10)}/{len(rows)} 카테고리")
    print(f"  ATE 비율 기하평균 EKF/RotVec = {float(np.exp(np.mean(np.log(ratios)))):.2f}x")

    # ── figure (2-panel) ──
    cats = [r[0] for r in rows]; x = np.arange(len(cats))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    w = 0.38
    ax[0].bar(x - w/2, [r[1] for r in rows], w, label="RotVec-DR (절대 yaw)", color="#1f77b4", edgecolor="black")
    ax[0].bar(x + w/2, [r[2] for r in rows], w, label="EKF (드리프트 yaw)", color="#d62728", edgecolor="black")
    ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(cats, rotation=30, ha="right", fontsize=8.5)
    ax[0].set_ylabel("ATE RMSE$_{xy}$ (m, log)"); ax[0].legend(fontsize=9)
    ax[0].set_title("(a) 카테고리별 ATE — RotVec-DR이 일관 우수", fontsize=10); ax[0].grid(axis="y", ls=":", alpha=0.4)
    bars = ax[1].bar(x, [r[4] for r in rows], 0.6, color="#ff7f0e", edgecolor="black")
    ax[1].axhline(10, ls="--", color="#2ca02c", lw=1.5, label="±10° 예산")
    for b, r in zip(bars, rows):
        ax[1].text(b.get_x()+b.get_width()/2, r[4]+3, f"{r[4]:.0f}°", ha="center", fontsize=8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(cats, rotation=30, ha="right", fontsize=8.5)
    ax[1].set_ylabel("EKF 누적 yaw 오차 (°, 말기평균)"); ax[1].legend(fontsize=9)
    ax[1].set_title("(b) EKF 누적 yaw — 전 카테고리 예산 초과", fontsize=10); ax[1].grid(axis="y", ls=":", alpha=0.4)
    fig.suptitle("그림 8. positive 일반화 — 전 카테고리에서 RotVec(절대 yaw) vs EKF(드리프트 yaw)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = LOG / "fig_4_8_positive_allcat.png"; fig.savefig(out, dpi=150); print(f"\n[OK] {out}")
    with open(LOG / "oxiod_t1_allcat.csv", "w", encoding="utf-8") as f:
        f.write("category,rotvec_dr_ate,ekf_ate,ratio,ekf_cumyaw_deg,ekf_maxyaw_deg\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.1f},{r[5]:.1f}\n")


if __name__ == "__main__":
    main()
