# -*- coding: utf-8 -*-
"""그림 7 재생성 — §4.4 EKF 스택(out_regression + TwoLayerImuTracker/ImuMSCKF,
mahalanobis_fail_scale=10, meascov scale=0.001) 그대로 사용해 §4.4 표와 정합.
EKF yaw 를 filter.get_evolving_state()[0] (R) 에서 시간축 추출.

실행:
  cd src/View
  KMP_DUPLICATE_LIB_OK=TRUE python oxiod_ekf_vs_rotvec_v2.py --seq oxford_handheld_13 --state 2 --scale 0.001
"""
import os, sys, json, argparse, logging
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
logging.disable(logging.WARNING)
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent; _SRC = _HERE.parent
for p in (_SRC / "Network", _SRC / "Trans", _SRC, _HERE):
    sys.path.insert(0, str(p))
for fn in ("Malgun Gothic", "맑은 고딕", "Gulim"):
    try:
        font_manager.findfont(fn, fallback_to_default=False); plt.rcParams["font.family"] = fn; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

from model_twolayer import TwoLayerModel                       # noqa
from visualize_comparison import load_npy, run_network, TwoLayerImuTracker  # noqa

DATA = Path(r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
MODEL_DIR = _SRC / "Network" / "out_regression"
WINDOW_LEN = 100
LOG = _SRC.parent / "logs"


def load_reg(md, dev):
    cfg = json.load(open(md / "config.json"))
    m = TwoLayerModel(cfg["model"]).to(dev)
    ck = torch.load(md / "checkpoints" / "best.pth", map_location=dev, weights_only=False)
    m.load_state_dict(ck["model"]); m.eval()
    return m, np.load(md / "norm_mean.npy"), np.load(md / "norm_std.npy")


def ekf_with_yaw(ts_us, gyr, acc_raw, quat, pos_gt, vel_gt, model, mean, std, state_id, scale):
    cfg = SimpleNamespace(
        sigma_na=np.sqrt(1e-3), sigma_ng=np.sqrt(1e-4), ita_ba=1e-4, ita_bg=1e-6,
        init_attitude_sigma=1.0/180*np.pi, init_yaw_sigma=0.1/180*np.pi,
        init_vel_sigma=1.0, init_pos_sigma=0.001, init_bg_sigma=1e-4, init_ba_sigma=0.02,
        g_norm=9.81, meascov_scale=1.0, mahalanobis_fail_scale=10.0)
    dev = next(model.parameters()).device
    tr = TwoLayerImuTracker(model, mean, std, cfg, update_freq=1.0, device=dev,
                            forced_state_id=state_id, meascov_override=scale,
                            gt_quat=None, gt_ts_us=None, use_soft_switching=False)
    tr.init_with_state_at_time(int(ts_us[0]), R.from_quat(quat[0]).as_matrix(),
                               vel_gt[0].reshape(3, 1), pos_gt[0].reshape(3, 1),
                               gyr[0].reshape(3, 1), acc_raw[0].reshape(3, 1))
    pos, yaw = [], []
    for i in range(1, len(ts_us)):
        tr.on_imu_measurement(int(ts_us[i]), gyr[i].reshape(3, 1), acc_raw[i].reshape(3, 1))
        st = tr.filter.get_evolving_state()
        Rm = np.asarray(st[0]); p = np.asarray(st[2]).flatten()
        pos.append(p[:2].copy())
        yaw.append(float(R.from_matrix(Rm).as_euler("zyx")[0]))
    return np.array(pos), np.array(yaw)


def rmse_anchor(traj_xy, pos_gt, win_idx):
    a = np.array([i + WINDOW_LEN for i in win_idx]); a = a[a < len(pos_gt)]
    if len(a) == 0: return float("nan")
    gt = pos_gt[a, :2]; tr = traj_xy[np.clip(a - 1, 0, len(traj_xy) - 1), :2]
    return float(np.sqrt(np.mean(np.sum((tr - gt) ** 2, axis=1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="oxford_handheld_1")
    ap.add_argument("--state", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.001)
    ap.add_argument("--seconds", type=float, default=0.0, help="panel(a) 표시 구간(s). 0=전체")
    args = ap.parse_args()
    dev = torch.device("cpu")
    print(f"[1] out_regression 로드"); model, mean, std = load_reg(MODEL_DIR, dev)
    npy = DATA / args.seq / "imu0_resampled.npy"
    print(f"[2] seq {args.seq}")
    ts_us, gyr, acc_raw, quat, pos_gt, vel_gt = load_npy(str(npy))

    print("[3] Network(RotVec-DR)")
    _gt, net_pos, _pred, anchor_gt, win_idx = run_network(model, str(npy), dev, mean, std, WINDOW_LEN, WINDOW_LEN)
    win_idx = list(win_idx)
    # §4.4(ekf_tune.py)와 동일한 Net RMSE 공식: anchor_gt vs net_pos[1:K+1]
    K = min(len(anchor_gt), len(net_pos) - 1)
    rmse_net = float(np.sqrt(np.mean(np.sum((net_pos[1:K+1, :2] - anchor_gt[:K, :2]) ** 2, axis=1))))

    print("[4] EKF(ImuMSCKF, §4.4 stack)")
    ekf_xy, ekf_yaw = ekf_with_yaw(ts_us, gyr, acc_raw, quat, pos_gt, vel_gt, model, mean, std, args.state, args.scale)
    rmse_ekf = rmse_anchor(ekf_xy, pos_gt, win_idx)
    print(f"    RotVec-DR(Net) ATE={rmse_net:.2f} m ; EKF ATE={rmse_ekf:.2f} m  ({rmse_ekf/rmse_net:.1f}x)")

    # yaw error vs GT
    T = min(len(ekf_yaw) + 1, len(pos_gt)); FS = 100.0
    sub = np.arange(0, len(ekf_yaw), 5)
    gt_yaw = np.array([R.from_quat(quat[min(i+1, len(quat)-1)]).as_euler("zyx")[0] for i in sub])
    off = gt_yaw[0] - ekf_yaw[sub][0]
    ekf_err = np.degrees(np.unwrap(ekf_yaw[sub] + off - gt_yaw))
    t_sec = (sub + 1) / FS
    print(f"    EKF yaw 오차: 말기평균 {np.mean(np.abs(ekf_err[-len(ekf_err)//5:])):.0f}° max {np.max(np.abs(ekf_err)):.0f}°")

    origin = pos_gt[0, :2]
    gt_xy = pos_gt[:, :2] - origin; nx = net_pos[:, :2] - origin; ex = ekf_xy - origin
    # panel(a): 가독성 위해 첫 N초 구간만 (§4.4 short_traj 방식). panel(b)·캡션 수치는 전체.
    if args.seconds and args.seconds > 0:
        S = int(args.seconds * 100); nA = int(args.seconds) + 1
        gtp = gt_xy[:S]; exp_ = ex[:min(S, len(ex))]; nxp = nx[:min(nA, len(nx))]
        seg = f"첫 {int(args.seconds)}초"
    else:
        gtp, exp_, nxp = gt_xy, ex, nx; seg = "전체"
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.0))
    ax[0].plot(gtp[:, 0], gtp[:, 1], "-", color="black", lw=2.4, label="GT", zorder=5)
    ax[0].plot(nxp[:, 0], nxp[:, 1], "-", color="#1f77b4", lw=2.0, marker="o", ms=3, label="RotVec-DR (안정 yaw)")
    ax[0].plot(exp_[:, 0], exp_[:, 1], "-", color="#d62728", lw=1.6, alpha=0.9, label="EKF (드리프트 yaw)")
    ax[0].scatter([0], [0], c="k", s=45, zorder=6, label="시작")
    ax[0].set_aspect("equal", "datalim"); ax[0].legend(fontsize=9, loc="best"); ax[0].grid(ls=":", alpha=0.4)
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Y (m)")
    ax[0].set_title(f"(a) 궤적 ({seg}) — EKF 발산 vs RotVec 추종", fontsize=10)
    ax[1].axhspan(-10, 10, color="#2ca02c", alpha=0.10, label="±10° 예산")
    ax[1].plot(t_sec, ekf_err, "-", color="#d62728", lw=1.6, label="EKF yaw 오차 (vs GT)")
    ax[1].axhline(0, color="#1f77b4", lw=2.0, label="RotVec yaw 오차 ≈ 0")
    ax[1].set_xlabel("시간 (s)"); ax[1].set_ylabel("yaw 오차 (°)"); ax[1].legend(fontsize=9, loc="best"); ax[1].grid(ls=":", alpha=0.4)
    ax[1].set_title("(b) yaw 오차 추이 — EKF 가 ±10° 예산 초과 드리프트", fontsize=10)
    fig.suptitle("그림 7. 실제 EKF yaw 드리프트 vs RotVec 절대 yaw (§4.4 스택)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = LOG / "fig_4_7_ekf_vs_rotvec.png"; fig.savefig(out, dpi=150)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
