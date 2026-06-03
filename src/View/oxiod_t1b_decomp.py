# -*- coding: utf-8 -*-
"""T1b — EKF 열등의 2성분 분해 (전 카테고리).
  overhead   = GT-meas EKF - Net        (측정 완벽해도 남는 EKF 동역학 손해)
  yaw_comp   = EKF - GT-meas EKF         (yaw 불안정 시 측정 OOD 성분)
yaw_comp 가 EKF yaw 오차와 함께 커지면 -> yaw 드리프트 메커니즘 일관.
slow 처럼 yaw 안정 카테고리는 yaw_comp 가 작아야 함(반례 아님 검증).

cd src/View; KMP_DUPLICATE_LIB_OK=TRUE python oxiod_t1b_decomp.py
"""
import os, sys, logging
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
logging.disable(logging.WARNING)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
except Exception:
    pass
from pathlib import Path
import numpy as np, torch
from scipy.spatial.transform import Rotation as R
_HERE = Path(__file__).resolve().parent; _SRC = _HERE.parent
for p in (_SRC / "Network", _SRC / "Trans", _SRC, _HERE):
    sys.path.insert(0, str(p))
from visualize_comparison import load_npy, run_network, run_ekf_imutracker  # type: ignore
from oxiod_ekf_vs_rotvec_v2 import load_reg, ekf_with_yaw, rmse_anchor       # type: ignore

DATA = Path(r"D:/EKF_DATASET/TLIO_Oxford_Dataset"); MODEL_DIR = _SRC / "Network" / "out_regression"; WL = 100
CATS = [("handbag", "oxford_handbag_1", 1, 0.001), ("handheld", "oxford_handheld_1", 2, 0.001),
        ("pocket", "oxford_pocket_1", 3, 0.001), ("running", "oxford_running_1", 4, 0.01),
        ("slow", "oxford_slow_walking_1", 5, 1.0), ("trolley", "oxford_trolley_1", 6, 0.001),
        ("large", "oxford_large_scale_1", 9, 0.1)]


def yaw_err(ekf_yaw, quat):
    sub = np.arange(0, len(ekf_yaw), 5)
    gy = np.array([R.from_quat(quat[min(i + 1, len(quat) - 1)]).as_euler("zyx")[0] for i in sub])
    off = gy[0] - ekf_yaw[sub][0]
    err = np.degrees((ekf_yaw[sub] + off - gy + np.pi) % (2 * np.pi) - np.pi)
    return float(np.mean(np.abs(err[len(err) // 2:])))


def main():
    dev = torch.device("cpu"); model, mean, std = load_reg(MODEL_DIR, dev)
    rows = []
    for cat, seq, sid, scale in CATS:
        f = str(DATA / seq / "imu0_resampled.npy")
        ts, gyr, acc, quat, pos, vel = load_npy(f)
        _g, net_pos, _p, anchor_gt, win = run_network(model, f, dev, mean, std, WL, WL); win = list(win)
        K = min(len(anchor_gt), len(net_pos) - 1)
        net = float(np.sqrt(np.mean(np.sum((net_pos[1:K + 1, :2] - anchor_gt[:K, :2]) ** 2, axis=1))))
        ekf = rmse_anchor(run_ekf_imutracker(ts, gyr, acc, quat, pos, vel, model, mean, std, WL,
              forced_state_id=sid, meascov_override=scale, mahalanobis_fail_scale=10.0, use_soft_switching=False)[:, :2], pos, win)
        gtm = rmse_anchor(run_ekf_imutracker(ts, gyr, acc, quat, pos, vel, model, mean, std, WL,
              use_gt_meas=True, forced_state_id=sid, meascov_override=scale, mahalanobis_fail_scale=10.0, use_soft_switching=False)[:, :2], pos, win)
        _xy, eyaw = ekf_with_yaw(ts, gyr, acc, quat, pos, vel, model, mean, std, sid, scale)
        ye = yaw_err(eyaw, quat)
        rows.append((cat, net, gtm, ekf, gtm - net, ekf - gtm, ye))
        print(f"{cat:<9} Net{net:5.2f} GTmeas{gtm:6.2f} EKF{ekf:6.2f} | overhead{gtm-net:+5.2f} yawComp{ekf-gtm:+6.2f} | yawErr{ye:4.0f}°")

    print("\n[분해 — yaw_comp 가 yaw_err 과 함께 커지는가?]")
    yc = np.array([r[5] for r in rows]); ye = np.array([r[6] for r in rows])
    print(f"  Pearson r(yaw_err, yaw_comp) = {np.corrcoef(ye, yc)[0,1]:+.2f}")
    print(f"  yaw 안정(<20°) 카테고리 yaw_comp 평균 = {np.mean([r[5] for r in rows if r[6]<20]):+.2f} m")
    print(f"  yaw 불안정(>40°) 카테고리 yaw_comp 평균 = {np.mean([r[5] for r in rows if r[6]>40]):+.2f} m")
    with open(_SRC.parent / "logs" / "oxiod_t1b_decomp.csv", "w", encoding="utf-8") as fp:
        fp.write("category,net,gtmeas,ekf,overhead,yaw_comp,yaw_err_deg\n")
        for r in rows:
            fp.write(",".join(str(round(x, 3)) if isinstance(x, float) else str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
