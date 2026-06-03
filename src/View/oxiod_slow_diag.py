# -*- coding: utf-8 -*-
"""slow walking EKF 실패 원인 진단 (yaw 2°인데 EKF 열등 -> 다른 요인?).
변형: Net / EKF@best(1.0) / EKF@0.001 / GT-meas EKF(GT 변위 주입) @1.0,0.001.
 · GT-meas 가 회복되면 -> 병목은 *측정값(네트워크)* 품질 (yaw 무관 저하 가능).
 · GT-meas 도 나쁘면 -> 병목은 *EKF 동역학*(속도/바이어스 적분).
비교용으로 handheld(yaw 드리프트 모드)도 함께.

cd src/View; KMP_DUPLICATE_LIB_OK=TRUE python oxiod_slow_diag.py
"""
import os, sys, logging
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
logging.disable(logging.WARNING)
from pathlib import Path
import numpy as np, torch
_HERE = Path(__file__).resolve().parent; _SRC = _HERE.parent
for p in (_SRC / "Network", _SRC / "Trans", _SRC, _HERE):
    sys.path.insert(0, str(p))
from visualize_comparison import load_npy, run_network, run_ekf_imutracker  # type: ignore
from oxiod_ekf_vs_rotvec_v2 import load_reg, rmse_anchor  # type: ignore

DATA = Path(r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
MODEL_DIR = _SRC / "Network" / "out_regression"
WL = 100


def diag(model, mean, std, seq, sid, best_scale, dev):
    ts, gyr, acc, quat, pos, vel = load_npy(str(DATA / seq / "imu0_resampled.npy"))
    _g, net_pos, _p, anchor_gt, win = run_network(model, str(DATA / seq / "imu0_resampled.npy"), dev, mean, std, WL, WL)
    win = list(win)
    K = min(len(anchor_gt), len(net_pos) - 1)
    rmse_net = float(np.sqrt(np.mean(np.sum((net_pos[1:K + 1, :2] - anchor_gt[:K, :2]) ** 2, axis=1))))

    def ekf(scale, gtmeas):
        p = run_ekf_imutracker(ts, gyr, acc, quat, pos, vel, model, mean, std, WL,
                               use_gt_meas=gtmeas, forced_state_id=sid,
                               meascov_override=scale, mahalanobis_fail_scale=10.0,
                               use_soft_switching=False)
        return rmse_anchor(p[:, :2], pos, win)

    print(f"\n=== {seq} (state {sid}, best_scale {best_scale}) ===")
    print(f"  Net(RotVec-DR)          : {rmse_net:6.2f} m")
    print(f"  EKF @best({best_scale})       : {ekf(best_scale, False):6.2f} m")
    print(f"  EKF @0.001              : {ekf(0.001, False):6.2f} m")
    print(f"  GT-meas EKF @best       : {ekf(best_scale, True):6.2f} m   <- GT 변위 주입")
    print(f"  GT-meas EKF @0.001      : {ekf(0.001, True):6.2f} m")


def main():
    dev = torch.device("cpu"); model, mean, std = load_reg(MODEL_DIR, dev)
    diag(model, mean, std, "oxford_slow_walking_1", 5, 1.0, dev)
    diag(model, mean, std, "oxford_handheld_1", 2, 0.001, dev)   # 비교: yaw 드리프트 모드


if __name__ == "__main__":
    main()
