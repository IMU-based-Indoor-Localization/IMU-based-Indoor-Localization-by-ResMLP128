"""
EKF 업데이트가 실제로 동작하는지 진단.
GT 측정값으로 처음 30초만 추적하고 위치 오차를 출력.
"""
import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC / "Network"))
sys.path.insert(0, str(_SRC / "Trans"))
sys.path.insert(0, str(_SRC))

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace

from model_twolayer import TwoLayerModel
from tracker.scekf import ImuMSCKF
from tracker.imu_buffer import ImuBuffer
from utils.from_scipy import compute_euler_from_matrix
from utils.math_utils import mat_exp

G_NORM = 9.81

DATA  = "TLIO_Oxford_Dataset/oxford_handheld_21/imu0_resampled.npy"

def load_npy(data_path):
    data = np.load(data_path)
    ts_us   = data[:, 0].astype(np.int64)
    gyr     = data[:, 1:4].astype(np.float64)
    acc_lin = data[:, 4:7].astype(np.float64)
    grav_b  = data[:, 7:10].astype(np.float64)
    quat    = data[:, 14:18].astype(np.float64)
    pos_gt  = data[:, 18:21].astype(np.float64)
    vel_gt  = data[:, 21:24].astype(np.float64)
    acc_raw = acc_lin - grav_b * G_NORM
    return ts_us, gyr, acc_raw, quat, pos_gt, vel_gt

def main():
    ts_us, gyr, acc_raw, quat, pos_gt, vel_gt = load_npy(DATA)
    print(f"데이터 로드: {len(ts_us)} 샘플 ({len(ts_us)/100:.1f}s @ 100Hz)")

    # 처음 10 샘플 타임스탬프 간격 확인
    dts = np.diff(ts_us[:10]) / 1e6
    print(f"타임스탬프 간격 (초): {dts}")

    # IMU 첫 샘플 확인
    print(f"\nacc_raw[0]={acc_raw[0]}  norm={np.linalg.norm(acc_raw[0]):.3f} (should be ~9.81)")
    print(f"gyr[0]={gyr[0]}")
    print(f"pos_gt[0]={pos_gt[0]}  pos_gt[100]={pos_gt[100]}")
    print(f"GT 변위 (1초): {pos_gt[100]-pos_gt[0]}  norm={np.linalg.norm(pos_gt[100]-pos_gt[0]):.3f}m")

    # 간단한 EKF 초기화 (첫 30초만 실행)
    ekf_cfg = SimpleNamespace(
        sigma_na=np.sqrt(1e-3), sigma_ng=np.sqrt(1e-4),
        ita_ba=1e-4, ita_bg=1e-6,
        init_attitude_sigma=1.0/180*np.pi, init_yaw_sigma=0.1/180*np.pi,
        init_vel_sigma=0.01, init_pos_sigma=0.001,
        init_bg_sigma=1e-4, init_ba_sigma=0.02,
        g_norm=9.81, meascov_scale=1.0,
        mahalanobis_fail_scale=0, mahalanobis_factor=0,  # 게이팅 완전 비활성
    )
    ekf = ImuMSCKF(config=ekf_cfg)
    R0 = R.from_quat(quat[0]).as_matrix().T  # Oxford quat = R_BW (world-to-body); .T gives R_WB (body-to-world)
    v0 = vel_gt[0].reshape(3,1)
    p0 = pos_gt[0].reshape(3,1)
    ekf.initialize_with_state(int(ts_us[0]), R0, v0, p0, np.zeros((3,1)), np.zeros((3,1)))

    # 100초치 데이터 처리
    N_samples = min(len(ts_us), 10001)  # 100초
    print(f"\n=== EKF 100초 추적 (GT 측정값, 게이팅 없음) ===")
    print(f"{'시간(s)':>8} {'EKF_pos':>30} {'GT_pos':>30} {'오차(m)':>8}")

    next_aug_t_us = int(ts_us[0])
    dt_update_us = 1_000_000  # 1초
    next_update_t_us = int(ts_us[0]) + dt_update_us * 2  # 2초 후 첫 업데이트

    clones_R = []
    clones_p = []
    clones_t = []

    for i in range(1, N_samples):
        t_us = int(ts_us[i])
        acc = acc_raw[i].reshape(3,1)
        gyr_i = gyr[i].reshape(3,1)

        do_aug = (t_us >= next_aug_t_us)
        t_aug = next_aug_t_us if do_aug else None

        ekf.propagate(acc, gyr_i, t_us, t_augmentation_us=t_aug)

        if do_aug:
            _, p_aug = ekf.get_past_state(next_aug_t_us)
            R_aug, _ = ekf.get_past_state(next_aug_t_us)
            clones_R.append(R_aug)
            clones_p.append(p_aug.copy())
            clones_t.append(next_aug_t_us)
            next_aug_t_us += dt_update_us

            # GT 업데이트 (2초 간격)
            if len(clones_t) >= 3:
                t_begin = clones_t[-3]
                t_end   = clones_t[-1]
                idx_old = int(np.searchsorted(ts_us, t_begin))
                idx_end = int(np.searchsorted(ts_us, t_end))
                dp_world = (pos_gt[idx_end] - pos_gt[idx_old]).reshape(3,1)

                R_old, _ = ekf.get_past_state(t_begin)
                ri_z = compute_euler_from_matrix(R_old, "xyz", extrinsic=True)[0,2]
                Ri_z = np.array([
                    [np.cos(ri_z), -np.sin(ri_z), 0],
                    [np.sin(ri_z),  np.cos(ri_z), 0],
                    [0, 0, 1],
                ])
                meas = Ri_z.T @ dp_world
                meas_cov = 1e-4 * np.eye(3)

                try:
                    ekf.update(meas, meas_cov, t_begin, t_end)
                    oldest_idx = ekf.state.si_timestamps_us.index(t_begin)
                    ekf.marginalize(oldest_idx)
                    clones_R = clones_R[-2:]
                    clones_p = clones_p[-2:]
                    clones_t = clones_t[-2:]
                except Exception as e:
                    print(f"  [업데이트 실패] t={t_us/1e6:.1f}s  {e}")

        if i % 1000 == 0:  # 10초마다 출력
            _, _, p_ekf, _, _ = ekf.get_evolving_state()
            gt_p = pos_gt[i]
            err = np.linalg.norm(p_ekf.flatten() - gt_p)
            t_sec = ts_us[i] / 1e6 - ts_us[0] / 1e6
            print(f"{t_sec:8.1f} {str(p_ekf.flatten().round(3)):>30} {str(gt_p.round(3)):>30} {err:8.3f}")

    # 최종 위치 오차
    _, _, p_ekf, _, _ = ekf.get_evolving_state()
    err_final = np.linalg.norm(p_ekf.flatten() - pos_gt[N_samples-1])
    print(f"\n최종 오차 (100초 시점): {err_final:.3f}m")

if __name__ == "__main__":
    main()
