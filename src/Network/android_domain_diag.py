# -*- coding: utf-8 -*-
"""Android raw 과소예측 원인 진단 — 입력 분포(중력정렬 acc/gyr) 비교.
Android vs OxIOD(학습 도메인) vs norm_mean/std(학습 정규화).
스케일/오프셋 어디서 어긋나는지 채널별로 본다.
cd src/Network; KMP_DUPLICATE_LIB_OK=TRUE python android_domain_diag.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).parent))
from offline_eval import WINDOW_LEN, FS, load_android, load_oxiod, load_model, window_to_gravity_aligned  # type: ignore

AND = r"D:\mobile\imu_android\csv\imu_csv\imu_record_1780543327203.csv"
OXF = r"D:/EKF_DATASET/TLIO_Oxford_Dataset/oxford_handheld_1/imu0_resampled.npy"
CH = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]


def ga_stack(data):
    acc, gyr, quat = data["acc"], data["gyr"], data["quat"]
    T = len(acc); out = []
    for s in range(0, T - WINDOW_LEN, WINDOW_LEN):
        out.append(window_to_gravity_aligned(acc, gyr, quat, s, s + WINDOW_LEN, frame="ga"))
    return np.concatenate(out, axis=0)  # [N,6]


def main():
    net, _p, mean, std = load_model("src/Network/out_classifier2")
    da = load_android(AND, calib_sec=2.0, linacc_scale=1.0)
    do = load_oxiod(OXF)
    ga_a, ga_o = ga_stack(da), ga_stack(do)
    ma, sa = ga_a.mean(0), ga_a.std(0)
    mo, so = ga_o.mean(0), ga_o.std(0)

    print(f"{'channel':>7} | {'norm_mean':>9} {'norm_std':>9} | {'OxIOD mean':>10} {'OxIOD std':>9} | {'AND mean':>9} {'AND std':>8} | {'std AND/OxIOD':>13}")
    print("-" * 95)
    for i, c in enumerate(CH):
        ratio = sa[i] / so[i] if so[i] > 1e-9 else float("nan")
        print(f"{c:>7} | {mean[i]:>9.3f} {std[i]:>9.3f} | {mo[i]:>10.3f} {so[i]:>9.3f} | {ma[i]:>9.3f} {sa[i]:>8.3f} | {ratio:>12.2f}x")

    # 수평 가속도 동적 진폭 (변위를 만드는 성분)
    hor_a = np.sqrt(ga_a[:, 0] ** 2 + ga_a[:, 1] ** 2)
    hor_o = np.sqrt(ga_o[:, 0] ** 2 + ga_o[:, 1] ** 2)
    print(f"\n수평 acc 동적 진폭 RMS:  OxIOD {np.sqrt((hor_o**2).mean()):.3f} m/s²  vs  Android {np.sqrt((hor_a**2).mean()):.3f} m/s²"
          f"  (비 {np.sqrt((hor_a**2).mean())/np.sqrt((hor_o**2).mean()):.2f}x)")
    print(f"acc z 평균(중력 포함 여부):  OxIOD {mo[2]:.2f}  Android {ma[2]:.2f}  (norm_mean_z {mean[2]:.2f})")


if __name__ == "__main__":
    main()
