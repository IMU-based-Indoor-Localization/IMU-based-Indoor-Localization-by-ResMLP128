"""
oxiod_drift_decompose.py — yaw-drift disp 오차의 크기/방향 분해 (지표 오염 검증)
==============================================================================
문제: oxiod_yaw_drift_ablation.py 의 input_only disp-RMSE 4.07× 는 *벡터* 오차
‖d − gt‖. drift 입력의 yaw 추출이 pitch/roll 과 커플링하며 남는 *잔여 프레임 회전*
이 오차에 섞일 수 있음 → "예측 품질 저하" vs "좌표계 회전 아티팩트" 미분리.

본 스크립트는 input_only 모드에서 네 가지 지표를 분리 측정한다 (8 카테고리 평균):

  (1) vec_mismatch : ‖d − gt_true‖   — GT 를 *true* ga 프레임에서 비교 (현재 지표).
                     프레임 회전 잔차 포함. = 실세계 좌표 총 효과.
  (2) vec_matched  : ‖d − gt_drift‖  — GT 를 *drift* ga 프레임에서 비교 (yaw0_drift 사용).
                     프레임 회전 제거 → *순수 네트워크 예측 오차*. (§4.5 핵심)
  (3) mag_err      : | |d| − |gt| |  — 회전 불변. 크기 예측이 망가지는지. (genuine OOD)
  (4) mean|d|, mean|gt| : 예측 크기 collapse/inflate 체크.

판정:
  · (3) mag_err 와 (2) vec_matched 가 baseline 근처 유지 + (1) 만 급증
        → 4.07× 는 *좌표계 아티팩트*. §4.5 "예측 품질 저하" 는 기하 효과이지 OOD 아님.
  · (3)/(2) 도 drift 따라 증가
        → 네트워크 예측 자체 저하 = genuine OOD. §4.5 성립.

사용:
  python src/Network/oxiod_drift_decompose.py \\
      --data_dir D:/EKF_DATASET/TLIO_Oxford_Dataset \\
      --model_dir src/Network/out_classifier2 --out logs/oxiod_drift_decompose.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from offline_eval import (   # type: ignore
    WINDOW_LEN, FS, load_oxiod, load_model,
    window_to_gravity_aligned, window_yaw0,
)
from oxiod_preproc_ablation import select_longest_sequence, CATEGORIES   # type: ignore
from oxiod_yaw_drift_ablation import make_drifted_quat   # type: ignore


DRIFT_RATES = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0]


def evaluate_decompose(net, data, mean, std, drift_rate: float):
    """input_only 모드. 반환: dict of RMS metrics over windows."""
    import torch

    acc, gyr, quat_true, pos = data["acc"], data["gyr"], data["quat"], data["pos"]
    quat_in = make_drifted_quat(quat_true, drift_rate)   # 입력만 drift

    T = len(acc)
    starts = list(range(0, T - WINDOW_LEN, WINDOW_LEN))

    vec_mis, vec_mat, mag_err = [], [], []
    abs_d, abs_gt = [], []
    with torch.no_grad():
        for s in starts:
            e = s + WINDOW_LEN
            imu = window_to_gravity_aligned(acc, gyr, quat_in, s, e, frame="ga")
            imu_n = (imu - mean) / std
            x = torch.from_numpy(imu_n.T.copy()).float().unsqueeze(0)
            y, _c, _l = net(x)
            d = y[0].numpy()[:2]                          # 예측 disp (입력 ga 프레임)

            dp = pos[e, :2] - pos[s, :2]                  # GT world 변위
            # true ga 프레임 GT
            yt = window_yaw0(quat_true, s)
            ct, stt = np.cos(-yt), np.sin(-yt)
            gt_true = np.array([ct * dp[0] - stt * dp[1], stt * dp[0] + ct * dp[1]])
            # drift ga 프레임 GT (입력 프레임과 정합)
            yd = window_yaw0(quat_in, s)
            cd, sd = np.cos(-yd), np.sin(-yd)
            gt_drift = np.array([cd * dp[0] - sd * dp[1], sd * dp[0] + cd * dp[1]])

            vec_mis.append(np.hypot(d[0] - gt_true[0], d[1] - gt_true[1]))
            vec_mat.append(np.hypot(d[0] - gt_drift[0], d[1] - gt_drift[1]))
            dn = np.hypot(d[0], d[1]); gn = np.hypot(dp[0], dp[1])
            mag_err.append(abs(dn - gn))
            abs_d.append(dn); abs_gt.append(gn)

    rms = lambda a: float(np.sqrt(np.mean(np.array(a) ** 2)))
    return {
        "vec_mismatch": rms(vec_mis),
        "vec_matched":  rms(vec_mat),
        "mag_err":      rms(mag_err),
        "mean_abs_d":   float(np.mean(abs_d)),
        "mean_abs_gt":  float(np.mean(abs_gt)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
    ap.add_argument("--model_dir", default="src/Network/out_classifier2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[1] 모델 로드: {args.model_dir}")
    net, _para, mean, std = load_model(args.model_dir)

    print(f"[2] 카테고리당 가장 긴 시퀀스")
    seqs = {}
    for c in CATEGORIES:
        sd = select_longest_sequence(data_dir, c)
        if sd is not None:
            seqs[c] = load_oxiod(sd / "imu0_resampled.npy")
    print(f"    {len(seqs)} 카테고리")

    rows = []
    print()
    print(f"[3] input_only 분해 추론 ({len(DRIFT_RATES)} drift × {len(seqs)} cat)")
    for dr in DRIFT_RATES:
        for cat, data in seqs.items():
            m = evaluate_decompose(net, data, mean, std, dr)
            rows.append((dr, cat, m))

    def avg(dr, key):
        return float(np.mean([r[2][key] for r in rows if r[0] == dr]))

    print()
    print("=" * 90)
    print("[A] input_only — disp 오차 분해 (8 카테고리 평균, m)")
    print("=" * 90)
    print(f"{'drift':>7} | {'(1)vec_mismatch':>15} | {'(2)vec_matched':>15} | "
          f"{'(3)mag_err':>11} | {'mean|d|':>8} | {'mean|gt|':>9}")
    print(f"{'(°/s)':>7} | {'현재지표(오염)':>15} | {'프레임정합(순수)':>15} | "
          f"{'회전불변':>11} | {'':>8} | {'':>9}")
    print("-" * 90)
    base = {k: avg(0.0, k) for k in ("vec_mismatch", "vec_matched", "mag_err")}
    for dr in DRIFT_RATES:
        print(f"{dr:>7.2f} | {avg(dr,'vec_mismatch'):>15.3f} | {avg(dr,'vec_matched'):>15.3f} | "
              f"{avg(dr,'mag_err'):>11.3f} | {avg(dr,'mean_abs_d'):>8.3f} | {avg(dr,'mean_abs_gt'):>9.3f}")

    print()
    print("=" * 90)
    print("[B] drift 0 → 5°/s 배수 (판정)")
    print("=" * 90)
    for k, label in (("vec_mismatch", "(1) 현재지표 (프레임 불일치)"),
                     ("vec_matched",  "(2) 프레임 정합 = 순수 예측오차"),
                     ("mag_err",      "(3) 크기오차 = 회전불변")):
        v0, v5 = base[k], avg(5.0, k)
        x = v5 / v0 if v0 > 0 else float("nan")
        print(f"  {label:<32}: {v0:.3f} → {v5:.3f}  ({x:.2f}x)")

    print()
    print("판정 가이드:")
    mat_x = avg(5.0, "vec_matched") / base["vec_matched"]
    mag_x = avg(5.0, "mag_err") / base["mag_err"]
    mis_x = avg(5.0, "vec_mismatch") / base["vec_mismatch"]
    if mat_x < 1.5 and mag_x < 1.5 and mis_x > 2.0:
        print("  → (2)(3) 평탄 + (1) 급증 ⇒ 4.07x 는 *좌표계 아티팩트*. 예측 자체는 견고.")
        print("    §4.5 '입력 OOD 가 예측 품질 저하' 는 약함 — 효과는 기하(출력 프레임)임.")
    elif mat_x >= 2.0 or mag_x >= 2.0:
        print("  → (2)/(3) 도 급증 ⇒ 네트워크 예측 자체 저하 = genuine OOD. §4.5 성립.")
    else:
        print("  → 부분적: 예측 저하 + 프레임 효과 혼재. 정량 분해값으로 서술 권장.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("drift_deg_s,category,vec_mismatch,vec_matched,mag_err,mean_abs_d,mean_abs_gt\n")
            for dr, cat, m in rows:
                f.write(f"{dr},{cat},{m['vec_mismatch']:.4f},{m['vec_matched']:.4f},"
                        f"{m['mag_err']:.4f},{m['mean_abs_d']:.4f},{m['mean_abs_gt']:.4f}\n")
        print(f"\n[OK] CSV 저장: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
