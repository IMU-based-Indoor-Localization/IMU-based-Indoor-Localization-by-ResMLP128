"""
oxiod_drift_onset.py — yaw drift OOD 발현 임계점(cliff) 특정
============================================================
oxiod_yaw_drift_ablation.py 에서 0.5°/s 가 이미 포화(disp 0.60m)임을 확인.
→ 진짜 cliff 는 그 아래. 0.0~0.5°/s 를 세밀 sweep 하여 OOD 발현 임계 yaw 오차를 특정.

input_only 모드 (입력 정렬만 drift, 출력은 GT yaw) 로 *신경망 예측 품질* 만 격리 측정.
지표: per-window |disp_xy| RMSE (m) + ATE RMSE_xy (m), 8 카테고리 평균.

목적: 논문 §5 "required yaw accuracy" 정량 목표 도출 — "yaw 오차 X° 이내면 OOD 회피".

사용:
  python src/Network/oxiod_drift_onset.py \\
      --data_dir D:/EKF_DATASET/TLIO_Oxford_Dataset \\
      --model_dir src/Network/out_classifier2 \\
      --out logs/oxiod_drift_onset.csv
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
from offline_eval import FS, load_oxiod, load_model   # type: ignore
from oxiod_preproc_ablation import select_longest_sequence, CATEGORIES   # type: ignore
from oxiod_yaw_drift_ablation import evaluate_drift   # type: ignore


# 세밀 sweep — 0 부터 0.5 까지 + 누적량 환산 (600s 시퀀스 기준 총 yaw 회전)
DRIFT_RATES = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


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

    print(f"[2] 카테고리당 가장 긴 시퀀스 + 길이")
    seqs, durations = {}, {}
    for cat in CATEGORIES:
        seq = select_longest_sequence(data_dir, cat)
        if seq is not None:
            seqs[cat] = load_oxiod(seq / "imu0_resampled.npy")
            durations[cat] = len(seqs[cat]["acc"]) / FS
    print(f"    {len(seqs)} 카테고리, 평균 길이 {np.mean(list(durations.values())):.0f}s")

    rows = []   # (drift, cat, disp, ate)
    print()
    print(f"[3] input_only 모드 세밀 sweep ({len(DRIFT_RATES)} drift × {len(seqs)} cat)")
    for dr in DRIFT_RATES:
        for cat, data in seqs.items():
            ate, disp = evaluate_drift(net, data, mean, std, dr, "input_only")
            rows.append((dr, cat, disp, ate))

    # ── 출력: drift 별 카테고리 평균 + 누적 yaw 환산 ──────────────────
    print()
    print("=" * 78)
    print("[A] drift onset — input_only (예측 품질 격리)")
    print("    누적 yaw = drift_rate × 시퀀스 길이 (평균 ~590s 기준 환산)")
    print("=" * 78)
    avg_dur = float(np.mean(list(durations.values())))
    print(f"{'drift(°/s)':>10} | {'누적 yaw(°)':>11} | {'disp-RMSE(m)':>13} | {'ATE(m)':>9} | {'disp 배수':>9}")
    print("-" * 78)
    base_disp = None
    for dr in DRIFT_RATES:
        disps = [r[2] for r in rows if r[0] == dr]
        ates  = [r[3] for r in rows if r[0] == dr]
        md = float(np.mean(disps)); ma = float(np.mean(ates))
        if base_disp is None:
            base_disp = md
        cum_yaw = dr * avg_dur
        ratio = md / base_disp if base_disp > 0 else float("nan")
        print(f"{dr:>10.2f} | {cum_yaw:>11.0f} | {md:>13.3f} | {ma:>9.2f} | {ratio:>8.2f}x")

    # ── cliff 판정: disp 가 baseline 의 2배를 처음 넘는 drift ───────────
    print()
    print("=" * 78)
    print("[B] cliff 판정 (disp-RMSE 가 baseline 2× 초과하는 첫 지점)")
    print("=" * 78)
    for dr in DRIFT_RATES:
        if dr == 0.0:
            continue
        disps = [r[2] for r in rows if r[0] == dr]
        md = float(np.mean(disps))
        if md > base_disp * 2.0:
            cum = dr * avg_dur
            print(f"  → cliff ≈ {dr:.2f}°/s (누적 yaw ~{cum:.0f}°) 에서 disp {md:.3f}m "
                  f"({md/base_disp:.1f}x) — OOD 발현")
            print(f"  → 권장 yaw accuracy 목표: 누적 yaw 오차 < ~{cum:.0f}° "
                  f"(rate < {dr:.2f}°/s @ {avg_dur:.0f}s)")
            break
    else:
        print("  baseline 2× 초과 없음 — 더 세밀/낮은 drift 필요 또는 robust")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("drift_deg_s,category,disp_rmse_m,ate_rmse_m\n")
            for dr, cat, disp, ate in rows:
                f.write(f"{dr},{cat},{disp:.4f},{ate:.4f}\n")
        print(f"\n[OK] CSV 저장: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
