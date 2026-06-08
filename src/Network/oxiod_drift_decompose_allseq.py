"""
oxiod_drift_decompose_allseq.py — §4.6.2 방향/크기 해리의 *전체 시퀀스* 통계
==============================================================================
§4.6.2 의 핵심 수치(방향 7.70× / 크기 1.00×)는 카테고리당 가장 긴 시퀀스 1개
(longest n=1/cat, 총 8개)에서 산출된 점추정이다. §4.6.1 은 전체 152 시퀀스에
Wilcoxon·bootstrap CI 를 붙였으나 §4.6.2 는 그렇지 못해 rigor 가 불일치한다.

본 스크립트는 oxiod_drift_decompose.py 의 input_only 분해를 *전체 시퀀스*에
적용해, 시퀀스마다 방향 배수(dir_err@5 / dir_err@0)와 크기 배수(mag_err@5 /
mag_err@0)를 구하고, §4.6.1 과 동일하게 per-sequence 비율의 기하평균 + bootstrap
95% CI 로 보고한다. 이로써 "7.70× 는 longest 편향 아니냐"는 질문을 봉쇄한다.

지표는 oxiod_drift_decompose.py 와 동일:
  dir_err  = ‖d − gt_drift‖  (입력 프레임 정합 = 순수 방향 예측오차)
  mag_err  = | |d| − |gt| |   (회전 불변 = 크기 예측오차)

사용:
  python src/Network/oxiod_drift_decompose_allseq.py \\
      --data_dir D:/EKF_DATASET/TLIO_Oxford_Dataset \\
      --model_dir src/Network/out_classifier2 --out logs/oxiod_drift_allseq.csv
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
    WINDOW_LEN, load_oxiod, load_model,
    window_to_gravity_aligned, window_yaw0,
)
from oxiod_preproc_ablation import CATEGORIES   # type: ignore
from oxiod_yaw_drift_ablation import make_drifted_quat   # type: ignore
from oxiod_preproc_significance import all_sequences, boot_ci_geomean   # type: ignore

DRIFTS = [0.0, 5.0]   # baseline 과 high-drift 만 — 배수 산출용
MIN_WINDOWS = 5       # 너무 짧은 시퀀스(<5 window)는 비율 불안정 → 제외


def decompose_seq(net, data, mean, std, drift_rate: float):
    """반환: (dir_rms, mag_rms, n_win).  input_only."""
    import torch
    acc, gyr, quat_true, pos = data["acc"], data["gyr"], data["quat"], data["pos"]
    quat_in = make_drifted_quat(quat_true, drift_rate)
    T = len(acc)
    starts = list(range(0, T - WINDOW_LEN, WINDOW_LEN))
    dir_e, mag_e = [], []
    with torch.no_grad():
        for s in starts:
            e = s + WINDOW_LEN
            imu = window_to_gravity_aligned(acc, gyr, quat_in, s, e, frame="ga")
            x = torch.from_numpy(((imu - mean) / std).T.copy()).float().unsqueeze(0)
            y, _c, _l = net(x)
            d = y[0].numpy()[:2]
            dp = pos[e, :2] - pos[s, :2]
            yd = window_yaw0(quat_in, s)
            cd, sd = np.cos(-yd), np.sin(-yd)
            gt = np.array([cd * dp[0] - sd * dp[1], sd * dp[0] + cd * dp[1]])
            dir_e.append(np.hypot(d[0] - gt[0], d[1] - gt[1]))
            mag_e.append(abs(np.hypot(*d) - np.hypot(*dp)))
    rms = lambda a: float(np.sqrt(np.mean(np.array(a) ** 2)))
    return rms(dir_e), rms(mag_e), len(starts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
    ap.add_argument("--model_dir", default="src/Network/out_classifier2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[1] 모델 로드: {args.model_dir}")
    net, _para, mean, std = load_model(args.model_dir)

    print(f"[2] 전체 시퀀스 방향/크기 배수 산출 (drift 0 vs 5°/s)")
    recs = []
    for cat in CATEGORIES:
        seqs = all_sequences(data_dir, cat)
        for sd in seqs:
            data = load_oxiod(sd / "imu0_resampled.npy")
            d0, m0, nw = decompose_seq(net, data, mean, std, 0.0)
            d5, m5, _  = decompose_seq(net, data, mean, std, 5.0)
            if nw < MIN_WINDOWS or d0 <= 0 or m0 <= 0:
                continue
            recs.append({"cat": cat, "seq": sd.name, "nw": nw,
                         "dir0": d0, "dir5": d5, "mag0": m0, "mag5": m5,
                         "dir_x": d5 / d0, "mag_x": m5 / m0})
        print(f"    {cat:<10} {len(seqs):>3} 시퀀스")

    N = len(recs)
    dir_x = np.array([r["dir_x"] for r in recs])
    mag_x = np.array([r["mag_x"] for r in recs])

    def summ(name, ratios):
        gm = float(np.exp(np.mean(np.log(ratios))))
        md = float(np.median(ratios))
        lo, hi = boot_ci_geomean(ratios)
        return gm, md, lo, hi

    gd, mdd, lod, hid = summ("dir", dir_x)
    gmag, mdmag, lomag, himag = summ("mag", mag_x)

    print("\n" + "=" * 76)
    print(f"[A] 전체 {N} 시퀀스 — yaw 드리프트 0→5°/s 방향/크기 배수 (per-sequence)")
    print("=" * 76)
    print(f"  방향 배수: 기하평균 {gd:.2f}× (95% CI {lod:.2f}–{hid:.2f}), 중앙값 {mdd:.2f}×")
    print(f"  크기 배수: 기하평균 {gmag:.2f}× (95% CI {lomag:.2f}–{himag:.2f}), 중앙값 {mdmag:.2f}×")
    n_dirgt = int(np.sum(dir_x > mag_x))
    print(f"  방향배수 > 크기배수 시퀀스: {n_dirgt}/{N} ({100*n_dirgt/N:.0f}%)")
    print(f"  pooled 방향오차 0→5: {np.mean([r['dir0'] for r in recs]):.3f} → "
          f"{np.mean([r['dir5'] for r in recs]):.3f} m")
    print(f"  pooled 크기오차 0→5: {np.mean([r['mag0'] for r in recs]):.3f} → "
          f"{np.mean([r['mag5'] for r in recs]):.3f} m")

    print("\n[B] 카테고리별 방향 배수 기하평균")
    for cat in CATEGORIES:
        sub = [r for r in recs if r["cat"] == cat]
        if not sub:
            continue
        g = float(np.exp(np.mean(np.log([r["dir_x"] for r in sub]))))
        gm2 = float(np.exp(np.mean(np.log([r["mag_x"] for r in sub]))))
        print(f"  {cat:<10} n={len(sub):>2}  방향 {g:>5.2f}×  크기 {gm2:>4.2f}×")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("category,sequence,n_win,dir0,dir5,mag0,mag5,dir_x,mag_x\n")
            for r in recs:
                f.write(f"{r['cat']},{r['seq']},{r['nw']},{r['dir0']:.4f},{r['dir5']:.4f},"
                        f"{r['mag0']:.4f},{r['mag5']:.4f},{r['dir_x']:.4f},{r['mag_x']:.4f}\n")
        print(f"\n[OK] CSV 저장: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
