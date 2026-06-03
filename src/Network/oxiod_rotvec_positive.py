# -*- coding: utf-8 -*-
"""RotVec positive 실험 — 절대 yaw(RotVec/자력계) vs gyro-only(드리프트) 시스템 비교.
§4.6.2가 '드리프트→붕괴'(음성)를 보였다면, 본 실험은 '절대 yaw가 정확도를 회복'(양성)을 보인다.

설계:
  · 시스템 수준 비교를 위해 입력+출력 yaw 를 동일 소스로 사용(both 모드 = 실제 운용).
  · 절대(RotVec/mag proxy) = GT yaw (드리프트 0).
  · gyro-only = 등속 yaw 드리프트(현실적 gyro bias proxy) 주입.
  · 지표: ATE RMSE_xy (8 카테고리), 절대 대비 배수, 누적 yaw.

해석: 절대 yaw(=RotVec)는 baseline ATE 유지, gyro-only 는 누적 yaw 가 ~10° 예산을
초과하며 ATE 급증 → 절대 yaw 융합이 필요충분 해법임을 양성으로 입증.

실행:
  KMP_DUPLICATE_LIB_OK=TRUE python src/Network/oxiod_rotvec_positive.py \
    --data_dir D:/EKF_DATASET/TLIO_Oxford_Dataset --model_dir src/Network/out_classifier2 \
    --out logs/oxiod_rotvec_positive.csv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
except Exception:
    pass
for fn in ("Malgun Gothic", "맑은 고딕", "Gulim"):
    try:
        font_manager.findfont(fn, fallback_to_default=False); plt.rcParams["font.family"] = fn; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent))
from offline_eval import FS, load_oxiod, load_model  # type: ignore
from oxiod_preproc_ablation import select_longest_sequence, CATEGORIES  # type: ignore
from oxiod_yaw_drift_ablation import evaluate_drift  # type: ignore

# gyro-only yaw drift rate (°/s) proxies. 0 = 절대(RotVec/mag).
RATES = [0.0, 0.02, 0.05, 0.10, 0.20]
LOG = Path(r"D:\mobile\imu_android\logs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
    ap.add_argument("--model_dir", default="src/Network/out_classifier2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    print("[1] 모델 로드"); net, _p, mean, std = load_model(args.model_dir)

    seqs, durs = {}, {}
    for c in CATEGORIES:
        s = select_longest_sequence(data_dir, c)
        if s is not None:
            seqs[c] = load_oxiod(s / "imu0_resampled.npy")
            durs[c] = len(seqs[c]["acc"]) / FS
    avg_dur = float(np.mean(list(durs.values())))
    print(f"[2] {len(seqs)} 카테고리, 평균 길이 {avg_dur:.0f}s")

    # ate[rate][cat]
    rows = []
    print("[3] both 모드 ATE (절대 vs gyro 드리프트)")
    ate = {r: {} for r in RATES}
    for r in RATES:
        for c, data in seqs.items():
            a, _disp = evaluate_drift(net, data, mean, std, r, "both")
            ate[r][c] = a
            rows.append((r, c, a))
        gm = float(np.exp(np.mean(np.log([ate[r][c] for c in seqs]))))
        print(f"    drift {r:.2f}°/s  geomean ATE = {gm:.2f} m")

    base = {c: ate[0.0][c] for c in seqs}
    print()
    print("=" * 78)
    print("[A] 절대 yaw(RotVec) vs gyro-only — ATE RMSE_xy (m)")
    print("=" * 78)
    hdr = f"{'Category':<10} " + " ".join(f"{('abs' if r==0 else f'{r:g}'):>8}" for r in RATES)
    print(hdr); print("-" * 78)
    for c in seqs:
        print(f"{c:<10} " + " ".join(f"{ate[r][c]:>8.2f}" for r in RATES))
    print("-" * 78)
    geo = lambda vals: float(np.exp(np.mean(np.log(vals))))
    print(f"{'geomean':<10} " + " ".join(f"{geo([ate[r][c] for c in seqs]):>8.2f}" for r in RATES))
    print(f"{'vs abs':<10} " + " ".join(f"{geo([ate[r][c]/base[c] for c in seqs]):>7.2f}x" for r in RATES))
    print()
    print("[B] 누적 yaw(°) = rate × 평균길이")
    for r in RATES:
        print(f"    {r:.2f}°/s → 누적 {r*avg_dur:>5.0f}°   (vs abs {geo([ate[r][c]/base[c] for c in seqs]):.2f}x)")

    # ── figure ──
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.6))
    cum = [r * avg_dur for r in RATES]
    ratio = [geo([ate[r][c] / base[c] for c in seqs]) for r in RATES]
    ax.plot(cum, ratio, "o-", color="#d62728", lw=2.2)
    ax.scatter([0], [1.0], color="#1f77b4", s=120, zorder=6, label="절대 yaw (RotVec/자력계)")
    ax.axhspan(0, 1.5, color="#2ca02c", alpha=0.08)
    ax.axvline(10, ls=":", color="#2ca02c", lw=1.2)
    ax.text(11, 1.05, "~10° 예산", color="#2ca02c", fontsize=9)
    for c_, rt in zip(cum, ratio):
        if c_ > 0:
            ax.annotate(f"{rt:.1f}×", (c_, rt), textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_xlabel("gyro-only 누적 yaw 오차 (°)")
    ax.set_ylabel("ATE 배수 (절대 yaw 대비)")
    ax.set_title("RotVec(절대 yaw) vs gyro-only — 절대 yaw가 정확도 회복", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(LOG / "fig_4_7_rotvec_positive.png", dpi=150)
    print("\n[OK] saved fig_4_7_rotvec_positive.png")

    if args.out:
        op = Path(args.out); op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w", encoding="utf-8") as f:
            f.write("drift_deg_s,category,ate_rmse_m\n")
            for r, c, a in rows:
                f.write(f"{r},{c},{a:.4f}\n")
        print(f"[OK] CSV {op}")


if __name__ == "__main__":
    main()
