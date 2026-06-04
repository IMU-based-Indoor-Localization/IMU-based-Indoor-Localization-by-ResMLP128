# -*- coding: utf-8 -*-
"""A1 — ~10° yaw 예산을 *순수 예측(frame-matched)* 지표로 재유도 (순환성 제거).
기존 onset(표 4.6.3)은 gt_true(오염) 지표 = 기하 프레임 회전 포함.
본 스크립트는 세밀 rate sweep 에서 세 지표를 분리해 각각의 예산을 뽑는다:
  · frame-matched 방향오차 (vec_matched)  : 입력 프레임 내 순수 예측 OOD
  · 회전불변 크기오차      (mag_err)        : 크기 예측 품질(기하 무관)
  · 오염 총효과            (vec_mismatch)   : 기존 예산이 쓴 지표(기하+OOD)
→ 셋의 누적 yaw 의존성을 비교해, 예산이 *진짜 OOD* 인지 *기하 아티팩트* 인지 판정.

실행:
  KMP_DUPLICATE_LIB_OK=TRUE python src/Network/oxiod_a1_framematched_budget.py \
    --data_dir D:/EKF_DATASET/TLIO_Oxford_Dataset --model_dir src/Network/out_classifier2 \
    --out logs/oxiod_a1_framematched.csv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
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
from offline_eval import FS, load_oxiod, load_model           # type: ignore
from oxiod_preproc_ablation import select_longest_sequence, CATEGORIES  # type: ignore
from oxiod_drift_decompose import evaluate_decompose          # type: ignore

RATES = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 2.0, 5.0]
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
            seqs[c] = load_oxiod(s / "imu0_resampled.npy"); durs[c] = len(seqs[c]["acc"]) / FS
    avg = float(np.mean(list(durs.values())))
    print(f"[2] {len(seqs)} 카테고리, 평균 길이 {avg:.0f}s")

    rows = []
    for r in RATES:
        ms = [evaluate_decompose(net, d, mean, std, r) for d in seqs.values()]
        vm = float(np.mean([m["vec_matched"] for m in ms]))    # frame-matched 방향
        me = float(np.mean([m["mag_err"] for m in ms]))        # 회전불변 크기
        vc = float(np.mean([m["vec_mismatch"] for m in ms]))   # 오염 총효과
        rows.append((r, r * avg, vm, me, vc))
        print(f"  rate {r:>4.2f}°/s  누적 {r*avg:>5.0f}°  | frame-matched {vm:.3f}  크기 {me:.3f}  오염 {vc:.3f}")

    b_vm, b_me, b_vc = rows[0][2], rows[0][3], rows[0][4]
    print("\n[3] baseline 대비 배수")
    print(f"{'rate':>5} {'누적°':>6} | {'frame-matched':>14} {'크기(회전불변)':>14} {'오염(총)':>10}")
    for r, cy, vm, me, vc in rows:
        print(f"{r:>5.2f} {cy:>6.0f} | {vm/b_vm:>13.2f}x {me/b_me:>13.2f}x {vc/b_vc:>9.2f}x")

    def thresh(idx, factor):
        for r, cy, vm, me, vc in rows:
            v = (vm, me, vc)[idx - 2]
            base = (b_vm, b_me, b_vc)[idx - 2]
            if v / base >= factor:
                return cy
        return None
    print("\n[4] 예산(누적 yaw, 해당 지표가 baseline 1.5×/2× 처음 초과)")
    for name, idx in (("frame-matched 방향", 2), ("크기(회전불변)", 3), ("오염(총효과)", 4)):
        t15, t2 = thresh(idx, 1.5), thresh(idx, 2.0)
        print(f"  {name:<16}: 1.5× @ {t15}°,  2× @ {t2}°")
    print("\n[판정] 오염 지표만 누적 yaw 따라 급증하고 frame-matched·크기는 평탄 → 예산은 *기하 아티팩트*.")
    print("       frame-matched 도 누적 yaw 따라 급증 → 예산은 *진짜 입력 OOD* (메커니즘 견고).")

    # figure
    cy = [r[1] for r in rows]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.8))
    ax.plot(cy, [r[2] / b_vm for r in rows], "o-", color="#d62728", lw=2, label="frame-matched 방향오차 (순수 OOD)")
    ax.plot(cy, [r[3] / b_me for r in rows], "s-", color="#1f77b4", lw=2, label="크기오차 (회전불변)")
    ax.plot(cy, [r[4] / b_vc for r in rows], "^--", color="#7f7f7f", lw=1.8, label="오염 총효과 (기존 예산 지표)")
    ax.axvline(10, ls=":", color="#2ca02c", lw=1.4); ax.text(11, ax.get_ylim()[1]*0.9, "~10°", color="#2ca02c")
    ax.set_xlabel("누적 yaw 오차 (°)"); ax.set_ylabel("baseline 대비 배수")
    ax.set_title("A1. 예산 재유도 — 순수 예측(frame-matched) vs 오염 지표", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(LOG / "fig_a1_framematched_budget.png", dpi=150)
    print(f"\n[OK] {LOG / 'fig_a1_framematched_budget.png'}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("rate_deg_s,cum_yaw_deg,frame_matched,mag_err,contaminated\n")
            for r in rows:
                f.write(",".join(f"{x:.4f}" for x in r) + "\n")
        print(f"[OK] CSV {args.out}")


if __name__ == "__main__":
    main()
