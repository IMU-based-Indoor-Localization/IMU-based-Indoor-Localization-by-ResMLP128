"""
oxiod_fig7_sensitivity.py — 합성 yaw 드리프트 민감도 2패널 그림 재생성
======================================================================
보고서 그림7(fig_4_6_2_yaw_drift.png) 생성기가 repo에 없어 재작성한다.
oxiod_drift_decompose.evaluate_decompose(input_only, frame-matched 방향오차 +
회전불변 크기오차)를 8개 카테고리 최장 시퀀스에 적용해:
  (a) 방향 vs 크기 분해 — yaw 드리프트 0→5°/s 에서 방향만 선택적 붕괴(~7.7×), 크기 보존(~1.0×)
  (b) 드리프트 onset — 누적 yaw(=rate×600s 환산) 대 baseline 대비 윈도우 오차, ±10° 예산 띠
제목에 '그림 N.' 번호를 넣지 않는다(번호는 보고서 캡션에서 부여). 표 8·9 수치와 정합.

사용:
  python src/Network/oxiod_fig7_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for fn in ("Malgun Gothic", "맑은 고딕", "Gulim"):
    try:
        font_manager.findfont(fn, fallback_to_default=False)
        plt.rcParams["font.family"] = fn
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent))
from offline_eval import load_oxiod, load_model            # type: ignore
from oxiod_preproc_ablation import select_longest_sequence, CATEGORIES   # type: ignore
from oxiod_drift_decompose import evaluate_decompose        # type: ignore

LOG = Path(r"D:\mobile\imu_android\logs")
DATA = Path(r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
T_REF = 550.0  # 누적 yaw 환산 기준(s) — 표9의 0.02°/s→11°, 0.20°/s→110° 와 정합

RATES_A = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0]          # 패널(a)
RATES_B = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50] # 패널(b) onset


def main():
    print("[1] 모델 로드"); net, _p, mean, std = load_model("src/Network/out_classifier2")
    seqs = {}
    for c in CATEGORIES:
        sd = select_longest_sequence(DATA, c)
        if sd is not None:
            seqs[c] = load_oxiod(sd / "imu0_resampled.npy")
    print(f"[2] {len(seqs)} 카테고리")

    rates_all = sorted(set(RATES_A) | set(RATES_B))
    dirr, magg, mism = {}, {}, {}
    for dr in rates_all:
        ds, ms, mi = [], [], []
        for data in seqs.values():
            m = evaluate_decompose(net, data, mean, std, dr)
            ds.append(m["vec_matched"]); ms.append(m["mag_err"]); mi.append(m["vec_mismatch"])
        dirr[dr] = float(np.mean(ds)); magg[dr] = float(np.mean(ms)); mism[dr] = float(np.mean(mi))
        print(f"    drift {dr:>4}: 방향(matched) {dirr[dr]:.3f}  크기 {magg[dr]:.3f}  실좌표(mismatch) {mism[dr]:.3f}")

    base_dir = dirr[0.0]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # ── (a) 방향 vs 크기 분해 ────────────────────────────────
    da = [dirr[r] for r in RATES_A]; ma = [magg[r] for r in RATES_A]
    x = list(range(len(RATES_A)))
    ax[0].plot(x, da, "-o", color="#d62728", lw=2.0, ms=6, label="방향 예측오차 (frame-matched)")
    ax[0].plot(x, ma, "-s", color="#1f77b4", lw=2.0, ms=6, label="크기 예측오차 (회전불변)")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{r:g}" for r in RATES_A])
    ax[0].set_xlabel("yaw 드리프트 (°/s)"); ax[0].set_ylabel("윈도우 예측오차 (m)")
    ax[0].set_title("(a) 방향 vs 크기 분해 — 방향만 선택적 붕괴", fontsize=11)
    ax[0].grid(ls=":", alpha=0.4); ax[0].legend(fontsize=9, loc="center left")
    ax[0].annotate(f"{da[-1]/da[0]:.2f}×", xy=(x[-2], da[-2]), fontsize=13,
                   color="#d62728", fontweight="bold")
    ax[0].annotate(f"{ma[-1]/ma[0]:.2f}× (보존)", xy=(x[-3], ma[-1] + 0.06), fontsize=11,
                   color="#1f77b4", fontweight="bold")

    # ── (b) 드리프트 onset (실좌표 윈도우오차 vec_mismatch 기준 — 표9와 정합) ──
    base_mism = mism[0.0]
    cum = [r * T_REF for r in RATES_B]
    rel = [mism[r] / base_mism for r in RATES_B]
    ax[1].axvspan(0, 10, color="#2ca02c", alpha=0.12)
    ax[1].plot(cum, rel, "-o", color="#2ca02c", lw=2.0, ms=6)
    for r in (0.02, 0.05, 0.20):
        c, y = r * T_REF, mism[r] / base_mism
        ax[1].annotate(f"{c:.0f}°\n{y:.2f}×", xy=(c, y), xytext=(c + 8, y - 0.18),
                       fontsize=9, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax[1].axhline(1.5, color="gray", ls=":", lw=1.0)
    ax[1].text(cum[-1] * 0.45, 1.42, "누적 yaw ~10° 이내\n= 입력 OOD 회피 기준",
               color="#2ca02c", fontsize=9)
    ax[1].set_xlabel("누적 yaw 오차 (°)"); ax[1].set_ylabel("윈도우 예측오차 (baseline 대비)")
    ax[1].set_title("(b) 드리프트 onset — 누적량이 본질", fontsize=11)
    ax[1].grid(ls=":", alpha=0.4)

    fig.suptitle("합성 yaw 드리프트 민감도 (input_only, 8 카테고리 평균)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = LOG / "fig_4_6_2_yaw_drift.png"
    fig.savefig(out, dpi=150)
    print(f"[OK] {out}  (방향 0→5: {da[-1]/da[0]:.2f}×, 크기 {ma[-1]/ma[0]:.2f}×)")


if __name__ == "__main__":
    main()
