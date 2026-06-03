# -*- coding: utf-8 -*-
"""§4.6 그림 생성: (A) 프레임 정렬 ablation, (B) yaw 드리프트 민감도.
fig A = 논문 표 4.6.1/4.6.2 승인값(하드코딩), fig B = decompose/onset CSV 평균(표와 정합)."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 (Windows Malgun Gothic)
for fn in ("Malgun Gothic", "맑은 고딕", "Gulim"):
    try:
        font_manager.findfont(fn, fallback_to_default=False); plt.rcParams["font.family"] = fn; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
LOG = r"D:\mobile\imu_android\logs"

def read_csv(name):
    with open(LOG + "\\" + name, encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ───────────────────────── Figure A : 4.6.1 프레임 정렬 ablation ─────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
# 좌: pooled (표 4.6.1)
pooled_lab = ["body→ga\n(회전 정렬 생략)", "yaw→ga\n(정렬 입도)"]
pooled_val = [2.44, 1.08]
pooled_ci  = [(2.08, 2.86), (1.02, 1.13)]
err = [[v-lo for v,(lo,hi) in zip(pooled_val,pooled_ci)], [hi-v for v,(lo,hi) in zip(pooled_val,pooled_ci)]]
bars = ax[0].bar(pooled_lab, pooled_val, yerr=err, capsize=6,
                 color=["#d62728", "#9ecae1"], edgecolor="black", width=0.55)
ax[0].axhline(1.0, ls="--", c="gray", lw=1)
for b, v in zip(bars, pooled_val):
    ax[0].text(b.get_x()+b.get_width()/2, v+0.12, f"{v:.2f}×", ha="center", fontweight="bold")
ax[0].set_ylabel("ATE 비율 (정렬 OFF / ga)")
ax[0].set_title("(a) 전체 152 시퀀스 — 정렬 효과\n대효과: 정렬 존재  /  소효과: 입도", fontsize=10)
ax[0].set_ylim(0, 3.3)
# 우: per-category body/ga (표 4.6.2)
cat = ["pocket","handbag","large","multi","handheld","running","slow","trolley"]
val = [5.62, 2.38, 2.23, 2.17, 1.33, 1.10, 0.86, 0.80]
colors = ["#d62728" if v >= 2 else ("#ff9896" if v >= 1.2 else "#c7c7c7") for v in val]
bars = ax[1].bar(cat, val, color=colors, edgecolor="black", width=0.7)
ax[1].axhline(1.0, ls="--", c="gray", lw=1)
for b, v in zip(bars, val):
    ax[1].text(b.get_x()+b.get_width()/2, v+0.12, f"{v:.2f}", ha="center", fontsize=8.5)
ax[1].set_ylabel("body/ga (회전 정렬 효과)")
ax[1].set_title("(b) 휴대 상태별 회전 정렬 효과\n자세 변동 큰 상태일수록 정렬 중요", fontsize=10)
ax[1].set_ylim(0, 6.4)
plt.setp(ax[1].get_xticklabels(), rotation=30, ha="right", fontsize=8.5)
fig.suptitle("그림 4. 입력 전처리 정렬 ablation (OxIOD GT, 1D-ResMLP128)", fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(LOG + r"\fig_4_6_1_frame_ablation.png", dpi=150)
print("saved fig_4_6_1_frame_ablation.png")

# ───────────────────────── Figure B : 4.6.2 yaw 드리프트 민감도 ─────────────────────────
dec = read_csv("oxiod_drift_decompose.csv")
ons = read_csv("oxiod_drift_onset.csv")

def mean_by(rows, key, col):
    drs = sorted(set(float(r[key]) for r in rows))
    return drs, [float(np.mean([float(r[col]) for r in rows if float(r[key]) == d])) for d in drs]

drs, direction = mean_by(dec, "drift_deg_s", "vec_matched")   # 방향 (frame-matched)
_,   magnitude = mean_by(dec, "drift_deg_s", "mag_err")        # 크기 (회전불변)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(len(drs))
ax[0].plot(x, direction, "o-", c="#d62728", lw=2, label="방향 예측오차 (frame-matched)")
ax[0].plot(x, magnitude, "s-", c="#1f77b4", lw=2, label="크기 예측오차 (회전불변)")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{d:g}" for d in drs])
ax[0].set_xlabel("yaw 드리프트 (°/s)"); ax[0].set_ylabel("윈도우 예측오차 (m)")
ax[0].annotate(f"7.70×", xy=(x[-1], direction[-1]), xytext=(x[-1]-2.4, direction[-1]+0.12),
               color="#d62728", fontweight="bold")
ax[0].annotate(f"1.00× (보존)", xy=(x[-1], magnitude[-1]), xytext=(x[-1]-2.0, 0.18),
               color="#1f77b4", fontweight="bold")
ax[0].legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 0.86)); ax[0].set_ylim(0, 1.4)
ax[0].set_title("(a) 방향 vs 크기 분해 — 방향만 선택적 붕괴", fontsize=10)

# onset: 누적 yaw vs baseline 대비 윈도우 오차
AVGDUR = 553.0  # s (논문: 0.02→11°, 0.05→28°, 0.2→110°)
drs2, disp = mean_by(ons, "drift_deg_s", "disp_rmse_m")
base = disp[0]
cum = [d*AVGDUR for d in drs2]
ratio = [v/base for v in disp]
ax[1].plot(cum, ratio, "o-", c="#2ca02c", lw=2)
ax[1].axhline(1.5, ls=":", c="gray", lw=1)
offs = {11:(16, 0.10), 28:(14, -0.20), 110:(8, -0.55)}
for d, c, v in zip(drs2, cum, ratio):
    key = 110 if abs(c-110) < 2 else round(c)
    if key in offs:
        dx, dy = offs[key]
        ax[1].annotate(f"{c:.0f}°\n{v:.2f}×", xy=(c, v), xytext=(c+dx, v+dy), fontsize=8,
                       arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
ax[1].axvspan(0, 10, color="#2ca02c", alpha=0.10)
ax[1].text(118, 1.25, "누적 yaw ~10° 이내\n= 입력 OOD 회피 기준", fontsize=8.5, color="#2ca02c")
ax[1].set_xlabel("누적 yaw 오차 (°)"); ax[1].set_ylabel("윈도우 예측오차 (baseline 대비)")
ax[1].set_title("(b) 드리프트 onset — 누적량이 본질", fontsize=10)
ax[1].set_ylim(0.8, 4.4)
fig.suptitle("그림 5. 합성 yaw 드리프트 민감도 (input_only, 8 카테고리 평균)", fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(LOG + r"\fig_4_6_2_yaw_drift.png", dpi=150)
print("saved fig_4_6_2_yaw_drift.png")
print(f"direction 0->5: {direction[0]:.3f} -> {direction[-1]:.3f} = {direction[-1]/direction[0]:.2f}x")
print(f"magnitude 0->5: {magnitude[0]:.3f} -> {magnitude[-1]:.3f} = {magnitude[-1]/magnitude[0]:.2f}x")
print(f"onset cum/ratio: " + ", ".join(f"{c:.0f}d={r:.2f}x" for c,r in zip(cum, ratio)))
