# -*- coding: utf-8 -*-
"""§4.6.1 재설계: 자세 구분 없이 ga/yaw/body 세 전처리를 풀(pooled) 직접 비교.
fig5 와 동일한 선형(line) 스타일. 전체 152 시퀀스(oxiod_significance.csv)."""
import csv, math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for fn in ("Malgun Gothic", "맑은 고딕", "Gulim"):
    try:
        font_manager.findfont(fn, fallback_to_default=False); plt.rcParams["font.family"] = fn; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
LOG = r"D:\mobile\imu_android\logs"

with open(LOG + r"\oxiod_significance.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
ga  = np.array([float(r["ga"])   for r in rows])
yaw = np.array([float(r["yaw"])  for r in rows])
body= np.array([float(r["body"]) for r in rows])
N = len(rows)
pct = (np.arange(1, N + 1) / N) * 100
geo = lambda a: math.exp(float(np.mean(np.log(a))))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

# (a) 프레임별 ATE 분포 — 각 프레임 독립 정렬(역CDF), log-y
ax[0].plot(pct, np.sort(ga),  "-", color="#1f77b4", lw=2.2, label="ga  (중력 정렬 ON · 매 시각 자세)")
ax[0].plot(pct, np.sort(yaw), "-", color="#2ca02c", lw=2.2, label="yaw (시작 시점 단일 회전)")
ax[0].plot(pct, np.sort(body),"-", color="#d62728", lw=2.2, label="body (회전 정렬 OFF)")
ax[0].set_yscale("log")
ax[0].set_xlabel("시퀀스 백분위 (%)")
ax[0].set_ylabel("ATE RMSE$_{xy}$ (m, log)")
ax[0].legend(fontsize=8.5, loc="upper left")
ax[0].set_title("(a) 프레임별 ATE 분포 (전체 152 시퀀스)\nbody 곡선이 전 구간에서 위 — 정렬 존재가 지배", fontsize=10)
ax[0].grid(True, which="both", ls=":", alpha=0.4)

# (b) 쌍별 비율 (정렬 OFF / ga), 시퀀스별 오름차순, log-y
bg = np.sort(body / ga); yg = np.sort(yaw / ga)
gm_bg, gm_yg = geo(body / ga), geo(yaw / ga)
fr_bg = float(np.mean(body / ga > 1) * 100); fr_yg = float(np.mean(yaw / ga > 1) * 100)
ax[1].plot(pct, bg, "-", color="#d62728", lw=2.2, label=f"body/ga  (기하평균 {gm_bg:.2f}×)")
ax[1].plot(pct, yg, "-", color="#1f77b4", lw=2.2, label=f"yaw/ga  (기하평균 {gm_yg:.2f}×)")
ax[1].axhline(1.0, ls="--", color="gray", lw=1)
ax[1].axhline(gm_bg, ls=":", color="#d62728", lw=1.2)
ax[1].axhline(gm_yg, ls=":", color="#1f77b4", lw=1.2)
ax[1].set_yscale("log")
ax[1].set_xlabel("시퀀스 백분위 (비율 오름차순, %)")
ax[1].set_ylabel("ATE 비율 (정렬 OFF / ga)")
ax[1].legend(fontsize=8.5, loc="upper left")
ax[1].set_title(f"(b) 쌍별 비율 — 자세 구분 없음\nbody/ga: {fr_bg:.0f}%가 >1 (대효과) · yaw/ga: {fr_yg:.0f}%가 >1 (소효과)", fontsize=10)
ax[1].grid(True, which="both", ls=":", alpha=0.4)

fig.suptitle("그림 4. 입력 전처리 정렬 ablation — ga vs yaw vs body 풀 비교 (자세 구분 없음)", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(LOG + r"\fig_4_6_1_frame_ablation.png", dpi=150)
print("saved fig_4_6_1_frame_ablation.png")
print(f"body/ga geomean={gm_bg:.3f}  ({fr_bg:.0f}% >1) ; yaw/ga geomean={gm_yg:.3f}  ({fr_yg:.0f}% >1)")
print(f"ATE median ga={np.median(ga):.2f} yaw={np.median(yaw):.2f} body={np.median(body):.2f}")
