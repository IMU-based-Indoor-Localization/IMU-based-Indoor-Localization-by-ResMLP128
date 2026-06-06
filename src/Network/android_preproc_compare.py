# -*- coding: utf-8 -*-
"""실측 Android walk 전처리 비교 — 같은 입력, 프레임만 ga/yaw(window-start)/body.
A(단위보정 ÷9.81) 적용 후 스케일 정상 상태에서 전처리 효과를 본다.
출력 heading 은 셋 다 rotVec window-start yaw0 동일 → *입력 전처리* 효과만 격리.
cd src/Network; KMP_DUPLICATE_LIB_OK=TRUE python android_preproc_compare.py [csv] [scale]
"""
from __future__ import annotations
import sys
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
from offline_eval import WINDOW_LEN, FS, load_android, load_model, window_to_gravity_aligned, window_yaw0  # type: ignore

CSV = sys.argv[1] if len(sys.argv) > 1 else r"D:\mobile\imu_android\csv\imu_csv\imu_record_1780543437088.csv"
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0 / 9.81
LOG = Path(r"D:\mobile\imu_android\logs")
FRAMES = [("ga", "#1f77b4", "ga (전처리 ON, per-sample 중력정렬)"),
          ("yaw", "#2ca02c", "window-start (전처리 OFF, §3.3 근사)"),
          ("body", "#d62728", "body (전처리 없음)")]


def dr(net, data, mean, std, frame):
    import torch
    acc, gyr, quat = data["acc"], data["gyr"], data["quat"]
    T = len(acc); starts = list(range(0, T - WINDOW_LEN, WINDOW_LEN))
    pred = [np.zeros(2)]
    with torch.no_grad():
        for s in starts:
            e = s + WINDOW_LEN
            imu = window_to_gravity_aligned(acc, gyr, quat, s, e, frame=frame)
            x = torch.from_numpy(((imu - mean) / std).T.copy()).float().unsqueeze(0)
            y, _c, _l = net(x); d = y[0].numpy()
            y0 = window_yaw0(quat, s); c, sn = np.cos(y0), np.sin(y0)
            pred.append(pred[-1] + np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]]))
    return np.array(pred)


def main():
    print(f"[1] 모델 로드 (csv={Path(CSV).name}, scale={SCALE:.4f})")
    net, _p, mean, std = load_model("src/Network/out_classifier2")
    data = load_android(CSV, calib_sec=2.0, linacc_scale=SCALE)
    dur = len(data["acc"]) / FS
    paths = {}
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    for k, (frame, col, lab) in enumerate(FRAMES):
        p = dr(net, data, mean, std, frame)
        plen = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        net_off = float(np.linalg.norm(p[-1] - p[0]))
        paths[frame] = p
        print(f"  {frame:<5}: path {plen:6.2f}m  net {net_off:6.2f}m  끝({p[-1,0]:.1f},{p[-1,1]:.1f})")
        ax[k].plot(p[:, 0], p[:, 1], "-", color=col, lw=1.7)
        ax[k].scatter([0], [0], c="g", s=60, zorder=5); ax[k].scatter([p[-1,0]],[p[-1,1]], c="r", marker="x", s=70, zorder=5)
        ax[k].set_aspect("equal", "datalim"); ax[k].grid(ls=":", alpha=0.4)
        ax[k].set_title(f"{lab}\npath {plen:.1f}m · net {net_off:.1f}m", fontsize=8.5)
        ax[k].set_xlabel("X (m)"); ax[k].set_ylabel("Y (m)")
    # overlay
    for (frame, col, lab) in FRAMES:
        p = paths[frame]
        ax[3].plot(p[:, 0], p[:, 1], "-", color=col, lw=1.5, label=frame)
    ax[3].scatter([0],[0], c="g", s=60, zorder=5); ax[3].set_aspect("equal","datalim")
    ax[3].grid(ls=":", alpha=0.4); ax[3].legend(fontsize=8); ax[3].set_title("겹쳐보기", fontsize=9)
    ax[3].set_xlabel("X (m)"); ax[3].set_ylabel("Y (m)")
    # 프레임 간 끝점 거리 (전처리 민감도 정량)
    import itertools
    print("\n[프레임 간 끝점 거리 m]")
    for a, b in itertools.combinations([f[0] for f in FRAMES], 2):
        d = float(np.linalg.norm(paths[a][-1] - paths[b][-1]))
        print(f"  {a:<5} vs {b:<5}: {d:.2f} m")
    fig.suptitle(f"실측 Android 전처리 비교 — {Path(CSV).name} ({dur:.0f}s, scale={SCALE:.3f})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = LOG / f"android_preproc_compare_{Path(CSV).stem}.png"; fig.savefig(out, dpi=140)
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
