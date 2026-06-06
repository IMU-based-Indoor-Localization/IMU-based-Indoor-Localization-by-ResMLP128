# -*- coding: utf-8 -*-
"""단말 경로 내보내기(track_PATH_B_*.csv) = on-device PATH_B 궤적 확인.
오프라인 하니스(imu_record raw)와 달리 단말 PDR-hybrid/adaptive-scale 적용된 최종 출력.
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

CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\mobile\imu_android\csv\track\track_PATH_B_1780543240642.csv")
LOG = Path(r"D:\mobile\imu_android\logs")


def main():
    xs, ys = [], []
    for line in CSV.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or line.startswith("x_m") or not line.strip():
            continue
        a, b = line.split(",")
        xs.append(float(a)); ys.append(float(b))
    p = np.column_stack([xs, ys])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    plen = float(seg.sum())
    net = float(np.linalg.norm(p[-1] - p[0]))
    bbox = (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max())
    extent = (bbox[1] - bbox[0], bbox[3] - bbox[2])
    print(f"n_points     : {len(p)}")
    print(f"path length  : {plen:.2f} m")
    print(f"net (시작-끝) : {net:.2f} m")
    print(f"끝점         : ({p[-1,0]:.2f}, {p[-1,1]:.2f})")
    print(f"bbox extent  : {extent[0]:.2f} x {extent[1]:.2f} m")
    print(f"seg mean/med : {seg.mean()*100:.1f} / {np.median(seg)*100:.1f} cm/point")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(p[:, 0], p[:, 1], "-", color="#1565C0", lw=1.6)
    ax.scatter([p[0, 0]], [p[0, 1]], c="g", s=80, zorder=5, label="시작")
    ax.scatter([p[-1, 0]], [p[-1, 1]], c="r", marker="x", s=90, zorder=5, label="끝")
    ax.set_aspect("equal", "datalim"); ax.grid(ls=":", alpha=0.4); ax.legend()
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"단말 PATH_B (on-device) — {CSV.name}\n"
                 f"path {plen:.1f}m · net {net:.1f}m · bbox {extent[0]:.0f}×{extent[1]:.0f}m", fontsize=10)
    fig.tight_layout()
    out = LOG / f"android_track_look_{CSV.stem}.png"; fig.savefig(out, dpi=140)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
