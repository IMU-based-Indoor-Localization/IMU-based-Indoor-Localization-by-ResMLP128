# -*- coding: utf-8 -*-
"""현장 4 walk 점검 — track(궤적) + marks(웨이포인트 est) 오버레이·통계.
파일↔경로 매핑 확정, 마크가 궤적 위에 있는지, 루프/왕복 복귀오차 확인.
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
D = Path(r"D:\mobile\imu_android\csv\walks"); LOG = Path(r"D:\mobile\imu_android\logs")

WALKS = [
    ("A (최신, 마크5) → 경로4? 321-1 루프", "track_PATH_B_1780882569833.csv", "marks_1780882572108.csv"),
    ("B (마크3)", "track_PATH_B_1780881872837.csv", "marks_1780881874145.csv"),
    ("C (마크3)", "track_PATH_B_1780881680433.csv", "marks_1780881681638.csv"),
    ("D (최구, 마크3)", "track_PATH_B_1780881588099.csv", "marks_1780881586491.csv"),
]


def load_track(p):
    xs, ys = [], []
    for ln in Path(p).read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or ln.startswith("x_m") or not ln.strip():
            continue
        a, b = ln.split(","); xs.append(float(a)); ys.append(float(b))
    return np.column_stack([xs, ys])


def load_marks(p):
    r = []
    for ln in Path(p).read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or ln.startswith("idx") or not ln.strip():
            continue
        c = ln.split(","); r.append((float(c[1]), float(c[2]), int(c[3])))
    return r


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, (label, tf, mf) in zip(axes.ravel(), WALKS):
        tr = load_track(D / tf); mk = load_marks(D / mf)
        M = np.array([[m[0], m[1]] for m in mk])
        plen = float(np.linalg.norm(np.diff(tr, axis=0), axis=1).sum())
        net = float(np.linalg.norm(tr[-1] - tr[0]))
        lc = float(np.linalg.norm(M[-1] - M[0]))  # 첫↔끝 마크 (왕복/루프 복귀오차)
        # 마크간 거리(est)
        seg = [float(np.linalg.norm(M[i+1]-M[i])) for i in range(len(M)-1)]
        dur = (mk[-1][2] - mk[0][2]) / 1000.0
        print(f"[{label}]  track {tf.split('_')[-1]}")
        print(f"   마크 {len(mk)}개 · path {plen:.1f}m · track net {net:.1f}m · 마크#1↔#끝 {lc:.1f}m · 마크시간 {dur:.0f}s")
        print(f"   마크간 거리(est): {', '.join(f'{s:.1f}' for s in seg)} m")
        ax.plot(tr[:, 0], tr[:, 1], "-", color="#1f77b4", lw=1.3, alpha=0.8, label="궤적(track)")
        ax.plot(M[:, 0], M[:, 1], "o", color="#d62728", ms=9, label="마크")
        for i, (x, y, _t) in enumerate(mk):
            ax.annotate(str(i+1), (x, y), fontsize=11, fontweight="bold", color="#6A1B9A",
                        xytext=(5, 5), textcoords="offset points")
        ax.scatter([tr[0,0]],[tr[0,1]], c="g", s=70, marker="s", zorder=5, label="궤적시작")
        ax.set_aspect("equal","datalim"); ax.grid(ls=":", alpha=0.4); ax.legend(fontsize=7)
        ax.set_title(f"{label}\n마크{len(mk)} · path{plen:.0f}m · net{net:.0f}m · 복귀{lc:.1f}m", fontsize=9)
        ax.set_xlabel("X(m)"); ax.set_ylabel("Y(m)")
    fig.suptitle("현장 4 walk 점검 — 궤적 + 마크 (est)", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.96])
    out = LOG / "android_walk_inspect.png"; fig.savefig(out, dpi=130)
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
