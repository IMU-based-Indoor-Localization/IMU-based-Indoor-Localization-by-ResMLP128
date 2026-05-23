"""
overlay_tracks.py — 단말 exportPath() 로 저장된 trackPoints CSV 두 개를 겹쳐 비교
================================================================================
[P60] 옵션 B 의 외부 도구.

사용 흐름:
  1. 앱 메뉴 → "EKF 모드 (비교용)" → EKF_CURRENT 선택 → 시작 → 보행 → 정지 → 메뉴 → 경로 내보내기
     → /sdcard/Android/data/com.imulocal/files/track_EKF_CURRENT_<ts>.csv
  2. 다시 EKF_TLIO 로 같은 경로 보행 → 두 번째 CSV 저장
  3. adb pull 로 PC 에 두 파일 가져와 본 도구로 겹치기:
     python tools/overlay_tracks.py \\
        track_EKF_CURRENT_*.csv  track_EKF_TLIO_*.csv \\
        --out logs/ekf_mode_overlay.png

CSV 형식 (MainActivity.exportPath() — P60):
  # mode=<EkfMode>
  # n_points=<N>
  x_m,y_m
  <x>,<y>
  ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass


COLORS = ["#1565C0", "#E65100", "#388E3C", "#6A1B9A"]  # 파랑, 주황, 초록, 보라


def load_track(csv_path):
    """반환: (label, np.ndarray[N,2])."""
    mode = None
    pts = []
    with open(csv_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("#"):
                if ln.startswith("# mode="):
                    mode = ln.split("=", 1)[1].strip()
                continue
            if ln.startswith("x_m"):
                continue
            try:
                x, y = ln.split(",")[:2]
                pts.append((float(x), float(y)))
            except ValueError:
                continue
    arr = np.array(pts, dtype=np.float64)
    label = mode or Path(csv_path).stem
    return label, arr


def summarize(arr):
    if len(arr) < 2:
        return 0.0, 0.0
    path_len = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
    end_off  = float(np.linalg.norm(arr[-1] - arr[0]))
    return path_len, end_off


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="trackPoints CSV 둘 이상")
    ap.add_argument("--out", default=None, help="겹친 그림 PNG 경로")
    args = ap.parse_args()

    if len(args.csvs) < 2:
        print("[!] CSV 두 개 이상 필요 (비교 목적).")
        sys.exit(1)

    tracks = [load_track(p) for p in args.csvs]

    print(f"\n{'='*64}\n  trackPoints 겹치기 비교\n{'='*64}")
    print(f"  {'label':<24}  {'n':>5}  {'path(m)':>9}  {'end_off(m)':>11}")
    for (lbl, arr), src in zip(tracks, args.csvs):
        pl, eo = summarize(arr)
        print(f"  {lbl:<24}  {len(arr):>5}  {pl:>9.2f}  {eo:>11.2f}")
    print()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for fn in ("Malgun Gothic", "NanumGothic", "Gulim", "DejaVu Sans"):
            try:
                matplotlib.rcParams["font.family"] = fn
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
    except ImportError:
        print("  [skip] matplotlib 없음 — plot 생략")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i, ((lbl, arr), src) in enumerate(zip(tracks, args.csvs)):
        pl, eo = summarize(arr)
        c = COLORS[i % len(COLORS)]
        ax.plot(arr[:, 0], arr[:, 1], "-", lw=1.4, alpha=0.85,
                color=c,
                label=f"{lbl}  (n={len(arr)}, path={pl:.2f}m, end_off={eo:.2f}m)")
        if len(arr) >= 1:
            ax.scatter([arr[0, 0]],  [arr[0, 1]],  color="#388E3C", s=60, zorder=5)
            ax.scatter([arr[-1, 0]], [arr[-1, 1]], color=c, s=40, marker="x", zorder=5)
    ax.set_title("EKF 모드 비교 — trackPoints overlay")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        plt.savefig(args.out, dpi=120)
        print(f"  [OK] plot 저장: {args.out}")


if __name__ == "__main__":
    main()
