# -*- coding: utf-8 -*-
"""최근 pull 3개 Android CSV 궤적 분석 — RotVec(ga) dead-reckoning.
각: duration, path_len, 평균속도, 시작-끝 오프셋. 3-panel 궤적.
cd src/Network; KMP_DUPLICATE_LIB_OK=TRUE python android_analyze3.py
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

CSV_DIR = Path(r"D:\mobile\imu_android\csv\imu_csv")
FILES = ["imu_record_1780543327203.csv", "imu_record_1780543367670.csv", "imu_record_1780543437088.csv"]
LOG = Path(r"D:\mobile\imu_android\logs")


def dr(net, data, mean, std, frame="ga"):
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
    SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0 / 9.81
    print(f"[1] 모델 로드 (linacc_scale={SCALE:.4f})"); net, _p, mean, std = load_model("src/Network/out_classifier2")
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    for k, fn in enumerate(FILES):
        f = str(CSV_DIR / fn)
        data = load_android(f, calib_sec=2.0, linacc_scale=SCALE)
        dur = len(data["acc"]) / FS
        p = dr(net, data, mean, std, "ga")
        plen = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        endoff = float(np.linalg.norm(p[-1] - p[0]))
        spd = plen / dur
        print(f"  {fn}")
        print(f"     dur {dur:5.1f}s  path {plen:6.2f}m  평균속도 {spd:4.2f} m/s  시작-끝 {endoff:6.2f}m")
        ax[k].plot(p[:, 0], p[:, 1], "-", color="#1f77b4", lw=1.8)
        ax[k].scatter([0], [0], c="g", s=70, zorder=5, label="시작")
        ax[k].scatter([p[-1, 0]], [p[-1, 1]], c="r", marker="x", s=70, zorder=5, label="끝")
        ax[k].set_aspect("equal", "datalim"); ax[k].grid(ls=":", alpha=0.4); ax[k].legend(fontsize=8)
        ax[k].set_xlabel("X (m)"); ax[k].set_ylabel("Y (m)")
        ax[k].set_title(f"{fn.replace('imu_record_','').replace('.csv','')}\n{dur:.0f}s · path {plen:.1f}m · {spd:.2f}m/s · LC {endoff:.1f}m", fontsize=9)
    fig.suptitle("Android 최근 3개 궤적 — RotVec(ga) dead-reckoning", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = LOG / "android_analyze3.png"; fig.savefig(out, dpi=150)
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
