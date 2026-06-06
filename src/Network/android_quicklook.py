# -*- coding: utf-8 -*-
"""Android CSV 궤적 quicklook — RotVec(ga) vs body 적분 형태 확인 (GT 없음).
폐루프면 시작-끝 오프셋(loop-closure)을 GT-free 지표로 쓸 수 있다.
cd src/Network; KMP_DUPLICATE_LIB_OK=TRUE python android_quicklook.py --csv ../../clean_walk_30s_001.csv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).parent))
from offline_eval import WINDOW_LEN, FS, load_android, load_model, window_to_gravity_aligned, window_yaw0  # type: ignore
LOG = Path(r"D:\mobile\imu_android\logs")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model_dir", default="src/Network/out_classifier2")
    args = ap.parse_args()
    print("[1] 모델 로드"); net, _p, mean, std = load_model(args.model_dir)
    print(f"[2] Android CSV 로드: {args.csv}")
    data = load_android(args.csv, calib_sec=2.0, linacc_scale=1.0)
    dur = len(data["acc"]) / FS
    print(f"    {len(data['acc'])} samples, {dur:.1f}s @100Hz")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for k, (frame, col, lab) in enumerate([("ga", "#1f77b4", "ga (RotVec 절대 yaw)"), ("body", "#d62728", "body (정렬 OFF)")]):
        p = dr(net, data, mean, std, frame)
        plen = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        endoff = float(np.linalg.norm(p[-1] - p[0]))
        print(f"    {frame:<5}: path_len {plen:6.2f} m, start-end 오프셋(loop-closure) {endoff:6.2f} m  (LC/len {100*endoff/max(plen,1e-6):4.1f}%)")
        ax[k].plot(p[:, 0], p[:, 1], "-", color=col, lw=1.8)
        ax[k].scatter([0], [0], c="g", s=60, zorder=5, label="시작")
        ax[k].scatter([p[-1, 0]], [p[-1, 1]], c=col, marker="x", s=60, zorder=5, label="끝")
        ax[k].set_aspect("equal", "datalim"); ax[k].grid(ls=":", alpha=0.4); ax[k].legend(fontsize=8)
        ax[k].set_title(f"{lab}\npath {plen:.1f}m, LC {endoff:.1f}m", fontsize=10)
        ax[k].set_xlabel("X (m)"); ax[k].set_ylabel("Y (m)")
    fig.suptitle(f"Android quicklook — {Path(args.csv).name}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = LOG / f"android_quicklook_{Path(args.csv).stem}.png"; fig.savefig(out, dpi=150)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
