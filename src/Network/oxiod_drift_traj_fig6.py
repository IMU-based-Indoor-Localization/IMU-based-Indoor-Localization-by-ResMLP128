# -*- coding: utf-8 -*-
"""그림 6 재설계 — 입력 yaw 드리프트 궤적을 §4.4/그림7 스타일(첫 N초·한 사이클)로.
기존(handbag_3 전체, GT 블롭)이 루프형이라 가독성 낮음 → handheld_1 첫 30초(한 사이클)로.
input_only(출력은 정답 yaw0): 입력 정렬 yaw에만 드리프트 → 예측 방향 붕괴가 궤적으로 가시화.

실행:
  KMP_DUPLICATE_LIB_OK=TRUE python src/Network/oxiod_drift_traj_fig6.py \
    --data_dir D:/EKF_DATASET/TLIO_Oxford_Dataset --model_dir src/Network/out_classifier2 \
    --seq oxford_handheld_1 --sec 30 --rates 0,2,5
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from offline_eval import WINDOW_LEN, FS, load_oxiod, load_model, window_to_gravity_aligned, window_yaw0  # type: ignore
from oxiod_yaw_drift_ablation import make_drifted_quat  # type: ignore
LOG = Path(r"D:\mobile\imu_android\logs")


def dr_seg(net, data, mean, std, dr, S):
    import torch
    acc, gyr, quat, pos = data["acc"], data["gyr"], data["quat"], data["pos"]
    quat_in = make_drifted_quat(quat, dr)        # 입력만 드리프트
    T = len(acc); starts = list(range(0, T - WINDOW_LEN, WINDOW_LEN))
    pred = [np.zeros(2)]
    with torch.no_grad():
        for s in starts:
            e = s + WINDOW_LEN
            imu = window_to_gravity_aligned(acc, gyr, quat_in, s, e, frame="ga")
            x = torch.from_numpy(((imu - mean) / std).T.copy()).float().unsqueeze(0)
            y, _c, _l = net(x); d = y[0].numpy()
            y0 = window_yaw0(quat, s)             # 출력은 정답 yaw0
            c, sn = np.cos(y0), np.sin(y0)
            pred.append(pred[-1] + np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]]))
    pred = np.array(pred)
    idx = [0] + [s + WINDOW_LEN for s in starts]
    gt = pos[idx, :2] - pos[0, :2]
    keep = [k for k, ix in enumerate(idx) if ix <= S]
    pred, gt = pred[keep], gt[keep]
    n = min(len(pred), len(gt))
    ate = float(np.sqrt(np.mean(np.linalg.norm(pred[:n] - gt[:n], axis=1) ** 2)))
    return pred[:n], gt[:n], ate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
    ap.add_argument("--model_dir", default="src/Network/out_classifier2")
    ap.add_argument("--seq", default="oxford_handheld_1")
    ap.add_argument("--sec", type=float, default=30.0)
    ap.add_argument("--rates", default="0,2,5")
    args = ap.parse_args()
    print("[1] 모델 로드"); net, _p, mean, std = load_model(args.model_dir)
    data = load_oxiod(Path(args.data_dir) / args.seq / "imu0_resampled.npy")
    dur = len(data["acc"]) / FS
    eff = min(args.sec, dur)
    S = int(eff * FS)
    rates = [float(x) for x in args.rates.split(",")]
    cols = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    seg = "전체" if args.sec >= dur else f"첫 {eff:.0f}초"
    print(f"[2] {args.seq} {seg} (dur {dur:.0f}s), rates {rates}")

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.8))
    gt_drawn = False
    for k, dr in enumerate(rates):
        pred, gt, ate = dr_seg(net, data, mean, std, dr, S)
        if not gt_drawn:
            ax.plot(gt[:, 0], gt[:, 1], "-", color="black", lw=2.6, label="GT", zorder=5); gt_drawn = True
        cum = dr * eff
        lab = "드리프트 0 (정상)" if dr == 0 else f"드리프트 {dr:g}°/s (누적 {cum:.0f}°)"
        ax.plot(pred[:, 0], pred[:, 1], "-", color=cols[k % len(cols)], lw=1.9, alpha=0.9,
                label=f"{lab}  ATE {ate:.1f}m")
        print(f"    dr {dr:g}: 누적 {cum:.0f}°, ATE {ate:.2f} m")
    ax.scatter([0], [0], c="k", s=45, zorder=6, label="시작")
    ax.set_aspect("equal", "datalim"); ax.legend(fontsize=8.5, loc="best"); ax.grid(ls=":", alpha=0.4)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"입력 yaw 드리프트 궤적 (input_only, {seg}) — {args.seq}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = LOG / "fig_4_6_2_traj.png"; fig.savefig(out, dpi=150)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
