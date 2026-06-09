# -*- coding: utf-8 -*-
"""§4.6 궤적 비교 그림 (§4.4 스타일).
 (A) 프레임 정렬 ablation: ga/yaw/body dead-reckoning 궤적 vs GT
 (B) yaw 드리프트: 입력 정렬 drift 0/0.1/0.3°/s 궤적 vs GT (input_only)
실행:
  KMP_DUPLICATE_LIB_OK=TRUE python src/Network/oxiod_traj_figs.py \
    --data_dir D:/EKF_DATASET/TLIO_Oxford_Dataset --model_dir src/Network/out_classifier2
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
from offline_eval import (WINDOW_LEN, FS, load_oxiod, load_model,  # type: ignore
                          window_to_gravity_aligned, window_yaw0)
from oxiod_preproc_ablation import select_longest_sequence  # type: ignore
from oxiod_yaw_drift_ablation import make_drifted_quat  # type: ignore
LOG = Path(r"D:\mobile\imu_android\logs")


def dr_path(net, data, mean, std, frame="ga", quat_in=None, quat_out=None):
    """dead-reckoning 누적 궤적 + GT 반환. (oxiod_preproc_ablation.evaluate_ate 와 동일 적분)"""
    import torch
    acc, gyr, quat, pos = data["acc"], data["gyr"], data["quat"], data["pos"]
    if quat_in is None:  quat_in = quat
    if quat_out is None: quat_out = quat
    T = len(acc); starts = list(range(0, T - WINDOW_LEN, WINDOW_LEN))
    pred = [np.zeros(2)]
    with torch.no_grad():
        for s in starts:
            e = s + WINDOW_LEN
            imu = window_to_gravity_aligned(acc, gyr, quat_in, s, e, frame=frame)
            x = torch.from_numpy(((imu - mean) / std).T.copy()).float().unsqueeze(0)
            y, _c, _l = net(x); d = y[0].numpy()
            yaw0 = window_yaw0(quat_out, s)
            c, sn = np.cos(yaw0), np.sin(yaw0)
            pred.append(pred[-1] + np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]]))
    pred = np.array(pred)
    gt_idx = [0] + [s + WINDOW_LEN for s in starts]
    gt = pos[gt_idx, :2] - pos[0:1, :2]
    n = min(len(pred), len(gt))
    ate = float(np.sqrt(np.mean(np.linalg.norm(pred[:n] - gt[:n], axis=1) ** 2)))
    return pred, gt, ate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
    ap.add_argument("--model_dir", default="src/Network/out_classifier2")
    ap.add_argument("--catA", default="pocket")   # 프레임 ablation 예시
    ap.add_argument("--catB", default="handbag")  # 드리프트 예시
    ap.add_argument("--seqA", default=None, help="명시 시퀀스 폴더명 (catA 무시)")
    ap.add_argument("--seqB", default=None, help="명시 시퀀스 폴더명 (catB 무시)")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    print(f"[1] 모델 로드"); net, _p, mean, std = load_model(args.model_dir)

    # ── (A) 프레임 ablation 궤적 ──────────────────────────────
    seqA = (data_dir / args.seqA) if args.seqA else select_longest_sequence(data_dir, args.catA)
    dataA = load_oxiod(seqA / "imu0_resampled.npy")
    print(f"[2] (A) {args.catA}: {seqA.name}")
    frames = [("ga", "ga (정렬 ON, 매 시각)", "#1f77b4"),
              ("yaw", "yaw (시작 단일 회전)", "#2ca02c"),
              ("body", "body (정렬 OFF)", "#d62728")]
    pathsA = {}
    for fr, _lab, _c in frames:
        p, gtA, ate = dr_path(net, dataA, mean, std, frame=fr)
        pathsA[fr] = (p, ate)
        print(f"    {fr:<5} ATE={ate:.2f} m")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    # 좌: 전체(자동 스케일, body 발산 포함)
    ax[0].plot(gtA[:, 0], gtA[:, 1], "-", color="black", lw=2.5, label="GT", zorder=5)
    for fr, lab, col in frames:
        p, ate = pathsA[fr]
        ax[0].plot(p[:, 0], p[:, 1], "-", color=col, lw=1.6, alpha=0.9, label=f"{lab}  (ATE {ate:.1f}m)")
    ax[0].scatter([0], [0], c="k", marker="o", s=40, zorder=6)
    ax[0].set_aspect("equal", "datalim"); ax[0].legend(fontsize=8.5, loc="best")
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Y (m)")
    ax[0].set_title("(a) 전체 — body(정렬 OFF) 발산", fontsize=10); ax[0].grid(ls=":", alpha=0.4)
    # 우: GT/ga/yaw 줌(body 제외)
    ax[1].plot(gtA[:, 0], gtA[:, 1], "-", color="black", lw=2.5, label="GT", zorder=5)
    for fr, lab, col in frames[:2]:
        p, ate = pathsA[fr]
        ax[1].plot(p[:, 0], p[:, 1], "-", color=col, lw=1.8, alpha=0.9, label=f"{lab}  (ATE {ate:.1f}m)")
    ax[1].scatter([0], [0], c="k", marker="o", s=40, zorder=6)
    ax[1].set_aspect("equal", "datalim"); ax[1].legend(fontsize=8.5, loc="best")
    ax[1].set_xlabel("X (m)"); ax[1].set_ylabel("Y (m)")
    ax[1].set_title("(b) 줌 — ga~GT, yaw 근접 (body 제외)", fontsize=10); ax[1].grid(ls=":", alpha=0.4)
    fig.suptitle(f"입력 전처리 정렬 ablation 궤적 비교 — {args.catA} ({seqA.name})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(LOG / "fig_4_6_1_traj.png", dpi=150); print("    saved fig_4_6_1_traj.png")

    # ── (B) yaw 드리프트 궤적 (input_only) ────────────────────
    seqB = (data_dir / args.seqB) if args.seqB else select_longest_sequence(data_dir, args.catB)
    dataB = load_oxiod(seqB / "imu0_resampled.npy")
    print(f"[3] (B) {args.catB}: {seqB.name}")
    quat_true = dataB["quat"]
    drifts = [(0.0, "drift 0 (정상)", "#1f77b4"),
              (0.1, "drift 0.1°/s (누적 ~55°)", "#ff7f0e"),
              (0.3, "drift 0.3°/s (누적 ~166°)", "#d62728")]
    fig2, bx = plt.subplots(1, 1, figsize=(6.2, 5.6))
    _, gtB, _ = dr_path(net, dataB, mean, std, frame="ga")
    bx.plot(gtB[:, 0], gtB[:, 1], "-", color="black", lw=2.6, label="GT", zorder=5)
    for dr, lab, col in drifts:
        qin = make_drifted_quat(quat_true, dr)
        p, _g, ate = dr_path(net, dataB, mean, std, frame="ga", quat_in=qin, quat_out=quat_true)
        bx.plot(p[:, 0], p[:, 1], "-", color=col, lw=1.8, alpha=0.9, label=f"{lab}  (ATE {ate:.1f}m)")
        print(f"    drift {dr}: ATE={ate:.2f} m")
    bx.scatter([0], [0], c="k", marker="o", s=40, zorder=6)
    bx.set_aspect("equal", "datalim"); bx.legend(fontsize=8.5, loc="best")
    bx.set_xlabel("X (m)"); bx.set_ylabel("Y (m)")
    bx.set_title(f"입력 yaw 드리프트 궤적 (input_only) — {args.catB} ({seqB.name})", fontsize=10, fontweight="bold")
    bx.grid(ls=":", alpha=0.4)
    fig2.tight_layout()
    fig2.savefig(LOG / "fig_4_6_2_traj.png", dpi=150); print("    saved fig_4_6_2_traj.png")


if __name__ == "__main__":
    main()
