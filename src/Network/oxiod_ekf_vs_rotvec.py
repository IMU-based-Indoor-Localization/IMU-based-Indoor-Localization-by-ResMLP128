# -*- coding: utf-8 -*-
"""실제 EKF yaw(불규칙 드리프트) vs RotVec(안정 절대 yaw) — positive 비교.
§4.4 의 EKF(imu_ekf_py.ScEkf)를 그대로 실행해, EKF 가 추정하는 yaw 가 시간에 따라
GT 에서 누적 드리프트(>~10° 예산)하며 궤적이 발산함을 보이고, 동일 네트워크 출력을
'안정 절대 yaw(RotVec≈GT)'로 dead-reckoning 하면 GT 를 추종함을 대조한다.

  · EKF traj/yaw : ScEkf 상태 p, R (yaw=atan2(R[1,0],R[0,0]))
  · RotVec-DR    : window-start 정답 yaw0 로 누적(=§4.4 Net-only, RotVec 등가)
  · GT           : pos, quat

실행:
  KMP_DUPLICATE_LIB_OK=TRUE python src/Network/oxiod_ekf_vs_rotvec.py \
    --data_dir D:/EKF_DATASET/TLIO_Oxford_Dataset --model_dir src/Network/out_classifier2 \
    --cat handheld --scale 0.001
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
from oxiod_preproc_ablation import select_longest_sequence  # type: ignore
from imu_ekf_py import ScEkf, current_cfg  # type: ignore

UPDATE_HZ = 20
CLONE_STRIDE = int(FS / UPDATE_HZ)   # 5
WARMUP_SEC = 2.0


def _infer(net, acc, gyr, quat, s, e, mean, std):
    import torch
    imu = window_to_gravity_aligned(acc, gyr, quat, s, e, frame="ga")
    x = torch.from_numpy(((imu - mean) / std).T.copy()).float().unsqueeze(0)
    with torch.no_grad():
        y, y_cov, _ = net(x)
    d = y[0].numpy().astype(np.float64)
    cov = np.diag(np.exp(y_cov[0].numpy()).astype(np.float64))
    return d, cov


def run_ekf(net, data, mean, std, scale):
    acc, gyr, quat = data["acc"], data["gyr"], data["quat"]
    T = len(acc)
    ts = (np.arange(T) / FS * 1e6).astype(np.int64)
    n_cal = max(int(WARMUP_SEC * FS), 50)
    cfg = current_cfg(); cfg.meascov_scale = scale
    ekf = ScEkf(cfg); ekf.initialize(int(ts[0]), acc[:n_cal].mean(axis=0).astype(np.float64))
    traj = [ekf.state.p[:2].copy()]
    yaw  = [np.arctan2(ekf.state.R[1, 0], ekf.state.R[0, 0])]
    for i in range(1, T):
        is_clone = (i % CLONE_STRIDE == 0)
        t_aug = int(ts[i]) if is_clone else None
        ekf.propagate(acc[i].astype(np.float64), gyr[i].astype(np.float64), int(ts[i]), t_aug)
        if i < n_cal + WINDOW_LEN:
            ekf.state.v[:] = 0.0
        if is_clone and i >= n_cal + WINDOW_LEN:
            s = i - WINDOW_LEN
            d, cov = _infer(net, acc, gyr, quat, s, i, mean, std)
            ekf.update(d, cov, int(ts[s]), int(ts[i]))
            ekf.marginalize_until(int(ts[s]))
        traj.append(ekf.state.p[:2].copy())
        yaw.append(np.arctan2(ekf.state.R[1, 0], ekf.state.R[0, 0]))
    return np.array(traj), np.array(yaw)


def rotvec_dr(net, data, mean, std):
    """RotVec(=정답 yaw0) dead-reckoning. 반환 anchor traj + anchor idx."""
    import torch
    acc, gyr, quat, pos = data["acc"], data["gyr"], data["quat"], data["pos"]
    T = len(acc); starts = list(range(0, T - WINDOW_LEN, WINDOW_LEN))
    pred = [np.zeros(2)]
    with torch.no_grad():
        for s in starts:
            e = s + WINDOW_LEN
            imu = window_to_gravity_aligned(acc, gyr, quat, s, e, frame="ga")
            x = torch.from_numpy(((imu - mean) / std).T.copy()).float().unsqueeze(0)
            y, _c, _l = net(x); d = y[0].numpy()
            y0 = window_yaw0(quat, s); c, sn = np.cos(y0), np.sin(y0)
            pred.append(pred[-1] + np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]]))
    idx = [0] + [s + WINDOW_LEN for s in starts]
    return np.array(pred), idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
    ap.add_argument("--model_dir", default="src/Network/out_classifier2")
    ap.add_argument("--cat", default="handheld")
    ap.add_argument("--scale", type=float, default=0.001, help="§4.4 best meascov_scale")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    print("[1] 모델 로드"); net, _p, mean, std = load_model(args.model_dir)
    seq = select_longest_sequence(data_dir, args.cat)
    data = load_oxiod(seq / "imu0_resampled.npy")
    print(f"[2] {args.cat}: {seq.name}  ({len(data['acc'])/FS:.0f}s)")

    print("[3] EKF 실행")
    ekf_traj, ekf_yaw = run_ekf(net, data, mean, std, args.scale)
    print("[4] RotVec(정답 yaw0) DR")
    rv_anchor, idx = rotvec_dr(net, data, mean, std)

    pos = data["pos"]; quat = data["quat"]
    T = min(len(ekf_traj), len(pos))
    gt_full = pos[:T, :2] - pos[0, :2]
    # GT yaw per sample (subsample 5)
    sub = np.arange(0, T, 5)
    gt_yaw = np.array([window_yaw0(quat, int(i)) for i in sub])
    ekf_yaw_s = ekf_yaw[sub]
    # align initial yaw (EKF gravity-init yaw=0 vs GT 절대)
    off = gt_yaw[0] - ekf_yaw_s[0]
    ekf_err = np.degrees(np.unwrap(ekf_yaw_s + off - gt_yaw))
    t_sec = sub / FS

    # ATE
    rv_gt = pos[idx, :2][:len(rv_anchor)] - pos[0, :2]
    n = min(len(rv_anchor), len(rv_gt))
    ate_rv = float(np.sqrt(np.mean(np.linalg.norm(rv_anchor[:n] - rv_gt[:n], axis=1) ** 2)))
    ekf_at_anchor = ekf_traj[[min(i, len(ekf_traj)-1) for i in idx]][:n]
    ate_ekf = float(np.sqrt(np.mean(np.linalg.norm(ekf_at_anchor - rv_gt[:n], axis=1) ** 2)))
    print(f"    RotVec-DR ATE = {ate_rv:.2f} m ; EKF ATE = {ate_ekf:.2f} m  ({ate_ekf/ate_rv:.1f}x)")
    print(f"    EKF yaw 누적 드리프트(말기 평균 |err|) = {np.mean(np.abs(ekf_err[-len(ekf_err)//5:])):.0f}°  max {np.max(np.abs(ekf_err)):.0f}°")

    # ── figure ──
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.0))
    ax[0].plot(gt_full[:, 0], gt_full[:, 1], "-", color="black", lw=2.4, label="GT", zorder=5)
    ax[0].plot(rv_anchor[:, 0], rv_anchor[:, 1], "-", color="#1f77b4", lw=1.8,
               label=f"RotVec-DR (안정 yaw)  ATE {ate_rv:.1f}m")
    ax[0].plot(ekf_traj[:, 0], ekf_traj[:, 1], "-", color="#d62728", lw=1.4, alpha=0.85,
               label=f"EKF (드리프트 yaw)  ATE {ate_ekf:.1f}m")
    ax[0].scatter([0], [0], c="k", s=40, zorder=6)
    ax[0].set_aspect("equal", "datalim"); ax[0].legend(fontsize=9, loc="best")
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Y (m)"); ax[0].grid(ls=":", alpha=0.4)
    ax[0].set_title(f"(a) 궤적 — EKF 발산 vs RotVec 추종 ({args.cat})", fontsize=10)

    ax[1].axhspan(-10, 10, color="#2ca02c", alpha=0.10, label="±10° 예산")
    ax[1].plot(t_sec, ekf_err, "-", color="#d62728", lw=1.8, label="EKF yaw 오차 (vs GT)")
    ax[1].axhline(0, color="#1f77b4", lw=2.0, label="RotVec yaw 오차 ≈ 0 (절대 기준)")
    ax[1].set_xlabel("시간 (s)"); ax[1].set_ylabel("yaw 오차 (°)")
    ax[1].legend(fontsize=9, loc="best"); ax[1].grid(ls=":", alpha=0.4)
    ax[1].set_title("(b) yaw 오차 추이 — EKF 가 ~10° 예산을 초과 드리프트", fontsize=10)
    fig.suptitle("그림 7. 실제 EKF yaw 드리프트 vs RotVec 절대 yaw — positive 비교", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(r"D:\mobile\imu_android\logs") / "fig_4_7_ekf_vs_rotvec.png"
    fig.savefig(out, dpi=150); print(f"[OK] {out.name}")


if __name__ == "__main__":
    main()
