# -*- coding: utf-8 -*-
"""실측 Android walk 에 합성 yaw-drift 주입 — RotVec 절대 yaw(드리프트0)의 가치 입증.
같은 IMU, 자세(quat)에만 점진 yaw 오프셋 Δ(t)=rate·t 주입(피치/롤=중력정렬 유지).
입력 ga 전처리 + 출력 heading 모두 드리프트된 자세를 사용(단말 rotVec 단일 소스 현실 반영).
기준 = rate 0 (순수 rotVec) → 드리프트 클수록 궤적 열화. OxIOD §4.6 메커니즘의 실측 재현.
cd src/Network; KMP_DUPLICATE_LIB_OK=TRUE python android_yaw_drift.py [csv] [scale]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.spatial.transform import Rotation as R
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
RATES = [0.0, 0.5, 1.0, 2.0, 5.0]   # °/s
COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]


def dr(net, acc, gyr, quat, mean, std, frame="ga"):
    import torch
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


def drift_quat(quat, fs, rate_deg_s):
    """자세에 world-z 축 yaw 오프셋 Δ(t)=rate·t 주입 (피치/롤 유지)."""
    if rate_deg_s == 0.0:
        return quat
    N = len(quat); t = np.arange(N) / fs
    delta = np.deg2rad(rate_deg_s) * t
    out = np.empty_like(quat)
    rq = R.from_quat(quat)
    for i in range(N):
        out[i] = (R.from_euler("z", delta[i]) * rq[i]).as_quat()
    return out


def main():
    print(f"[1] 모델 로드 (csv={Path(CSV).name}, scale={SCALE:.4f})")
    net, _p, mean, std = load_model("src/Network/out_classifier2")
    data = load_android(CSV, calib_sec=2.0, linacc_scale=SCALE)
    acc, gyr, quat = data["acc"], data["gyr"], data["quat"]
    dur = len(acc) / FS
    print(f"[2] {dur:.1f}s, {len(acc)} samples")

    # 현실 anchor: 자이로 적분 yaw vs rotVec yaw 누적차 (자력계 융합 없을 때 실제 드리프트)
    gyro_yaw = np.cumsum(gyr[:, 2]) / FS
    rotvec_yaw = np.unwrap([R.from_quat(q).as_euler("zyx")[0] for q in quat])
    gyro_yaw -= gyro_yaw[0]; rotvec_yaw -= rotvec_yaw[0]
    real_drift_end = np.degrees(abs(gyro_yaw[-1] - rotvec_yaw[-1]))
    print(f"[3] 이 기기 자이로-only vs rotVec 누적 yaw 차(=자력계 없을 때 실제 드리프트): {real_drift_end:.1f}° / {dur:.0f}s "
          f"(~{real_drift_end/dur:.2f}°/s)")

    rows, paths = [], {}
    p_ref = None
    for r, col in zip(RATES, COLORS):
        q2 = drift_quat(quat, FS, r)
        p = dr(net, acc, gyr, q2, mean, std, "ga")
        paths[r] = p
        if r == 0.0:
            p_ref = p
        plen = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        end_dev = float(np.linalg.norm(p[-1] - p_ref[-1])) if p_ref is not None else 0.0
        K = min(len(p), len(p_ref))
        traj_dev = float(np.mean(np.linalg.norm(p[:K] - p_ref[:K], axis=1)))
        cum = r * dur
        rows.append((r, cum, end_dev, traj_dev, plen))
        print(f"  rate {r:>4.1f}°/s  누적 {cum:>5.0f}°  | 끝점편차 {end_dev:6.2f}m  평균편차 {traj_dev:6.2f}m  path {plen:.1f}m")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for r, col in zip(RATES, COLORS):
        p = paths[r]
        ax[0].plot(p[:, 0], p[:, 1], "-", color=col, lw=1.6, label=f"{r:.1f}°/s (누적 {r*dur:.0f}°)")
    ax[0].scatter([0], [0], c="k", s=50, zorder=5, label="시작")
    ax[0].set_aspect("equal", "datalim"); ax[0].grid(ls=":", alpha=0.4); ax[0].legend(fontsize=8)
    ax[0].set_title("yaw-drift 주입 시 궤적 (기준 0°/s = rotVec 절대)", fontsize=10)
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Y (m)")

    cum = [x[1] for x in rows]; endd = [x[2] for x in rows]
    ax[1].plot(cum, endd, "o-", color="#d62728", lw=2)
    ax[1].axvline(real_drift_end, ls=":", color="#1f77b4", lw=1.5)
    ax[1].text(real_drift_end + 2, max(endd) * 0.85, f"이 기기 실제\n자이로 드리프트\n{real_drift_end:.0f}°", color="#1f77b4", fontsize=8)
    ax[1].set_xlabel("누적 yaw 드리프트 (°)"); ax[1].set_ylabel("기준(rotVec) 대비 끝점편차 (m)")
    ax[1].set_title("yaw-drift → 측위 열화", fontsize=10); ax[1].grid(ls=":", alpha=0.4)
    fig.suptitle(f"실측 Android yaw-drift 민감도 — {Path(CSV).name} ({dur:.0f}s)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = LOG / f"android_yaw_drift_{Path(CSV).stem}.png"; fig.savefig(out, dpi=140)
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
