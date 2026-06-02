"""
oxiod_yaw_drift_ablation.py — 논문 §4.5 'yaw drift → OOD' 메커니즘 직접 검증
==============================================================================
논문 주장 (§4.5): EKF yaw 가 정답 yaw 에서 누적 드리프트하면, 신경망 입력이
훈련 분포를 이탈(OOD)하여 *예측 품질 자체* 가 저하된다.

이 스크립트는 합성 yaw drift 를 GT quaternion 에 주입하여 그 주장을 검증한다.
  q_drift(t) = R_z(drift_rate · t) ⊗ q_true(t)     (world 프레임에 yaw 오프셋 누적)

핵심: yaw drift 는 두 경로로 측위에 영향을 준다 — 둘을 *분리* 해서 측정해야
논문의 진짜 주장(입력 OOD)을 격리 검증할 수 있다.

  (1) 입력 정렬 경로  : drift 된 quat 으로 gravity-aligned 변환 → 신경망 입력이 바뀜.
                        논문이 지목한 OOD 효과. heading-agnostic 모델이면 window 시작
                        yaw 제거로 대부분 상쇄되어야 하므로, 효과가 작을 것으로 예상.
  (2) 출력 재구성 경로: drift 된 yaw0 으로 모델 disp 를 world 로 회전 → 궤적 방향 오차.
                        모델이 완벽해도 발생하는 *순수 기하* 효과.

3 모드:
  - 'both'       : 입력+출력 모두 drift (현실 EKF). 총 효과.
  - 'input_only' : 입력만 drift, 출력은 TRUE yaw0. → (1) 격리 = 논문 주장 직접 검증.
  - 'output_only': 출력만 drift, 입력은 TRUE quat. → (2) 격리 = 기하 효과.

지표:
  - ATE RMSE_xy (m)              : 전체 궤적 누적 오차
  - per-window |disp_xy| RMSE (m): window 단위 모델 출력 vs GT 변위 오차
                                   ← 입력 OOD 가 *예측 품질* 을 떨어뜨리는지 직접 측정

판정:
  · input_only 의 disp-RMSE / ATE 가 drift 와 함께 크게 증가 → 논문 주장 (입력 OOD) 성립.
  · input_only 는 거의 평탄한데 output_only/both ATE 만 증가 → 효과는 기하 회전이지
    입력 OOD 가 아님 → 논문 §4.5 의 *메커니즘 설명* 은 부정확 (현상은 맞아도 원인이 다름).

사용:
  python src/Network/oxiod_yaw_drift_ablation.py \\
      --data_dir D:/EKF_DATASET/TLIO_Oxford_Dataset \\
      --model_dir src/Network/out_classifier2 \\
      --out logs/oxiod_yaw_drift.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from offline_eval import (   # type: ignore
    WINDOW_LEN, FS, load_oxiod, load_model,
    window_to_gravity_aligned, window_yaw0,
)
from oxiod_preproc_ablation import select_longest_sequence, CATEGORIES   # type: ignore


DRIFT_RATES = [0.0, 0.5, 1.0, 2.0, 5.0]   # deg/s
MODES = ["both", "input_only", "output_only"]


def make_drifted_quat(quat: np.ndarray, drift_rate_deg_s: float) -> np.ndarray:
    """q_drift(t) = R_z(drift_rate · t) ⊗ q_true(t).  world 프레임 yaw 누적 드리프트."""
    if drift_rate_deg_s == 0.0:
        return quat
    from scipy.spatial.transform import Rotation as Rot
    n = len(quat)
    t = np.arange(n) / FS
    drift_rad = np.deg2rad(drift_rate_deg_s) * t
    Rz = Rot.from_euler("z", drift_rad)          # [N] world-yaw offset
    Rtrue = Rot.from_quat(quat)                   # body→world (true)
    Rdrift = Rz * Rtrue                           # apply Rtrue then Rz
    return Rdrift.as_quat().astype(np.float32)


def evaluate_drift(net, data, mean, std, drift_rate: float, mode: str):
    """drift 주입 후 ATE + per-window disp RMSE 산출."""
    import torch

    acc, gyr, quat_true, pos = data["acc"], data["gyr"], data["quat"], data["pos"]
    quat_drift = make_drifted_quat(quat_true, drift_rate)

    # 입력/출력 각각 어느 quat 을 쓰는지
    if mode == "both":
        quat_in, quat_out = quat_drift, quat_drift
    elif mode == "input_only":
        quat_in, quat_out = quat_drift, quat_true
    elif mode == "output_only":
        quat_in, quat_out = quat_true, quat_drift
    else:
        raise ValueError(mode)

    T = len(acc)
    starts = list(range(0, T - WINDOW_LEN, WINDOW_LEN))

    pred = [np.zeros(2)]
    disp_errs = []
    with torch.no_grad():
        for s in starts:
            e = s + WINDOW_LEN
            imu = window_to_gravity_aligned(acc, gyr, quat_in, s, e, frame="ga")
            imu_n = (imu - mean) / std
            x = torch.from_numpy(imu_n.T.copy()).float().unsqueeze(0)
            y, _cov, _logits = net(x)
            d = y[0].numpy()                                # ga-frame disp (3,)

            # 출력 재구성: yaw0 (출력 경로 quat) 로 world 회전
            yaw0_out = window_yaw0(quat_out, s)
            c, sn = np.cos(yaw0_out), np.sin(yaw0_out)
            dw = np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]])
            pred.append(pred[-1] + dw)

            # per-window disp err: 예측(ga) vs GT(true ga frame)
            yaw0_true = window_yaw0(quat_true, s)
            ct, st = np.cos(-yaw0_true), np.sin(-yaw0_true)
            dp = pos[e, :2] - pos[s, :2]
            gt_ga = np.array([ct * dp[0] - st * dp[1], st * dp[0] + ct * dp[1]])
            disp_errs.append(float(np.hypot(d[0] - gt_ga[0], d[1] - gt_ga[1])))

    pred = np.array(pred)
    gt_anchor_idx = [0] + [s + WINDOW_LEN for s in starts]
    gt = pos[gt_anchor_idx, :2] - pos[0:1, :2]
    n = min(len(pred), len(gt))
    ate = float(np.sqrt(np.mean(np.linalg.norm(pred[:n] - gt[:n], axis=1) ** 2)))
    disp_rmse = float(np.sqrt(np.mean(np.array(disp_errs) ** 2)))
    return ate, disp_rmse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
    ap.add_argument("--model_dir", default="src/Network/out_classifier2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[1] 모델 로드: {args.model_dir}")
    net, _para, mean, std = load_model(args.model_dir)

    print(f"[2] 카테고리당 가장 긴 시퀀스 선택")
    sequences = {}
    for cat in CATEGORIES:
        seq = select_longest_sequence(data_dir, cat)
        if seq is not None:
            sequences[cat] = seq
    print(f"    {len(sequences)} 카테고리: {list(sequences.keys())}")

    # data 캐시 (drift 별 재로드 회피)
    data_cache = {cat: load_oxiod(seq / "imu0_resampled.npy")
                  for cat, seq in sequences.items()}

    rows = []   # (cat, mode, drift, ate, disp_rmse)
    print()
    print(f"[3] {len(sequences)}cat × {len(MODES)}mode × {len(DRIFT_RATES)}drift 추론")
    for mode in MODES:
        for cat, data in data_cache.items():
            for dr in DRIFT_RATES:
                ate, disp = evaluate_drift(net, data, mean, std, dr, mode)
                rows.append((cat, mode, dr, ate, disp))

    # ── 출력: 모드별 drift sweep 표 (카테고리 평균) ──────────────────
    def cat_mean(mode, dr, metric_idx):
        vals = [r[metric_idx] for r in rows if r[1] == mode and r[2] == dr]
        return float(np.mean(vals)), float(np.median(vals))

    print()
    print("=" * 78)
    print("[A] per-window |disp_xy| RMSE (m) — 입력 OOD 가 *예측 품질* 떨어뜨리는지")
    print("    (논문 §4.5 직접 검증 지표. input_only 가 drift 따라 커지면 논문 성립)")
    print("=" * 78)
    print(f"{'drift(°/s)':>10} | " + " | ".join(f"{m:>16}" for m in MODES))
    print("-" * 78)
    for dr in DRIFT_RATES:
        cells = []
        for m in MODES:
            mn, md = cat_mean(m, dr, 4)
            cells.append(f"{mn:6.3f}(med {md:.3f})")
        print(f"{dr:>10.1f} | " + " | ".join(f"{c:>16}" for c in cells))

    print()
    print("=" * 78)
    print("[B] ATE RMSE_xy (m) — 전체 궤적 누적 오차 (카테고리 평균)")
    print("=" * 78)
    print(f"{'drift(°/s)':>10} | " + " | ".join(f"{m:>16}" for m in MODES))
    print("-" * 78)
    for dr in DRIFT_RATES:
        cells = []
        for m in MODES:
            mn, md = cat_mean(m, dr, 3)
            cells.append(f"{mn:6.2f}(med {md:.2f})")
        print(f"{dr:>10.1f} | " + " | ".join(f"{c:>16}" for c in cells))

    # ── 판정 보조: drift 0 → 5 증가율 ────────────────────────────────
    print()
    print("=" * 78)
    print("[C] drift 0 → 5°/s 증가율 (카테고리 평균 기준)")
    print("=" * 78)
    for m in MODES:
        d0_disp, _ = cat_mean(m, 0.0, 4)
        d5_disp, _ = cat_mean(m, 5.0, 4)
        d0_ate, _  = cat_mean(m, 0.0, 3)
        d5_ate, _  = cat_mean(m, 5.0, 3)
        disp_x = d5_disp / d0_disp if d0_disp > 0 else float("nan")
        ate_x  = d5_ate / d0_ate if d0_ate > 0 else float("nan")
        print(f"  {m:>12} : disp-RMSE {d0_disp:.3f}→{d5_disp:.3f} ({disp_x:.2f}x)  "
              f"ATE {d0_ate:.2f}→{d5_ate:.2f} ({ate_x:.2f}x)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("category,mode,drift_deg_s,ate_rmse_m,disp_rmse_m\n")
            for cat, mode, dr, ate, disp in rows:
                f.write(f"{cat},{mode},{dr},{ate:.4f},{disp:.4f}\n")
        print(f"\n[OK] CSV 저장: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
