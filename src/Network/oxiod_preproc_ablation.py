"""
oxiod_preproc_ablation.py — OxIOD GT 기반 전처리 ablation 정량 비교
====================================================================
논문 §3.3 의 주장 ("per-timestep 자세 정렬이 핵심, 윈도우 시작 회전 하나만으론 부족") 을
GT 가 있는 OxIOD 8 카테고리 대표 시퀀스에서 정량 검증.

비교 조합 (3 frame):
  - 'ga'   per-sample R(q_t) + window 시작 yaw 제거  (학습 동일, baseline)
  - 'yaw'  window 시작 R(q_s) 1개만 사용             (논문 §3.3 OFF 케이스)
  - 'body' 회전 없음 (raw body frame 그대로)         (가장 극단적 OFF)

평가 지표:
  ATE RMSE_xy = sqrt( (1/N) Σ_k ‖p_pred_xy(k) − p_gt_xy(k)‖² )
                k = 100, 200, 300, ... (stride=100 anchor, 논문 §4.1 정의)

시퀀스 선택: 각 카테고리에서 npy 행수 가장 많은 시퀀스 1개 (논문 §4.1 와 동일).

사용:
  python src/Network/oxiod_preproc_ablation.py \\
      --data_dir D:/EKF_DATASET/TLIO_Oxford_Dataset \\
      --model_dir src/Network/out_classifier2 \\
      --out logs/oxiod_ablation.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Windows cp949 콘솔 한글 안전
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from offline_eval import (   # type: ignore
    WINDOW_LEN, FS, load_oxiod, load_model,
    window_to_gravity_aligned, window_yaw0,
)


CATEGORIES = ["handbag", "handheld", "pocket", "running", "slow", "trolley", "large", "multi"]
FRAMES = ["ga", "yaw", "body"]
FRAME_LABELS = {
    "ga":   "ON (per-sample, baseline)",
    "yaw":  "OFF (window-start only)",
    "body": "OFF (no rotation)",
}


def select_longest_sequence(data_dir: Path, category: str) -> Path | None:
    """카테고리 prefix 의 시퀀스 중 npy 행수 가장 많은 폴더 반환."""
    candidates = sorted(data_dir.glob(f"oxford_{category}_*"))
    best, best_n = None, -1
    for seq_dir in candidates:
        npy_path = seq_dir / "imu0_resampled.npy"
        if not npy_path.exists():
            continue
        try:
            # mmap 로 행수만 읽기 (전체 로드 회피)
            data = np.load(npy_path, mmap_mode="r")
            n = data.shape[0]
        except Exception:
            continue
        if n > best_n:
            best_n = n
            best = seq_dir
    return best


def evaluate_ate(net, data, mean, std, frame: str) -> tuple[float, int]:
    """카테고리 시퀀스 1개에 대해 dead-reckoning 적분 후 ATE RMSE_xy 산출.

    GT: data['pos'][k] - data['pos'][0]  (anchor 0 정렬)
    Pred: window 단위 disp 누적 (yaw0 으로 world 회전)
    RMSE = sqrt(mean(‖p_pred(k) - p_gt(k)‖²))   anchor k ∈ {100, 200, ...}
    """
    import torch

    acc, gyr, quat, pos = data["acc"], data["gyr"], data["quat"], data["pos"]
    T = len(acc)
    stride = WINDOW_LEN
    starts = list(range(0, T - WINDOW_LEN, stride))

    pred = [np.zeros(2)]
    with torch.no_grad():
        for s in starts:
            e = s + WINDOW_LEN
            imu = window_to_gravity_aligned(acc, gyr, quat, s, e, frame=frame)
            imu_n = (imu - mean) / std
            x = torch.from_numpy(imu_n.T.copy()).float().unsqueeze(0)
            y, _y_cov, _logits = net(x)
            d = y[0].numpy()
            # ga frame → world: window 시작 절대 yaw 로 회전
            yaw0 = window_yaw0(quat, s)
            c, sn = np.cos(yaw0), np.sin(yaw0)
            dw = np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]])
            pred.append(pred[-1] + dw)
    pred = np.array(pred)   # [n_anchor+1, 2]

    # GT anchor: pos 의 anchor index = window end (= s + WINDOW_LEN). pred[0]=시작, pred[i] = anchor i-1 후 누적.
    gt_anchor_idx = [0] + [s + WINDOW_LEN for s in starts]
    gt = pos[gt_anchor_idx, :2] - pos[0:1, :2]   # GT xy 변위

    n = min(len(pred), len(gt))
    err = np.linalg.norm(pred[:n] - gt[:n], axis=1)   # [n]
    rmse_xy = float(np.sqrt(np.mean(err ** 2)))
    return rmse_xy, len(starts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=r"D:/EKF_DATASET/TLIO_Oxford_Dataset",
                    help="OxIOD 데이터셋 루트 (시퀀스 폴더들 포함)")
    ap.add_argument("--model_dir", default="src/Network/out_classifier2",
                    help="배포 모델 폴더 (config.json + checkpoints/best.pth + norm_*.npy)")
    ap.add_argument("--out", default=None,
                    help="결과 CSV 저장 경로 (옵션). 생략 시 콘솔만.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"[err] data_dir 없음: {data_dir}")
        return 1

    print(f"[1] 모델 로드: {args.model_dir}")
    net, _para, mean, std = load_model(args.model_dir)

    # 시퀀스 선택
    print(f"[2] 카테고리당 가장 긴 시퀀스 자동 선택 (논문 §4.1)")
    sequences = {}
    for cat in CATEGORIES:
        seq = select_longest_sequence(data_dir, cat)
        if seq is None:
            print(f"    [skip] {cat}: 사용 가능 시퀀스 없음")
            continue
        npy = seq / "imu0_resampled.npy"
        n = np.load(npy, mmap_mode="r").shape[0]
        duration_s = n / FS
        print(f"    {cat:<10} → {seq.name:<25}  ({n} samples, {duration_s:.1f}s)")
        sequences[cat] = seq

    if not sequences:
        print("[err] 시퀀스 없음 — 데이터 경로 확인")
        return 1

    # ablation 실행
    print()
    print(f"[3] {len(sequences)} 카테고리 × {len(FRAMES)} frame 추론 시작")
    print("-" * 90)
    header = f"{'Category':<10} {'n_win':>6}  " + "  ".join(
        f"{FRAME_LABELS[fr]:>30}" for fr in FRAMES
    )
    print(header)
    print("-" * 90)

    results = {}   # results[cat][frame] = rmse
    for cat, seq_dir in sequences.items():
        data = load_oxiod(seq_dir / "imu0_resampled.npy")
        row_rmse = {}
        for fr in FRAMES:
            rmse, n_win = evaluate_ate(net, data, mean, std, frame=fr)
            row_rmse[fr] = rmse
        results[cat] = row_rmse
        line = f"{cat:<10} {n_win:>6}  " + "  ".join(
            f"{row_rmse[fr]:>30.3f}" for fr in FRAMES
        )
        print(line)
    print("-" * 90)

    # 카테고리 평균
    print()
    print("[4] 카테고리 평균 ATE RMSE_xy (m)")
    print("-" * 60)
    for fr in FRAMES:
        vals = [results[c][fr] for c in results]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        print(f"  {FRAME_LABELS[fr]:<35} : {mean_v:.3f} ± {std_v:.3f} m")

    # 상대 비교
    print()
    print("[5] ga (baseline) 대비 상대 악화")
    print("-" * 60)
    print(f"  {'Category':<10} {'yaw-OFF':>15} {'body-OFF':>15}")
    for cat in results:
        ga_rmse = results[cat]["ga"]
        if ga_rmse <= 0:
            continue
        yaw_x = results[cat]["yaw"] / ga_rmse
        body_x = results[cat]["body"] / ga_rmse
        print(f"  {cat:<10} {yaw_x:>14.2f}x {body_x:>14.2f}x")

    # CSV 저장 (옵션)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("category,sequence," + ",".join(f"rmse_{fr}" for fr in FRAMES) + "\n")
            for cat, row in results.items():
                seq_name = sequences[cat].name
                f.write(f"{cat},{seq_name}," + ",".join(f"{row[fr]:.4f}" for fr in FRAMES) + "\n")
        print(f"\n[OK] CSV 저장: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
