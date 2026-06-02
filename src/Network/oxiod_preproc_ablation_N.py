"""
oxiod_preproc_ablation_N.py — 전처리 프레임 ablation, 카테고리당 N 시퀀스 평균
=============================================================================
oxiod_preproc_ablation.py 의 단일 시퀀스 한계(trolley/multi 역전 outlier 의심) 해소.
카테고리당 가장 긴 N개 시퀀스로 ATE RMSE_xy 산출 → 카테고리별 mean±std.
프레임별 카테고리 비율은 기하평균/중앙값으로 보고 (산술평균은 large 20m 가 왜곡).

사용:
  python src/Network/oxiod_preproc_ablation_N.py \\
      --data_dir D:/EKF_DATASET/TLIO_Oxford_Dataset \\
      --model_dir src/Network/out_classifier2 --n 5 \\
      --out logs/oxiod_ablation_N5.csv
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
from offline_eval import FS, load_oxiod, load_model   # type: ignore
from oxiod_preproc_ablation import CATEGORIES, evaluate_ate   # type: ignore

FRAMES = ["ga", "yaw", "body"]


def select_longest_n(data_dir: Path, category: str, n: int) -> list[Path]:
    """카테고리 prefix 시퀀스 중 npy 행수 상위 N개 폴더."""
    cand = []
    for seq_dir in sorted(data_dir.glob(f"oxford_{category}_*")):
        npy = seq_dir / "imu0_resampled.npy"
        if npy.exists():
            try:
                rows = np.load(npy, mmap_mode="r").shape[0]
            except Exception:
                continue
            cand.append((rows, seq_dir))
    cand.sort(key=lambda x: -x[0])
    return [d for _, d in cand[:n]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
    ap.add_argument("--model_dir", default="src/Network/out_classifier2")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[1] 모델 로드: {args.model_dir}")
    net, _para, mean, std = load_model(args.model_dir)

    print(f"[2] 카테고리당 상위 {args.n} 시퀀스")
    seqs = {c: select_longest_n(data_dir, c, args.n) for c in CATEGORIES}
    for c, ss in seqs.items():
        print(f"    {c:<10} {len(ss)}개: {', '.join(s.name.split('oxford_')[-1] for s in ss)}")

    # results[cat][frame] = list of per-seq rmse
    results = {c: {f: [] for f in FRAMES} for c in CATEGORIES}
    rows_csv = []
    print()
    print(f"[3] {sum(len(s) for s in seqs.values())} 시퀀스 × {len(FRAMES)} frame 추론")
    for cat, ss in seqs.items():
        for seq_dir in ss:
            data = load_oxiod(seq_dir / "imu0_resampled.npy")
            for fr in FRAMES:
                rmse, _n = evaluate_ate(net, data, mean, std, frame=fr)
                results[cat][fr].append(rmse)
                rows_csv.append((cat, seq_dir.name, fr, rmse))

    # ── 표: 카테고리별 mean±std (frame 3개) ──────────────────────────
    print()
    print("=" * 92)
    print(f"[A] 카테고리별 ATE RMSE_xy mean±std (N={args.n}) — frame 3종")
    print("=" * 92)
    print(f"{'Category':<10} | {'ga (ON)':>16} | {'yaw (OFF)':>16} | {'body (OFF)':>16} | {'yaw/ga':>7} {'body/ga':>8}")
    print("-" * 92)
    ratio_yaw, ratio_body = [], []
    for cat in CATEGORIES:
        cells = {}
        for fr in FRAMES:
            v = np.array(results[cat][fr])
            cells[fr] = (float(v.mean()), float(v.std()))
        ry = cells["yaw"][0] / cells["ga"][0] if cells["ga"][0] > 0 else float("nan")
        rb = cells["body"][0] / cells["ga"][0] if cells["ga"][0] > 0 else float("nan")
        ratio_yaw.append(ry); ratio_body.append(rb)
        print(f"{cat:<10} | {cells['ga'][0]:>7.3f}±{cells['ga'][1]:<7.3f} | "
              f"{cells['yaw'][0]:>7.3f}±{cells['yaw'][1]:<7.3f} | "
              f"{cells['body'][0]:>7.3f}±{cells['body'][1]:<7.3f} | "
              f"{ry:>6.2f}x {rb:>7.2f}x")

    # ── 종합 비율 (기하평균/중앙값) ──────────────────────────────────
    def geomean(a):
        a = np.array(a)
        return float(np.exp(np.mean(np.log(a))))
    print("-" * 92)
    print(f"{'기하평균 비율':<10} | {'':>16} | {'':>16} | {'':>16} | "
          f"{geomean(ratio_yaw):>6.2f}x {geomean(ratio_body):>7.2f}x")
    print(f"{'중앙값 비율':<10} | {'':>16} | {'':>16} | {'':>16} | "
          f"{float(np.median(ratio_yaw)):>6.2f}x {float(np.median(ratio_body)):>7.2f}x")

    # ── 단일 시퀀스(N=1) 와 비교: 역전 카테고리 재확인 ────────────────
    print()
    print("[B] yaw/ga 비율 (N=1 단일 시퀀스 결과와 비교 — outlier 검증)")
    print(f"    N=1 에서 역전했던 카테고리: trolley(0.94x), multi(0.65x)")
    for cat in ("trolley", "multi"):
        ry = np.mean(results[cat]["yaw"]) / np.mean(results[cat]["ga"])
        verdict = "여전히 역전(ga 열등)" if ry < 1.0 else "정상(ga 우위)으로 수렴"
        print(f"    {cat:<10}: N={args.n} yaw/ga = {ry:.2f}x → {verdict}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("category,sequence,frame,ate_rmse_m\n")
            for cat, seqn, fr, rmse in rows_csv:
                f.write(f"{cat},{seqn},{fr},{rmse:.4f}\n")
        print(f"\n[OK] CSV 저장: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
