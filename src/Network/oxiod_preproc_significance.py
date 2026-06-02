"""
oxiod_preproc_significance.py — 전처리 프레임 ablation 통계적 유의성 검정
==========================================================================
"per-sample vs window-start 차이는 부차적(~9%)" 이라는 (거의 null) 주장을 게재
등급으로 정당화하기 위한 paired 검정. 동시에 *전체 시퀀스* 를 사용해 longest-5
선택 편향도 제거 (한 번에 두 보강).

각 시퀀스마다 ga/yaw/body ATE RMSE_xy 산출 → 시퀀스 단위 paired 비교:
  - Wilcoxon signed-rank (non-parametric, paired): yaw vs ga, body vs ga
  - 효과크기: per-sequence 비율(yaw/ga, body/ga) 의 기하평균·중앙값 + bootstrap 95% CI
  - 카테고리별 + 전체 pooled

사용:
  python src/Network/oxiod_preproc_significance.py \\
      --data_dir D:/EKF_DATASET/TLIO_Oxford_Dataset \\
      --model_dir src/Network/out_classifier2 --out logs/oxiod_significance.csv
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
from offline_eval import load_oxiod, load_model   # type: ignore
from oxiod_preproc_ablation import CATEGORIES, evaluate_ate   # type: ignore

FRAMES = ["ga", "yaw", "body"]


def all_sequences(data_dir: Path, category: str) -> list[Path]:
    out = []
    for sd in sorted(data_dir.glob(f"oxford_{category}_*")):
        if (sd / "imu0_resampled.npy").exists():
            out.append(sd)
    return out


def boot_ci_geomean(ratios: np.ndarray, n_boot: int = 5000, seed: int = 12345):
    """비율 기하평균의 bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(ratios)
    logs = np.log(ratios)
    means = np.array([np.mean(logs[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(np.exp(means), [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
    ap.add_argument("--model_dir", default="src/Network/out_classifier2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scipy.stats import wilcoxon

    data_dir = Path(args.data_dir)
    print(f"[1] 모델 로드: {args.model_dir}")
    net, _para, mean, std = load_model(args.model_dir)

    print(f"[2] 전체 시퀀스 ATE 산출 (ga/yaw/body)")
    # rows: per-sequence dict {cat, seq, ga, yaw, body}
    recs = []
    for cat in CATEGORIES:
        seqs = all_sequences(data_dir, cat)
        for sd in seqs:
            data = load_oxiod(sd / "imu0_resampled.npy")
            r = {"cat": cat, "seq": sd.name}
            for fr in FRAMES:
                r[fr], _ = evaluate_ate(net, data, mean, std, frame=fr)
            recs.append(r)
        print(f"    {cat:<10} {len(seqs):>3} 시퀀스 완료")

    ga = np.array([r["ga"] for r in recs])
    yaw = np.array([r["yaw"] for r in recs])
    body = np.array([r["body"] for r in recs])
    N = len(recs)
    print(f"\n[3] 총 {N} 시퀀스 paired 검정")

    def report(name, a, b):
        # b vs a (b = OFF frame, a = ga baseline). H1: b > a (OFF 가 더 나쁨)
        diff = b - a
        ratios = b / a
        # Wilcoxon signed-rank (two-sided) + 단측 (greater)
        try:
            w_two = wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
            w_gt  = wilcoxon(b, a, zero_method="wilcox", alternative="greater")
            p_two, p_gt = float(w_two.pvalue), float(w_gt.pvalue)
        except Exception as e:
            p_two = p_gt = float("nan")
        gm = float(np.exp(np.mean(np.log(ratios))))
        md = float(np.median(ratios))
        lo, hi = boot_ci_geomean(ratios)
        n_worse = int(np.sum(diff > 0))
        print(f"\n  [{name}] (N={N})")
        print(f"    기하평균 비율 = {gm:.3f}× (95% CI {lo:.3f}–{hi:.3f}), 중앙값 {md:.3f}×")
        print(f"    OFF 가 더 나쁜 시퀀스 = {n_worse}/{N} ({100*n_worse/N:.0f}%)")
        print(f"    Wilcoxon p (two-sided) = {p_two:.2e}, p (greater) = {p_gt:.2e}")
        sig = "유의 (p<0.05)" if p_two < 0.05 else "유의하지 않음 (p≥0.05)"
        print(f"    → {sig}")
        return dict(name=name, gm=gm, lo=lo, hi=hi, md=md, p_two=p_two, p_gt=p_gt, n_worse=n_worse)

    print("=" * 70)
    print("[A] 전체 pooled paired 검정")
    print("=" * 70)
    res_yaw = report("yaw vs ga (per-sample 효과)", ga, yaw)
    res_body = report("body vs ga (회전 유무)", ga, body)

    # 카테고리별 (n>=6 만)
    print("\n" + "=" * 70)
    print("[B] 카테고리별 yaw vs ga (n≥6, Wilcoxon two-sided p + 기하평균비율)")
    print("=" * 70)
    for cat in CATEGORIES:
        sub = [r for r in recs if r["cat"] == cat]
        if len(sub) < 6:
            print(f"  {cat:<10} n={len(sub)} (검정 생략)")
            continue
        a = np.array([r["ga"] for r in sub]); b = np.array([r["yaw"] for r in sub])
        try:
            p = float(wilcoxon(b, a, alternative="two-sided").pvalue)
        except Exception:
            p = float("nan")
        gm = float(np.exp(np.mean(np.log(b / a))))
        mark = " *" if p < 0.05 else ""
        print(f"  {cat:<10} n={len(sub):>2}  yaw/ga geomean={gm:.2f}×  p={p:.3f}{mark}")

    # 해석
    print("\n" + "=" * 70)
    print("[C] 게재 해석")
    print("=" * 70)
    print(f"  per-sample (yaw vs ga): 기하평균 {res_yaw['gm']:.2f}× "
          f"(CI {res_yaw['lo']:.2f}–{res_yaw['hi']:.2f}), p={res_yaw['p_two']:.1e}")
    print(f"  회전유무 (body vs ga) : 기하평균 {res_body['gm']:.2f}× "
          f"(CI {res_body['lo']:.2f}–{res_body['hi']:.2f}), p={res_body['p_two']:.1e}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("category,sequence,ga,yaw,body\n")
            for r in recs:
                f.write(f"{r['cat']},{r['seq']},{r['ga']:.4f},{r['yaw']:.4f},{r['body']:.4f}\n")
        print(f"\n[OK] CSV 저장: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
