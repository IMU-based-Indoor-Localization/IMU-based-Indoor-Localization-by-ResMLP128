# -*- coding: utf-8 -*-
"""현재 배포 모델(out_classifier2) in-domain 강건성 — OxIOD 카테고리별 DR ATE.
evaluate() 재사용: 모델 dead-reckoning 궤적 vs GT 궤적 RMSE(ATE).
프레임 ga/yaw/body 3종 → in-domain 전처리 효과도 동시 확인(Android N=3 보완).
cd src/Network; KMP_DUPLICATE_LIB_OK=TRUE python oxiod_model_ate.py [data_dir]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).parent))
from offline_eval import FS, load_oxiod, load_model, evaluate  # type: ignore

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
SEQS = [("handbag", "oxford_handbag_1"), ("handheld", "oxford_handheld_1"),
        ("pocket", "oxford_pocket_1"), ("running", "oxford_running_1"),
        ("slow", "oxford_slow_walking_1"), ("trolley", "oxford_trolley_1"),
        ("large", "oxford_large_scale_1")]


def main():
    print("[1] 모델 로드 (out_classifier2)")
    net, _p, mean, std = load_model("src/Network/out_classifier2")
    print(f"[2] OxIOD ATE 벤치 — {DATA}\n")
    hdr = f"{'category':>9} | {'dur':>5} | {'DRpath':>7} {'GTpath':>7} | {'ATE_ga':>7} {'rel%':>5} | {'ATE_yaw':>7} {'ATE_body':>8}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for cat, seq in SEQS:
        f = DATA / seq / "imu0_resampled.npy"
        if not f.exists():
            print(f"{cat:>9} | (없음: {f})"); continue
        data = load_oxiod(f)
        rg = evaluate(net, data, mean, std, frame="ga")
        ry = evaluate(net, data, mean, std, frame="yaw")
        rb = evaluate(net, data, mean, std, frame="body")
        dur = rg["n_win"]  # 1s/window
        gtp = rg["gt_path_len"]; ate = rg["rmse"]
        rel = 100 * ate / gtp if gtp > 1e-6 else float("nan")
        rows.append((cat, dur, rg["path_len"], gtp, ate, rel, ry["rmse"], rb["rmse"]))
        print(f"{cat:>9} | {dur:>4}s | {rg['path_len']:>6.1f}m {gtp:>6.1f}m | {ate:>6.2f}m {rel:>4.0f}% | {ry['rmse']:>6.2f}m {rb['rmse']:>7.2f}m")

    if rows:
        a = np.array([[r[4], r[5], r[6], r[7]] for r in rows])
        print("-" * len(hdr))
        print(f"{'평균':>9} | {'':>5} | {'':>7} {'':>7} | {a[:,0].mean():>6.2f}m {a[:,1].mean():>4.0f}% | {a[:,2].mean():>6.2f}m {a[:,3].mean():>7.2f}m")
        print(f"\n[해석] ATE_ga = 현재 모델 in-domain 측위오차(작을수록 강건).")
        print(f"        ga≈yaw, body≫ga 면 '중력정렬 전처리 필수'가 in-domain 에서도 성립(Android N=3 과 일치).")


if __name__ == "__main__":
    main()
