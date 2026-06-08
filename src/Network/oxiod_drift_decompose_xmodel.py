"""
oxiod_drift_decompose_xmodel.py — yaw-drift 방향/크기 해리의 *백본 독립성* 검증
==============================================================================
§4.6.2 의 핵심 신규 주장 — "yaw 드리프트는 변위 *방향* 예측만 선택적으로 붕괴(7.7×)
시키되 *크기* 예측은 보존(1.0×)" — 이 ResMLP128 특유 현상인지, 학습형 관성
오도메트리의 일반 속성인지를 확인한다.

oxiod_drift_decompose.py 의 input_only 분해 파이프라인을 그대로 쓰되, *백본만* 교체:
  - resmlp        : TwoLayerModel (ResMLP128, 논문 배포 모델, out_regression)
  - resnet_small  : ResNet1DSmall(base_plane=19, fc_dim=166) ≈ 460K, ResMLP 와 동급
  - resnet_full   : ResNet1DSmall(base_plane=64, fc_dim=512) ≈ 5M (TLIO 원본 크기)

각 모델은 *자기* norm_mean/std 로 정규화한다 (compare_models.py 와 동일).
평가 시퀀스·드리프트·분해 지표는 oxiod_drift_decompose.py 와 1:1 동일하므로,
출력되는 (방향 배수, 크기 배수) 를 백본 간 직접 비교할 수 있다.

지표 (input_only, 8 카테고리 평균):
  (2) vec_matched : ‖d − gt_drift‖  — 입력 프레임 정합 GT 와 비교 = 순수 예측오차(방향)
  (3) mag_err     : | |d| − |gt| |  — 회전 불변 = 크기 예측오차
  배수 = drift 5°/s 값 / drift 0 값.

사용:
  python src/Network/oxiod_drift_decompose_xmodel.py \\
      --data_dir D:/EKF_DATASET/TLIO_Oxford_Dataset \\
      --resmlp_dir src/Network/out_regression \\
      --resnet_small_dir "D:/mobile/IMU-based-Indoor-Localization-by-ResMLP128/src/Network/out_resnet_small" \\
      --resnet_full_dir  "D:/mobile/IMU-based-Indoor-Localization-by-ResMLP128/src/Network/out_resnet" \\
      --out logs/oxiod_drift_decompose_xmodel.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from offline_eval import (   # type: ignore
    WINDOW_LEN, load_oxiod,
    window_to_gravity_aligned, window_yaw0,
)
from oxiod_preproc_ablation import select_longest_sequence, CATEGORIES   # type: ignore
from oxiod_yaw_drift_ablation import make_drifted_quat   # type: ignore

DRIFT_RATES = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0]
_INTER_DIM = 4   # window=100 @100Hz: 100→50→25→13→7→4 (compare_models.py 와 동일)


# ─────────────────────────────────────────────────────────────────────────────
# 백본 로더 — 각 모델 + 자기 norm 반환. predict 시 통일된 disp[:2] 추출.
# ─────────────────────────────────────────────────────────────────────────────
def _load_norm(model_dir: Path):
    return (np.load(model_dir / "norm_mean.npy").astype(np.float32),
            np.load(model_dir / "norm_std.npy").astype(np.float32))


def load_backbone(kind: str, model_dir: Path):
    """반환: (net, mean, std, is_resnet)."""
    import torch
    model_dir = Path(model_dir)
    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    if kind == "resmlp":
        from model_twolayer import TwoLayerModel
        net = TwoLayerModel(cfg["model"])
        ckpt = torch.load(model_dir / "checkpoints" / "best.pth", map_location="cpu",
                          weights_only=False)
        state = (ckpt.get("model_state_dict") or ckpt.get("model")
                 or ckpt.get("state_dict") or ckpt)
        net.load_state_dict(state, strict=False)
        is_resnet = False
    else:
        from model_resnet_small import ResNet1DSmall
        if kind == "resnet_small":
            bp = cfg.get("resnet_base_plane", 19)
            fd = cfg.get("resnet_fc_dim", 166)
        else:  # resnet_full
            bp = cfg.get("resnet_base_plane", 64)
            fd = cfg.get("resnet_fc_dim", 512)
        net = ResNet1DSmall(in_dim=6, out_dim=3, group_sizes=[2, 2, 2, 2],
                            inter_dim=_INTER_DIM, base_plane=bp, fc_dim=fd)
        ckpt = torch.load(model_dir / "checkpoints" / "best.pth", map_location="cpu",
                          weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        net.load_state_dict(state, strict=True)
        is_resnet = True

    net.eval()
    mean, std = _load_norm(model_dir)
    return net, mean, std, is_resnet


def predict_disp(net, x, is_resnet):
    """통일 추론 → ga-frame disp [3]."""
    if is_resnet:
        y, _cov = net(x)
    else:
        y, _cov, _logits = net(x)
    return y[0].numpy()


# ─────────────────────────────────────────────────────────────────────────────
# input_only 분해 — oxiod_drift_decompose.py 와 동일 로직
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_decompose(net, is_resnet, data, mean, std, drift_rate: float):
    import torch
    acc, gyr, quat_true, pos = data["acc"], data["gyr"], data["quat"], data["pos"]
    quat_in = make_drifted_quat(quat_true, drift_rate)   # 입력만 drift

    T = len(acc)
    starts = list(range(0, T - WINDOW_LEN, WINDOW_LEN))
    vec_mat, mag_err, abs_d, abs_gt = [], [], [], []

    with torch.no_grad():
        for s in starts:
            e = s + WINDOW_LEN
            imu = window_to_gravity_aligned(acc, gyr, quat_in, s, e, frame="ga")
            imu_n = (imu - mean) / std
            x = torch.from_numpy(imu_n.T.copy()).float().unsqueeze(0)
            d = predict_disp(net, x, is_resnet)[:2]      # 예측 disp (입력 ga 프레임)

            dp = pos[e, :2] - pos[s, :2]                 # GT world 변위
            yd = window_yaw0(quat_in, s)                 # 입력(drift) 프레임 정합
            cd, sd = np.cos(-yd), np.sin(-yd)
            gt_drift = np.array([cd * dp[0] - sd * dp[1], sd * dp[0] + cd * dp[1]])

            vec_mat.append(np.hypot(d[0] - gt_drift[0], d[1] - gt_drift[1]))
            dn = np.hypot(d[0], d[1]); gn = np.hypot(dp[0], dp[1])
            mag_err.append(abs(dn - gn)); abs_d.append(dn); abs_gt.append(gn)

    rms = lambda a: float(np.sqrt(np.mean(np.array(a) ** 2)))
    return {"vec_matched": rms(vec_mat), "mag_err": rms(mag_err),
            "mean_abs_d": float(np.mean(abs_d)), "mean_abs_gt": float(np.mean(abs_gt))}


def run_backbone(kind, model_dir, seqs):
    print(f"\n[load] {kind:<13} {model_dir}")
    net, mean, std, is_resnet = load_backbone(kind, Path(model_dir))
    n_par = sum(p.numel() for p in net.parameters())
    print(f"       params={n_par:,}  is_resnet={is_resnet}")
    rows = []
    for dr in DRIFT_RATES:
        for cat, data in seqs.items():
            rows.append((dr, cat, evaluate_decompose(net, is_resnet, data, mean, std, dr)))
    avg = lambda dr, k: float(np.mean([r[2][k] for r in rows if r[0] == dr]))
    base_dir = avg(0.0, "vec_matched"); base_mag = avg(0.0, "mag_err")
    end_dir = avg(5.0, "vec_matched");  end_mag = avg(5.0, "mag_err")
    summary = {
        "kind": kind, "n_par": n_par,
        "base_dir": base_dir, "end_dir": end_dir,
        "base_mag": base_mag, "end_mag": end_mag,
        "dir_x": end_dir / base_dir if base_dir > 0 else float("nan"),
        "mag_x": end_mag / base_mag if base_mag > 0 else float("nan"),
        "mean_abs_d_0": avg(0.0, "mean_abs_d"), "mean_abs_d_5": avg(5.0, "mean_abs_d"),
        "rows": rows,
    }
    print(f"       baseline 방향오차 {base_dir:.3f}m  크기오차 {base_mag:.3f}m  "
          f"평균|d| {summary['mean_abs_d_0']:.3f}m")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=r"D:/EKF_DATASET/TLIO_Oxford_Dataset")
    ap.add_argument("--resmlp_dir", default="src/Network/out_regression")
    ap.add_argument("--resnet_small_dir",
                    default=r"D:/mobile/IMU-based-Indoor-Localization-by-ResMLP128/src/Network/out_resnet_small")
    ap.add_argument("--resnet_full_dir",
                    default=r"D:/mobile/IMU-based-Indoor-Localization-by-ResMLP128/src/Network/out_resnet")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[1] 카테고리당 가장 긴 시퀀스 ({data_dir})")
    seqs = {}
    for c in CATEGORIES:
        sd = select_longest_sequence(data_dir, c)
        if sd is not None:
            seqs[c] = load_oxiod(sd / "imu0_resampled.npy")
            print(f"    {c:<10} {sd.name}")
    print(f"    → {len(seqs)} 카테고리")

    backbones = [("resmlp", args.resmlp_dir),
                 ("resnet_small", args.resnet_small_dir),
                 ("resnet_full", args.resnet_full_dir)]
    summaries = []
    for kind, mdir in backbones:
        if not (Path(mdir) / "checkpoints" / "best.pth").exists():
            print(f"\n[skip] {kind}: 체크포인트 없음 ({mdir})")
            continue
        try:
            summaries.append(run_backbone(kind, mdir, seqs))
        except Exception as e:
            print(f"\n[fail] {kind}: {type(e).__name__}: {e}")

    print("\n" + "=" * 78)
    print("[A] 백본 간 yaw-drift 방향/크기 해리 (input_only, 8 카테고리 평균)")
    print("=" * 78)
    print(f"{'backbone':<14} {'params':>10} | {'방향오차 0→5°/s':>22} | "
          f"{'크기오차 0→5°/s':>22}")
    print("-" * 78)
    for s in summaries:
        print(f"{s['kind']:<14} {s['n_par']:>10,} | "
              f"{s['base_dir']:.3f}→{s['end_dir']:.3f}m ({s['dir_x']:>5.2f}×) | "
              f"{s['base_mag']:.3f}→{s['end_mag']:.3f}m ({s['mag_x']:>5.2f}×)")
    print("=" * 78)
    print("판정: 모든 백본에서 방향배수 ≫ 크기배수(≈1.0×) 이면 → 방향 선택적 붕괴는")
    print("      백본 독립적 = 학습형 관성 오도메트리의 일반 속성 (§4.6.2 일반화).")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("backbone,drift_deg_s,category,vec_matched,mag_err,mean_abs_d,mean_abs_gt\n")
            for s in summaries:
                for dr, cat, m in s["rows"]:
                    f.write(f"{s['kind']},{dr},{cat},{m['vec_matched']:.4f},"
                            f"{m['mag_err']:.4f},{m['mean_abs_d']:.4f},{m['mean_abs_gt']:.4f}\n")
        print(f"\n[OK] CSV 저장: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
