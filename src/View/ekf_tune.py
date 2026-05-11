"""
ekf_tune.py
-----------
ResMLP128 (out_regression, no-classifier) + Method-B EKF 의
meascov_scale 을 Oxford 카테고리별로 그리드 서치로 자동 튜닝.

Method-B: classifier 없는 out_regression 모델을 EKF 와 결합.
          state_id 는 시퀀스 파일명에서 자동 추출 (handbag → 1 등).

실행:
    cd src/View
    python ekf_tune.py

출력:
    doc/ekf_compare/{category}.png   (GT / Network / 최적 EKF 비교)
    콘솔에 업데이트된 STATE_EKF_PARAMS 코드 출력
"""

import os
import sys
import json
import logging

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# scekf.py 의 INFO/WARNING 로그 억제 (Mahalanobis 경고 등)
logging.disable(logging.WARNING)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent   # src/View
_SRC  = _HERE.parent                      # src/
_ROOT = _SRC.parent                       # project root

sys.path.insert(0, str(_SRC / "Network"))
sys.path.insert(0, str(_SRC / "Trans"))
sys.path.insert(0, str(_SRC))

from model_twolayer import TwoLayerModel
from visualize_comparison import (
    load_npy,
    run_network,
    run_ekf_imutracker,
    STATE_EKF_PARAMS,
)
# ResMLPClassifier: 별도 분류기 (out_cls_oxford) — soft switching 에서 사용
try:
    from train_classifier import ResMLPClassifier
except ImportError:
    ResMLPClassifier = None

# ── 설정 ──────────────────────────────────────────────────────────────
MODEL_DIR     = _SRC / "Network" / "out_regression"    # 회귀 전용 모델 (use_classifier=False)
CLS_MODEL_DIR = _SRC / "Network" / "out_cls_oxford"    # 별도 분류기 모델 (ResMLPClassifier)
OXFORD_SPLIT  = _SRC / "Network" / "oxford_split"
OUTPUT_DIR    = _ROOT / "doc" / "ekf_compare"
WINDOW_LEN    = 100

# 카테고리 → state_id 매핑 (visualize_comparison.py 의 STATE_EKF_PARAMS 와 일치)
CATEGORY_TO_STATE: dict[str, int] = {
    "handbag":       1,
    "handheld":      2,
    "pocket":        3,
    "running":       4,
    "slow_walking":  5,
    "trolley":       6,
    "multi_devices": 7,
    "multi_users":   8,
    "large_scale":   9,
}

CATEGORIES = list(CATEGORY_TO_STATE.keys())

# 그리드 서치 후보 (log-scale, 작을수록 네트워크 신뢰)
GRID_MEASCOV = [
    0.001, 0.003, 0.005,
    0.01,  0.03,  0.05,
    0.1,   0.3,   0.5,
    1.0,   3.0,   10.0,
]

# ── 모델 로드 ─────────────────────────────────────────────────────────
def load_regression_model(model_dir: Path, device: torch.device):
    """out_regression (use_classifier=False, SimpleMean) 모델 로드."""
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    model_cfg = cfg["model"]
    model = TwoLayerModel(model_cfg).to(device)
    ck = torch.load(model_dir / "checkpoints" / "best.pth",
                    map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    mean = np.load(model_dir / "norm_mean.npy")
    std  = np.load(model_dir / "norm_std.npy")
    return model, mean, std


def load_classifier_model(model_dir: Path, device: torch.device):
    """out_cls_oxford (ResMLPClassifier, 별도 분류기) 모델 로드."""
    if ResMLPClassifier is None:
        raise ImportError("train_classifier.py 를 찾을 수 없습니다 (ResMLPClassifier 미정의)")
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    model_cfg = cfg["model"]
    model = ResMLPClassifier(model_cfg).to(device)
    ck = torch.load(model_dir / "checkpoints" / "best.pth",
                    map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    mean = np.load(model_dir / "norm_mean.npy")
    std  = np.load(model_dir / "norm_std.npy")
    return model, mean, std


# ── 시퀀스 탐색 (test → val → train 우선순위) ─────────────────────────
def find_sequence(category: str):
    for split in ["test", "val", "train"]:
        split_dir = OXFORD_SPLIT / split
        if not split_dir.exists():
            continue
        for d in sorted(split_dir.iterdir()):
            if d.is_dir() and category in d.name:
                npy = d / "imu0_resampled.npy"
                if npy.exists():
                    return npy, d.name, split
    return None, None, None


# ── RMSE_XY 계산 (윈도우 앵커 기반) ─────────────────────────────────
def compute_rmse_xy_anchor(ekf_pos: np.ndarray,
                           pos_gt: np.ndarray,
                           win_indices) -> float:
    """
    ekf_pos  : [T-1, 3]  EKF 위치 (ts[1]~ts[-1])
    pos_gt   : [T, 3]    GT 위치
    win_indices : 윈도우 시작 인덱스 목록
    """
    anchor_idxs = np.array([idx + WINDOW_LEN for idx in win_indices])
    valid = anchor_idxs < len(pos_gt)
    anchor_idxs = anchor_idxs[valid]
    if len(anchor_idxs) == 0:
        return float("nan")

    gt_xy  = pos_gt[anchor_idxs, :2]
    ekf_xy = ekf_pos[np.clip(anchor_idxs - 1, 0, len(ekf_pos) - 1), :2]
    err    = np.sqrt(np.sum((ekf_xy - gt_xy) ** 2, axis=1))
    return float(np.sqrt(np.mean(err ** 2)))


# ── 시각화 ────────────────────────────────────────────────────────────
def plot_ekf_result(category: str, seq_name: str, split: str,
                    pos_gt: np.ndarray, net_pos: np.ndarray,
                    ekf_pos: np.ndarray, grid_log: list,
                    best_scale: float, rmse_net: float, rmse_ekf: float,
                    output_dir: Path,
                    ekf_pos_gtyaw: np.ndarray = None,
                    rmse_gt_yaw: float = float("nan")):
    """
    Left : X-Y 경로 비교 (GT / Network / Best EKF / GT-yaw EKF)
    Right: meascov_scale vs RMSE_XY 그리드 서치 곡선
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("#f8f8f8")

    origin = pos_gt[0].copy()
    gt_xy  = pos_gt[:, :2]  - origin[:2]
    net_xy = net_pos[:, :2] - origin[:2]
    ekf_xy = ekf_pos[:, :2] - origin[:2]

    # ── Left: X-Y 경로 ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(gt_xy[:,0],  gt_xy[:,1],  lw=2.2, color="#2166ac", linestyle="-",
            label="GT")
    ax.plot(net_xy[:,0], net_xy[:,1], lw=1.8, color="#d7191c", linestyle="--",
            label=f"ResMLP128 (RMSE {rmse_net:.3f} m)")
    ax.plot(ekf_xy[:,0], ekf_xy[:,1], lw=1.8, color="#1a9641", linestyle="-.",
            label=f"EKF best  (scale={best_scale:.4g}, RMSE {rmse_ekf:.3f} m)")
    if ekf_pos_gtyaw is not None:
        gtyaw_xy = ekf_pos_gtyaw[:, :2] - origin[:2]
        ax.plot(gtyaw_xy[:,0], gtyaw_xy[:,1], lw=1.8, color="#ff7f00", linestyle=":",
                label=f"EKF+GT-yaw (RMSE {rmse_gt_yaw:.3f} m)")
    ax.plot(0, 0, "k^", markersize=9,  zorder=6, label="Start")
    ax.plot(*gt_xy[-1], "b*", markersize=11, zorder=6, label="End (GT)")

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(f"X–Y  |  {seq_name}", fontsize=10)
    ax.set_aspect("equal")
    ax.legend(fontsize=8.5, loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f2f2f2")

    # ── Right: 그리드 서치 곡선 ──────────────────────────────────────
    ax2 = axes[1]
    scales = [s for s, r in grid_log if not np.isnan(r)]
    rmses  = [r for s, r in grid_log if not np.isnan(r)]
    if scales:
        ax2.semilogx(scales, rmses, "o-", color="#7b2d8b", lw=1.8,
                     markersize=6, label="EKF RMSE_XY")
        ax2.axvline(best_scale, color="#d7191c", linestyle="--",
                    lw=1.5, label=f"Best = {best_scale:.4g}")
        ax2.axhline(rmse_net, color="#2166ac", linestyle=":",
                    lw=1.5, label=f"Network RMSE = {rmse_net:.3f} m")
        if not np.isnan(rmse_gt_yaw):
            ax2.axhline(rmse_gt_yaw, color="#ff7f00", linestyle="-.",
                        lw=1.5, label=f"EKF+GT-yaw = {rmse_gt_yaw:.3f} m")
        ax2.set_xlabel("meascov_scale (log)", fontsize=11)
        ax2.set_ylabel("RMSE_XY (m)", fontsize=11)
        ax2.set_title("Grid Search: meascov_scale vs RMSE_XY", fontsize=10)
        ax2.legend(fontsize=8.5, framealpha=0.85)
        ax2.grid(True, which="both", alpha=0.3)
        ax2.set_facecolor("#f2f2f2")

    fig.suptitle(
        f"Oxford  [{category}]  ({split} set)  —  Method-C: GT yaw 주입 검증",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = output_dir / f"{category}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  --> {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    print(">> ResMLP128 (out_regression) 로드...")
    model, norm_mean, norm_std = load_regression_model(MODEL_DIR, device)
    print("  [OK]")

    cls_model = cls_mean = cls_std = None
    if CLS_MODEL_DIR.exists():
        print(">> out_classifier2 (soft-switching 용) 로드...")
        try:
            cls_model, cls_mean, cls_std = load_classifier_model(CLS_MODEL_DIR, device)
            print("  [OK]\n")
        except Exception as e:
            print(f"  [WARN] 로드 실패: {e} → soft-switching 생략\n")
    else:
        print(f"  [WARN] {CLS_MODEL_DIR} 없음 → soft-switching 생략 (학습 후 재실행 필요)\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []  # (cat, seq, split, best_scale, rmse_net, rmse_ekf, rmse_gtyaw, rmse_gtmeas, rmse_gtyawnorm, rmse_soft, use_ekf, grid_log)
    network_only_sids_so_far: set = set()   # 누적 Network-only 상태 ID 집합

    for cat in CATEGORIES:
        npy_path, seq_name, split = find_sequence(cat)
        if npy_path is None:
            print(f"[{cat}]  시퀀스 없음, 스킵\n")
            continue

        state_id = CATEGORY_TO_STATE[cat]
        print(f"[{cat}]  {seq_name}  ({split})  state_id={state_id}")

        # IMU / GT 로드
        ts_us, gyr, acc_raw, quat, pos_gt, vel_gt = load_npy(str(npy_path))

        # Network 궤적 (1회만 실행)
        _gt, net_pos, pred_steps, anchor_gt, win_indices = run_network(
            model, str(npy_path), device, norm_mean, norm_std,
            WINDOW_LEN, WINDOW_LEN,
        )
        K = min(len(anchor_gt), len(net_pos) - 1)
        if K > 0:
            err_net = np.sqrt(np.sum(
                (net_pos[1:K+1, :2] - anchor_gt[:K, :2]) ** 2, axis=1
            ))
            rmse_net = float(np.sqrt(np.mean(err_net ** 2)))
        else:
            rmse_net = float("nan")

        # 그리드 서치
        best_scale = GRID_MEASCOV[-1]
        best_rmse  = float("inf")
        grid_log   = []

        for scale in GRID_MEASCOV:
            try:
                ekf_pos = run_ekf_imutracker(
                    ts_us, gyr, acc_raw, quat, pos_gt, vel_gt,
                    model, norm_mean, norm_std, WINDOW_LEN,
                    forced_state_id=state_id,
                    meascov_override=scale,
                    mahalanobis_fail_scale=10.0,
                )
                rmse = compute_rmse_xy_anchor(ekf_pos, pos_gt, list(win_indices))
                grid_log.append((scale, rmse))
                marker = ""
                if rmse < best_rmse:
                    best_rmse  = rmse
                    best_scale = scale
                    marker = " <<"
                print(f"    scale={scale:<8.4g}  RMSE_XY={rmse:.4f} m{marker}")
            except Exception as e:
                print(f"    scale={scale:.4g}  ERROR: {e}")
                grid_log.append((scale, float("nan")))

        print(f"  >> best: meascov_scale={best_scale:.4g}  "
              f"RMSE_XY={best_rmse:.4f} m  (network={rmse_net:.4f} m)")

        # 최적 스케일로 EKF 재실행 → 시각화 + 두 가지 진단
        rmse_gt_yaw      = float("nan")
        rmse_gt_meas     = float("nan")
        rmse_gt_yaw_norm = float("nan")
        try:
            ekf_pos_best = run_ekf_imutracker(
                ts_us, gyr, acc_raw, quat, pos_gt, vel_gt,
                model, norm_mean, norm_std, WINDOW_LEN,
                forced_state_id=state_id,
                meascov_override=best_scale,
                mahalanobis_fail_scale=10.0,
            )
            # ── 방안 C: GT yaw 주입 실험
            ekf_pos_gtyaw = run_ekf_imutracker(
                ts_us, gyr, acc_raw, quat, pos_gt, vel_gt,
                model, norm_mean, norm_std, WINDOW_LEN,
                forced_state_id=state_id,
                meascov_override=best_scale,
                mahalanobis_fail_scale=10.0,
                gt_yaw_inject=True,
            )
            rmse_gt_yaw = compute_rmse_xy_anchor(ekf_pos_gtyaw, pos_gt, list(win_indices))
            print(f"  >> GT yaw 주입:    RMSE_XY={rmse_gt_yaw:.4f} m "
                  f"(vs network={rmse_net:.4f} m, EKF={best_rmse:.4f} m)")

            # ── 방안 D: GT 측정값 사용 (IMU 전파 품질 진단)
            ekf_pos_gtmeas = run_ekf_imutracker(
                ts_us, gyr, acc_raw, quat, pos_gt, vel_gt,
                model, norm_mean, norm_std, WINDOW_LEN,
                forced_state_id=state_id,
                meascov_override=best_scale,
                mahalanobis_fail_scale=10.0,
                use_gt_meas=True,
            )
            rmse_gt_meas = compute_rmse_xy_anchor(ekf_pos_gtmeas, pos_gt, list(win_indices))
            print(f"  >> GT 측정값 사용:     RMSE_XY={rmse_gt_meas:.4f} m "
                  f"(vs network={rmse_net:.4f} m, EKF={best_rmse:.4f} m)")

            # ── 방안 E: GT yaw 정규화 + 네트워크 측정 (측정 프레임 수정)
            ekf_pos_gtyawnorm = run_ekf_imutracker(
                ts_us, gyr, acc_raw, quat, pos_gt, vel_gt,
                model, norm_mean, norm_std, WINDOW_LEN,
                forced_state_id=state_id,
                meascov_override=best_scale,
                mahalanobis_fail_scale=10.0,
                use_gt_yaw_norm=True,
            )
            rmse_gt_yaw_norm = compute_rmse_xy_anchor(ekf_pos_gtyawnorm, pos_gt, list(win_indices))
            print(f"  >> GT yaw 정규화+네트워크: RMSE_XY={rmse_gt_yaw_norm:.4f} m "
                  f"(vs network={rmse_net:.4f} m, EKF={best_rmse:.4f} m)")

            plot_ekf_result(
                cat, seq_name, split,
                pos_gt, net_pos, ekf_pos_best, grid_log,
                best_scale, rmse_net, best_rmse, OUTPUT_DIR,
                ekf_pos_gtyaw=ekf_pos_gtyaw, rmse_gt_yaw=rmse_gt_yaw,
            )
        except Exception as e:
            print(f"  [!] 시각화 실패: {e}")

        # ── Soft Switching EKF (out_regression + out_cls_oxford 별도 분류기) ──
        rmse_soft = float("nan")
        # 현재 카테고리 use_ekf 계산 및 누적 network_only 집합 갱신
        _use_ekf_cur = best_rmse < rmse_net
        if not _use_ekf_cur:
            network_only_sids_so_far.add(state_id)
        if cls_model is not None:
            try:
                ekf_pos_soft = run_ekf_imutracker(
                    ts_us, gyr, acc_raw, quat, pos_gt, vel_gt,
                    model, norm_mean, norm_std, WINDOW_LEN,   # 회귀 모델 (변위 추론)
                    mahalanobis_fail_scale=10.0,
                    use_soft_switching=True,                   # Context-Aware Soft Switching
                    cls_model=cls_model,                       # 별도 분류기 (meascov_scale 결정)
                    cls_norm_mean=cls_mean,
                    cls_norm_std=cls_std,
                )
                rmse_soft = compute_rmse_xy_anchor(ekf_pos_soft, pos_gt, list(win_indices))
                print(f"  >> Soft-Switch EKF:    RMSE_XY={rmse_soft:.4f} m "
                      f"(vs network={rmse_net:.4f} m, EKF={best_rmse:.4f} m)")
            except Exception as e:
                print(f"  [!] Soft-Switch 실패: {e}")

        use_ekf = best_rmse < rmse_net   # EKF RMSE 가 Network RMSE 보다 낮으면 EKF 사용
        results.append((cat, seq_name, split, best_scale,
                        rmse_net, best_rmse, rmse_gt_yaw, rmse_gt_meas, rmse_gt_yaw_norm, rmse_soft, use_ekf, grid_log))
        print()

    # ── 결과 요약 테이블 ─────────────────────────────────────────────
    print("\n" + "=" * 128)
    print(f"{'Category':<18} {'Split':<6} {'StateID':>7} {'Best Scale':>11} "
          f"{'Net RMSE':>10} {'EKF RMSE':>10} {'Soft-EKF':>10} {'GT-Meas':>9} {'GT-YawNorm':>11}  {'Winner':>8}")
    print("-" * 128)
    for cat, seq, split, scale, r_net, r_ekf, r_gtyaw, r_gtmeas, r_gtyawnorm, r_soft, use_ekf, _ in results:
        sid  = CATEGORY_TO_STATE.get(cat, -1)
        diff = r_net - r_ekf          # positive = EKF better
        mark = " [EKF+]" if use_ekf else " [NET+]"
        soft_str      = f"{r_soft:>10.4f}"      if not np.isnan(r_soft)      else f"{'N/A':>10}"
        gtmeas_str    = f"{r_gtmeas:>9.4f}"     if not np.isnan(r_gtmeas)    else f"{'N/A':>9}"
        gtyawnorm_str = f"{r_gtyawnorm:>11.4f}"  if not np.isnan(r_gtyawnorm) else f"{'N/A':>11}"
        print(f"{cat:<18} {split:<6} {sid:>7} {scale:>11.4g} "
              f"{r_net:>10.4f} {r_ekf:>10.4f} {soft_str} {gtmeas_str} {gtyawnorm_str}  {diff:>+7.4f}{mark}")
    print("=" * 128)

    # ── 업데이트된 STATE_EKF_PARAMS 코드 ────────────────────────────
    state_to_cat = {v: k for k, v in CATEGORY_TO_STATE.items()}
    cat_to_scale = {cat: scale for cat, _, _, scale, _, _, _, _, _, _, _, _ in results}

    labels = {
        -1: "unknown",      1: "handbag",   2: "handheld",
         3: "pocket",       4: "running",   5: "slow-walking",
         6: "trolley",      7: "multi_devices",
         8: "multi_users",  9: "large_scale",
    }
    # 기존 STATE_EKF_PARAMS 의 meascov_scale (튜닝되지 않은 state 에 fallback)
    default_scales = {
        -1: 0.001, 1: 0.05, 2: 0.01, 3: 0.005,
         4: 0.05,  5: 0.005, 6: 0.02,
         7: 0.01,  8: 0.01,  9: 0.01,
    }

    print("\n>> 아래 코드를 visualize_comparison.py 의 STATE_EKF_PARAMS 에 붙여넣으세요:")
    print("-" * 65)
    print("STATE_EKF_PARAMS = {")
    print("    # state_id: dict(meascov_scale, sigma_na, sigma_ng, ita_ba, ita_bg)")
    print("    # None 값은 run_ekf_imutracker 의 전역 파라미터를 그대로 사용함")
    for sid, label in labels.items():
        if sid == -1:
            scale = default_scales[-1]
        else:
            cat   = state_to_cat.get(sid)
            scale = cat_to_scale.get(cat, default_scales.get(sid, 0.01))
        print(f"    {sid:>2}: dict(meascov_scale={scale:<10.4g}, "
              f"sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # {label}")
    print("}")
    print("-" * 65)

    # ── NETWORK_ONLY_STATES 출력 ────────────────────────────────────────────
    network_only_cats = [
        cat for cat, _, _, _, r_net, r_ekf, _, _, _, _, use_ekf, _
        in results if not use_ekf
    ]
    network_only_sids = sorted(
        {CATEGORY_TO_STATE[cat] for cat in network_only_cats if cat in CATEGORY_TO_STATE}
    )

    print("\n>> 아래 코드를 visualize_comparison.py 의 NETWORK_ONLY_STATES 에 붙여넣으세요:")
    print("-" * 65)
    if network_only_sids:
        sid_list  = ", ".join(str(s) for s in network_only_sids)
        cat_names = ", ".join(network_only_cats)
        print(f"NETWORK_ONLY_STATES = {{{sid_list}}}  # {cat_names}")
    else:
        print("NETWORK_ONLY_STATES = set()  # 모든 카테고리에서 EKF 가 Network 보다 우수")
    print("-" * 65)

    print(f"\n[완료] 저장 위치: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
