"""
cls3 모델의 log_var 및 클래스 예측 분포를 분석해 STATE_EKF_PARAMS 설정값을 계산.

실행:
    cd d:/EKF_basic/src
    python analyze_logvar.py
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC / "Network"))
sys.path.insert(0, str(_SRC / "Trans"))
sys.path.insert(0, str(_SRC))

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

from model_twolayer import TwoLayerModel
from dataset import TLIONpySingleDataset

G_NORM = 9.81

CLASS_NAMES = {
    1: "handbag",
    2: "handheld",
    3: "pocket",
    4: "running",
    5: "slow-walk",
    6: "trolley",
    7: "noise",
}

SEQUENCES = {
    "handheld_21": "TLIO_Oxford_Dataset/oxford_handheld_21/imu0_resampled.npy",
    "handbag_1":   "TLIO_Oxford_Dataset/oxford_handbag_1/imu0_resampled.npy",
    "large_scale_6": "TLIO_Oxford_Dataset/oxford_large_scale_6/imu0_resampled.npy",
}

MODEL_PATH = "outputs/out_classifier3/checkpoints/best.pth"
NORM_MEAN  = "outputs/out_classifier3/norm_mean.npy"
NORM_STD   = "outputs/out_classifier3/norm_std.npy"
WINDOW_LEN = 200
STRIDE     = 200


def load_model(device):
    patch_len = WINDOW_LEN // 10
    para = {
        "input_len": WINDOW_LEN, "input_channel": 6, "patch_len": patch_len,
        "feature_dim": 128, "out_dim": 3, "active_func": "GELU",
        "extractor": {"name": "ResMLP", "layer_num": 6, "expansion": 2, "dropout": 0.0},
        "reg":       {"name": "PoseCondMean", "layer_num": 3, "dropout": 0.0},
        "classifier": {"num_classes": 7, "layer_num": 2, "dropout": 0.0, "pooling_type": "mean"},
        "use_classifier": True,
    }
    model = TwoLayerModel(para).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def analyze_sequence(seq_name, data_path, model, norm_mean, norm_std, device):
    data    = np.load(data_path)
    pos_gt  = data[:, 18:21].astype(np.float64)
    quat    = data[:, 14:18].astype(np.float32)

    dataset = TLIONpySingleDataset(
        npy_path=data_path, window_len=WINDOW_LEN, stride=STRIDE,
        normalize=True, precomputed_stats=(norm_mean, norm_std),
        is_train=False, fmt="oxford", with_label=False,
    )

    all_preds, all_logvars, all_gt_disp = [], [], []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    y_hats, log_vars, class_ids = [], [], []
    for imu, _ in loader:
        yh, lv, cl = model(imu.to(device))
        y_hats.append(yh.cpu().numpy())
        log_vars.append(lv.cpu().numpy())
        class_ids.append(cl.argmax(dim=1).cpu().numpy() + 1)  # 1-based

    y_hats    = np.concatenate(y_hats,    axis=0)  # [K, 3]
    log_vars  = np.concatenate(log_vars,  axis=0)  # [K, 3]
    class_ids = np.concatenate(class_ids, axis=0)  # [K]

    # GT 변위 계산 (world 프레임, yaw 정규화 없이)
    for k, start in enumerate(dataset.indices):
        end = start + WINDOW_LEN
        if end >= len(pos_gt):
            end = len(pos_gt) - 1
        gt_disp = pos_gt[end] - pos_gt[start]
        all_gt_disp.append(gt_disp)

    gt_disp = np.array(all_gt_disp)  # [K, 3]
    K = min(len(y_hats), len(gt_disp))
    y_hats   = y_hats[:K]
    log_vars = log_vars[:K]
    class_ids= class_ids[:K]
    gt_disp  = gt_disp[:K]

    # per-window 오차 (GT와 pred의 norm 차이)
    per_err = np.linalg.norm(y_hats - gt_disp, axis=1)  # [K]
    per_gt_norm = np.linalg.norm(gt_disp, axis=1)        # [K]

    exp_lv_mean = np.exp(np.clip(log_vars, -4.0, 4.0))   # [K, 3]

    print(f"\n{'='*60}")
    print(f"시퀀스: {seq_name}  (K={K} 윈도우)")
    print(f"  GT 이동 거리:  mean={per_gt_norm.mean():.3f}m  max={per_gt_norm.max():.3f}m")
    print(f"  pred 오차:     mean={per_err.mean():.3f}m  median={np.median(per_err):.3f}m")
    print(f"  log_var (원본) mean=[{log_vars[:,0].mean():.2f}, {log_vars[:,1].mean():.2f}, {log_vars[:,2].mean():.2f}]")
    print(f"  exp(log_var)   mean=[{exp_lv_mean[:,0].mean():.4f}, {exp_lv_mean[:,1].mean():.4f}, {exp_lv_mean[:,2].mean():.4f}]")
    print(f"  exp(log_var)   diag_mean={exp_lv_mean.mean():.4f}")

    # 클래스별 분포
    print(f"\n  클래스별 분포:")
    for cls_id in sorted(set(class_ids)):
        mask = class_ids == cls_id
        cnt = mask.sum()
        lv_sub = log_vars[mask]
        ev_sub = exp_lv_mean[mask]
        err_sub = per_err[mask]
        cls_name = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
        print(f"    {cls_id}={cls_name:10s}: {cnt:3d}개 | "
              f"exp_lv_mean={ev_sub.mean():.4f} | "
              f"err_mean={err_sub.mean():.3f}m | "
              f"err_max={err_sub.max():.3f}m")

    # Mahalanobis 통과 조건 계산
    # d² = sum_i (meas_i² / R_ii) < 11.345  (chi², 3dof, 99%)
    # R_ii = STATE_scale × exp(log_var_i)
    # 보수 추정: meas ≈ GT 변위 (yaw-정규화 프레임에서는 크기 동일)
    # 필요 STATE_scale = per_err_mean² / (exp_lv_mean * 11.345/3)
    # 여기서는 단순화: R_ii 등방 가정, meas_err = per_err (근사)
    exp_lv_diag = exp_lv_mean.mean(axis=1)  # [K]
    err_sq = per_err ** 2
    # d² = 3 * err² / (STATE_scale * exp_lv) < 11.345
    # STATE_scale > 3 * err² / (exp_lv * 11.345)
    required_scale = 3.0 * err_sq / (exp_lv_diag * 11.345)
    print(f"\n  Mahalanobis 통과 필요 STATE_scale:")
    print(f"    mean={required_scale.mean():.3f}  "
          f"p50={np.percentile(required_scale,50):.3f}  "
          f"p90={np.percentile(required_scale,90):.3f}  "
          f"p95={np.percentile(required_scale,95):.3f}  "
          f"max={required_scale.max():.3f}")

    # 90th percentile 기준 권장값
    recommended = float(np.percentile(required_scale, 90))
    print(f"  → 권장 STATE_scale (p90 기준): {recommended:.2f}")
    return required_scale, class_ids, exp_lv_mean


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")

    norm_mean = np.load(NORM_MEAN)
    norm_std  = np.load(NORM_STD)
    model     = load_model(device)

    print("\n[모델 로드 완료]")

    results = {}
    for seq_name, data_path in SEQUENCES.items():
        full_path = _SRC / data_path
        if not full_path.exists():
            print(f"  파일 없음: {full_path}")
            continue
        req_scale, class_ids, exp_lv = analyze_sequence(
            seq_name, str(full_path), model, norm_mean, norm_std, device
        )
        results[seq_name] = (req_scale, class_ids)

    # 전체 합산 권장값
    print("\n" + "="*60)
    print("전체 권장 STATE_EKF_PARAMS 설정")
    print("="*60)

    all_req = np.concatenate([v[0] for v in results.values()])
    print(f"전체 p90={np.percentile(all_req,90):.3f}  "
          f"p95={np.percentile(all_req,95):.3f}  "
          f"max={all_req.max():.3f}")
    print()
    print("제안 (p90 기준, 클래스 무관 단일값으로 시작):")
    p90 = np.percentile(all_req, 90)
    print(f"  STATE_EKF_PARAMS 모든 클래스: meascov_scale = {p90:.1f}")
    print()
    print("참고: exp(log_var)이 클수록 모델이 자신의 불확실도를 높게 봄.")
    print("      STATE_scale이 너무 크면 측정값을 무시하고 IMU 적분만으로 진행.")
    print("      너무 작으면 Mahalanobis 게이팅 실패 → 업데이트 거부.")


if __name__ == "__main__":
    main()
