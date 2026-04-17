import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

from model_twolayer import TwoLayerModel
from dataset import TLIONpySingleDataset


def plot_trajectory(model_path, data_path, device,
                    norm_mean_path, norm_std_path,
                    stride=200):
    # 1. 모델 설정
    model_para = {
        "input_len": 200,
        "input_channel": 6,
        "patch_len": 20,
        "feature_dim": 128,
        "out_dim": 3,
        "active_func": "GELU",
        "extractor": {"name": "ResMLP", "layer_num": 6, "expansion": 2, "dropout": 0.0},
        "reg": {"name": "SimpleMean", "layer_num": 3, "dropout": 0.0},
        "use_classifier": False,
    }

    # 2. 모델 로드
    model = TwoLayerModel(model_para).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"모델 로드 완료: {model_path}")

    # 3. 학습 시 저장한 정규화 통계 로드
    norm_mean = np.load(norm_mean_path)
    norm_std = np.load(norm_std_path)

    # 4. 데이터셋 로드
    #    주의: stride=1로 두면 1초 displacement가 서로 심하게 겹치므로
    #    trajectory 복원용으로는 non-overlap(stride=200) 권장
    dataset = TLIONpySingleDataset(
        npy_path=data_path,
        window_len=200,
        stride=stride,
        normalize=True,
        precomputed_stats=(norm_mean, norm_std),
        is_train=False,
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    print(f"데이터 로드 완료: {data_path} (샘플 수: {len(dataset)})")

    # 5. 원본 absolute GT 로드
    raw = np.load(data_path)
    quat = raw[:, 7:11].astype(np.float32)
    gt_pos_full = raw[:, 11:14].astype(np.float32)

    # 6. 추론
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imu, target in loader:
            imu = imu.to(device)
            y_hat, _, _ = model(imu)
            all_preds.append(y_hat.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    preds_local = np.concatenate(all_preds, axis=0)     # [K, 3]
    targets_local = np.concatenate(all_targets, axis=0) # [K, 3]

    # 7. local displacement -> world displacement 변환
    pred_world_steps = []
    target_world_steps = []
    gt_anchor_positions = []

    for k, start in enumerate(dataset.indices):
        end = start + dataset.window_len

        # window 시작 시점 yaw만 사용 (dataset.py와 동일한 정의)
        r_start = R.from_quat(quat[start])
        yaw = r_start.as_euler("zyx", degrees=False)[0]
        R_yaw = R.from_euler("z", yaw).as_matrix()

        pred_world = R_yaw @ preds_local[k]
        target_world = R_yaw @ targets_local[k]

        pred_world_steps.append(pred_world)
        target_world_steps.append(target_world)
        gt_anchor_positions.append(gt_pos_full[end])  # 각 윈도우 끝 absolute GT

    pred_world_steps = np.asarray(pred_world_steps, dtype=np.float32)
    target_world_steps = np.asarray(target_world_steps, dtype=np.float32)
    gt_anchor_positions = np.asarray(gt_anchor_positions, dtype=np.float32)

    # 8. 예측 궤적 복원
    #    첫 위치는 실제 GT 시작점으로 고정
    pred_pos = [gt_pos_full[0].copy()]
    for step in pred_world_steps:
        pred_pos.append(pred_pos[-1] + step)
    pred_pos = np.asarray(pred_pos, dtype=np.float32)

    # 9. GT 비교용 step trajectory
    #    target을 world로 되돌린 뒤 누적하면 dataset target 정의가 맞는지 같이 확인 가능
    target_recon = [gt_pos_full[0].copy()]
    for step in target_world_steps:
        target_recon.append(target_recon[-1] + step)
    target_recon = np.asarray(target_recon, dtype=np.float32)

    # 10. 간단한 수치 점검
    print("GT full range min:", gt_pos_full.min(axis=0))
    print("GT full range max:", gt_pos_full.max(axis=0))
    print("Pred traj range min:", pred_pos.min(axis=0))
    print("Pred traj range max:", pred_pos.max(axis=0))

    # 3D step-endpoint RMSE
    rmse_3d = np.sqrt(np.mean(np.sum((pred_pos[1:] - gt_anchor_positions) ** 2, axis=1)))
    print(f"Step-endpoint RMSE (3D): {rmse_3d:.4f} m")

    # XY 기준 step-endpoint RMSE
    rmse_xy = np.sqrt(
        np.mean(
            np.sum((pred_pos[1:, :2] - gt_anchor_positions[:, :2]) ** 2, axis=1)
        )
    )
    print(f"Step-endpoint RMSE (XY): {rmse_xy:.4f} m")

    # 11. 2D XY plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # GT
    ax.plot(
        gt_pos_full[:, 0], gt_pos_full[:, 1],
        label="GT", color="blue", alpha=0.7, linewidth=2.0
    )

    # Prediction
    ax.plot(
        pred_pos[:, 0], pred_pos[:, 1],
        label="Prediction", color="red", alpha=0.9, linewidth=2.0
    )

    # Start / End points
    ax.scatter(
        gt_pos_full[0, 0], gt_pos_full[0, 1],
        marker="o", s=60, label="Start"
    )
    ax.scatter(
        gt_pos_full[-1, 0], gt_pos_full[-1, 1],
        marker="x", s=70, label="GT End"
    )
    ax.scatter(
        pred_pos[-1, 0], pred_pos[-1, 1],
        marker="x", s=70, label="Pred End"
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Trajectory Reconstruction (XY)\n"
        f"{Path(data_path).name} | stride={stride} | RMSE_XY={rmse_xy:.3f} m"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    MODEL_W = "./out_tlio_6ch_128/checkpoints/best.pth"
    DATA_NPY = "./dataset_split/test/145820422949970/imu0_resampled.npy"
    NORM_MEAN = "./out_tlio_6ch_128/norm_mean.npy"
    NORM_STD = "./out_tlio_6ch_128/norm_std.npy"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plot_trajectory(
        model_path=MODEL_W,
        data_path=DATA_NPY,
        device=DEVICE,
        norm_mean_path=NORM_MEAN,
        norm_std_path=NORM_STD,
        stride=200,   # non-overlap 권장
    )