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

    # step endpoint 기준 RMSE
    # pred_pos[1:] <-> gt_anchor_positions
    rmse = np.sqrt(np.mean(np.sum((pred_pos[1:] - gt_anchor_positions) ** 2, axis=1)))
    print(f"Step-endpoint RMSE: {rmse:.4f} m")

    # 11. 3D plot
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 전체 GT absolute trajectory
    ax.plot(
        gt_pos_full[:, 0], gt_pos_full[:, 1], gt_pos_full[:, 2],
        label="Ground Truth (absolute)", color="blue", alpha=0.45
    )

    # GT step endpoints
    ax.plot(
        gt_anchor_positions[:, 0], gt_anchor_positions[:, 1], gt_anchor_positions[:, 2],
        label="GT step endpoints", color="cyan", alpha=0.8
    )

    # 모델 예측 trajectory
    ax.plot(
        pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2],
        label="Prediction (reconstructed)", color="red", alpha=0.9
    )

    # target 복원 trajectory (디버그용)
    ax.plot(
        target_recon[:, 0], target_recon[:, 1], target_recon[:, 2],
        label="Target recon (debug)", color="green", alpha=0.7
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Trajectory Reconstruction\n{Path(data_path).name} | stride={stride}")
    ax.legend()
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