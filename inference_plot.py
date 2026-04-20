import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

from networks.model_twolayer import TwoLayerModel
from networks.dataset import ImuDataset, build_dataloaders


def load_model(model_path, device, use_classifier=True, input_len=100, patch_len=10):
    model_para = {
        "input_len": input_len, "input_channel": 6, "patch_len": patch_len,
        "feature_dim": 128, "out_dim": 3, "active_func": "GELU",
        "extractor": {"name": "ResMLP", "layer_num": 6, "expansion": 2, "dropout": 0.0},
        "reg": {"name": "PoseCondMean" if use_classifier else "SimpleMean",
                "layer_num": 3, "dropout": 0.0},
        "classifier": {"num_classes": 7, "layer_num": 2, "dropout": 0.0, "pooling_type": "mean"},
        "use_classifier": use_classifier,
    }
    model = TwoLayerModel(model_para).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"모델 로드 완료: {model_path}")
    return model


def infer_single_file(model, data_path, device, norm_mean, norm_std,
                      stride=100, fmt="oxford", window_len=100):
    # Oxford: quat=14:18, pos=18:21 / TLIO: quat=7:11, pos=11:14
    if fmt == "oxford":
        quat_col, pos_col = (14, 18), (18, 21)
    else:
        quat_col, pos_col = (7, 11), (11, 14)

    dataset = TLIONpySingleDataset(
        npy_path=data_path,
        window_len=window_len,
        stride=stride,
        normalize=True,
        precomputed_stats=(norm_mean, norm_std),
        is_train=False,
        fmt=fmt,
        with_label=False,
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    raw = np.load(data_path)
    quat        = raw[:, quat_col[0]:quat_col[1]].astype(np.float32)
    gt_pos_full = raw[:, pos_col[0]: pos_col[1]].astype(np.float32)

    all_preds = []
    with torch.no_grad():
        for imu, _ in loader:
            imu = imu.to(device)
            y_hat, _, _ = model(imu)
            all_preds.append(y_hat.cpu().numpy())

    preds_local = np.concatenate(all_preds, axis=0)  # [K, 3]

    pred_world_steps   = []
    gt_anchor_positions = []

    for k, start in enumerate(dataset.indices):
        end = start + dataset.window_len
        r_start = R.from_quat(quat[start])
        yaw     = r_start.as_euler("zyx", degrees=False)[0]
        R_yaw   = R.from_euler("z", yaw).as_matrix()

        pred_world_steps.append(R_yaw @ preds_local[k])
        gt_anchor_positions.append(gt_pos_full[end])

    pred_world_steps    = np.array(pred_world_steps,    dtype=np.float32)
    gt_anchor_positions = np.array(gt_anchor_positions, dtype=np.float32)

    pred_pos = [gt_pos_full[0].copy()]
    for step in pred_world_steps:
        pred_pos.append(pred_pos[-1] + step)
    pred_pos = np.array(pred_pos, dtype=np.float32)

    rmse_xy = float(np.sqrt(
        np.mean(np.sum((pred_pos[1:, :2] - gt_anchor_positions[:, :2]) ** 2, axis=1))
    ))

    return gt_pos_full, pred_pos, rmse_xy


def plot_multiple_trajectories(model_path, data_root, device,
                               norm_mean_path, norm_std_path,
                               window_len=100, stride=100, max_files=10,
                               fmt="oxford", use_classifier=True):
    patch_len = window_len // 10  # 패치 수 10개 고정
    model     = load_model(model_path, device, use_classifier=use_classifier,
                           input_len=window_len, patch_len=patch_len)
    norm_mean = np.load(norm_mean_path)
    norm_std  = np.load(norm_std_path)

    # 클래스 순서로 정렬: label col 기준
    label_map = {1:'handbag',2:'handheld',3:'pocket',4:'running',5:'slow',6:'trolley',-1:'noise'}
    label_order = ['handbag','handheld','pocket','running','slow','trolley','noise']
    all_files = sorted(Path(data_root).rglob("imu0_resampled.npy"))
    label_col = 13 if fmt == "oxford" else None
    def cls_key(p):
        if label_col is None:
            return (99, p.name)
        lbl = int(np.load(p)[0, label_col])
        name = label_map.get(lbl, 'noise')
        return (label_order.index(name) if name in label_order else 99, p.name)
    npy_files = sorted(all_files, key=cls_key)[:max_files]
    if not npy_files:
        raise FileNotFoundError(f"{data_root} 아래에서 imu0_resampled.npy를 찾지 못했습니다.")

    n    = len(npy_files)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.0))
    axes = np.array(axes).reshape(-1)

    rmse_list = []
    for ax, npy_path in zip(axes, npy_files):
        gt_pos_full, pred_pos, rmse_xy = infer_single_file(
            model=model, data_path=npy_path, device=device,
            norm_mean=norm_mean, norm_std=norm_std,
            stride=stride, fmt=fmt, window_len=window_len,
        )
        rmse_list.append(rmse_xy)

        # origin 기준 정규화
        origin = gt_pos_full[0].copy()
        gt   = gt_pos_full - origin
        pred = pred_pos    - origin

        ax.plot(gt[:,0],   gt[:,1],   label="GT",   alpha=0.7, linewidth=2.0)
        ax.plot(pred[:,0], pred[:,1], label="Pred",  alpha=0.9, linewidth=2.0)
        ax.scatter(0, 0, marker="o", s=35, label="Start", zorder=5)
        ax.scatter(gt[-1,0],   gt[-1,1],   marker="x", s=45, label="GT End",   zorder=5)
        ax.scatter(pred[-1,0], pred[-1,1], marker="^", s=45, label="Pred End", zorder=5)

        ax.set_title(f"{npy_path.parent.name}\nRMSE_XY={rmse_xy:.3f}m", fontsize=9)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    fig.suptitle(
        f"XY Trajectories | window={window_len}(1s@100Hz) stride={stride} | {n} files | "
        f"mean RMSE_XY={np.mean(rmse_list):.3f}m", fontsize=13
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("inference_result.png", dpi=150)
    print(f"\n저장: inference_result.png")
    print(f"파일별 RMSE_XY: {[f'{r:.3f}' for r in rmse_list]}")
    print(f"평균 RMSE_XY: {np.mean(rmse_list):.3f}m")
    plt.show()


if __name__ == "__main__":
    # joint model (out_classifier2, window_len=100) + Oxford test set
    MODEL_W   = "./out_classifier2/checkpoints/best.pth"
    DATA_ROOT = "./cls_split_7way/test"
    NORM_MEAN = "./out_classifier2/norm_mean.npy"
    NORM_STD  = "./out_classifier2/norm_std.npy"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plot_multiple_trajectories(
        model_path=MODEL_W,
        data_root=DATA_ROOT,
        device=DEVICE,
        norm_mean_path=NORM_MEAN,
        norm_std_path=NORM_STD,
        window_len=100,
        stride=100,
        max_files=30,
        fmt="oxford",
        use_classifier=True,
    )
