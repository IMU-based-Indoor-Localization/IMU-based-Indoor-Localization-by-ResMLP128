import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ekf import ImuEKF
from model_twolayer import TwoLayerModel
from dataset import IMU_COLS, TARGET_COLS

def run_ekf_on_oxiod(csv_path, model_path, config=None):
    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 기본 설정 - 학습 시 config와 일치해야 합니다.
    if config is None:
        config = {
            "input_len": 100,
            "input_channel": 6,
            "patch_len": 25,
            "feature_dim": 512,
            "out_dim": 3,
            "active_func": "GELU",
            "extractor": {
                "name": "ResMLP",
                "layer_num": 6,
                "expansion": 2,
                "dropout": 0.2,
            },
            "reg": {
                "name": "MeanMLP",
                "layer_num": 3,
            }
        }
    
    model = TwoLayerModel(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # 체크포인트 형식에 따라 state_dict 로드
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 2. 데이터 로드
    df = pd.read_csv(csv_path)
    imu_data = df[IMU_COLS].values # [T, 6]
    
    # Ground Truth가 있는 경우 비교를 위해 로드
    has_gt = all(col in df.columns for col in TARGET_COLS)
    if has_gt:
        gt_pos_delta = df[TARGET_COLS].values
        gt_pos = np.cumsum(gt_pos_delta, axis=0)
    
    # 타임스탬프 처리 (기본 200Hz -> 5ms)
    if 'Time' in df.columns:
        timestamps = (df['Time'].values * 1e6).astype(np.int64)
    else:
        timestamps = (np.arange(len(df)) * 0.005 * 1e6).astype(np.int64)

    # 3. EKF 초기화
    ekf = ImuEKF()
    # 첫 번째 IMU 샘플로 초기 위치 설정
    ekf.initialize(timestamps[0], imu_data[0, :3].reshape((3, 1)))

    ekf_pos = []
    window_len = config['input_len']
    stride = 10 # 10개 샘플마다 모델 예측 수행 (성능과 정확도 사이의 절충)
    
    print(f"Running EKF on {csv_path}...")
    
    for i in range(1, len(imu_data)):
        # IMU를 이용한 상태 전파 (Propagation)
        acc = imu_data[i, 0:3].reshape((3, 1))
        gyr = imu_data[i, 3:6].reshape((3, 1))
        
        # 윈도우 기반 업데이트를 위해 일정 간격으로 상태 복제(Augmentation)
        augment = (i % stride == 0)
        ekf.propagate(timestamps[i], acc, gyr, augment=augment)
        
        # 모델 기반 측정 업데이트 (Measurement Update)
        if i >= window_len and augment:
            window_start = i - window_len
            window_imu = imu_data[window_start:i] # [100, 6]
            
            # 입력 데이터 전처리 (입형이 [Batch, Channel, Length] 형식이어야 함)
            inp = torch.from_numpy(window_imu.T).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                preds = model(inp)
                # TwoLayerModel은 (out1, out2) 혹은 (out1, out2, pose_logits)를 반환함
                pred_dp = preds[0].cpu().numpy().flatten()
                pred_log_var = preds[1].cpu().numpy().flatten()
                
                # 로그 분산을 공분산 행렬로 변환
                pred_var = np.exp(pred_log_var)
                meas_cov = np.diag(pred_var)
            
            # EKF 업데이트 수행
            t_start = timestamps[window_start]
            t_end = timestamps[i]
            ekf.update_displacement(pred_dp, meas_cov, t_start, t_end)
            
            # 계산 효율을 위해 오래된 복제 상태 제거 (Marginalization)
            if ekf.state.N > 15:
                ekf.marginalize(1)
        
        ekf_pos.append(ekf.state.p.flatten().copy())

    ekf_pos = np.array(ekf_pos)
    
    # 4. 결과 시각화
    plt.figure(figsize=(10, 6))
    if has_gt:
        plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'k--', label='Ground Truth', alpha=0.5)
    plt.plot(ekf_pos[:, 0], ekf_pos[:, 1], 'r-', label='EKF Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Trajectory Comparison - {Path(csv_path).name}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('trajectory_comparison.png')
    print("Trajectory plot saved as trajectory_comparison.png")
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python test_ekf.py <csv_path> <model_path>")
    else:
        run_ekf_on_oxiod(sys.argv[1], sys.argv[2])
