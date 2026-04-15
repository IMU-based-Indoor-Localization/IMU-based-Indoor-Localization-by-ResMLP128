import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# 1. 단일 .npy 파일 처리 Dataset
# ---------------------------------------------------------------------------
class TLIONpySingleDataset(Dataset):
    """
    하나의 TLIO .npy 파일에서 슬라이딩 윈도우 샘플을 생성합니다.
    """
    def __init__(
        self,
        npy_path: Union[str, Path],
        window_len: int = 200,
        stride: int = 10,
        normalize: bool = True,
        precomputed_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        is_train: bool = True,  # <--- 이 줄 추가
    ):
        self.is_train = is_train # <--- 이 줄 추가
        self.npy_path = Path(npy_path)
        self.window_len = window_len
        self.stride = stride

        # 1. NPY 파일 로드 (Shape: [N, 17])
        data = np.load(self.npy_path)
        
        # 2. JSON 명세에 따른 컬럼 슬라이싱
        gyr_world = data[:, 1:4]
        acc_world = data[:, 4:7]
        self.quat_data = data[:, 7:11].astype(np.float32)
        self.pos_data = data[:, 11:14].astype(np.float32)

        # 3. [핵심] World 프레임을 Device(Body) 프레임으로 역회전
        rotations = R.from_quat(self.quat_data)
        rotations_inv = rotations.inv()
        
        gyr_body = rotations_inv.apply(gyr_world)
        acc_body = rotations_inv.apply(acc_world)
        
        # 4. 신경망 입력을 위한 6ch 구성 (acc + gyro)
        self.imu_data = np.concatenate([acc_body, gyr_body], axis=1).astype(np.float32)

        # 5. 정규화 (train 통계 우선 사용)
        if normalize:
            if precomputed_stats is not None:
                self.mean, self.std = precomputed_stats
            else:
                self.mean = self.imu_data.mean(axis=0)
                self.std = self.imu_data.std(axis=0) + 1e-8
            self.imu_data = (self.imu_data - self.mean) / self.std
        else:
            self.mean, self.std = None, None

        # 6. 슬라이딩 윈도우 인덱스
        T = len(self.imu_data)
        self.indices = list(range(0, T - window_len, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        end = start + self.window_len

        # [1] Input: 6ch IMU 데이터 [6, window_len]
        imu_np = self.imu_data[start:end].T.copy()
        
        # [2] Target: Gravity-Aligned Delta Position
        pos_start = self.pos_data[start]
        pos_end = self.pos_data[end]
        
        # World 좌표계 기준 변위
        delta_p_world = pos_end - pos_start 
        
        # 윈도우 시작점의 방향에서 Yaw(수평 회전)만 추출하여 수평 좌표계 형성
        r_start = R.from_quat(self.quat_data[start])
        yaw, pitch, roll = r_start.as_euler('zyx', degrees=False)
        R_yaw_inv = R.from_euler('z', yaw).inv().as_matrix()
        
        # 타겟 변위를 시작점의 수평 좌표계로 회전 (Z축인 중력방향은 유지됨)
        target_np = R_yaw_inv @ delta_p_world 
        
    # [3] Augmentation (Yaw Rotation)
        if self.is_train: 
            theta = float(np.random.uniform(-np.pi, np.pi))
            c, s_ = np.cos(theta), np.sin(theta)
            R2 = np.array([[c, -s_], [s_, c]], dtype=np.float32)
            
            # Accel, Gyro 수평 회전
            imu_np[0:2] = R2 @ imu_np[0:2]
            imu_np[3:5] = R2 @ imu_np[3:5]
            
            # Target 변위 수평 회전
            target_np[:2] = R2 @ target_np[:2]

        # if문이 끝나고 바로 return으로 넘어가야 합니다.
        return torch.from_numpy(imu_np), torch.from_numpy(target_np.astype(np.float32))

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mean, self.std


# ---------------------------------------------------------------------------
# 2. 다중 폴더/파일 병합 Dataset (ConcatDataset)
# ---------------------------------------------------------------------------
class TLIOMultiDataset(Dataset):
    """
    여러 개의 .npy 파일(여러 폴더)을 하나의 Dataset으로 합칩니다.
    """
    def __init__(
        self,
        base_dir: Union[str, Path],
        window_len: int = 200,
        stride: int = 10,
        normalize: bool = True,
        precomputed_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        is_train: bool = True,
    ):
        base_dir = Path(base_dir)
        
        # [매우 중요] rglob을 사용하여 하위 폴더의 모든 .npy 파일을 재귀적으로 찾습니다.
        npy_paths = sorted(base_dir.rglob("*.npy"))
        
        if not npy_paths:
            raise ValueError(f"[{base_dir}] 경로에서 .npy 파일을 찾을 수 없습니다.")

        # 정규화 통계 계산 (train의 경우 전체 데이터 기준)
        if normalize and precomputed_stats is None:
            precomputed_stats = self._compute_global_stats(npy_paths)

        # 개별 파일별 Dataset 생성
        sub_datasets = [
            TLIONpySingleDataset(
                npy_path=p,
                window_len=window_len,
                stride=stride,
                normalize=normalize,
                precomputed_stats=precomputed_stats,
                is_train=is_train,  # <--- 이 줄 추가
            )
            for p in npy_paths
        ]

        self._dataset = ConcatDataset(sub_datasets)
        self._stats = precomputed_stats
        
        print(f"[{base_dir.name}] 로드된 파일 수: {len(sub_datasets)} | 총 샘플 수: {len(self._dataset)}")

    def _compute_global_stats(self, npy_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """전체 데이터의 컬럼별 mean/std 계산"""
        all_imu = []
        print("전체 데이터 정규화 통계 계산 중...")
        for p in npy_paths:
            data = np.load(p)
            gyr_world = data[:, 1:4]
            acc_world = data[:, 4:7]
            quat_data = data[:, 7:11].astype(np.float32)
            
            rotations_inv = R.from_quat(quat_data).inv()
            gyr_body = rotations_inv.apply(gyr_world)
            acc_body = rotations_inv.apply(acc_world)
            
            imu_data = np.concatenate([acc_body, gyr_body], axis=1).astype(np.float32)
            all_imu.append(imu_data)
            
        concat = np.concatenate(all_imu, axis=0)
        mean = concat.mean(axis=0)
        std = concat.std(axis=0) + 1e-8
        return mean, std

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._stats

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._dataset[idx]


# ---------------------------------------------------------------------------
# 3. DataLoader 빌더 (train.py에서 호출하는 함수)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 3. DataLoader 빌더 (train.py에서 호출하는 함수)
# ---------------------------------------------------------------------------
def build_dataloaders(
    train_paths: str,
    val_paths: str,
    test_paths: Optional[str] = None,
    window_len: int = 200,
    train_stride: int = 10,
    eval_stride: int = 50,
    batch_size: int = 128,
    num_workers: int = 4,
    **kwargs # 호환성을 위한 쓰레기값 받기
) -> dict:
    """
    train.py에서 호출하는 최종 DataLoader 생성 함수
    """
    
    # 1. Train 데이터셋 로드 (정규화 통계 자동 계산)
    train_ds = TLIOMultiDataset(
        base_dir=train_paths,
        window_len=window_len,
        stride=train_stride,
        normalize=True, 
        is_train=True
    )
    stats = train_ds.get_stats()

    # 2. Val 데이터셋 로드 (Train 통계로 정규화)
    val_ds = TLIOMultiDataset(
        base_dir=val_paths,
        window_len=window_len,
        stride=eval_stride,
        normalize=True, 
        precomputed_stats=stats, 
        is_train=False
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True),
        "stats": stats,
    }

    # 3. Test 데이터셋 로드 (선택)
    if test_paths and Path(test_paths).exists():
        test_ds = TLIOMultiDataset(
            base_dir=test_paths,
            window_len=window_len,
            stride=eval_stride,
            normalize=True, 
            precomputed_stats=stats, 
            is_train=False
        )
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                        num_workers=num_workers, pin_memory=True)

    return loaders