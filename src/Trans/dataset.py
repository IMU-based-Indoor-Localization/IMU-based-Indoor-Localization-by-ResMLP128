import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.spatial.transform import Rotation as R

# 포맷별 컬럼 인덱스 정의
# TLIO: ts(1)+gyr(3)+acc(3)+quat(4)+pos(3)+vel(3) = 17col
# Oxford: ts(1)+gyr(3)+acc(3)+gravity(3)+attitude(3)+label(1)+qxyzw(4)+pos(3)+vel(3) = 24col
_COL = {
    "tlio":   {"quat": (7,  11), "pos": (11, 14), "label": None},
    "oxford": {"quat": (14, 18), "pos": (18, 21), "label": 13},
}

# ---------------------------------------------------------------------------
# 1. 단일 .npy 파일 처리 Dataset
# ---------------------------------------------------------------------------
class TLIONpySingleDataset(Dataset):
    """
    하나의 TLIO .npy 파일에서 슬라이딩 윈도우 샘플을 생성합니다.
    fmt='tlio'   : TLIO 17col 포맷 (label 없음)
    fmt='oxford' : Oxford 24col 포맷 (label col 13 포함)
    with_label   : True 시 __getitem__ 반환값에 label(int) 추가
    """
    def __init__(
        self,
        npy_path: Union[str, Path],
        window_len: int = 200,
        stride: int = 10,
        normalize: bool = True,
        precomputed_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        is_train: bool = True,
        fmt: str = "tlio",
        with_label: bool = False,
    ):
        assert fmt in _COL, f"fmt must be 'tlio' or 'oxford', got '{fmt}'"
        self.is_train = is_train
        self.npy_path = Path(npy_path)
        self.window_len = window_len
        self.stride = stride
        self.with_label = with_label
        self.fmt = fmt

        col = _COL[fmt]

        # 1. NPY 파일 로드
        data = np.load(self.npy_path)

        # 2. 컬럼 슬라이싱
        self.gyr_body = data[:, 1:4].astype(np.float32)
        self.acc_body = data[:, 4:7].astype(np.float32)
        self.quat_data = data[:, col["quat"][0]:col["quat"][1]].astype(np.float32)
        self.pos_data  = data[:, col["pos"][0] :col["pos"][1] ].astype(np.float32)

        # label: Oxford는 시퀀스 전체가 동일한 정수값 → 첫 행에서 읽음
        if with_label and col["label"] is not None:
            self.label = int(data[0, col["label"]])
        else:
            self.label = -1  # label 미사용 시 sentinel

        # 3. 정규화 통계
        self.normalize = normalize
        if normalize:
            if precomputed_stats is not None:
                self.mean, self.std = precomputed_stats
            else:
                self.mean, self.std = self._compute_local_stats()
        else:
            self.mean, self.std = None, None

        # 4. 슬라이딩 윈도우 인덱스
        T = len(self.acc_body)
        self.indices = list(range(0, T - window_len, stride))

    def __len__(self) -> int:
        return len(self.indices)
    def _window_to_gravity_aligned(self, start: int, end: int) -> np.ndarray:
        """
        [start:end] 구간의 raw body IMU를
        '윈도우 시작점 yaw 기준 local gravity-aligned frame'으로 변환.
        반환 shape: [window_len, 6]  (acc_xyz + gyro_xyz)
        """
        acc_body = self.acc_body[start:end]   # [L, 3]
        gyr_body = self.gyr_body[start:end]   # [L, 3]

        # 각 시점 body -> world 회전
        R_all = R.from_quat(self.quat_data[start:end]).as_matrix().astype(np.float32)  # [L, 3, 3]

        # 윈도우 시작점 yaw만 추출
        r_start = R.from_quat(self.quat_data[start])
        yaw0 = r_start.as_euler('zyx', degrees=False)[0]
        R_yaw_inv = R.from_euler('z', yaw0).inv().as_matrix().astype(np.float32)  # [3, 3]

        # body -> world
        acc_world = np.einsum('tij,tj->ti', R_all, acc_body)  # [L, 3]
        gyr_world = np.einsum('tij,tj->ti', R_all, gyr_body)  # [L, 3]

        # world -> local gravity-aligned (시작 yaw 제거)
        acc_ga = (R_yaw_inv @ acc_world.T).T  # [L, 3]
        gyr_ga = (R_yaw_inv @ gyr_world.T).T  # [L, 3]

        imu_np = np.concatenate([acc_ga, gyr_ga], axis=1).astype(np.float32)  # [L, 6]
        return imu_np

    def _compute_local_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        단일 파일 fallback용 통계 계산.
        gravity-aligned window 입력 기준으로 mean/std 계산한다.
        """
        ch_sum = np.zeros(6, dtype=np.float64)
        ch_sumsq = np.zeros(6, dtype=np.float64)
        count = 0

        for start in range(0, len(self.acc_body) - self.window_len, self.stride):
            end = start + self.window_len
            imu_win = self._window_to_gravity_aligned(start, end)  # [L, 6]

            ch_sum += imu_win.sum(axis=0)
            ch_sumsq += (imu_win ** 2).sum(axis=0)
            count += imu_win.shape[0]

        mean = ch_sum / max(count, 1)
        var = ch_sumsq / max(count, 1) - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

        return mean.astype(np.float32), std + 1e-8
    

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        end = start + self.window_len

        # [1] Input: window별 local gravity-aligned IMU [window_len, 6]
        imu_np = self._window_to_gravity_aligned(start, end)

        # [2] Target: Gravity-Aligned Delta Position
        pos_start = self.pos_data[start]
        pos_end = self.pos_data[end]

        # World 기준 변위
        delta_p_world = pos_end - pos_start

        # 윈도우 시작점 yaw만 제거한 local gravity-aligned target
        r_start = R.from_quat(self.quat_data[start])
        yaw0 = r_start.as_euler('zyx', degrees=False)[0]
        R_yaw_inv = R.from_euler('z', yaw0).inv().as_matrix().astype(np.float32)

        target_np = (R_yaw_inv @ delta_p_world).astype(np.float32)

        # [3] Augmentation (Yaw Rotation) - train일 때만
        # local gravity-aligned frame에서 수평 회전 augmentation
        if self.is_train:
            theta = float(np.random.uniform(-np.pi, np.pi))
            c, s_ = np.cos(theta), np.sin(theta)
            R2 = np.array([[c, -s_], [s_, c]], dtype=np.float32)

            # acc x,y 회전
            imu_np[:, 0:2] = imu_np[:, 0:2] @ R2.T
            # gyro x,y 회전
            imu_np[:, 3:5] = imu_np[:, 3:5] @ R2.T
            # target x,y 회전
            target_np[:2] = R2 @ target_np[:2]

        # [4] Normalize
        if self.normalize:
            imu_np = (imu_np - self.mean) / self.std

        # [5] 모델 입력 형태 [6, window_len]
        imu_np = imu_np.T.copy()

        imu_t    = torch.from_numpy(imu_np)
        target_t = torch.from_numpy(target_np.astype(np.float32))

        if self.with_label:
            return imu_t, target_t, torch.tensor(self.label, dtype=torch.long)
        return imu_t, target_t


# ---------------------------------------------------------------------------
# 2. 다중 폴더/파일 병합 Dataset (ConcatDataset)
# ---------------------------------------------------------------------------
class TLIOMultiDataset(Dataset):
    """
    여러 개의 .npy 파일(여러 폴더)을 하나의 Dataset으로 합칩니다.
    fmt='tlio' | 'oxford'  포맷 선택
    with_label=True 시 (imu, displacement, label) 3-tuple 반환
    """
    def __init__(
        self,
        base_dir: Union[str, Path],
        window_len: int = 200,
        stride: int = 10,
        normalize: bool = True,
        precomputed_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        is_train: bool = True,
        fmt: str = "tlio",
        with_label: bool = False,
    ):
        base_dir = Path(base_dir)
        self.fmt = fmt
        self.with_label = with_label

        npy_paths = sorted(base_dir.rglob("*.npy"))
        if not npy_paths:
            raise ValueError(f"[{base_dir}] 경로에서 .npy 파일을 찾을 수 없습니다.")

        # 정규화 통계 계산 (train의 경우 전체 데이터 기준)
        if normalize and precomputed_stats is None:
            precomputed_stats = self._compute_global_stats(
                npy_paths=npy_paths,
                window_len=window_len,
                stride=stride,
                fmt=fmt,
            )
        # 개별 파일별 Dataset 생성
        sub_datasets = [
            TLIONpySingleDataset(
                npy_path=p,
                window_len=window_len,
                stride=stride,
                normalize=normalize,
                precomputed_stats=precomputed_stats,
                is_train=is_train,
                fmt=fmt,
                with_label=with_label,
            )
            for p in npy_paths
        ]

        self._dataset = ConcatDataset(sub_datasets)
        self._stats = precomputed_stats
        
        print(f"[{base_dir.name}] 로드된 파일 수: {len(sub_datasets)} | 총 샘플 수: {len(self._dataset)}")

    def _compute_global_stats(
        self,
        npy_paths: List[Path],
        window_len: int,
        stride: int,
        fmt: str = "tlio",
    ) -> Tuple[np.ndarray, np.ndarray]:

        col = _COL[fmt]
        ch_sum = np.zeros(6, dtype=np.float64)
        ch_sumsq = np.zeros(6, dtype=np.float64)
        count = 0

        print("전체 데이터 정규화 통계 계산 중...")

        for p in npy_paths:
            data = np.load(p)

            gyr_body  = data[:, 1:4].astype(np.float32)
            acc_body  = data[:, 4:7].astype(np.float32)
            quat_data = data[:, col["quat"][0]:col["quat"][1]].astype(np.float32)

            T = len(acc_body)

            for start in range(0, T - window_len, stride):
                end = start + window_len

                acc_win = acc_body[start:end]  # [L, 3]
                gyr_win = gyr_body[start:end]  # [L, 3]

                # 각 시점 body -> world
                R_all = R.from_quat(quat_data[start:end]).as_matrix().astype(np.float32)  # [L,3,3]

                # 시작점 yaw 제거
                r_start = R.from_quat(quat_data[start])
                yaw0 = r_start.as_euler('zyx', degrees=False)[0]
                R_yaw_inv = R.from_euler('z', yaw0).inv().as_matrix().astype(np.float32)

                acc_world = np.einsum('tij,tj->ti', R_all, acc_win)
                gyr_world = np.einsum('tij,tj->ti', R_all, gyr_win)

                acc_ga = (R_yaw_inv @ acc_world.T).T
                gyr_ga = (R_yaw_inv @ gyr_world.T).T

                imu_win = np.concatenate([acc_ga, gyr_ga], axis=1).astype(np.float32)  # [L, 6]

                ch_sum += imu_win.sum(axis=0)
                ch_sumsq += (imu_win ** 2).sum(axis=0)
                count += imu_win.shape[0]

        mean = ch_sum / max(count, 1)
        var = ch_sumsq / max(count, 1) - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

        return mean.astype(np.float32), std + 1e-8

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
    fmt: str = "tlio",
    with_label: bool = False,
    **kwargs
) -> dict:
    """
    train.py에서 호출하는 최종 DataLoader 생성 함수.
    fmt='oxford' + with_label=True 로 joint training 데이터 로드 가능.
    """
    ds_kwargs = dict(window_len=window_len, normalize=True, fmt=fmt, with_label=with_label)

    # 1. Train 데이터셋 로드 (정규화 통계 자동 계산)
    train_ds = TLIOMultiDataset(base_dir=train_paths, stride=train_stride, is_train=True, **ds_kwargs)
    stats = train_ds.get_stats()

    # 2. Val 데이터셋 로드 (Train 통계로 정규화)
    val_ds = TLIOMultiDataset(
        base_dir=val_paths, stride=eval_stride, is_train=False,
        precomputed_stats=stats, **ds_kwargs
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "stats": stats,
    }

    # 3. Test 데이터셋 로드 (선택)
    if test_paths and Path(test_paths).exists():
        test_ds = TLIOMultiDataset(
            base_dir=test_paths, stride=eval_stride, is_train=False,
            precomputed_stats=stats, **ds_kwargs
        )
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)

    return loaders