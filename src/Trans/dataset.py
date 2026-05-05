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

# TLIO 데이터에서 포즈 레이블 없음을 나타내는 sentinel
_LABEL_NO_CLASS = -100


# ---------------------------------------------------------------------------
# Augmentation 헬퍼 (TLIO-master/src/dataloader/data_transform.py 기반)
# ---------------------------------------------------------------------------

def _rvec_to_rot_np(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues 회전 벡터 → 3x3 회전 행렬 (numpy 구현)."""
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = (rvec / angle).astype(np.float32)
    K = np.array([
        [0.0,       -axis[2],  axis[1]],
        [ axis[2],   0.0,     -axis[0]],
        [-axis[1],   axis[0],  0.0   ],
    ], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _aug_bias_noise(
    imu_np: np.ndarray,
    accel_bias_range: float = 0.2,
    gyro_bias_range: float  = 0.05,
    accel_noise_std: float  = 0.0,
    gyro_noise_std: float   = 0.0,
) -> np.ndarray:
    """
    IMU에 랜덤 상수 바이어스 + 가우시안 노이즈를 추가합니다.
    imu_np: [T, 6]  acc(3) + gyro(3)  (Oxford 컨벤션)
    TLIO TransformAddNoiseBias의 numpy 등가 구현.
    """
    imu = imu_np.copy()
    T   = imu.shape[0]
    accel_bias = (np.random.rand(3).astype(np.float32) - 0.5) * (accel_bias_range / 0.5)
    gyro_bias  = (np.random.rand(3).astype(np.float32) - 0.5) * (gyro_bias_range  / 0.5)
    imu[:, :3]  += accel_bias
    imu[:, 3:6] += gyro_bias
    if accel_noise_std > 0.0:
        imu[:, :3]  += np.random.randn(T, 3).astype(np.float32) * accel_noise_std
    if gyro_noise_std > 0.0:
        imu[:, 3:6] += np.random.randn(T, 3).astype(np.float32) * gyro_noise_std
    return imu


def _aug_gravity_perturb(
    imu_np: np.ndarray,
    theta_range_deg: float = 5.0,
) -> np.ndarray:
    """
    수평 축 중심으로 랜덤 틸트 회전을 적용합니다 (중력 방향 섭동).
    imu_np: [T, 6]  acc(3) + gyro(3)
    TLIO TransformPerturbGravity의 numpy 등가 구현.
    """
    imu = imu_np.copy()
    phi   = np.random.rand() * 2.0 * np.pi
    axis  = np.array([np.cos(phi), np.sin(phi), 0.0], dtype=np.float32)
    theta = np.random.rand() * np.pi * theta_range_deg / 180.0
    R_mat = _rvec_to_rot_np(axis * float(theta))
    imu[:, :3]  = (R_mat @ imu[:, :3].T).T
    imu[:, 3:6] = (R_mat @ imu[:, 3:6].T).T
    return imu


# ---------------------------------------------------------------------------
# 1. 단일 .npy 파일 처리 Dataset (Oxford / TLIO 포맷)
# ---------------------------------------------------------------------------
class TLIONpySingleDataset(Dataset):
    """
    하나의 TLIO .npy 파일에서 슬라이딩 윈도우 샘플을 생성합니다.
    fmt='tlio'   : TLIO 17col 포맷 (label 없음)
    fmt='oxford' : Oxford 24col 포맷 (label col 13 포함)
    with_label   : True 시 __getitem__ 반환값에 label(int) 추가
    is_train=True: 증강 적용 (bias_noise + gravity_perturb + yaw_rotation)
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
        self.is_train   = is_train
        self.npy_path   = Path(npy_path)
        self.window_len = window_len
        self.stride     = stride
        self.with_label = with_label
        self.fmt        = fmt

        col  = _COL[fmt]
        data = np.load(self.npy_path)

        self.gyr_body  = data[:, 1:4].astype(np.float32)
        self.acc_body  = data[:, 4:7].astype(np.float32)
        self.quat_data = data[:, col["quat"][0]:col["quat"][1]].astype(np.float32)
        self.pos_data  = data[:, col["pos"][0] :col["pos"][1] ].astype(np.float32)

        if with_label and col["label"] is not None:
            self.label = int(data[0, col["label"]])
        else:
            self.label = -1

        self.normalize = normalize
        if normalize:
            if precomputed_stats is not None:
                self.mean, self.std = precomputed_stats
            else:
                self.mean, self.std = self._compute_local_stats()
        else:
            self.mean, self.std = None, None

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
        acc_body = self.acc_body[start:end]
        gyr_body = self.gyr_body[start:end]

        R_all = R.from_quat(self.quat_data[start:end]).as_matrix().astype(np.float32)

        r_start   = R.from_quat(self.quat_data[start])
        yaw0      = r_start.as_euler('zyx', degrees=False)[0]
        R_yaw_inv = R.from_euler('z', yaw0).inv().as_matrix().astype(np.float32)

        acc_world = np.einsum('tij,tj->ti', R_all, acc_body)
        gyr_world = np.einsum('tij,tj->ti', R_all, gyr_body)

        acc_ga = (R_yaw_inv @ acc_world.T).T
        gyr_ga = (R_yaw_inv @ gyr_world.T).T

        return np.concatenate([acc_ga, gyr_ga], axis=1).astype(np.float32)

    def _compute_local_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        ch_sum   = np.zeros(6, dtype=np.float64)
        ch_sumsq = np.zeros(6, dtype=np.float64)
        count    = 0
        for start in range(0, len(self.acc_body) - self.window_len, self.stride):
            end     = start + self.window_len
            imu_win = self._window_to_gravity_aligned(start, end)
            ch_sum   += imu_win.sum(axis=0)
            ch_sumsq += (imu_win ** 2).sum(axis=0)
            count    += imu_win.shape[0]
        mean = ch_sum / max(count, 1)
        var  = ch_sumsq / max(count, 1) - mean ** 2
        std  = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
        return mean.astype(np.float32), std + 1e-8

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        end   = start + self.window_len

        # [1] Input: window별 local gravity-aligned IMU [window_len, 6]
        imu_np = self._window_to_gravity_aligned(start, end)

        # [2] Target: Gravity-Aligned Delta Position
        delta_p_world = self.pos_data[end] - self.pos_data[start]
        r_start   = R.from_quat(self.quat_data[start])
        yaw0      = r_start.as_euler('zyx', degrees=False)[0]
        R_yaw_inv = R.from_euler('z', yaw0).inv().as_matrix().astype(np.float32)
        target_np = (R_yaw_inv @ delta_p_world).astype(np.float32)

        # [3] Augmentation (학습 시에만)
        # TLIO-master 기반: bias_noise → gravity_perturb → yaw_rotation
        if self.is_train:
            # bias 추가 (TLIO TransformAddNoiseBias)
            imu_np = _aug_bias_noise(imu_np)
            # 중력 섭동 (TLIO TransformPerturbGravity)
            imu_np = _aug_gravity_perturb(imu_np)
            # Yaw 회전 (TLIO TransformInYawPlane) + target 동일 변환
            theta = float(np.random.uniform(-np.pi, np.pi))
            c, s_ = np.cos(theta), np.sin(theta)
            R2    = np.array([[c, -s_], [s_, c]], dtype=np.float32)
            imu_np[:, 0:2] = imu_np[:, 0:2] @ R2.T   # acc x,y
            imu_np[:, 3:5] = imu_np[:, 3:5] @ R2.T   # gyro x,y
            target_np[:2]  = R2 @ target_np[:2]

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
# 2. 다중 폴더/파일 병합 Dataset (Oxford / TLIO 포맷)
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
        base_dir       = Path(base_dir)
        self.fmt       = fmt
        self.with_label = with_label

        npy_paths = sorted(base_dir.rglob("*.npy"))
        if not npy_paths:
            raise ValueError(f"[{base_dir}] 경로에서 .npy 파일을 찾을 수 없습니다.")

        if normalize and precomputed_stats is None:
            precomputed_stats = self._compute_global_stats(
                npy_paths=npy_paths,
                window_len=window_len,
                stride=stride,
                fmt=fmt,
            )

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
        self._stats   = precomputed_stats

        print(f"[{base_dir.name}] 로드된 파일 수: {len(sub_datasets)} | 총 샘플 수: {len(self._dataset)}")

    def _compute_global_stats(
        self,
        npy_paths: List[Path],
        window_len: int,
        stride: int,
        fmt: str = "tlio",
    ) -> Tuple[np.ndarray, np.ndarray]:
        col      = _COL[fmt]
        ch_sum   = np.zeros(6, dtype=np.float64)
        ch_sumsq = np.zeros(6, dtype=np.float64)
        count    = 0

        print("전체 데이터 정규화 통계 계산 중...")

        for p in npy_paths:
            data      = np.load(p)
            gyr_body  = data[:, 1:4].astype(np.float32)
            acc_body  = data[:, 4:7].astype(np.float32)
            quat_data = data[:, col["quat"][0]:col["quat"][1]].astype(np.float32)
            T         = len(acc_body)

            for start in range(0, T - window_len, stride):
                end     = start + window_len
                acc_win = acc_body[start:end]
                gyr_win = gyr_body[start:end]

                R_all = R.from_quat(quat_data[start:end]).as_matrix().astype(np.float32)

                r_start   = R.from_quat(quat_data[start])
                yaw0      = r_start.as_euler('zyx', degrees=False)[0]
                R_yaw_inv = R.from_euler('z', yaw0).inv().as_matrix().astype(np.float32)

                acc_world = np.einsum('tij,tj->ti', R_all, acc_win)
                gyr_world = np.einsum('tij,tj->ti', R_all, gyr_win)
                acc_ga    = (R_yaw_inv @ acc_world.T).T
                gyr_ga    = (R_yaw_inv @ gyr_world.T).T
                imu_win   = np.concatenate([acc_ga, gyr_ga], axis=1).astype(np.float32)

                ch_sum   += imu_win.sum(axis=0)
                ch_sumsq += (imu_win ** 2).sum(axis=0)
                count    += imu_win.shape[0]

        mean = ch_sum / max(count, 1)
        var  = ch_sumsq / max(count, 1) - mean ** 2
        std  = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
        return mean.astype(np.float32), std + 1e-8

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._stats

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._dataset[idx]


# ---------------------------------------------------------------------------
# 3. TLIO Golden Dataset (200Hz → 100Hz 서브샘플, 포즈 레이블 없음)
# ---------------------------------------------------------------------------
class TLIOGoldenDataset(Dataset):
    """
    golden-new-format-cc-by-nc-with-imus 형식의 단일 시퀀스를 로드합니다.
    200Hz 원본 데이터를 100Hz로 서브샘플 (every 2nd sample).
    포즈 레이블 없음 → with_label=True 시 label=_LABEL_NO_CLASS(-100) 반환.

    TLIO 17-col 포맷:
      ts(1) | gyr_world(3) | acc_world(3) | qxyzw(4) | pos(3) | vel(3)
    gyr/acc는 이미 world frame으로 회전된 값 (body→world 회전 완료).
    """
    def __init__(
        self,
        npy_path: Union[str, Path],
        window_len: int = 100,
        stride: int = 10,
        normalize: bool = True,
        precomputed_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        is_train: bool = True,
        with_label: bool = False,
        aug_params: Optional[dict] = None,
    ):
        self.npy_path   = Path(npy_path)
        self.window_len = window_len
        self.stride     = stride
        self.is_train   = is_train
        self.with_label = with_label
        self.aug_params = aug_params or {}

        # 200Hz → 100Hz 서브샘플
        data = np.load(self.npy_path)[::2]

        # TLIO 17-col 파싱
        self.gyr_world = data[:, 1:4].astype(np.float32)   # gyr_compensated_rotated_in_World
        self.acc_world = data[:, 4:7].astype(np.float32)   # acc_compensated_rotated_in_World
        self.quat_data = data[:, 7:11].astype(np.float32)  # qxyzw_World_Device
        self.pos_data  = data[:, 11:14].astype(np.float32) # pos_World_Device

        T = len(self.acc_world)
        self.indices = list(range(0, T - window_len, stride))

        self.normalize = normalize
        if normalize:
            if precomputed_stats is not None:
                self.mean, self.std = precomputed_stats
            else:
                self.mean, self.std = self._compute_stats()
        else:
            self.mean, self.std = None, None

    def __len__(self) -> int:
        return len(self.indices)

    def _window_to_gravity_aligned(self, start: int, end: int) -> np.ndarray:
        """
        이미 world frame인 gyr/acc를 시작점 yaw 정규화 로컬 프레임으로 변환.
        반환 shape: [window_len, 6]  acc(3) + gyro(3)  (Oxford 컨벤션과 동일)
        """
        gyr_w     = self.gyr_world[start:end]
        acc_w     = self.acc_world[start:end]
        r_start   = R.from_quat(self.quat_data[start])
        yaw0      = r_start.as_euler('zyx', degrees=False)[0]
        R_yaw_inv = R.from_euler('z', yaw0).inv().as_matrix().astype(np.float32)
        acc_ga    = (R_yaw_inv @ acc_w.T).T
        gyr_ga    = (R_yaw_inv @ gyr_w.T).T
        return np.concatenate([acc_ga, gyr_ga], axis=1).astype(np.float32)

    def _compute_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        ch_sum   = np.zeros(6, dtype=np.float64)
        ch_sumsq = np.zeros(6, dtype=np.float64)
        count    = 0
        for start in range(0, len(self.acc_world) - self.window_len, self.stride):
            imu_win   = self._window_to_gravity_aligned(start, start + self.window_len)
            ch_sum   += imu_win.sum(0)
            ch_sumsq += (imu_win ** 2).sum(0)
            count    += imu_win.shape[0]
        mean = (ch_sum / max(count, 1)).astype(np.float32)
        var  = ch_sumsq / max(count, 1) - mean ** 2
        std  = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32) + 1e-8
        return mean, std

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        end   = start + self.window_len

        imu_np = self._window_to_gravity_aligned(start, end)

        delta_p_world = self.pos_data[end] - self.pos_data[start]
        r_start   = R.from_quat(self.quat_data[start])
        yaw0      = r_start.as_euler('zyx', degrees=False)[0]
        R_yaw_inv = R.from_euler('z', yaw0).inv().as_matrix().astype(np.float32)
        target_np = (R_yaw_inv @ delta_p_world).astype(np.float32)

        if self.is_train:
            # TLIO 기반 증강: bias_noise → gravity_perturb → yaw_rotation
            bias_p = self.aug_params.get("bias_noise", {})
            imu_np = _aug_bias_noise(imu_np, **bias_p)
            grav_p = self.aug_params.get("gravity_perturb", {})
            imu_np = _aug_gravity_perturb(imu_np, **grav_p)
            theta  = float(np.random.uniform(-np.pi, np.pi))
            c, s_  = np.cos(theta), np.sin(theta)
            R2     = np.array([[c, -s_], [s_, c]], dtype=np.float32)
            imu_np[:, 0:2] = imu_np[:, 0:2] @ R2.T
            imu_np[:, 3:5] = imu_np[:, 3:5] @ R2.T
            target_np[:2]  = R2 @ target_np[:2]

        if self.normalize:
            imu_np = (imu_np - self.mean) / self.std

        imu_np = imu_np.T.copy()

        imu_t    = torch.from_numpy(imu_np)
        target_t = torch.from_numpy(target_np)

        if self.with_label:
            return imu_t, target_t, torch.tensor(_LABEL_NO_CLASS, dtype=torch.long)
        return imu_t, target_t


# ---------------------------------------------------------------------------
# 4. TLIO Golden Multi-Sequence Dataset
# ---------------------------------------------------------------------------
class TLIOGoldenMultiDataset(Dataset):
    """
    golden-new-format-cc-by-nc-with-imus 폴더의 여러 시퀀스를 로드합니다.
    {base_dir}/{split}_list.txt 에서 시퀀스 ID를 읽어 사용합니다.
    포즈 레이블 없음 → with_label=True 시 label=-100 반환.
    """
    # TLIO-master 기본 증강 파라미터
    _AUG_DEFAULT = {
        "bias_noise":      {"accel_bias_range": 0.2, "gyro_bias_range": 0.05},
        "gravity_perturb": {"theta_range_deg": 5.0},
    }

    def __init__(
        self,
        base_dir: Union[str, Path],
        window_len: int = 100,
        stride: int = 10,
        normalize: bool = True,
        precomputed_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        is_train: bool = True,
        with_label: bool = False,
        aug_params: Optional[dict] = None,
        split: str = "train",
    ):
        base_dir   = Path(base_dir)
        self.split = split

        split_file = base_dir / f"{split}_list.txt"
        with open(split_file) as f:
            seq_ids = [s.strip() for s in f if s.strip()]

        npy_paths = [base_dir / sid / "imu0_resampled.npy" for sid in seq_ids]
        npy_paths = [p for p in npy_paths if p.exists()]

        if not npy_paths:
            raise ValueError(f"[TLIOGoldenMultiDataset] {split}_list.txt에 유효한 시퀀스 없음")

        # 정규화 통계: 외부 제공(Oxford stats) 우선, 없으면 자체 계산
        if normalize and precomputed_stats is None:
            precomputed_stats = self._compute_global_stats(npy_paths, window_len, stride)
        self._stats = precomputed_stats

        _aug = aug_params if aug_params is not None else self._AUG_DEFAULT

        sub_datasets = [
            TLIOGoldenDataset(
                npy_path=p,
                window_len=window_len,
                stride=stride,
                normalize=normalize,
                precomputed_stats=precomputed_stats,
                is_train=is_train,
                with_label=with_label,
                aug_params=_aug,
            )
            for p in npy_paths
        ]
        self._dataset = ConcatDataset(sub_datasets)
        print(f"[TLIO Golden/{split}] {len(npy_paths)}개 시퀀스 | 총 {len(self._dataset)}개 샘플")

    @staticmethod
    def _compute_global_stats(
        npy_paths: List[Path],
        window_len: int,
        stride: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """TLIO golden 데이터 전체에서 정규화 통계 계산."""
        ch_sum   = np.zeros(6, dtype=np.float64)
        ch_sumsq = np.zeros(6, dtype=np.float64)
        count    = 0
        print("TLIO Golden 데이터 정규화 통계 계산 중...")
        for p in npy_paths:
            data      = np.load(p)[::2]            # 200Hz → 100Hz
            gyr_world = data[:, 1:4].astype(np.float32)
            acc_world = data[:, 4:7].astype(np.float32)
            quat_data = data[:, 7:11].astype(np.float32)
            T         = len(acc_world)
            for start in range(0, T - window_len, stride):
                end       = start + window_len
                gyr_w     = gyr_world[start:end]
                acc_w     = acc_world[start:end]
                r_start   = R.from_quat(quat_data[start])
                yaw0      = r_start.as_euler('zyx', degrees=False)[0]
                R_yaw_inv = R.from_euler('z', yaw0).inv().as_matrix().astype(np.float32)
                acc_ga    = (R_yaw_inv @ acc_w.T).T
                gyr_ga    = (R_yaw_inv @ gyr_w.T).T
                imu_win   = np.concatenate([acc_ga, gyr_ga], axis=1)
                ch_sum   += imu_win.sum(0)
                ch_sumsq += (imu_win ** 2).sum(0)
                count    += imu_win.shape[0]
        mean = (ch_sum / max(count, 1)).astype(np.float32)
        var  = ch_sumsq / max(count, 1) - mean ** 2
        std  = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32) + 1e-8
        return mean, std

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._stats

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._dataset[idx]


# ---------------------------------------------------------------------------
# 5. DataLoader 빌더 (train.py에서 호출하는 함수)
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
    tlio_aug_dir: Optional[str] = None,   # TLIO golden 데이터 경로 (없으면 None)
    tlio_aug_stride: int = 10,            # TLIO 데이터 stride
    **kwargs,
) -> dict:
    """
    train.py에서 호출하는 최종 DataLoader 생성 함수.
    tlio_aug_dir 지정 시 TLIO golden 데이터를 Oxford 학습 데이터와 혼합.
    TLIO 데이터는 포즈 레이블 없음(label=-100); 분류 손실은 마스킹됨.
    """
    ds_kwargs = dict(window_len=window_len, normalize=True, fmt=fmt, with_label=with_label)

    # 1. Oxford 학습 데이터 (정규화 통계 자동 계산)
    train_ds = TLIOMultiDataset(base_dir=train_paths, stride=train_stride, is_train=True, **ds_kwargs)
    stats    = train_ds.get_stats()

    # 2. TLIO golden 증강 데이터 혼합 (선택)
    if tlio_aug_dir is not None and Path(tlio_aug_dir).exists():
        print(f"[TLIO Aug] golden 데이터 로드: {tlio_aug_dir}")
        tlio_train_ds = TLIOGoldenMultiDataset(
            base_dir=tlio_aug_dir,
            window_len=window_len,
            stride=tlio_aug_stride,
            normalize=True,
            precomputed_stats=stats,   # Oxford 통계로 정규화 (추론 일관성 유지)
            is_train=True,
            with_label=with_label,
            split="train",
        )
        combined_train_ds = ConcatDataset([train_ds, tlio_train_ds])
        print(
            f"[Combined] Oxford: {len(train_ds)} | TLIO: {len(tlio_train_ds)} "
            f"| Total: {len(combined_train_ds)}"
        )
    else:
        combined_train_ds = train_ds

    # 3. Val 데이터 (Oxford만, 학습 통계로 정규화)
    val_ds = TLIOMultiDataset(
        base_dir=val_paths, stride=eval_stride, is_train=False,
        precomputed_stats=stats, **ds_kwargs,
    )

    loaders = {
        "train": DataLoader(
            combined_train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "stats": stats,
    }

    # 4. Test 데이터셋 (선택)
    if test_paths and Path(test_paths).exists():
        test_ds = TLIOMultiDataset(
            base_dir=test_paths, stride=eval_stride, is_train=False,
            precomputed_stats=stats, **ds_kwargs,
        )
        loaders["test"] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    return loaders
