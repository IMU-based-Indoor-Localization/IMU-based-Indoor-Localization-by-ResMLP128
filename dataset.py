"""
dataset.py
----------
OXIOD 포맷 IMU 데이터셋 로더

CSV 컬럼 구조:
    Time, placement_label,
    user_acc_x, user_acc_y, user_acc_z,          ← IMU acc  (3ch)
    rotation_rate_x, rotation_rate_y, rotation_rate_z,  ← IMU gyro (3ch)
    gravity_x, gravity_y, gravity_z,              ← (선택)
    attitude_roll, attitude_pitch, attitude_yaw,  ← (선택)
    target_delta_x, target_delta_y, target_delta_z,     ← 회귀 타깃 (3)
    target_delta_roll, target_delta_pitch, target_delta_yaw

OXIOD placement_label 매핑:
    -1 → 0  (non-phone / noise correction only)
     1 → 1  handbag
     2 → 2  handheld
     3 → 3  pocket
     4 → 4  running
     5 → 5  slow walking
     6 → 6  trolley
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

# 사용할 IMU 채널 (6-axis: acc + gyro)
IMU_COLS = [
    "user_acc_x(m/s^2)", "user_acc_y(m/s^2)", "user_acc_z(m/s^2)",
    "rotation_rate_x(rad/s)", "rotation_rate_y(rad/s)", "rotation_rate_z(rad/s)",
]

# 추가 센서 채널 (선택 사용)
GRAVITY_COLS = ["gravity_x(m/s^2)", "gravity_y(m/s^2)", "gravity_z(m/s^2)"]
ATTITUDE_COLS = ["attitude_roll(rad)", "attitude_pitch(rad)", "attitude_yaw(rad)"]

# 회귀 타깃
TARGET_COLS = ["target_delta_x", "target_delta_y", "target_delta_z"]

LABEL_COL = "placement_label"

# placement_label → class index
LABEL_MAP = {-1: 0, 0: 6, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = ["noise", "handbag", "handheld", "pocket",
               "running", "slow_walking", "trolley"]
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# 단일 CSV → Dataset
# ---------------------------------------------------------------------------

class OXIODSingleFileDataset(Dataset):
    """
    하나의 CSV 파일에서 슬라이딩 윈도우 샘플을 생성합니다.

    Args:
        csv_path      : CSV 파일 경로
        window_len    : 윈도우 크기 (timestep 수), 기본 100
        stride        : 윈도우 이동 간격, 기본 10
        input_cols    : 사용할 입력 컬럼 리스트 (기본: 6-axis IMU)
        target_cols   : 회귀 타깃 컬럼 (기본: delta x/y/z)
        normalize     : True이면 입력을 z-score 정규화 (mean/std는 파일 단위)
        precomputed_stats : (mean, std) 튜플. normalize=True이고 외부 통계를 
                            적용할 때 사용 (e.g. train set 통계를 val/test에 적용)

    Returns (per item):
        imu    : FloatTensor [input_channel, window_len]
        target : FloatTensor [3]  — cumulative delta position
        label  : LongTensor  []   — class index (0~6)
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        window_len: int = 100,
        stride: int = 10,
        input_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        normalize: bool = True,
        precomputed_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.csv_path   = Path(csv_path)
        self.window_len = window_len
        self.stride     = stride
        self.input_cols = input_cols if input_cols is not None else IMU_COLS
        self.target_cols = target_cols if target_cols is not None else TARGET_COLS

        df = pd.read_csv(self.csv_path)
        self._validate_columns(df)

        # ---- 레이블 변환 ----
        raw_labels = df[LABEL_COL].astype(int).values
        self.labels = np.array([LABEL_MAP.get(l, 0) for l in raw_labels], dtype=np.int64)

        # ---- 입력 / 타깃 numpy 배열 ----
        self.imu_data    = df[self.input_cols].values.astype(np.float32)   # [T, C]
        self.target_data = df[self.target_cols].values.astype(np.float32)  # [T, 3]

        # ---- 정규화 ----
        if normalize:
            if precomputed_stats is not None:
                self.mean, self.std = precomputed_stats
            else:
                self.mean = self.imu_data.mean(axis=0)
                self.std  = self.imu_data.std(axis=0) + 1e-8
            self.imu_data = (self.imu_data - self.mean) / self.std
        else:
            self.mean, self.std = None, None

        # ---- 슬라이딩 윈도우 인덱스 생성 ----
        T = len(self.imu_data)
        self.indices = list(range(0, T - window_len + 1, stride))

    def _validate_columns(self, df: pd.DataFrame):
        missing = [c for c in self.input_cols + self.target_cols + [LABEL_COL]
                   if c not in df.columns]
        if missing:
            raise ValueError(f"[{self.csv_path.name}] 누락된 컬럼: {missing}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        end   = start + self.window_len

        # imu: [C, window_len]  ← 모델 입력 형식에 맞게 transpose
        imu = torch.from_numpy(self.imu_data[start:end].T.copy())

        # target: 윈도우 내 delta의 합 → 윈도우 전체 변위
        target = torch.from_numpy(
            self.target_data[start:end].sum(axis=0).copy()
        )

        # label: 윈도우 중간 시점의 레이블 사용 (다수결도 가능하지만 단순화)
        label = torch.tensor(self.labels[start + self.window_len // 2],
                             dtype=torch.long)

        return imu, target, label

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """정규화에 사용된 (mean, std) 반환. train set 통계를 val/test에 전달할 때 사용."""
        if self.mean is None:
            raise RuntimeError("normalize=False로 생성된 Dataset입니다.")
        return self.mean, self.std


# ---------------------------------------------------------------------------
# 다중 파일 → Dataset
# ---------------------------------------------------------------------------

class OXIODDataset(Dataset):
    """
    여러 CSV 파일을 하나의 Dataset으로 합칩니다.

    사용 예:
        train_ds = OXIODDataset(
            csv_paths=["data/running_1.csv", "data/handheld_1.csv", ...],
            window_len=100,
            stride=10,
            normalize=True,
        )
        val_ds = OXIODDataset(
            csv_paths=["data/running_2.csv"],
            window_len=100,
            stride=50,
            normalize=True,
            precomputed_stats=train_ds.get_stats(),  # train 통계 재사용
        )

    Args:
        csv_paths         : CSV 파일 경로 리스트 또는 디렉터리 경로 (str/Path)
        window_len        : 윈도우 크기
        stride            : 윈도우 이동 간격
        input_cols        : 입력 컬럼 목록
        target_cols       : 타깃 컬럼 목록
        normalize         : 정규화 여부
        precomputed_stats : 외부 (mean, std). None이면 전체 데이터로 계산
        file_glob         : csv_paths가 디렉터리일 때 파일 패턴 (기본 "*.csv")
    """

    def __init__(
        self,
        csv_paths: Union[List[Union[str, Path]], str, Path],
        window_len: int = 100,
        stride: int = 10,
        input_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        normalize: bool = True,
        precomputed_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        file_glob: str = "*.csv",
    ):
        self.input_cols  = input_cols  or IMU_COLS
        self.target_cols = target_cols or TARGET_COLS

        # ---- 파일 목록 정리 ----
        if isinstance(csv_paths, (str, Path)) and Path(csv_paths).is_dir():
            csv_paths = sorted(Path(csv_paths).glob(file_glob))
        else:
            if isinstance(csv_paths, (str, Path)):
                csv_paths = [csv_paths]
            csv_paths = [Path(p) for p in csv_paths]

        if not csv_paths:
            raise ValueError("CSV 파일을 찾을 수 없습니다.")

        # ---- 정규화 통계 결정 ----
        if normalize and precomputed_stats is None:
            precomputed_stats = self._compute_global_stats(csv_paths)

        # ---- 개별 Dataset 생성 후 합치기 ----
        sub_datasets = [
            OXIODSingleFileDataset(
                csv_path          = p,
                window_len        = window_len,
                stride            = stride,
                input_cols        = self.input_cols,
                target_cols       = self.target_cols,
                normalize         = normalize,
                precomputed_stats = precomputed_stats,
            )
            for p in csv_paths
        ]

        self._dataset = ConcatDataset(sub_datasets)
        self._stats   = precomputed_stats
        self.num_files = len(sub_datasets)
        self.file_lengths = [len(ds) for ds in sub_datasets]

        print(f"[OXIODDataset] 로드된 파일 수: {self.num_files}")
        print(f"[OXIODDataset] 총 샘플 수   : {len(self._dataset)}")

    def _compute_global_stats(
        self, csv_paths: List[Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """전체 데이터의 컬럼별 mean/std 계산."""
        all_data = []
        for p in csv_paths:
            df = pd.read_csv(p)
            all_data.append(df[self.input_cols].values.astype(np.float32))
        concat = np.concatenate(all_data, axis=0)
        mean = concat.mean(axis=0)
        std  = concat.std(axis=0) + 1e-8
        return mean, std

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """정규화 통계 반환 (val/test set 생성 시 전달)."""
        if self._stats is None:
            raise RuntimeError("normalize=False로 생성된 Dataset입니다.")
        return self._stats

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._dataset[idx]


# ---------------------------------------------------------------------------
# 파일 단위 분리 헬퍼
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DataLoader 팩토리
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_paths:  List[Union[str, Path]],
    val_paths:    List[Union[str, Path]],
    test_paths:   Optional[List[Union[str, Path]]] = None,
    window_len:   int  = 100,
    train_stride: int  = 10,
    eval_stride:  int  = 50,
    batch_size:   int  = 256,
    num_workers:  int  = 4,
    normalize:    bool = True,
    input_cols:   Optional[List[str]] = None,
) -> dict:
    """
    이미 분리된 파일 경로 리스트를 받아 DataLoader를 생성합니다.
    train 통계가 val/test 정규화에 자동 적용됩니다.

    Args:
        train_paths  : train CSV 경로 리스트 또는 디렉터리
        val_paths    : val   CSV 경로 리스트 또는 디렉터리
        test_paths   : test  CSV 경로 리스트 또는 디렉터리 (없으면 생략)

    Returns:
        {
            "train" : DataLoader,
            "val"   : DataLoader,
            "test"  : DataLoader  (test_paths가 None이면 미포함),
            "stats" : (mean, std) ← train 정규화 통계 (추론 시 재사용)
        }
    """
    # train Dataset → 정규화 통계 계산
    train_ds = OXIODDataset(
        csv_paths  = train_paths,
        window_len = window_len,
        stride     = train_stride,
        normalize  = normalize,
        input_cols = input_cols,
    )
    stats = train_ds.get_stats() if normalize else None

    # val/test는 train 통계로 정규화 (필수)
    val_ds = OXIODDataset(
        csv_paths         = val_paths,
        window_len        = window_len,
        stride            = eval_stride,
        normalize         = normalize,
        precomputed_stats = stats,
        input_cols        = input_cols,
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            drop_last=True),
        "val"  : DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "stats": stats,
    }

    if test_paths:
        test_ds = OXIODDataset(
            csv_paths         = test_paths,
            window_len        = window_len,
            stride            = eval_stride,
            normalize         = normalize,
            precomputed_stats = stats,
            input_cols        = input_cols,
        )
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)

    return loaders


# ---------------------------------------------------------------------------
# 빠른 검증
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    csv_file = sys.argv[1] if len(sys.argv) > 1 else "running_1.csv"

    ds = OXIODSingleFileDataset(csv_file, window_len=100, stride=10)
    print(f"샘플 수: {len(ds)}")

    imu, target, label = ds[0]
    print(f"imu   : {imu.shape}   dtype={imu.dtype}")     # [6, 100]
    print(f"target: {target.shape} dtype={target.dtype}")  # [3]
    print(f"label : {label.item()} ({CLASS_NAMES[label.item()]})")

    loader = DataLoader(ds, batch_size=32, shuffle=True)
    batch_imu, batch_target, batch_label = next(iter(loader))
    print(f"\nBatch imu   : {batch_imu.shape}")     # [32, 6, 100]
    print(f"Batch target: {batch_target.shape}")    # [32, 3]
    print(f"Batch label : {batch_label.shape}")     # [32]
