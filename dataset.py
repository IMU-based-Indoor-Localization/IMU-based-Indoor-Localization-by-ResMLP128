import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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

# 사용할 IMU 채널 (14-axis: acc + gyro + gravity + attitude[sin/cos 변환])
# [결정 1] gravity 3채널은 "중력 방향 = 절대 수직" 기준을 제공해
#          pitch/roll 모호성을 해결한다.
# [결정 2] attitude (iOS CMAttitude, world-frame 기준)를 추가해 yaw 모호성도 해결.
#          handbag/pocket 등 폰이 회전하는 placement에서 결정적.
# [결정 3] roll/yaw는 ±π wraparound가 있어 raw로 넣으면 학습 저해.
#          → sin/cos로 분해해 연속적 feature로 변환.
#          pitch는 π/2 근처에 고정되어 wrap 없음 → raw 사용.
# LLIO 논문 원본: 6-axis (world-frame acc + body-frame gyro)
# 세션 간 일반화가 가장 좋음 — attitude 절대값 의존 없음
IMU_COLS_6 = [
    "user_acc_x(m/s^2)", "user_acc_y(m/s^2)", "user_acc_z(m/s^2)",
    "rotation_rate_x(rad/s)", "rotation_rate_y(rad/s)", "rotation_rate_z(rad/s)",
]

# 확장 14채널 (gravity + attitude sin/cos 추가)
IMU_COLS_14 = [
    "user_acc_x(m/s^2)", "user_acc_y(m/s^2)", "user_acc_z(m/s^2)",
    "rotation_rate_x(rad/s)", "rotation_rate_y(rad/s)", "rotation_rate_z(rad/s)",
    "gravity_x(m/s^2)", "gravity_y(m/s^2)", "gravity_z(m/s^2)",
    "att_roll_sin", "att_roll_cos",
    "attitude_pitch(rad)",
    "att_yaw_sin",  "att_yaw_cos",
]

# 기본값: 14채널 (gravity + attitude 포함 → orientation 정보 제공)
IMU_COLS = IMU_COLS_14

# 추가 센서 채널 (선택 사용 — input_cols로 직접 넘길 때 참조용)
GRAVITY_COLS  = ["gravity_x(m/s^2)", "gravity_y(m/s^2)", "gravity_z(m/s^2)"]
ATTITUDE_COLS = ["attitude_roll(rad)", "attitude_pitch(rad)", "attitude_yaw(rad)"]

# attitude → sin/cos 파생 feature 이름
ATTITUDE_DERIVED_COLS = ["att_roll_sin", "att_roll_cos", "att_yaw_sin", "att_yaw_cos"]

# 회귀 타깃
TARGET_COLS = ["target_delta_x", "target_delta_y", "target_delta_z"]

LABEL_COL = "placement_label"

# placement_label → class index
# OXIOD raw label 체계:
#   -1: noise(학습 제외용), 0: trolley, 1: handbag, 2: handheld,
#   3: pocket, 4: running, 5: slow walking
# class index(학습용)는 0~6으로 재배치: trolley를 마지막(6)에 둠.
# 예상 밖의 값이 들어오면 LABEL_MAP.get(l, 0) 에 의해 noise(0)로 처리.
LABEL_MAP = {-1: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
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
        augment_yaw: bool = False,
        augment_noise: bool = False,
        noise_std: float = 0.05,    # 정규화 후 기준 std. 너무 크면 학습 불안정.
        augment_scale: bool = False,
        scale_range: Tuple[float, float] = (0.8, 1.25),
    ):
        self.csv_path   = Path(csv_path)
        self.window_len = window_len
        self.stride     = stride
        self.input_cols = input_cols if input_cols is not None else IMU_COLS
        self.target_cols = target_cols if target_cols is not None else TARGET_COLS
        self.augment_yaw   = augment_yaw
        self.augment_noise = augment_noise
        self.noise_std     = noise_std
        self.augment_scale = augment_scale
        self.scale_range   = scale_range

        df = pd.read_csv(self.csv_path)

        # [추가] attitude → sin/cos 파생 feature 생성
        # wraparound(±π)가 있는 roll/yaw만 변환. pitch는 π/2 근처 고정이라 생략.
        # [중요] yaw는 session-relative로 정규화한다:
        #   iOS CMAttitude(xArbitraryZVertical)의 X축은 "세션 시작 시점의
        #   임의 수평 방향"이라 파일마다 yaw=0 기준이 다르다. 원본 그대로
        #   넣으면 모델이 학습한 "yaw 값 → world 방향" 매핑이 다른 세션에
        #   일반화되지 않아 체계적 편향(궤적이 통째로 회전)을 일으킨다.
        #   → 각 파일 첫 샘플의 yaw를 빼서 "세션 시작 = 0" 기준으로 통일.
        if "attitude_roll(rad)" in df.columns:
            df["att_roll_sin"] = np.sin(df["attitude_roll(rad)"].values)
            df["att_roll_cos"] = np.cos(df["attitude_roll(rad)"].values)
        if "attitude_yaw(rad)" in df.columns:
            yaw_abs = df["attitude_yaw(rad)"].values
            yaw_rel = yaw_abs - yaw_abs[0]
            # [-π, π] 범위로 wrap
            yaw_rel = np.arctan2(np.sin(yaw_rel), np.cos(yaw_rel))
            df["att_yaw_sin"]  = np.sin(yaw_rel)
            df["att_yaw_cos"]  = np.cos(yaw_rel)

        self._validate_columns(df)

        # ---- 레이블 변환 ----
        raw_labels = df[LABEL_COL].astype(int).values
        self.labels = np.array([LABEL_MAP.get(l, 0) for l in raw_labels], dtype=np.int64)

        # ---- 입력 / 타깃 numpy 배열 ----
        self.imu_data    = df[self.input_cols].values.astype(np.float32)   # [T, C]
        self.target_data = df[self.target_cols].values.astype(np.float32)  # [T, 3]

        # ---- 이상치 클리핑 ----
        # 100Hz 기준 1프레임=0.01초, 사람 최대 속도 ~10m/s → 0.1m/frame
        # 실제 데이터에 -46.5m 같은 이상치가 존재하여 MSE를 왜곡함
        DELTA_CLIP = 0.1
        self.target_data = np.clip(self.target_data, -DELTA_CLIP, DELTA_CLIP)

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

        # imu: [C, window_len]
        imu_np    = self.imu_data[start:end].T.copy()           # [C, W]
        target_np = self.target_data[start:end].sum(axis=0).copy()  # [3]

        # [추가] random horizontal (yaw) rotation augmentation
        # 채널 순서(IMU_COLS):
        #   0-2: user_acc x,y,z       (world frame → X,Y 회전)
        #   3-5: rotation_rate x,y,z  (body frame  → 그대로)
        #   6-8: gravity x,y,z        (world frame → X,Y 회전)
        #   9-10: att_roll_sin, cos   (roll은 yaw 무관 → 그대로)
        #   11:   attitude_pitch      (pitch는 yaw 무관 → 그대로)
        #   12-13: att_yaw_sin, cos   (yaw 자체 → theta 만큼 회전)
        # 6채널(legacy) 및 14채널(현재) 모두 지원.
        if self.augment_yaw:
            theta = float(np.random.uniform(-np.pi, np.pi))
            c, s_ = np.cos(theta), np.sin(theta)
            R2 = np.array([[c, -s_], [s_, c]], dtype=np.float32)  # 2-D 회전

            n_ch = imu_np.shape[0]
            if n_ch == 6:
                # 6채널: user_acc(ch0-2)는 world frame → 타겟과 함께 회전 필수
                # rotation_rate(ch3-5)는 body frame → 그대로 (비선형 변환 필요해 생략)
                imu_np[0:2] = R2 @ imu_np[0:2]
            elif n_ch >= 14:
                # 14(+2)채널 yaw augmentation (세계 수평면 기준 회전):
                #   user_acc x,y (world-frame) → 회전
                imu_np[0:2] = R2 @ imu_np[0:2]
                #   rotation_rate (body frame) → 그대로
                #   gravity (body frame, iOS CMMotionManager) → 그대로
                #   att_roll_sin/cos (ch9,10), attitude_pitch (ch11) → 그대로 (yaw 무관)
                #   att_yaw_sin/cos (ch12,13) → 회전
                imu_np[12:14] = R2 @ imu_np[12:14]
                #   pc1_x, pc1_y (ch14,15) → world-frame 벡터이므로 함께 회전
                # (PC1 채널이 있는 경우 여기서 회전할 수 있으나, 현재는 14ch 기준)

            # target X,Y 회전, Z 그대로
            target_np[:2] = R2 @ target_np[:2]
            target_np = target_np.astype(np.float32)

        # 속도(보행 속도) 스케일 augmentation
        # acc/gyro 신호와 타깃 변위를 동일 배율로 스케일링해 걷는 속도 차이를 모사.
        # gravity/attitude 채널은 orientation 정보이므로 변경하지 않음.
        if self.augment_scale:
            scale = float(np.random.uniform(self.scale_range[0], self.scale_range[1]))
            imu_np[0:6] = imu_np[0:6] * scale   # acc(0-2) + gyro(3-5)
            target_np   = target_np   * scale

        # IMU 센서 노이즈 augmentation
        # 정규화된 imu_np에 Gaussian noise를 추가해 센서 개체차/환경 변화를 모사.
        # att sin/cos 채널(9,10,12,13)은 [-1,1] 범위라 noise 후 재정규화.
        if self.augment_noise:
            noise = np.random.randn(*imu_np.shape).astype(np.float32) * self.noise_std
            imu_np = imu_np + noise
            # sin/cos 채널 재정규화 (14채널일 때만, 채널 9-10, 12-13이 존재할 때)
            if imu_np.shape[0] >= 14:
                for c0, c1 in [(9, 10), (12, 13)]:
                    norm = np.sqrt(imu_np[c0]**2 + imu_np[c1]**2 + 1e-8)
                    imu_np[c0] /= norm
                    imu_np[c1] /= norm

        imu    = torch.from_numpy(imu_np)
        target = torch.from_numpy(target_np)

        # label: 윈도우 중간 시점
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
        augment_yaw: bool = False,
        augment_noise: bool = False,
        noise_std: float = 0.05,
        augment_scale: bool = False,
        scale_range: Tuple[float, float] = (0.8, 1.25),
    ):
        self.input_cols    = input_cols  or IMU_COLS
        self.target_cols   = target_cols or TARGET_COLS
        self.augment_yaw   = augment_yaw
        self.augment_noise = augment_noise
        self.noise_std     = noise_std
        self.augment_scale = augment_scale
        self.scale_range   = scale_range

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
                augment_yaw       = self.augment_yaw,
                augment_noise     = self.augment_noise,
                noise_std         = self.noise_std,
                augment_scale     = self.augment_scale,
                scale_range       = self.scale_range,
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
            # [추가] attitude 파생 feature (IMU_COLS 와 동일 로직)
            # yaw는 session-relative 로 정규화 (OXIODSingleFileDataset과 동일)
            if "attitude_roll(rad)" in df.columns:
                df["att_roll_sin"] = np.sin(df["attitude_roll(rad)"].values)
                df["att_roll_cos"] = np.cos(df["attitude_roll(rad)"].values)
            if "attitude_yaw(rad)" in df.columns:
                yaw_abs = df["attitude_yaw(rad)"].values
                yaw_rel = yaw_abs - yaw_abs[0]
                yaw_rel = np.arctan2(np.sin(yaw_rel), np.cos(yaw_rel))
                df["att_yaw_sin"]  = np.sin(yaw_rel)
                df["att_yaw_cos"]  = np.cos(yaw_rel)
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
    train_paths:   List[Union[str, Path]],
    val_paths:     List[Union[str, Path]],
    test_paths:    Optional[List[Union[str, Path]]] = None,
    window_len:    int  = 100,
    train_stride:  int  = 10,
    eval_stride:   int  = 50,
    batch_size:    int  = 256,
    num_workers:   int  = 4,
    normalize:     bool = True,
    input_cols:    Optional[List[str]] = None,
    augment_noise: bool  = False,
    noise_std:     float = 0.02,
    augment_scale: bool  = False,
    scale_range:   Tuple[float, float] = (0.8, 1.25),
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
        csv_paths     = train_paths,
        window_len    = window_len,
        stride        = train_stride,
        normalize     = normalize,
        input_cols    = input_cols,
        augment_yaw   = True,
        augment_noise = augment_noise,
        noise_std     = noise_std,
        augment_scale = augment_scale,
        scale_range   = scale_range,
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
