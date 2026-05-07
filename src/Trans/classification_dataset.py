import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from dataset import TLIONpySingleDataset, TLIOMultiDataset

# Oxford raw label → 0-indexed class id
# raw: -1=noise, 1=handbag, 2=handheld, 3=pocket, 4=running, 5=slow, 6=trolley
LABEL_REMAP: Dict[int, int] = {-1: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

# Class names ordered by class index 0-6
CLASS_NAMES: List[str] = ["handbag", "handheld", "pocket", "running", "slow", "trolley", "noise"]

# Folder keyword → class name
_FOLDER_TO_CLASS: Dict[str, str] = {
    "handbag":      "handbag",
    "handheld":     "handheld",
    "pocket":       "pocket",
    "running":      "running",
    "slow_walking": "slow",
    "slow":         "slow",
    "trolley":      "trolley",
}


def infer_label_from_path(seq_dir: Path) -> Optional[str]:
    """Returns class name from folder name, or None if not identifiable."""
    name = seq_dir.name.lower()
    for keyword, cls_name in _FOLDER_TO_CLASS.items():
        if keyword in name:
            return cls_name
    return None


class _ClsWrapper(Dataset):
    """
    TLIONpySingleDataset 위에 씌우는 래퍼.
    (imu, displacement) 대신 (imu, class_idx) 를 반환하며
    noise / scale 증강을 추가로 지원.
    """

    def __init__(
        self,
        single_ds: TLIONpySingleDataset,
        cls_idx: int,
        augment_noise: bool = False,
        noise_std: float = 0.01,
        augment_scale: bool = False,
        scale_range: Tuple[float, float] = (0.95, 1.05),
    ):
        self._ds = single_ds
        self._cls_idx = cls_idx
        self._augment_noise = augment_noise
        self._noise_std = noise_std
        self._augment_scale = augment_scale
        self._scale_min, self._scale_max = scale_range

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        imu, _target = self._ds[idx]   # imu: [6, window_len]
        if self._augment_noise:
            imu = imu + torch.randn_like(imu) * self._noise_std
        if self._augment_scale:
            scale = torch.empty(1).uniform_(self._scale_min, self._scale_max).item()
            imu = imu * scale
        label = torch.tensor(self._cls_idx, dtype=torch.long)
        return imu, label


def _build_split_dataset(
    dir_path: Path,
    window_len: int,
    stride: int,
    stats: Tuple[np.ndarray, np.ndarray],
    is_train: bool,
    augment_noise: bool,
    noise_std: float,
    augment_scale: bool,
    scale_range: Tuple[float, float],
) -> Tuple[Optional[ConcatDataset], Dict[str, int]]:
    """
    split 디렉토리 안의 시퀀스 폴더들을 순회하여
    ConcatDataset과 클래스별 윈도우 수를 반환.
    """
    class_counts: Dict[str, int] = {name: 0 for name in CLASS_NAMES}
    wrappers: List[_ClsWrapper] = []

    for seq_dir in sorted(dir_path.iterdir()):
        if not seq_dir.is_dir():
            continue
        cls_name = infer_label_from_path(seq_dir)
        if cls_name is None:
            continue
        cls_idx = CLASS_NAMES.index(cls_name)

        npy_paths = sorted(seq_dir.rglob("*.npy"))
        for npy_path in npy_paths:
            try:
                single_ds = TLIONpySingleDataset(
                    npy_path=npy_path,
                    window_len=window_len,
                    stride=stride,
                    normalize=True,
                    precomputed_stats=stats,
                    is_train=is_train,
                    fmt="oxford",
                    with_label=False,
                )
                if len(single_ds) == 0:
                    continue
                wrapper = _ClsWrapper(
                    single_ds=single_ds,
                    cls_idx=cls_idx,
                    augment_noise=augment_noise,
                    noise_std=noise_std,
                    augment_scale=augment_scale,
                    scale_range=scale_range,
                )
                class_counts[cls_name] += len(single_ds)
                wrappers.append(wrapper)
            except Exception as e:
                print(f"[경고] {npy_path} 로드 실패: {e}")
                continue

    if not wrappers:
        return None, class_counts
    return ConcatDataset(wrappers), class_counts


def build_classification_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    window_len: int = 200,
    train_stride: int = 100,
    eval_stride: int = 100,
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = True,
    augment_noise: bool = False,
    noise_std: float = 0.01,
    augment_scale: bool = False,
    scale_range: Tuple[float, float] = (0.95, 1.05),
) -> Dict:
    """
    train/val/test DataLoader를 생성하고 관련 메타데이터를 반환.

    반환 dict 키:
        train, val, [test]              DataLoader
        class_names                     List[str]
        stats                           (mean, std) np.ndarray 튜플
        train_window_class_counts       Dict[str, int]
        val_window_class_counts         Dict[str, int]
        [test_window_class_counts]      Dict[str, int]
    """
    # 1. Train 전체 통계 계산 (TLIOMultiDataset 활용)
    print(f"[classification_dataset] train 통계 계산 중: {train_dir}")
    stats_ds = TLIOMultiDataset(
        base_dir=train_dir,
        window_len=window_len,
        stride=train_stride,
        normalize=True,
        fmt="oxford",
        with_label=False,
        is_train=False,
    )
    stats: Tuple[np.ndarray, np.ndarray] = stats_ds.get_stats()

    # 2. 각 split 데이터셋 구성
    train_ds, train_counts = _build_split_dataset(
        Path(train_dir), window_len, train_stride, stats, True,
        augment_noise, noise_std, augment_scale, scale_range,
    )
    val_ds, val_counts = _build_split_dataset(
        Path(val_dir), window_len, eval_stride, stats, False,
        False, noise_std, False, scale_range,
    )

    assert train_ds is not None, f"train_dir에서 유효한 시퀀스를 찾을 수 없습니다: {train_dir}"
    assert val_ds is not None, f"val_dir에서 유효한 시퀀스를 찾을 수 없습니다: {val_dir}"

    loaders: Dict = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "class_names": CLASS_NAMES,
        "stats": stats,
        "train_window_class_counts": train_counts,
        "val_window_class_counts": val_counts,
    }

    if test_dir and Path(test_dir).exists():
        test_ds, test_counts = _build_split_dataset(
            Path(test_dir), window_len, eval_stride, stats, False,
            False, noise_std, False, scale_range,
        )
        if test_ds is not None:
            loaders["test"] = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            )
            loaders["test_window_class_counts"] = test_counts

    return loaders


def compute_class_weights_from_counts(counts: Dict[str, int]) -> torch.Tensor:
    """
    클래스별 윈도우 수로부터 역제곱근 가중치를 계산.
    CLASS_NAMES 순서(0-6)로 정렬된 Tensor 반환.

    주의: 샘플이 0개인 클래스(noise 등)는 weight=0 으로 처리하여 loss 에서 배제.
    (샘플이 없는 클래스에 큰 weight 를 주면 label_smoothing 과 결합 시 collapse 발생)
    """
    counts_arr = np.array(
        [counts.get(name, 0) for name in CLASS_NAMES], dtype=np.float32
    )
    # 샘플이 0인 클래스는 weight=0 (학습에서 제외), 나머지만 역제곱근 가중치
    has_samples = counts_arr > 0
    weights = np.zeros_like(counts_arr)
    weights[has_samples] = 1.0 / np.sqrt(counts_arr[has_samples])
    # 유효 클래스 수 기준으로 정규화
    n_valid = has_samples.sum()
    if weights.sum() > 0:
        weights = weights / weights.sum() * n_valid
    return torch.from_numpy(weights.astype(np.float32))
