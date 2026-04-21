import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from classification_dataset import CLASS_NAMES, infer_label_from_path


def find_sequence_dirs(dataset_root: Path) -> List[Path]:
    """
    전체 dataset 폴더 아래에서 '.npy를 포함하는 최상위 시퀀스 폴더'를 찾는다.
    예:
      dataset_root/oxford_handbag_1/imu0_resampled.npy
    -> 시퀀스 폴더는 oxford_handbag_1
    """
    seq_dirs = set()
    for npy_path in dataset_root.rglob("*.npy"):
        rel = npy_path.relative_to(dataset_root)
        seq_dir = dataset_root / rel.parts[0]
        seq_dirs.add(seq_dir)
    return sorted(seq_dirs)


def allocate_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if n < 3:
        raise ValueError(f"클래스별 시퀀스 수가 최소 3개는 필요합니다. 현재 n={n}")

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, n - n_train - n_val)

    # 합이 맞지 않으면 조정
    while n_train + n_val + n_test > n:
        # train을 우선 줄이고, 그다음 val을 줄인다. 단 각 split 최소 1 유지.
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    while n_train + n_val + n_test < n:
        n_train += 1

    return n_train, n_val, n_test



def split_by_class(seq_dirs: List[Path], seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    by_class: Dict[str, List[Path]] = defaultdict(list)
    for seq_dir in seq_dirs:
        label = infer_label_from_path(seq_dir)
        by_class[label].append(seq_dir)

    rng = random.Random(seed)
    split_map = {"train": [], "val": [], "test": []}
    summary = {}

    for cls in CLASS_NAMES:
        items = sorted(by_class.get(cls, []))
        if not items:
            continue
        rng.shuffle(items)
        n_train, n_val, n_test = allocate_counts(len(items), train_ratio, val_ratio, test_ratio)

        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:n_train + n_val + n_test]

        split_map["train"].extend(train_items)
        split_map["val"].extend(val_items)
        split_map["test"].extend(test_items)

        summary[cls] = {
            "total": len(items),
            "train": [p.name for p in train_items],
            "val": [p.name for p in val_items],
            "test": [p.name for p in test_items],
        }

    return split_map, summary



def copy_or_link(src: Path, dst: Path, mode: str):
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
    elif mode == "copy":
        shutil.copytree(src, dst)
    else:
        raise ValueError(mode)



def build_output_structure(dataset_root: Path, output_root: Path, split_map: Dict[str, List[Path]], mode: str):
    output_root.mkdir(parents=True, exist_ok=True)
    for split_name, seq_dirs in split_map.items():
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for seq_dir in seq_dirs:
            dst = split_dir / seq_dir.name
            copy_or_link(seq_dir, dst, mode)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True, help="모든 oxford_* 시퀀스 폴더가 모여 있는 루트")
    ap.add_argument("--output_root", type=str, required=True, help="train/val/test를 만들 출력 루트")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--mode", type=str, default="copy", choices=["copy", "symlink"])
    args = ap.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test 비율의 합은 1.0이어야 합니다.")

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    seq_dirs = find_sequence_dirs(dataset_root)
    if not seq_dirs:
        raise FileNotFoundError(f"{dataset_root} 아래에서 .npy를 포함한 시퀀스 폴더를 찾지 못했습니다.")

    split_map, summary = split_by_class(
        seq_dirs=seq_dirs,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    build_output_structure(dataset_root, output_root, split_map, args.mode)

    report = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "summary": summary,
    }
    with open(output_root / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("완료")
    for split_name in ["train", "val", "test"]:
        print(f"[{split_name}] {len(split_map[split_name])} sequences")


if __name__ == "__main__":
    main()
