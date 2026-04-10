import os
import shutil
from pathlib import Path
import numpy as np
from collections import defaultdict

# 설정
RAW_DATA_DIR = "dataset"
OUT_DATA_DIR = "data"      # 분할된 데이터가 저장될 폴더
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

CLASS_NAMES = ["noise", "handbag", "handheld", "pocket", "running", "slow_walking", "trolley"]

def create_split_folders():
    all_files = sorted(Path(RAW_DATA_DIR).glob("*.csv"))
    if not all_files:
        print(f"파일을 찾을 수 없습니다: {RAW_DATA_DIR}")
        return

    rng = np.random.default_rng(SEED)
    class_groups = defaultdict(list)
    
    # 1. 파일 정보 수집
    for f in all_files:
        with open(f, 'r', encoding='utf-8') as file:
            num_lines = sum(1 for _ in file) - 1
            
        c_name = "unknown"
        for cls in CLASS_NAMES:
            if cls in f.name.lower():
                c_name = cls
                break
        class_groups[c_name].append({'path': f, 'length': num_lines})

    train_files, val_files, test_files = [], [], []

    # 2. 클래스별 분할 알고리즘
    for c_name, items in class_groups.items():
        rng.shuffle(items)
        total_len = sum(item['length'] for item in items)
        target_val_len = total_len * VAL_RATIO
        target_test_len = total_len * TEST_RATIO
        
        cur_val_len, cur_test_len = 0, 0
        
        for item in items:
            if cur_test_len < target_test_len:
                test_files.append(item['path'])
                cur_test_len += item['length']
            elif cur_val_len < target_val_len:
                val_files.append(item['path'])
                cur_val_len += item['length']
            else:
                train_files.append(item['path'])

    # 3. 파일 물리적 복사
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    for split_name, files in splits.items():
        split_dir = Path(OUT_DATA_DIR) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for f in files:
            dest = split_dir / f.name
            shutil.copy(f, dest)
            
        print(f"[{split_name.upper()}] {len(files)}개 파일 복사 완료 -> {split_dir}")

if __name__ == "__main__":
    create_split_folders()