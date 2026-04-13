from collections import defaultdict
import pandas as pd

def split_files_by_session(
    data_dir:   Union[str, Path],
    val_ratio:  float = 0.15,
    test_ratio: float = 0.15,
    seed:       int   = 42,
    file_glob:  str   = "*.csv",
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    클래스(자세)별 불균형과 파일별 길이(샘플 수) 차이를 모두 고려하여
    Train / Val / Test 세트로 안전하게 분리합니다.
    """
    all_files = sorted(Path(data_dir).glob(file_glob))
    if not all_files:
        raise ValueError(f"CSV 파일을 찾을 수 없습니다: {data_dir}/{file_glob}")

    rng = np.random.default_rng(seed)
    
    # ---------------------------------------------------------
    # 1. 파일별 정보 수집 (클래스 추론 및 데이터 길이 계산)
    # ---------------------------------------------------------
    class_groups = defaultdict(list)
    
    for f in all_files:
        # 파일 길이(샘플 수) 계산 (pandas로 전체를 메모리에 올리지 않고 줄 수만 카운트)
        with open(f, 'r', encoding='utf-8') as file:
            num_lines = sum(1 for _ in file) - 1  # 헤더 제외
            
        # 파일명에서 클래스 추론 (예: 'running_1.csv' -> 'running')
        # 매칭되는 클래스가 없으면 'unknown'으로 분류
        c_name = "unknown"
        for cls in CLASS_NAMES:
            if cls in f.name.lower():
                c_name = cls
                break
                
        class_groups[c_name].append({'path': f, 'length': num_lines})

    train_files, val_files, test_files = [], [], []

    # ---------------------------------------------------------
    # 2. 클래스별로 비율과 길이를 고려하여 분배 (Stratified & Greedy)
    # ---------------------------------------------------------
    for c_name, items in class_groups.items():
        # 특정 클래스 내에서 편향이 없도록 무작위 섞기
        rng.shuffle(items)
        
        # 해당 클래스의 전체 샘플 수 계산
        total_len = sum(item['length'] for item in items)
        target_val_len = total_len * val_ratio
        target_test_len = total_len * test_ratio
        
        cur_val_len, cur_test_len = 0, 0
        
        # 탐욕(Greedy) 알고리즘: 목표 길이에 도달할 때까지 파일을 할당
        for item in items:
            f_path = item['path']
            f_len = item['length']
            
            if cur_test_len < target_test_len:
                test_files.append(f_path)
                cur_test_len += f_len
            elif cur_val_len < target_val_len:
                val_files.append(f_path)
                cur_val_len += f_len
            else:
                train_files.append(f_path)
                
    # ---------------------------------------------------------
    # 3. 예외 처리: 데이터가 너무 적어 세트가 비어버리는 경우 방지
    # ---------------------------------------------------------
    if len(train_files) == 0:
        raise ValueError("Train 세트에 할당된 파일이 없습니다. 데이터가 너무 적습니다.")
    
    # 평가 세트가 비어있다면, 훈련 세트에서 파일 1개씩 강제 차출
    if len(val_files) == 0 and len(train_files) > 1:
        val_files.append(train_files.pop())
    if len(test_files) == 0 and len(train_files) > 1 and test_ratio > 0:
        test_files.append(train_files.pop())

    print(f"[Stratified Split] 완료")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val  : {len(val_files)} files")
    print(f"  Test : {len(test_files)} files")
    
    return train_files, val_files, test_files