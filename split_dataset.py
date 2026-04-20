import os
import shutil
from pathlib import Path

# --- 설정 경로 ---
# 원본 데이터셋이 있는 폴더
source_dir = Path("TLIO_dataset")

# 분할된 폴더들이 저장될 새 최상위 폴더 (학습 코드 config와 매칭)
output_dir = Path("dataset_split") 

# 리스트 파일명 매핑
list_files = {
    "train": "train_list.txt",
    "val": "val_list.txt",
    "test": "test_list.txt"
}

def split_folders():
    for split_name, txt_file in list_files.items():
        # 1. 이동할 목적지 폴더 생성 (예: dataset_split/train)
        target_dir = output_dir / split_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. txt 파일 읽어서 ID 리스트 추출 (공백 및 줄바꿈 제거)
        if not os.path.exists(txt_file):
            print(f"[오류] {txt_file} 파일을 찾을 수 없습니다.")
            continue
            
        with open(txt_file, 'r', encoding='utf-8') as f:
            folder_ids = [line.strip() for line in f.readlines() if line.strip()]
            
        print(f"\n[{split_name.upper()}] {len(folder_ids)}개의 폴더를 이동합니다...")
        
        # 3. 폴더 이동 수행
        success_count = 0
        for folder_id in folder_ids:
            src_path = source_dir / folder_id
            dst_path = target_dir / folder_id
            
            if src_path.exists() and src_path.is_dir():
                # 폴더 전체를 잘라내기(이동) 합니다.
                # ※ 원본을 보존하고 복사만 하려면 아래 줄을 shutil.copytree(src_path, dst_path) 로 변경하세요.
                shutil.move(str(src_path), str(dst_path))
                success_count += 1
            else:
                print(f"  -> [경고] {src_path} 폴더가 존재하지 않습니다.")
                
        print(f"  => {success_count} / {len(folder_ids)} 완료")

if __name__ == "__main__":
    print("데이터셋 분할을 시작합니다...")
    split_folders()
    print("\n모든 작업이 완료되었습니다!")