"""
prepare_assets.py
=================
Android assets 폴더에 필요한 파일들을 준비한다.

실행 방법 (프로젝트 루트에서):
    # 기본: src/outputs/out_classifier2 의 통계 사용
    python android/prepare_assets.py

    # 다른 모델 폴더 지정
    python android/prepare_assets.py --model_dir src/outputs/out_xxx

수행 작업:
  1. norm_mean.npy / norm_std.npy → assets/norm_mean.txt, norm_std.txt
  2. mobile_assets/ 안의 imu_model.ptl, model_meta.json → assets/ 복사

[P13] 정규화 통계 일치 보장:
  과거에는 NORM_MEAN/NORM_STD 가 out_tlio_6ch_128 (분류기 없는 회귀 전용,
  std≈[5.76, 6.07, 2.97]) 로 하드코딩되어 있었으나,
  실제 빌드되는 모델은 out_classifier2 (분류기 포함 통합, std≈[0.12, 0.12, 0.15]).
  통계 불일치로 어플리케이션에 들어가는 정규화된 입력이 학습 분포의 약 1/48 →
  분류기가 거의 모든 입력을 "running" 으로 편향 → R_scale=50× 적용 →
  EKF 측정 무시 → 회전 시 발산. 본 수정으로 out_classifier2 통계로 교체.

※ Eigen은 CMakeLists.txt의 FetchContent가 빌드 시 자동으로 받으므로
   별도 설치 불필요.
"""

import argparse
import os
import shutil

# ── 경로 설정 ──────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(ROOT, "android", "app", "src", "main", "assets")
MOBILE_DIR = os.path.join(ROOT, "mobile_assets")

# [P13] 기본 모델 폴더: out_classifier2 (분류기 포함 input_len=100 통합 모델)
DEFAULT_MODEL_DIR = os.path.join(ROOT, "src", "outputs", "out_classifier2")

parser = argparse.ArgumentParser(description="Android assets 준비")
parser.add_argument(
    "--model_dir",
    default=DEFAULT_MODEL_DIR,
    help="norm_mean.npy / norm_std.npy 가 있는 학습 출력 폴더 "
         "(기본: src/outputs/out_classifier2)",
)
parser.add_argument(
    "--norm_mean",
    default=None,
    help="norm_mean.npy 경로를 직접 지정 (--model_dir 보다 우선)",
)
parser.add_argument(
    "--norm_std",
    default=None,
    help="norm_std.npy 경로를 직접 지정 (--model_dir 보다 우선)",
)
args = parser.parse_args()

NORM_MEAN = args.norm_mean or os.path.join(args.model_dir, "norm_mean.npy")
NORM_STD  = args.norm_std  or os.path.join(args.model_dir, "norm_std.npy")

os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Step 1: norm 파라미터 변환 ─────────────────────────────────
print("[1/2] norm 파라미터 변환 중 ...")
print(f"   source mean: {NORM_MEAN}")
print(f"   source std : {NORM_STD}")
try:
    import numpy as np
    mean = np.load(NORM_MEAN).flatten()
    std  = np.load(NORM_STD).flatten()

    with open(os.path.join(ASSETS_DIR, "norm_mean.txt"), "w") as f:
        f.write(", ".join(f"{v:.8f}" for v in mean))
    with open(os.path.join(ASSETS_DIR, "norm_std.txt"), "w") as f:
        f.write(", ".join(f"{v:.8f}" for v in std))

    print(f"   mean shape={mean.shape}: {mean[:3].tolist()} ...")
    print(f"   std  shape={std.shape}:  {std[:3].tolist()} ...")
    print("   norm_mean.txt / norm_std.txt 저장 완료")
except Exception as e:
    print(f"   실패: {e}")

# ── Step 2: 모델 파일 복사 ─────────────────────────────────────
print("\n[2/2] 모델 파일 복사 중 ...")
FILES_TO_COPY = ["imu_model.ptl", "model_meta.json"]
all_ok = True
for fname in FILES_TO_COPY:
    src = os.path.join(MOBILE_DIR, fname)
    dst = os.path.join(ASSETS_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        size_kb = os.path.getsize(dst) / 1024
        print(f"   복사: {fname}  ({size_kb:.1f} KB)")
    else:
        print(f"   ⚠️  파일 없음: {src}")
        print(f"       convert_to_pytorch_mobile.py 를 먼저 실행하세요.")
        all_ok = False

# ── 최종 요약 ──────────────────────────────────────────────────
print("\n── assets 폴더 최종 내용 ──")
for f in sorted(os.listdir(ASSETS_DIR)):
    kb = os.path.getsize(os.path.join(ASSETS_DIR, f)) / 1024
    print(f"   {f:<30} {kb:>8.1f} KB")

if all_ok:
    print("\n✅ 준비 완료! Android Studio 에서 android/ 폴더를 열어 빌드하세요.")
else:
    print("\n⚠️  일부 파일이 없습니다. convert_and_prepare.bat 을 실행하세요.")
