"""
export_resmlp128_cls.py
-----------------------
out_classifier2/best.pth 를 TwoLayerModel(use_classifier=True) 로 인스턴스화
후 PyTorch Mobile Lite (.ptl) 로 익스포트.

핵심:
  - 모델: ResMLP128 백본 + PoseClassifier(7-way) + PoseConditionedPoolingReg
  - 입력 [1, 6, 100], 출력 (disp[3], cov[3], cls_logits[7]) → 3-tuple
  - optimize_for_mobile() *제외* (P41 진단: prepacked conv2d 가 XNNPACK
    호환성 문제로 SIGBUS BUS_ADRALN 유발)

실행:
    # anaconda 환경 활성 후
    python src/export_resmlp128_cls.py
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "Network"))
from model_twolayer import TwoLayerModel


# ── 경로 ────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).parent.parent
CKPT_DIR  = PROJ_ROOT / "src" / "Network" / "out_classifier2"
CKPT      = CKPT_DIR / "checkpoints" / "best.pth"
CONFIG    = CKPT_DIR / "config.json"
NORM_M    = CKPT_DIR / "norm_mean.npy"
NORM_S    = CKPT_DIR / "norm_std.npy"

OUT_PTL   = PROJ_ROOT / "mobile_assets" / "imu_model.ptl"
ASSETS    = PROJ_ROOT / "android" / "app" / "src" / "main" / "assets"
ASSETS_PTL  = ASSETS / "imu_model.ptl"
ASSETS_MEAN = ASSETS / "norm_mean.txt"
ASSETS_STD  = ASSETS / "norm_std.txt"


def main():
    # ── 1. config 로드 ────────────────────────────────────────
    print(f"[1/5] config 로드: {CONFIG}")
    if not CONFIG.exists():
        raise FileNotFoundError(
            f"{CONFIG} 없음.\n"
            f"먼저 git show origin/EKF:src/outputs/out_classifier2/{{config.json,checkpoints/best.pth,norm_mean.npy,norm_std.npy}} "
            f"로 파일들을 받아 {CKPT_DIR} 에 배치하세요."
        )
    with open(CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    model_para = cfg["model"]
    print(f"  use_classifier = {model_para.get('use_classifier')}")
    print(f"  input_len = {model_para.get('input_len')}, "
          f"feature_dim = {model_para.get('feature_dim')}")

    # ── 2. 모델 생성 + 가중치 로드 ─────────────────────────────
    print(f"[2/5] TwoLayerModel 인스턴스 + checkpoint 로드: {CKPT}")
    net = TwoLayerModel(model_para)
    ckpt = torch.load(CKPT, map_location="cpu")

    # checkpoint 가 dict({'model_state_dict': ...}) 형식인지 직접 state_dict 인지 구분
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt  # 직접 state_dict

    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠ missing keys ({len(missing)} 개) — 첫 5개: {missing[:5]}")
    if unexpected:
        print(f"  ⚠ unexpected keys ({len(unexpected)} 개) — 첫 5개: {unexpected[:5]}")
    net.eval()

    # ── 3. trace (optimize_for_mobile 제외) ───────────────────
    print("[3/5] torch.jit.trace 실행 (입력 [1,6,100])")
    example = torch.zeros(1, 6, model_para["input_len"])
    with torch.no_grad():
        out = net(example)
    print(f"  forward 출력 tuple 길이 = {len(out)}")
    for i, t in enumerate(out):
        if t is None:
            print(f"    [{i}] None")
        else:
            print(f"    [{i}] shape={tuple(t.shape)} dtype={t.dtype}")

    traced = torch.jit.trace(net, example, strict=False)
    traced = torch.jit.freeze(traced)

    # ── 4. .ptl 저장 (Mobile Lite) ────────────────────────────
    print(f"[4/5] _save_for_lite_interpreter → {OUT_PTL}")
    OUT_PTL.parent.mkdir(parents=True, exist_ok=True)
    traced._save_for_lite_interpreter(str(OUT_PTL))

    # assets 에도 동일 복사
    ASSETS.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(OUT_PTL, ASSETS_PTL)
    size_mb = os.path.getsize(OUT_PTL) / 1024 / 1024
    print(f"  ✓ {OUT_PTL.name} ({size_mb:.2f} MB)")
    print(f"  ✓ {ASSETS_PTL}")

    # ── 5. norm params → assets/norm_*.txt 변환 ───────────────
    print("[5/5] norm params 변환 (npy → txt)")
    mean = np.load(NORM_M)
    std  = np.load(NORM_S)
    with open(ASSETS_MEAN, "w", encoding="utf-8") as f:
        f.write(",".join(f"{x:.10f}" for x in mean) + "\n")
    with open(ASSETS_STD, "w", encoding="utf-8") as f:
        f.write(",".join(f"{x:.10f}" for x in std) + "\n")
    print(f"  ✓ {ASSETS_MEAN}  mean={mean.tolist()}")
    print(f"  ✓ {ASSETS_STD}   std ={std.tolist()}")

    print("\n=== 완료 ===")
    print(f"다음 단계: Android Studio Build → Run → Replay 측정 → cls 분포 logcat 확인")


if __name__ == "__main__":
    main()
