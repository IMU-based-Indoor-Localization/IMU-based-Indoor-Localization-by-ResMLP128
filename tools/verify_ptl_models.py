"""
verify_ptl_models.py
====================
mobile_assets/ 의 PyTorch Mobile (Lite Interpreter) .ptl 모델들을 로드해서:
  - 출력 형태 (tuple len, 각 tensor shape)
  - forward 결정론성 (같은 입력에 같은 출력 → BatchNorm eval 정상)
  - 출력 값 범위 (NaN / Inf 없는지)
  - 두 가지 입력 (zero, random) 에서 출력 변화 (학습된 모델인지 sanity check)

확인.
P44 SIGBUS 우려로 Android 배포 전 사전 검증.

실행:
    cd D:\\mobile\\imu_android
    python tools\\verify_ptl_models.py

요구사항:
  - torch installed (pip install torch — Lite Interpreter 함수는 torch 1.7+)
"""
import os
import sys
import traceback

import torch


PATHS = [
    "mobile_assets/imu_model.ptl",
    "mobile_assets/imu_model_cls.ptl",
    "mobile_assets/imu_model_resmlp128.ptl",
    "mobile_assets/imu_model_resnet_working.ptl",
    "mobile_assets/imu_model_resnet_p41_backup.ptl",
    "android/app/src/main/assets/imu_model.ptl",
]


def desc(out):
    """tuple/list/tensor 출력을 사람이 읽기 좋은 문자열로."""
    if isinstance(out, (tuple, list)):
        return f"tuple/list len={len(out)} " + " ".join(
            f"[{i}]={tuple(o.shape)}({o.dtype})" if hasattr(o, "shape") else f"[{i}]={type(o).__name__}"
            for i, o in enumerate(out)
        )
    if hasattr(out, "shape"):
        return f"tensor {tuple(out.shape)} ({out.dtype})"
    return f"unknown {type(out).__name__}"


def check_finite(out):
    """NaN / Inf 검사."""
    tensors = out if isinstance(out, (tuple, list)) else [out]
    bad = []
    for i, t in enumerate(tensors):
        if not hasattr(t, "isnan"):
            continue
        if torch.isnan(t).any():
            bad.append(f"[{i}] NaN")
        if torch.isinf(t).any():
            bad.append(f"[{i}] Inf")
    return bad


def out_diff(a, b):
    if isinstance(a, (tuple, list)):
        return [(a[i] - b[i]).abs().max().item() for i in range(len(a))]
    return [(a - b).abs().max().item()]


def value_summary(out):
    """tuple/list 안 각 tensor 의 min/mean/max 요약."""
    tensors = out if isinstance(out, (tuple, list)) else [out]
    parts = []
    for i, t in enumerate(tensors):
        if not hasattr(t, "min"):
            parts.append(f"[{i}]?")
            continue
        tf = t.float()
        parts.append(
            f"[{i}] min={tf.min().item():+.4f} mean={tf.mean().item():+.4f} max={tf.max().item():+.4f}"
        )
    return " | ".join(parts)


def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"cwd: {os.getcwd()}")
    print()

    # Determ test: torch.manual_seed 고정 후 random window 1개 + zero window 1개
    torch.manual_seed(42)
    x_rand = torch.randn(1, 6, 100)
    x_zero = torch.zeros(1, 6, 100)

    fail = 0
    for path in PATHS:
        print(f"=== {path} ===")
        if not os.path.exists(path):
            print("  파일 없음 — 스킵")
            print()
            continue
        size_kb = os.path.getsize(path) / 1024
        print(f"  size: {size_kb:.1f} KB")
        try:
            m = torch._C._load_for_lite_interpreter(path)
        except Exception:
            print("  로드 실패:")
            traceback.print_exc(limit=3)
            fail += 1
            print()
            continue

        # forward 1
        try:
            out1 = m(x_rand)
        except Exception:
            print("  forward(random) 실패:")
            traceback.print_exc(limit=3)
            fail += 1
            print()
            continue
        print(f"  forward(random): {desc(out1)}")
        print(f"    {value_summary(out1)}")
        bad = check_finite(out1)
        if bad:
            print(f"    !!! NaN/Inf: {bad}")

        # determinism
        try:
            out2 = m(x_rand)
            diffs = out_diff(out1, out2)
            print(f"  same input twice — max diff per output: {diffs}")
            if any(d > 1e-5 for d in diffs):
                print("    !!! non-deterministic (BatchNorm not in eval mode?)")
        except Exception:
            print("  forward 2 실패:")
            traceback.print_exc(limit=3)

        # zero input
        try:
            out_z = m(x_zero)
            print(f"  forward(zeros): {value_summary(out_z)}")
        except Exception:
            print("  forward(zeros) 실패:")
            traceback.print_exc(limit=3)

        print()

    print(f"=== 종합: 실패 {fail}/{len([p for p in PATHS if os.path.exists(p)])} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
