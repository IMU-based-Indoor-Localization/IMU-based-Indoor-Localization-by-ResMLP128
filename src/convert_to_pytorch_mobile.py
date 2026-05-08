"""
convert_to_pytorch_mobile.py
-----------------------------
TorchScript(.pt) → PyTorch Mobile Lite(.ptl) 변환.

사용 예:
    python src/convert_to_pytorch_mobile.py \
        --pt_path   mobile_assets/model_torchscript.pt \
        --out_path  mobile_assets/imu_model.ptl
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile


def convert(pt_path: str, out_path: str, cpu: bool = True):
    device = torch.device("cpu") if cpu or not torch.cuda.is_available() \
             else torch.device("cuda:0")

    print(f"[1/3] TorchScript 로드: {pt_path}")
    model = torch.jit.load(pt_path, map_location=device)
    model.eval()

    print("[2/3] Mobile 최적화 적용 ...")
    optimized = optimize_for_mobile(model)

    print(f"[3/3] .ptl 저장: {out_path}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    optimized._save_for_lite_interpreter(out_path)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"✅ 완료: {out_path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorchScript → PyTorch Mobile Lite 변환")
    parser.add_argument("--pt_path",  required=True, help="입력 .pt 파일 경로")
    parser.add_argument("--out_path", required=True, help="출력 .ptl 파일 경로")
    parser.add_argument("--cpu", action="store_true", default=True)
    args = parser.parse_args()
    convert(args.pt_path, args.out_path, args.cpu)
