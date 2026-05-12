"""
convert_model_to_torchscript.py
--------------------------------
학습된 TwoLayerModel 체크포인트(.pth)를 TorchScript(.pt)로 변환한다.

사용 예:
    python convert_model_to_torchscript.py \
        --model_path outputs/out_regression/checkpoints/best.pth \
        --config_path outputs/out_regression/config.json \
        --out_dir    outputs/out_regression

config.json 에 "model" 키가 있으면 그 설정을 사용하고,
없으면 --window_len / --patch_len 인자로 수동 지정한다.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from pprint import pprint

import torch
sys.path.insert(0, str(Path(__file__).parent))          # src/ 를 경로에 추가
from Network.model_twolayer import TwoLayerModel
from utils.logging import logging


# ---------------------------------------------------------------------------
# 기본 모델 파라미터 (config.json 이 없을 때 fallback)
# train.py DEFAULT_CFG["model"] 과 동일하게 유지할 것
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PARA = {
    "input_len":     100,
    "input_channel": 6,
    "patch_len":     10,
    "feature_dim":   128,
    "out_dim":       3,
    "active_func":   "GELU",
    "extractor": {"name": "ResMLP", "layer_num": 6, "expansion": 2, "dropout": 0.0},
    "reg":       {"name": "SimpleMean", "layer_num": 3, "dropout": 0.0},
    "use_classifier": False,
}


def load_and_convert(model_path: str, model_para: dict,
                     out_dir: str, cpu: bool = True):
    device = torch.device("cpu") if (cpu or not torch.cuda.is_available()) \
             else torch.device("cuda:0")

    # ── 모델 구성 및 가중치 로드 ──────────────────────────────────────────
    net = TwoLayerModel(model_para).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # 체크포인트 키 자동 탐지 (train.py 는 "model", 구버전은 "model_state_dict")
    state_key = "model" if "model" in ckpt else "model_state_dict"
    net.load_state_dict(ckpt[state_key])
    net.eval()

    # ── TorchScript 변환 (trace) ──────────────────────────────────────────
    window_len = model_para["input_len"]
    in_ch      = model_para["input_channel"]
    dummy      = torch.zeros(1, in_ch, window_len, device=device)

    with torch.no_grad():
        traced = torch.jit.trace(net, dummy)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_torchscript.pt")
    traced.save(out_path)
    logging.info(f"TorchScript 저장 완료: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TwoLayerModel → TorchScript 변환")

    parser.add_argument("--model_path",  required=True,  help="체크포인트 (.pth) 경로")
    parser.add_argument("--config_path", default=None,   help="config.json 경로 (model 키 포함)")
    parser.add_argument("--out_dir",     default="./",   help="출력 폴더")
    parser.add_argument("--cpu",         action="store_true", default=True)

    # config.json 이 없을 때 수동 지정
    parser.add_argument("--window_len",  type=int,   default=100)
    parser.add_argument("--patch_len",   type=int,   default=10)
    parser.add_argument("--feature_dim", type=int,   default=128)
    parser.add_argument("--layer_num",   type=int,   default=6)

    args = parser.parse_args()

    # ── 모델 파라미터 결정 ────────────────────────────────────────────────
    if args.config_path is not None and Path(args.config_path).exists():
        with open(args.config_path) as f:
            cfg = json.load(f)
        model_para = cfg.get("model", DEFAULT_MODEL_PARA)
        logging.info(f"config.json 에서 모델 설정 로드: {args.config_path}")
    else:
        model_para = {**DEFAULT_MODEL_PARA,
                      "input_len":   args.window_len,
                      "patch_len":   args.patch_len,
                      "feature_dim": args.feature_dim,
                      "extractor":   {**DEFAULT_MODEL_PARA["extractor"],
                                      "layer_num": args.layer_num}}
        logging.info("config.json 없음 → 인자 기반 모델 설정 사용")

    logging.info("모델 파라미터:")
    pprint(model_para)

    load_and_convert(args.model_path, model_para, args.out_dir, args.cpu)
