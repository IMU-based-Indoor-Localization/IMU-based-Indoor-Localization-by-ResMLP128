"""
export_resmlp128.py
-------------------
out_regression/checkpoints/best.pth → mobile_assets/imu_model_resmlp128.ptl

실행:
    cd D:/mobile/imu_android
    python src/export_resmlp128.py

주의:
    optimize_for_mobile() 미사용 — P41 SIGBUS(at::infer_size_dimvector) 원인.
    _save_for_lite_interpreter() 만 사용.
"""

import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Network"))

import torch
import torch.nn as nn
from model_twolayer import TwoLayerModel

# ── 설정 ────────────────────────────────────────────────────────
CKPT_PATH = "src/Network/out_regression/checkpoints/best.pth"
OUT_PATH  = "mobile_assets/imu_model_resmlp128.ptl"
ASSET_PATH = "android/app/src/main/assets/imu_model.ptl"

MODEL_CFG = {
    "input_len":     100,
    "input_channel": 6,
    "patch_len":     10,
    "feature_dim":   128,
    "out_dim":       3,
    "active_func":   "GELU",
    "extractor": {"name": "ResMLP", "layer_num": 6, "expansion": 2, "dropout": 0.2},
    "reg":       {"name": "SimpleMean", "layer_num": 3, "dropout": 0.2},
    "use_classifier": False,
}


class _ExportWrapper(nn.Module):
    """forward() → (disp[3], dispCov[3])  2-tuple.
    TwoLayerModel 은 (y, y_cov, None) 3-tuple 을 반환하는데,
    PyTorch Mobile 은 None 을 튜플 원소로 지원하지 않아 파싱 오류 발생.
    InferenceEngine 이 기대하는 2-tuple 로 래핑.
    """
    def __init__(self, model: TwoLayerModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        y, y_cov, _ = self.model(x)
        return y, y_cov


def main():
    print(f"[1/4] 체크포인트 로드: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    epoch    = ckpt.get("epoch", "?")
    val_rmse = ckpt.get("val_rmse", "?")
    print(f"      epoch={epoch}  val_rmse={val_rmse:.5f}" if isinstance(val_rmse, float) else f"      epoch={epoch}")

    model = TwoLayerModel(MODEL_CFG)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"      파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    print("[2/4] Export 래퍼 + trace + freeze")
    wrapper = _ExportWrapper(model)
    wrapper.eval()

    dummy = torch.zeros(1, 6, 100)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"      trace 전 출력 확인: disp={out[0].shape}  dispCov={out[1].shape}")

    traced = torch.jit.trace(wrapper, dummy)
    traced = torch.jit.freeze(traced)

    # 동일 입력으로 원본 vs traced 출력 비교
    with torch.no_grad():
        ref  = wrapper(dummy)
        trc  = traced(dummy)
    disp_err = (ref[0] - trc[0]).abs().max().item()
    cov_err  = (ref[1] - trc[1]).abs().max().item()
    print(f"      원본 vs traced 최대 오차: disp={disp_err:.2e}  cov={cov_err:.2e}")
    assert disp_err < 1e-5 and cov_err < 1e-5, "trace 오차가 너무 큼"

    print(f"[3/4] .ptl 저장: {OUT_PATH}")
    os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
    traced._save_for_lite_interpreter(OUT_PATH)
    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"      크기: {size_mb:.2f} MB")

    print(f"[4/4] assets 에 복사: {ASSET_PATH}")
    import shutil
    os.makedirs(os.path.dirname(os.path.abspath(ASSET_PATH)), exist_ok=True)
    shutil.copy(OUT_PATH, ASSET_PATH)
    print(f"      완료: {ASSET_PATH}")

    print("\n✅ 변환 완료.")
    print("   다음 단계: Android Studio 에서 빌드 후 Replay 버튼으로 R0 측정")


if __name__ == "__main__":
    main()
