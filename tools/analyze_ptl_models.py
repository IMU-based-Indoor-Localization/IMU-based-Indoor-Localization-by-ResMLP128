"""
analyze_ptl_models.py
=====================
mobile_assets/ 의 모든 .ptl 모델을 *동일 입력*에 대해 forward 한 뒤 disp 출력 비교.

단계 (verify_ptl_models.py 의 a/b 단계에 추가):
  (c) 모델 정체 — 출력 shape, 안정성, learned 여부
  (d) **같은 Android window 에 모델별 disp 차이** (정량)
  (e) **OxIOD handheld_1 window 에 모델별 disp 차이** (학습 분포 내 행동)
  (f) Android window 와 OxIOD window 의 모델별 disp 비교 — OoD 정도 측정

핵심 질문:
  Q1. 어느 모델이 Android 입력에 가장 작은 disp 를 출력하나? (3.5× 과대 추정 완화)
  Q2. OxIOD 입력에서 어느 모델이 GT 와 가장 가까운가? (학습 성능 비교)
  Q3. Android - OxIOD 출력 차이가 모델마다 다른가? (OoD 민감도 차이)

출력:
  - 콘솔 표
  - logs/ptl_disp_comparison.txt (저장)

실행:
    cd D:\\mobile\\imu_android
    python tools\\analyze_ptl_models.py
"""
import csv
import io
import os
import sys
import traceback
from collections import defaultdict

import numpy as np
import torch


# ─── 입력 ─────────────────────────────────────────────────────
MODEL_PATHS = [
    "mobile_assets/imu_model.ptl",
    "mobile_assets/imu_model_cls.ptl",
    "mobile_assets/imu_model_resmlp128.ptl",
    "mobile_assets/imu_model_resnet_working.ptl",
    "mobile_assets/imu_model_resnet_p41_backup.ptl",
    "android/app/src/main/assets/imu_model.ptl",
]
ANDROID_CSV = "latest.csv"
OXIOD_NPY   = "src/TLIO_Oxford_Dataset/oxford_handheld_1/imu0_resampled.npy"

# norm 후보 (which 모델이 어떤 norm 으로 학습됐는지 모를 때 둘 다 시도)
NORM_CANDIDATES = {
    "out_classifier2": (
        "src/Network/out_classifier2/norm_mean.npy",
        "src/Network/out_classifier2/norm_std.npy",
    ),
    "out_regression": (
        "src/Network/out_regression/norm_mean.npy",
        "src/Network/out_regression/norm_std.npy",
    ),
}


# ─── 데이터 로딩 헬퍼 ────────────────────────────────────────
def load_android_window(csv_path: str, window_start_sec: float = 5.0, window_sec: float = 1.0) -> np.ndarray:
    """latest.csv 의 [start, start+1s] 구간 6채널 100Hz window. shape (6, 100). m/s² + rad/s 단위."""
    with open(csv_path, "rb") as f:
        raw = f.read()
    nul = raw.find(b"\x00")
    real = raw[:nul if nul >= 0 else len(raw)].decode("utf-8", errors="ignore")
    sensors = defaultdict(list)
    rdr = csv.reader(io.StringIO(real))
    next(rdr)
    for r in rdr:
        if len(r) < 6:
            continue
        try:
            sensors[r[0]].append((int(r[1]), float(r[2]), float(r[3]), float(r[4])))
        except Exception:
            pass

    linAcc = np.array(sensors["linAcc"], dtype=np.float64)
    gyr    = np.array(sensors["gyr"],    dtype=np.float64)

    t0 = min(linAcc[0, 0], gyr[0, 0])
    t_start_ns = t0 + int(window_start_sec * 1e9)
    t_end_ns   = t_start_ns + int(window_sec * 1e9)

    def resample(ts_ns, xyz, n: int):
        t_new = np.linspace(t_start_ns, t_end_ns, n) / 1e9
        ts = ts_ns / 1e9
        return np.stack([np.interp(t_new, ts, xyz[:, k]) for k in range(3)], axis=1)

    n = 100
    lin_w = resample(linAcc[:, 0], linAcc[:, 1:4], n)  # m/s²
    gyr_w = resample(gyr[:, 0],    gyr[:, 1:4],    n)  # rad/s
    window = np.concatenate([lin_w, gyr_w], axis=1).T  # (6, 100)
    return window.astype(np.float32)


def load_oxiod_window(npy_path: str, start_idx: int = 1000) -> np.ndarray:
    """OxIOD handheld_1 의 [start, start+100] 인덱스 6채널 window. shape (6, 100).
    OxIOD acc = g 단위 → ×9.81 로 m/s² 화. gyr = rad/s 그대로.
    """
    d = np.load(npy_path)
    gyr = d[start_idx:start_idx + 100, 1:4]
    acc = d[start_idx:start_idx + 100, 4:7] * 9.81
    return np.concatenate([acc, gyr], axis=1).T.astype(np.float32)  # (6, 100)


def load_norm(name: str):
    mean_p, std_p = NORM_CANDIDATES[name]
    if not (os.path.exists(mean_p) and os.path.exists(std_p)):
        return None
    return np.load(mean_p).astype(np.float32), np.load(std_p).astype(np.float32)


def normalize(window: np.ndarray, mean: np.ndarray, std: np.ndarray, scale_acc: float = 1.0) -> np.ndarray:
    """window (6, 100) — channel 0-2 = acc 에 scale_acc 적용 후 (x-mean)/std."""
    w = window.copy()
    w[:3] = w[:3] / scale_acc
    return (w - mean[:, None]) / std[:, None]


def desc(out):
    if isinstance(out, (tuple, list)):
        return f"tuple len={len(out)}: " + " ".join(
            f"[{i}]{tuple(o.shape)}" if hasattr(o, "shape") else f"[{i}]{type(o).__name__}"
            for i, o in enumerate(out)
        )
    if hasattr(out, "shape"):
        return f"tensor {tuple(out.shape)}"
    return str(type(out).__name__)


def disp_of(out):
    """tuple/tensor 출력에서 disp (첫 element) 만 [3] 배열로 추출."""
    t = out[0] if isinstance(out, (tuple, list)) else out
    return t.detach().numpy().flatten()[:3]


def forward_safe(model, x: np.ndarray):
    """x: (6, 100) numpy → tensor [1, 6, 100] → forward. 결과 + 예외 string 반환."""
    xt = torch.from_numpy(x).unsqueeze(0).float()
    try:
        return model(xt), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ─── 메인 ─────────────────────────────────────────────────────
def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"cwd: {os.getcwd()}")
    print()

    # 1. window 준비
    print("[1] Android window 로딩 (latest.csv, 5~6초 보행 추정 구간)")
    try:
        ad_w_raw = load_android_window(ANDROID_CSV, window_start_sec=5.0, window_sec=1.0)
    except Exception:
        print("    Android window 로딩 실패:")
        traceback.print_exc(limit=3)
        return 1
    print(f"    shape={ad_w_raw.shape}  std (ch별)={ad_w_raw.std(axis=1)}")

    print("[2] OxIOD window 로딩 (handheld_1, idx 1000~1100 보행)")
    try:
        ox_w_raw = load_oxiod_window(OXIOD_NPY, start_idx=1000)
    except Exception:
        print("    OxIOD window 로딩 실패:")
        traceback.print_exc(limit=3)
        return 1
    print(f"    shape={ox_w_raw.shape}  std (ch별)={ox_w_raw.std(axis=1)}")
    print()

    # 2. norm 후보 로드
    norms = {}
    for name in NORM_CANDIDATES:
        n = load_norm(name)
        if n is not None:
            norms[name] = n
            mean, std = n
            print(f"[3.{name}] mean={mean.tolist()}\n         std ={std.tolist()}")
        else:
            print(f"[3.{name}] norm 파일 없음 — 스킵")
    print()

    # 3. 각 모델 × 각 norm × (Android raw / Android g변환 / OxIOD m/s² / OxIOD g) forward
    print("=" * 110)
    print(f"{'model':<48s} | {'input':<24s} | {'norm':<16s} | {'disp x':<8s} {'disp y':<8s} {'disp z':<8s} | {'|d_xy|':<7s} | desc")
    print("=" * 110)

    rows = []
    for mpath in MODEL_PATHS:
        if not os.path.exists(mpath):
            print(f"{mpath:<48s} | (파일 없음)")
            continue
        try:
            model = torch._C._load_for_lite_interpreter(mpath)
        except Exception as e:
            print(f"{mpath:<48s} | (로드 실패: {type(e).__name__})")
            continue

        for norm_name, (mean, std) in norms.items():
            # Android m/s² 그대로
            ad_norm = normalize(ad_w_raw, mean, std, scale_acc=1.0)
            out, err = forward_safe(model, ad_norm)
            if err:
                print(f"{os.path.basename(mpath):<48s} | Android m/s²            | {norm_name:<16s} | forward 실패 {err[:40]}")
            else:
                d = disp_of(out)
                xy = float(np.sqrt(d[0]**2 + d[1]**2))
                print(f"{os.path.basename(mpath):<48s} | Android m/s²            | {norm_name:<16s} | {d[0]:+7.4f} {d[1]:+7.4f} {d[2]:+7.4f} | {xy:6.4f} | {desc(out)[:30]}")
                rows.append((mpath, "Android m/s²", norm_name, d, desc(out)))

            # Android g 단위 (linAcc /9.81)
            ad_norm_g = normalize(ad_w_raw, mean, std, scale_acc=9.81)
            out, err = forward_safe(model, ad_norm_g)
            if err:
                print(f"{os.path.basename(mpath):<48s} | Android g (/9.81)       | {norm_name:<16s} | forward 실패")
            else:
                d = disp_of(out)
                xy = float(np.sqrt(d[0]**2 + d[1]**2))
                print(f"{os.path.basename(mpath):<48s} | Android g (/9.81)       | {norm_name:<16s} | {d[0]:+7.4f} {d[1]:+7.4f} {d[2]:+7.4f} | {xy:6.4f} |")
                rows.append((mpath, "Android g (/9.81)", norm_name, d, desc(out)))

            # OxIOD m/s² 단위
            ox_norm = normalize(ox_w_raw, mean, std, scale_acc=1.0)
            out, err = forward_safe(model, ox_norm)
            if err:
                pass
            else:
                d = disp_of(out)
                xy = float(np.sqrt(d[0]**2 + d[1]**2))
                print(f"{os.path.basename(mpath):<48s} | OxIOD m/s²              | {norm_name:<16s} | {d[0]:+7.4f} {d[1]:+7.4f} {d[2]:+7.4f} | {xy:6.4f} |")
                rows.append((mpath, "OxIOD m/s²", norm_name, d, desc(out)))

            # OxIOD g (학습 분포)
            ox_norm_g = normalize(ox_w_raw, mean, std, scale_acc=9.81)
            out, err = forward_safe(model, ox_norm_g)
            if err:
                pass
            else:
                d = disp_of(out)
                xy = float(np.sqrt(d[0]**2 + d[1]**2))
                print(f"{os.path.basename(mpath):<48s} | OxIOD g (학습분포)         | {norm_name:<16s} | {d[0]:+7.4f} {d[1]:+7.4f} {d[2]:+7.4f} | {xy:6.4f} |")
                rows.append((mpath, "OxIOD g (학습분포)", norm_name, d, desc(out)))
        print("-" * 110)

    print()
    print("=== 해석 ===")
    print("  - OxIOD g + 같은 norm 학습 데이터 → 가장 학습 분포와 가까운 입력. |d_xy| 는 ~0.5-1.5m (보행 1초)")
    print("  - Android m/s² + out_classifier2 norm = 현 Android 동작 동일")
    print("  - Android g + out_classifier2 norm = USE_OOD_FIX=true 동일")
    print("  - 같은 input 에 모델별 disp 차이가 *크면* 모델 교체 효과 큼")
    print("  - 같은 input 에 모델별 disp 차이가 *작으면* 모델 교체 효과 미미 → fine-tune 필요")
    return 0


if __name__ == "__main__":
    sys.exit(main())
