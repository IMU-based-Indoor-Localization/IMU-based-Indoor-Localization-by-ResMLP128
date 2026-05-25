"""
preproc_ablation.py — 전처리 단계 ablation 비교 (PC offline)
============================================================
같은 Android CSV 입력에 대해 *입력 전처리 토글* 효과만 격리해 비교한다.
단말 변경 0, PC 에서 재현. 학술 보고용 figure 산출.

비교 대상 전처리 (모두 단말 ImuCollector/InferenceEngine 의 단계 모방):
  1) calib   : 시작 2초 평균 acc/gyr 차감 (bias 보정)
  2) LPF     : 5 Hz Butterworth low-pass (P47 시도, 학습 분포 어긋남 검증)
  3) norm    : norm_mean / norm_std 정규화 (모델 입력 표준화)

후처리 (PATH_B 의 적응 scale / clamp / EMA / static/turn) 은 *동일* — 입력
전처리 효과만 격리. 모델 dead-reckoning (방향+크기 그대로 누적) 으로 단순화
— PATH_B 의 rotVec heading 치환은 단말만의 특화라 PC 비교에서 제외.

옵션 (argparse):
  --android <csv>          : Android ImuTestActivity CSV
  --model_dir <dir>        : out_classifier2 등
  --out <png>              : 결과 figure 경로

4 조합 자동 실행 (기본 + 각 단독 OFF/ON):
  (1) baseline (all on)        : calib=on,  LPF=off, norm=on   ← 현재 단말 동등
  (2) -calib                   : calib=off, LPF=off, norm=on
  (3) +LPF                     : calib=on,  LPF=on,  norm=on
  (4) -norm                    : calib=on,  LPF=off, norm=off
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Windows cp949 콘솔도 한글 안전
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from offline_eval import (   # type: ignore
    WINDOW_LEN, FS, load_android, load_model,
    window_to_gravity_aligned, window_yaw0,
)


def apply_lpf(data: np.ndarray, cutoff_hz: float = 5.0, fs: float = FS, order: int = 2):
    """5 Hz Butterworth low-pass — P47 시도 LPF 모방.
    data: [N, C] (N=샘플, C=채널). axis=0 따라 필터 적용."""
    from scipy.signal import butter, filtfilt
    b, a = butter(order, cutoff_hz / (fs / 2.0), btype="low")
    return filtfilt(b, a, data, axis=0)


def evaluate_with_preproc(data, net, mean, std, *,
                          use_calib: bool, use_lpf: bool, use_norm: bool,
                          stride: int = WINDOW_LEN):
    """전처리 토글 적용 후 모델 dead-reckoning. 반환: recon_xy[N,2], stats dict."""
    import torch

    acc = data["acc"].copy()
    gyr = data["gyr"].copy()
    quat = data["quat"]

    # ── 입력 전처리 토글 (load_android 가 이미 calib 적용했으므로 use_calib=False 시 *역적용*)
    # 정확한 격리 위해 calib 효과를 *되돌림* 으로써 비교.
    # 단순화: 처음 2 초 평균을 *다시 더해서* 차감 효과 무효화.
    if not use_calib:
        n_cal = min(int(2.0 * FS), len(acc))
        if n_cal > 10:
            acc_bias = acc[:n_cal].mean(axis=0)
            gyr_bias = gyr[:n_cal].mean(axis=0)
            acc = acc + acc_bias   # ← 차감 분 되돌림 (no_calib = bias 그대로)
            gyr = gyr + gyr_bias

    if use_lpf:
        acc = apply_lpf(acc)
        gyr = apply_lpf(gyr)

    # ── window 별 추론
    T = len(acc)
    starts = list(range(0, T - WINDOW_LEN, stride))
    disp_xy = []
    recon = [np.zeros(2)]
    with torch.no_grad():
        for s in starts:
            e = s + WINDOW_LEN
            imu = window_to_gravity_aligned(acc, gyr, quat, s, e, frame="ga")
            imun = (imu - mean) / std if use_norm else imu       # ← norm toggle
            x = torch.from_numpy(imun.T.copy()).float().unsqueeze(0)
            y, _y_cov, _logits = net(x)
            d = y[0].numpy()
            disp_xy.append(float(np.hypot(d[0], d[1])))
            # ga frame → world (window 시작 yaw 로 회전)
            yaw0 = window_yaw0(quat, s)
            c, sn = np.cos(yaw0), np.sin(yaw0)
            dw = np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]])
            recon.append(recon[-1] + dw)
    recon = np.array(recon)
    disp_xy = np.array(disp_xy)
    path = float(np.linalg.norm(np.diff(recon, axis=0), axis=1).sum())
    end_off = float(np.linalg.norm(recon[-1] - recon[0]))
    return {
        "recon": recon, "disp_xy": disp_xy,
        "n_win": len(starts),
        "path": path, "end_off": end_off,
        "disp_mean": float(disp_xy.mean()) if len(disp_xy) else 0.0,
        "disp_max":  float(disp_xy.max())  if len(disp_xy) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--android", required=True, help="Android CSV")
    ap.add_argument("--model_dir", required=True, help="모델 폴더")
    ap.add_argument("--out", default="logs/preproc_ablation.png", help="결과 figure")
    ap.add_argument("--gt_perimeter", type=float, default=None,
                    help="GT 둘레 (m). 있으면 drift 비율 산출 (예: 25x12 -> 74)")
    args = ap.parse_args()

    print(f"[1] 모델 로드: {args.model_dir}")
    net, _para, mean, std = load_model(args.model_dir)

    print(f"[2] Android CSV: {args.android}")
    data = load_android(args.android, calib_sec=2.0)  # *기본* calib (use_calib=False 시 되돌림)

    configs = [
        ("baseline",   dict(use_calib=True,  use_lpf=False, use_norm=True)),
        ("-calib",     dict(use_calib=False, use_lpf=False, use_norm=True)),
        ("+LPF",       dict(use_calib=True,  use_lpf=True,  use_norm=True)),
        ("-norm",      dict(use_calib=True,  use_lpf=False, use_norm=False)),
    ]
    results = []
    print()
    print(f"{'config':<30} {'n_win':>5} {'path(m)':>10} {'end_off(m)':>11} {'disp mean':>10} {'disp max':>10}")
    print("-" * 80)
    for label, kw in configs:
        r = evaluate_with_preproc(data, net, mean, std, **kw)
        results.append((label, r))
        drift_str = f"drift={r['end_off']/args.gt_perimeter*100:.1f}%" if args.gt_perimeter else ""
        print(f"{label:<30} {r['n_win']:>5} {r['path']:>10.2f} {r['end_off']:>11.2f} "
              f"{r['disp_mean']:>10.3f} {r['disp_max']:>10.3f}  {drift_str}")

    # ── 시각화
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for fn in ("Malgun Gothic", "NanumGothic", "DejaVu Sans"):
            try:
                matplotlib.rcParams["font.family"] = fn
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
    except ImportError:
        print("[skip] matplotlib 없음 — figure 생략")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    colors  = ["#1565C0", "#E65100", "#388E3C", "#6A1B9A"]
    linestyles = ["-", "--", "-", "-"]   # -calib 만 점선 (baseline 위에 겹침)
    widths  = [3.5, 1.8, 2.2, 2.2]       # baseline 가장 굵게

    # 검출: -calib 결과가 baseline 과 동일한지 (수치 비교)
    eq_to_base = []
    base_recon = results[0][1]["recon"]
    for label, r in results:
        same = (len(r["recon"]) == len(base_recon)
                and np.allclose(r["recon"], base_recon, atol=1e-6))
        eq_to_base.append(same)

    ax = axes[0]
    for (label, r), c, ls, w, same in zip(results, colors, linestyles, widths, eq_to_base):
        suffix = " (= baseline)" if (same and label != "baseline") else ""
        ax.plot(r["recon"][:, 0], r["recon"][:, 1], ls, lw=w, color=c, alpha=0.9,
                label=f"{label}{suffix}  path={r['path']:.1f}m  end_off={r['end_off']:.2f}m")
        ax.scatter([r['recon'][0,0]], [r['recon'][0,1]], color="#388E3C", s=80, zorder=5)
        ax.scatter([r['recon'][-1,0]], [r['recon'][-1,1]], color=c, s=60, marker="x", zorder=5)
    ax.set_title(f"전처리 ablation — dead-reckoning 궤적 ({Path(args.android).name})",
                 fontsize=14)
    ax.set_xlabel("x (m)", fontsize=12); ax.set_ylabel("y (m)", fontsize=12)
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="best")

    ax2 = axes[1]
    for (label, r), c, ls, w, same in zip(results, colors, linestyles, widths, eq_to_base):
        suffix = " (= baseline)" if (same and label != "baseline") else ""
        ax2.plot(r["disp_xy"], ls, lw=w, color=c, alpha=0.9,
                 label=f"{label}{suffix}  mean={r['disp_mean']:.3f}m")
    ax2.axhline(1.5, color="r", ls=":", lw=1, label="정상 보행 상한 ~1.5m")
    ax2.set_title("window별 |disp_xy|", fontsize=14)
    ax2.set_xlabel("window idx", fontsize=12); ax2.set_ylabel("|disp_xy| (m)", fontsize=12)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\n[OK] figure 저장: {args.out}")


if __name__ == "__main__":
    main()
