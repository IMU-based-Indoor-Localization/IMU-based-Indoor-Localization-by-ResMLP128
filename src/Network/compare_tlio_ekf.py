"""
compare_tlio_ekf.py — EKF 계수만 다른 두 변형의 궤적 오프라인 비교
================================================================
같은 IMU CSV + 같은 모델 출력 시퀀스를 두 EKF cfg 변형(현재 단말 cfg vs
TLIO 논문 cfg) 에 동일하게 입력해, **EKF 계수 차이만으로** 궤적이 어떻게
달라지는지 격리 측정한다. 단말 코드는 일절 변경하지 않는다.

비교 항목 (imu_ekf_py.tlio_cfg() 와 current_cfg() 차이):
  - init_vel_sigma : 1.0   → 0.1  (TLIO §V-E)
  - init_ba_sigma  : 0.02  → 0.2
  - meascov_scale  : 1.0   → 10.0 (TLIO §V-D 끝, temporal correlation 보정)
  - 그 외(σ_θ, σ_bg, σ_y, χ²=11.345) 는 두 cfg 가 이미 동일.

EKF 식은 imu_ekf.cpp 와 1:1 동등한 Python 포팅(imu_ekf_py.py) — 식이 같고
계수만 다르므로 *계수 효과만* 깔끔히 격리된다.

실행 (anaconda 환경)
-------------------
# 1) Android CSV (ImuTestActivity 기록)
python src/Network/compare_tlio_ekf.py \\
    --model_dir src/Network/out_classifier2 \\
    --android latest.csv \\
    --plot logs/tlio_compare.png

# 2) OxIOD baseline (있는 GT 와 함께 비교)
python src/Network/compare_tlio_ekf.py \\
    --model_dir src/Network/out_classifier2 \\
    --oxiod src/TLIO_Oxford_Dataset/oxford_handheld_1 \\
    --plot logs/tlio_compare_oxiod.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Windows cp949 콘솔에서도 한글/유니코드(—, ·) 출력되도록.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

# 같은 폴더의 offline_eval / imu_ekf_py 재사용
sys.path.insert(0, str(Path(__file__).parent))
from offline_eval import (  # type: ignore
    GRAVITY, WINDOW_LEN, FS,
    load_android, load_oxiod, load_model,
    window_to_gravity_aligned, window_yaw0,
)
from imu_ekf_py import ScEkf, current_cfg, tlio_cfg  # type: ignore


UPDATE_HZ = 20            # TLIO §V-D 와 동일 20Hz
CLONE_STRIDE_SAMPLES = int(FS / UPDATE_HZ)   # 100/20 = 5
WARMUP_SEC = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# 1) 모델 추론 — 단일 윈도우 → (disp_ga, cov_ga)
# ─────────────────────────────────────────────────────────────────────────────
def _infer_window(net, acc, gyr, quat, s, e, mean, std):
    """반환: disp[3] (ga-frame), cov[3,3] (ga-frame, diag)."""
    import torch
    imu  = window_to_gravity_aligned(acc, gyr, quat, s, e, frame="ga")
    imun = (imu - mean) / std
    x    = torch.from_numpy(imun.T.copy()).float().unsqueeze(0)
    with torch.no_grad():
        y, y_cov, _logits = net(x)
    d = y[0].numpy().astype(np.float64)
    # y_cov 는 모델 종류에 따라 logvar 또는 cov 출력 — 대각 가정
    cov_diag = np.exp(y_cov[0].numpy()).astype(np.float64)   # log-var → var
    return d, np.diag(cov_diag)


# ─────────────────────────────────────────────────────────────────────────────
# 2) 측정값(meas) 을 begin-clone gravity-aligned frame 으로 변환
#    EKF.update 는 begin clone 의 yaw 회전 R_z 가 적용된 frame 을 가정.
#    학습 모델 disp 는 *window 시작 yaw 를 제거한* gravity-aligned frame.
#    두 frame 은 정의가 동일 — disp 그대로 측정값으로 사용.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 3) 메인 비교 루프 — 두 EKF 동기 실행
# ─────────────────────────────────────────────────────────────────────────────
def run_compare(net, data, mean, std, label_a="current", label_b="TLIO",
                cfg_a=None, cfg_b=None):
    """반환 dict per cfg: traj[N,2], path_len, end_offset, n_updates, n_skipped."""
    if cfg_a is None: cfg_a = current_cfg()
    if cfg_b is None: cfg_b = tlio_cfg()

    acc, gyr, quat = data["acc"], data["gyr"], data["quat"]
    pos_gt = data.get("pos")
    T = len(acc)

    # IMU 100Hz 균일 그리드 가정 → 합성 ts (μs)
    ts = (np.arange(T) / FS * 1e6).astype(np.int64)

    # 초기 정적 평균 acc (rot from gravity) + warm-up 무업데이트
    n_cal = max(int(WARMUP_SEC * FS), 50)
    acc_static = acc[:n_cal].mean(axis=0).astype(np.float64)

    ekf_a = ScEkf(cfg_a)
    ekf_b = ScEkf(cfg_b)
    ekf_a.initialize(int(ts[0]), acc_static)
    ekf_b.initialize(int(ts[0]), acc_static)

    traj_a = [ekf_a.state.p[:2].copy()]
    traj_b = [ekf_b.state.p[:2].copy()]
    keep_clones = WINDOW_LEN // CLONE_STRIDE_SAMPLES + 3   # 20 + 여유

    stats = {label_a: {"n_upd": 0, "n_skip": 0, "skip_reasons": {}},
             label_b: {"n_upd": 0, "n_skip": 0, "skip_reasons": {}}}

    for i in range(1, T):
        is_clone_step = (i % CLONE_STRIDE_SAMPLES == 0)
        t_aug = int(ts[i]) if is_clone_step else None
        for ekf in (ekf_a, ekf_b):
            ekf.propagate(acc[i].astype(np.float64), gyr[i].astype(np.float64),
                          int(ts[i]), t_aug)
            # warm-up: 첫 첫 업데이트 직전까지 v 고정 0 (단말 ZUPT 효과 모방).
            # 그렇지 않으면 IMU bias 잔여로 v 가 발산해 첫 update 시 χ² gate
            # 가 매번 reject → cfg 차이 비교가 불가능.
            if i < n_cal + WINDOW_LEN:
                ekf.state.v[:] = 0.0

        # update: warm-up 이후 + 윈도우 1초 채워진 후
        if is_clone_step and i >= n_cal + WINDOW_LEN:
            s = i - WINDOW_LEN
            t_begin = int(ts[s])
            t_end   = int(ts[i])
            disp_ga, cov_ga = _infer_window(net, acc, gyr, quat,
                                            s, i, mean, std)
            # 두 EKF 모두 같은 측정값으로 update
            for ekf, lbl in [(ekf_a, label_a), (ekf_b, label_b)]:
                r = ekf.update(disp_ga, cov_ga, t_begin, t_end)
                if r["applied"]:
                    stats[lbl]["n_upd"] += 1
                else:
                    stats[lbl]["n_skip"] += 1
                    rk = r["reason"].split("(")[0] or "unknown"
                    stats[lbl]["skip_reasons"][rk] = (
                        stats[lbl]["skip_reasons"].get(rk, 0) + 1)
                ekf.marginalize_until(t_begin)

        traj_a.append(ekf_a.state.p[:2].copy())
        traj_b.append(ekf_b.state.p[:2].copy())

    traj_a = np.array(traj_a)
    traj_b = np.array(traj_b)

    def _summary(tr):
        path_len = float(np.linalg.norm(np.diff(tr, axis=0), axis=1).sum())
        end_off  = float(np.linalg.norm(tr[-1] - tr[0]))
        return path_len, end_off

    pl_a, eo_a = _summary(traj_a)
    pl_b, eo_b = _summary(traj_b)

    res = {
        label_a: {
            "cfg": cfg_a, "traj_xy": traj_a,
            "path_len": pl_a, "end_offset": eo_a,
            **stats[label_a],
        },
        label_b: {
            "cfg": cfg_b, "traj_xy": traj_b,
            "path_len": pl_b, "end_offset": eo_b,
            **stats[label_b],
        },
    }

    # GT (OxIOD)
    if pos_gt is not None:
        gt_xy = pos_gt[:T, :2].astype(np.float64) - pos_gt[0, :2]
        n = min(len(traj_a), len(gt_xy))
        res["gt_xy"] = gt_xy[:n]
        for lbl, tr in [(label_a, traj_a[:n]), (label_b, traj_b[:n])]:
            res[lbl]["rmse"] = float(
                np.sqrt(((tr - gt_xy[:n]) ** 2).sum(axis=1).mean()))
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 4) 출력
# ─────────────────────────────────────────────────────────────────────────────
def print_compare(res, src_label):
    print(f"\n{'='*72}\n  비교 결과 — {src_label}\n{'='*72}")
    headers = ["cfg", "n_upd", "n_skip", "path_len(m)", "end_off(m)"]
    if "rmse" in res.get("current", {}):
        headers.append("RMSE_XY(m)")
    print("  " + "  ".join(f"{h:>14}" for h in headers))
    for lbl in ("current", "TLIO"):
        r = res[lbl]
        row = [
            lbl,
            r["n_upd"], r["n_skip"],
            f"{r['path_len']:.2f}", f"{r['end_offset']:.2f}",
        ]
        if "rmse" in r:
            row.append(f"{r['rmse']:.2f}")
        print("  " + "  ".join(f"{str(v):>14}" for v in row))
        if r["skip_reasons"]:
            br = ", ".join(f"{k}={v}" for k, v in r["skip_reasons"].items())
            print(f"  {'':>14}  skip 사유: {br}")

    # 핵심 cfg 차이 요약
    print("\n  [cfg 차이 — TLIO 가 다른 값만]:")
    print(f"    init_vel_sigma   : current=1.0  →  TLIO=0.1   m/s")
    print(f"    init_ba_sigma    : current=0.02 →  TLIO=0.2   m/s²")
    print(f"    meascov_scale    : current=1.0  →  TLIO=10.0  (TLIO §V-D)")


def make_plot(res, src_label, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Windows 한글 폰트 — Malgun Gothic 가장 흔함. 없으면 fallback 영문.
        for fn in ("Malgun Gothic", "NanumGothic", "Gulim", "DejaVu Sans"):
            try:
                matplotlib.rcParams["font.family"] = fn
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
    except ImportError:
        print("  [skip] matplotlib 없음 — plot 생략")
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    a = res["current"]["traj_xy"]
    b = res["TLIO"]["traj_xy"]
    ax.plot(a[:, 0], a[:, 1], "-", lw=1.4, alpha=0.85,
            color="#1565C0",
            label=f"current cfg (path={res['current']['path_len']:.2f}m, "
                  f"end={res['current']['end_offset']:.2f}m, "
                  f"upd/skip={res['current']['n_upd']}/{res['current']['n_skip']})")
    ax.plot(b[:, 0], b[:, 1], "-", lw=1.4, alpha=0.85,
            color="#E65100",
            label=f"TLIO cfg    (path={res['TLIO']['path_len']:.2f}m, "
                  f"end={res['TLIO']['end_offset']:.2f}m, "
                  f"upd/skip={res['TLIO']['n_upd']}/{res['TLIO']['n_skip']})")
    if "gt_xy" in res:
        g = res["gt_xy"]
        ax.plot(g[:, 0], g[:, 1], "k--", lw=1.0, alpha=0.6,
                label=f"GT (length={float(np.linalg.norm(np.diff(g,axis=0),axis=1).sum()):.2f}m)")
    # 시작·종점 마커
    for tr, c in [(a, "#1565C0"), (b, "#E65100")]:
        ax.scatter([tr[0, 0]], [tr[0, 1]], color="#388E3C", s=60, zorder=5)
        ax.scatter([tr[-1, 0]], [tr[-1, 1]], color=c, s=40, marker="x", zorder=5)
    ax.set_title(f"EKF 계수 비교 (식 동일, 계수만 변경) — {src_label}")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\n  [OK] plot 저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5) main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True,
                    help="모델 폴더 (config.json + checkpoints/best.pth + norm_*.npy)")
    ap.add_argument("--android", default=None,
                    help="Android ImuTestActivity CSV")
    ap.add_argument("--oxiod", default=None,
                    help="OxIOD 시퀀스 폴더 또는 imu0_resampled.npy")
    ap.add_argument("--linacc_scale", type=float, default=1.0,
                    help="Android linAcc 계수 (1.0=m/s², 0.10194=1/9.81=g)")
    ap.add_argument("--calib_sec", type=float, default=2.0)
    ap.add_argument("--plot", default=None, help="비교 plot PNG 경로")
    ap.add_argument("--abs_innov_m", type=float, default=1e9,
                    help="절대 innov 게이트 (m). 1e9=비활성(TLIO 논문 식). "
                         "단말 EKF 동작 재현 시 3.0 권장.")
    args = ap.parse_args()

    if not args.android and not args.oxiod:
        print("[!] --android 또는 --oxiod 중 하나 지정")
        sys.exit(1)

    print(f"[1] 모델 로드: {args.model_dir}")
    net, model_para, mean, std = load_model(args.model_dir)
    print(f"  use_classifier={model_para.get('use_classifier')}  "
          f"feature_dim={model_para.get('feature_dim')}")

    if args.oxiod:
        p = Path(args.oxiod)
        npy = p if p.suffix == ".npy" else p / "imu0_resampled.npy"
        print(f"\n[2] OxIOD 로드: {npy}")
        data = load_oxiod(npy)
        src  = f"OxIOD {Path(npy).parent.name}"
    else:
        print(f"\n[2] Android CSV 로드: {args.android}")
        data = load_android(args.android, calib_sec=args.calib_sec,
                            linacc_scale=args.linacc_scale)
        src  = f"Android {Path(args.android).name}"

    print(f"\n[3] EKF 두 변형 동기 실행 (식 동일, 계수만 변경)")
    cfg_a = current_cfg(); cfg_a.max_innov_norm = args.abs_innov_m
    cfg_b = tlio_cfg();    cfg_b.max_innov_norm = args.abs_innov_m
    res = run_compare(net, data, mean, std,
                      label_a="current", label_b="TLIO",
                      cfg_a=cfg_a, cfg_b=cfg_b)
    print_compare(res, src)

    if args.plot:
        make_plot(res, src, args.plot)


if __name__ == "__main__":
    main()
