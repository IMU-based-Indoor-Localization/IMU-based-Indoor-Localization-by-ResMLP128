"""
offline_eval.py — Phase 0 오프라인 재현 하니스
==============================================
EKF 를 완전히 배제하고, "IMU 시퀀스 → 학습 동일 전처리 → 동일 모델 → window 별 disp"
를 PC 에서 재현한다. 모델 OoD 여부를 EKF 발산과 분리해 격리 측정하는 것이 목적.

해결하려는 핵심 문제
--------------------
단말 실측에서 모델 disp 가 8~13배 과대 (handoff P51). 그런데 OxIOD test RMSE 는
handheld 1.57m (정상). → 배포 환경에서만 망가지는 OoD. 지금까지의 진단은 단말에서
상수 하나씩 바꿔 replay+logcat 읽기 — 모델 오차와 EKF 오차가 섞여 측정 불가능했다.

이 하니스가 하는 일
-------------------
1. OxIOD 시퀀스를 동일 모델에 통과 → GT 와 비교해 RMSE 산출 (하니스 + 모델 + norm 검증).
   OxIOD RMSE 가 ~1.5m 면 하니스가 학습 파이프라인을 정확히 재현하는 것.
2. Android CSV 를 *동일* 전처리로 통과 → window 별 disp 출력. EKF 없음.
3. --diagnose: Android 를 단위 스케일 {1.0 = m/s², 1/9.81 = g} × frame {ga, body}
   매트릭스로 sweep. 어느 조합이 정상 disp 를 내는지 GT 없이도 판정 가능
   (5m 왕복은 약한 GT: 시작=끝, 경로길이 ~10m, 1초 window |xy| ~1.0-1.5m).

핵심 가설 (이 하니스가 검증)
---------------------------
OxIOD acc(col 4:7) 는 g 단위 (gravity norm ≈ 1.0 확인). 모델 norm_std[0:3] ≈ 0.12-0.15
도 g 단위와 정합. 그런데 Android linAcc 는 m/s². 단위 그대로 입력하면 9.81배 과대
→ 정규화 후 학습 분포 밖 → 거대한 disp. "8~13배 과대" 가 "9.81배 단위 오차" 와 근접.
on-device P46 A/B 는 GT 가 없어 분류 분포로만 판정 → 이 하니스가 GT 로 확정한다.

실행 (anaconda 환경 — torch + scipy 필요)
------------------------------------------
# 1) 하니스 자체 검증 — OxIOD 가 ~1.5m RMSE 나오는지
python src/Network/offline_eval.py \
    --model_dir src/Network/out_classifier2 \
    --oxiod src/TLIO_Oxford_Dataset/oxford_handheld_1

# 2) Android 진단 매트릭스
python src/Network/offline_eval.py \
    --model_dir src/Network/out_classifier2 \
    --android latest.csv --diagnose

# 3) OxIOD baseline 과 Android 동시 비교
python src/Network/offline_eval.py \
    --model_dir src/Network/out_classifier2 \
    --oxiod src/TLIO_Oxford_Dataset/oxford_handheld_1 \
    --android latest.csv --diagnose --plot logs/offline_eval.png

# 분리 회귀 모델(A) 로도 동일 비교 — Phase 3
python src/Network/offline_eval.py --model_dir src/Network/out_regression ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

WINDOW_LEN = 100          # 학습 window (1초 @ 100Hz)
FS         = 100.0        # 모델 입력 샘플레이트
GRAVITY    = 9.81

# OxIOD 24-col: ts(1)+gyr(3)+acc(3)+gravity(3)+attitude(3)+label(1)+qxyzw(4)+pos(3)+vel(3)
_OX = {"gyr": (1, 4), "acc": (4, 7), "grav": (7, 10), "quat": (14, 18), "pos": (18, 21)}


# ─────────────────────────────────────────────────────────────────────────────
# 1. 학습 동일 전처리 — dataset.py _window_to_gravity_aligned 와 1:1 동일
# ─────────────────────────────────────────────────────────────────────────────
def window_to_gravity_aligned(acc, gyr, quat, start, end, frame="ga"):
    """[start:end] 구간의 body-frame IMU 를 학습 입력 프레임으로 변환.

    frame:
      'ga'   — 학습과 동일: 각 시점 body→world (per-timestep quat) 회전 후
               window 시작 yaw 제거. (dataset.py 와 정확히 일치)
      'body' — 회전 없이 raw body frame 그대로. (프레임 변환을 건너뛴 경우)
      'yaw'  — per-timestep R_all 없이 window 시작 회전 1개로만 변환.
               (Android transformWindowToWorldFrame 의 R_begin 1개 방식 근사)
    반환: [window_len, 6]  (acc_xyz + gyr_xyz)
    """
    from scipy.spatial.transform import Rotation as Rot

    a = acc[start:end].astype(np.float64)
    g = gyr[start:end].astype(np.float64)

    if frame == "body":
        return np.concatenate([a, g], axis=1).astype(np.float32)

    R_all = Rot.from_quat(quat[start:end]).as_matrix()           # [W,3,3] body→world
    yaw0  = Rot.from_quat(quat[start]).as_euler("zyx")[0]
    R_yaw_inv = Rot.from_euler("z", yaw0).inv().as_matrix()

    if frame == "yaw":
        # window 시작 회전 1개만 적용 (per-timestep 무시)
        R_begin = R_all[0]
        acc_w = (R_begin @ a.T).T
        gyr_w = (R_begin @ g.T).T
    else:  # 'ga' — 학습 동일
        acc_w = np.einsum("tij,tj->ti", R_all, a)
        gyr_w = np.einsum("tij,tj->ti", R_all, g)

    acc_ga = (R_yaw_inv @ acc_w.T).T
    gyr_ga = (R_yaw_inv @ gyr_w.T).T
    return np.concatenate([acc_ga, gyr_ga], axis=1).astype(np.float32)


def window_yaw0(quat, start):
    """window 시작 시점의 절대 yaw (rad) — 궤적 재구성용."""
    from scipy.spatial.transform import Rotation as Rot
    return float(Rot.from_quat(quat[start]).as_euler("zyx")[0])


# ─────────────────────────────────────────────────────────────────────────────
# 2. 데이터 로더
# ─────────────────────────────────────────────────────────────────────────────
def load_oxiod(npy_path):
    """OxIOD 시퀀스 → dict(acc[g], gyr[rad/s], quat[xyzw], pos[m]).  100Hz."""
    data = np.load(npy_path)
    if data.shape[1] < 21:
        raise ValueError(f"OxIOD npy 컬럼 부족: {data.shape}")
    return {
        "acc":  data[:, _OX["acc"][0]:_OX["acc"][1]].astype(np.float32),
        "gyr":  data[:, _OX["gyr"][0]:_OX["gyr"][1]].astype(np.float32),
        "quat": data[:, _OX["quat"][0]:_OX["quat"][1]].astype(np.float32),
        "pos":  data[:, _OX["pos"][0]:_OX["pos"][1]].astype(np.float32),
        "src":  f"OxIOD:{Path(npy_path).parent.name}",
    }


def _read_android_long(csv_path):
    """Android ImuTestActivity long-format CSV (sensor,ts_ns,x,y,z,w) → 센서별 dict."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline().strip()
        if not header.startswith("sensor,"):
            raise ValueError(f"CSV 헤더 형식 다름: {header!r}")
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split(",")
            if len(p) != 6:
                continue
            try:
                rows.append((p[0], int(p[1]), float(p[2]), float(p[3]),
                             float(p[4]), float(p[5])))
            except ValueError:
                continue
    out = {}
    for sensor in ("acc", "gyr", "linAcc", "rotVec"):
        sub = [r for r in rows if r[0] == sensor]
        if not sub:
            continue
        ts = np.array([r[1] for r in sub], dtype=np.int64)
        if sensor == "rotVec":
            v = np.array([[r[2], r[3], r[4], r[5]] for r in sub], dtype=np.float64)
        else:
            v = np.array([[r[2], r[3], r[4]] for r in sub], dtype=np.float64)
        order = np.argsort(ts)
        out[sensor] = {"ts": ts[order], "v": v[order]}
    return out


def load_android(csv_path, calib_sec=2.0, linacc_scale=1.0):
    """Android CSV → 100Hz 균일 그리드로 리샘플한 dict(acc, gyr, quat, pos=None).

    - 모델 입력 acc 채널 = linAcc (TYPE_LINEAR_ACCELERATION, 중력 제거).
    - quat = rotVec (TYPE_ROTATION_VECTOR, 자력계 융합 절대 자세 — yaw drift 없음).
      → 학습 _window_to_gravity_aligned 의 per-timestep 자세로 그대로 사용.
      이것이 "프레임이 정확할 때" 모델 입력 = best-case. EKF yaw drift 와 분리됨.
    - calib_sec: ImuCollector 의 2초 영점보정 재현 — 첫 N초 평균을 linAcc/gyr 에서 차감.
    - linacc_scale: linAcc 에 곱하는 계수. 1.0 = m/s² 그대로, 1/9.81 = g 단위 변환.
    """
    from scipy.spatial.transform import Rotation as Rot
    from scipy.spatial.transform import Slerp

    raw = _read_android_long(csv_path)
    for need in ("linAcc", "gyr", "rotVec"):
        if need not in raw:
            raise ValueError(f"CSV 에 {need} 센서 없음 — ImuTestActivity 4센서 기록 필요")

    la, gy, rv = raw["linAcc"], raw["gyr"], raw["rotVec"]
    # 공통 시간 구간 (ns) — 모든 센서가 값을 갖는 구간만
    t0 = max(la["ts"][0], gy["ts"][0], rv["ts"][0])
    t1 = min(la["ts"][-1], gy["ts"][-1], rv["ts"][-1])
    if t1 <= t0:
        raise ValueError("센서 시간 구간 겹침 없음")
    n_grid = int((t1 - t0) / 1e9 * FS)
    grid = t0 + (np.arange(n_grid) / FS * 1e9).astype(np.int64)
    gf = grid.astype(np.float64)

    def interp3(s):
        sf = s["ts"].astype(np.float64)
        return np.stack([np.interp(gf, sf, s["v"][:, k]) for k in range(3)], axis=1)

    acc = interp3(la) * float(linacc_scale)
    gyr = interp3(gy)

    # rotVec: 단위 quaternion slerp
    rv_q = rv["v"] / np.linalg.norm(rv["v"], axis=1, keepdims=True)
    slerp = Slerp(rv["ts"].astype(np.float64), Rot.from_quat(rv_q))
    quat = slerp(np.clip(gf, rv["ts"][0], rv["ts"][-1])).as_quat().astype(np.float32)

    # 영점 보정 (ImuCollector 2초 캘리브 재현)
    if calib_sec > 0:
        n_cal = min(int(calib_sec * FS), len(acc))
        if n_cal > 10:
            acc -= acc[:n_cal].mean(axis=0)
            gyr -= gyr[:n_cal].mean(axis=0)

    return {
        "acc": acc.astype(np.float32), "gyr": gyr.astype(np.float32),
        "quat": quat, "pos": None,
        "src": f"Android:{Path(csv_path).name}(scale={linacc_scale:.4f})",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. 모델
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_dir):
    """out_classifier2 / out_regression 폴더 → (model, model_para, mean, std)."""
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from model_twolayer import TwoLayerModel

    model_dir = Path(model_dir)
    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    model_para = cfg["model"]

    net = TwoLayerModel(model_para)
    ckpt = torch.load(model_dir / "checkpoints" / "best.pth", map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys {len(missing)}: {missing[:3]}")
    if unexpected:
        print(f"  [warn] unexpected keys {len(unexpected)}: {unexpected[:3]}")
    net.eval()

    mean = np.load(model_dir / "norm_mean.npy").astype(np.float32)
    std  = np.load(model_dir / "norm_std.npy").astype(np.float32)
    return net, model_para, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# 4. window 별 추론 + 궤적 재구성
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["handbag", "handheld", "pocket", "running", "slow_walk", "trolley", "unknown"]


def evaluate(net, data, mean, std, frame="ga", stride=WINDOW_LEN):
    """비겹침(stride=WINDOW_LEN) window 추론.

    반환 dict:
      n_win, disp_xy[], disp_z[], cls[]    — window 별
      recon_xy[N,2]                        — dead-reckoning 재구성 궤적 (world)
      gt_xy[N,2] 또는 None                 — OxIOD GT 궤적 (있을 때)
      path_len, end_offset, rmse, net_only_rmse
    """
    import torch

    acc, gyr, quat, pos = data["acc"], data["gyr"], data["quat"], data["pos"]
    T = len(acc)
    starts = list(range(0, T - WINDOW_LEN, stride))

    disp_xy, disp_z, cls_list = [], [], []
    recon = [np.zeros(2)]
    gt = [np.zeros(2)] if pos is not None else None

    with torch.no_grad():
        for s in starts:
            e = s + WINDOW_LEN
            imu = window_to_gravity_aligned(acc, gyr, quat, s, e, frame=frame)
            imu_n = (imu - mean) / std                          # [W,6]
            x = torch.from_numpy(imu_n.T.copy()).float().unsqueeze(0)  # [1,6,W]
            y, y_cov, logits = net(x)
            d = y[0].numpy()                                    # [3] ga-frame disp

            disp_xy.append(float(np.hypot(d[0], d[1])))
            disp_z.append(float(d[2]))
            if logits is not None:
                cls_list.append(int(np.argmax(logits[0].numpy())))
            else:
                cls_list.append(-1)

            # ga-frame disp → world: window 시작 절대 yaw 로 회전
            yaw0 = window_yaw0(quat, s)
            c, sn = np.cos(yaw0), np.sin(yaw0)
            dw = np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]])
            recon.append(recon[-1] + dw)

            if gt is not None:
                gdp = pos[e] - pos[s]
                gt.append(gt[-1] + gdp[:2])

    recon = np.array(recon)
    res = {
        "src": data["src"], "frame": frame, "n_win": len(starts),
        "disp_xy": np.array(disp_xy), "disp_z": np.array(disp_z),
        "cls": np.array(cls_list), "recon_xy": recon,
        "path_len": float(np.abs(np.diff(recon, axis=0)).sum()),
        "end_offset": float(np.hypot(*recon[-1])),
        "gt_xy": None, "rmse": None, "net_rmse": None,
    }
    if gt is not None:
        gt = np.array(gt)
        res["gt_xy"] = gt
        n = min(len(recon), len(gt))
        res["rmse"] = float(np.sqrt(((recon[:n] - gt[:n]) ** 2).sum(axis=1).mean()))
        res["net_rmse"] = res["rmse"]   # EKF 없음 → recon 자체가 net-only dead-reckoning
        res["gt_path_len"] = float(np.abs(np.diff(gt, axis=0)).sum())
    return res


def print_result(res):
    d = res["disp_xy"]
    print(f"\n── {res['src']}  [frame={res['frame']}]  window {res['n_win']}개 ──")
    if len(d) == 0:
        print("  window 없음 (시퀀스가 너무 짧음)")
        return
    print(f"  window별 |disp_xy| (1초 변위, 정상 보행 GT ~= 1.0~1.5m):")
    print(f"    mean={d.mean():.3f}  median={np.median(d):.3f}  "
          f"p95={np.percentile(d,95):.3f}  max={d.max():.3f}  min={d.min():.3f} m")
    print(f"    |disp_z| mean={np.abs(res['disp_z']).mean():.3f} m")
    cls = res["cls"]
    if (cls >= 0).any():
        uniq, cnt = np.unique(cls[cls >= 0], return_counts=True)
        dist = "  ".join(f"{CLASS_NAMES[u]}={c/len(cls)*100:.0f}%"
                         for u, c in zip(uniq, cnt))
        print(f"  cls 분포: {dist}")
    print(f"  재구성 궤적: 경로길이={res['path_len']:.2f}m  "
          f"종점offset={res['end_offset']:.2f}m")
    if res["rmse"] is not None:
        print(f"  [GT] RMSE_XY = {res['rmse']:.3f} m   "
              f"(GT 경로길이 {res['gt_path_len']:.2f}m)")
        if res["rmse"] < 2.5:
            print(f"    -> 하니스/모델/norm 정상 (OxIOD test RMSE ~1.5m 수준)")
        else:
            print(f"    -> RMSE 큼: 하니스 전처리 또는 모델 로드 점검 필요")


# ─────────────────────────────────────────────────────────────────────────────
# 5. plot
# ─────────────────────────────────────────────────────────────────────────────
def make_plot(results, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib 없음 — plot 생략")
        return
    n = len(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for res in results:
        lbl = f"{res['src']} [{res['frame']}]"
        axes[0].plot(res["recon_xy"][:, 0], res["recon_xy"][:, 1], "-o",
                     ms=2, label=lbl)
        if res["gt_xy"] is not None:
            axes[0].plot(res["gt_xy"][:, 0], res["gt_xy"][:, 1], "k--",
                         label=f"{res['src']} GT")
    axes[0].set_title("재구성 궤적 (dead-reckoning, EKF 없음)")
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
    axes[0].axis("equal"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7)

    for res in results:
        axes[1].plot(res["disp_xy"], "-", label=f"{res['src']} [{res['frame']}]")
    axes[1].axhline(1.5, color="r", ls=":", label="정상 보행 상한 ~1.5m")
    axes[1].set_title("window별 |disp_xy|")
    axes[1].set_xlabel("window idx"); axes[1].set_ylabel("|disp_xy| (m)")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=110)
    print(f"\n  [OK] plot 저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True,
                    help="모델 폴더 (config.json + checkpoints/best.pth + norm_*.npy)")
    ap.add_argument("--oxiod", default=None,
                    help="OxIOD 시퀀스 폴더 또는 imu0_resampled.npy 경로")
    ap.add_argument("--android", default=None, help="Android ImuTestActivity CSV")
    ap.add_argument("--frame", default="ga", choices=["ga", "body", "yaw"],
                    help="Android 입력 프레임 (ga=학습동일, body=무회전, yaw=시작회전1개)")
    ap.add_argument("--linacc_scale", type=float, default=1.0,
                    help="Android linAcc 계수 (1.0=m/s², 0.10194=1/9.81=g)")
    ap.add_argument("--calib_sec", type=float, default=2.0,
                    help="Android 영점보정 구간 (초). 0=비활성")
    ap.add_argument("--diagnose", action="store_true",
                    help="Android 를 {scale 1.0, 1/9.81} × {frame ga, body} 매트릭스 sweep")
    ap.add_argument("--plot", default=None, help="궤적/disp plot PNG 경로")
    args = ap.parse_args()

    print(f"[1] 모델 로드: {args.model_dir}")
    net, model_para, mean, std = load_model(args.model_dir)
    print(f"  use_classifier={model_para.get('use_classifier')}  "
          f"feature_dim={model_para.get('feature_dim')}")
    print(f"  norm_mean={mean.round(4).tolist()}")
    print(f"  norm_std ={std.round(4).tolist()}")

    results = []

    # OxIOD — 하니스 검증 baseline
    if args.oxiod:
        p = Path(args.oxiod)
        npy = p if p.suffix == ".npy" else p / "imu0_resampled.npy"
        print(f"\n[2] OxIOD 로드: {npy}")
        ox = load_oxiod(npy)
        res = evaluate(net, ox, mean, std, frame="ga")
        print_result(res)
        results.append(res)

    # Android
    if args.android:
        print(f"\n[3] Android 로드: {args.android}")
        if args.diagnose:
            print("  --diagnose: 단위 스케일 × frame 매트릭스 sweep")
            combos = [
                (1.0,            "ga",   "m/s² 그대로 + 학습동일 프레임 (현재 단말 동작)"),
                (1.0 / GRAVITY,  "ga",   "g 단위 변환 + 학습동일 프레임 (단위가설)"),
                (1.0,            "body", "m/s² + 무회전 (프레임 변환 영향 격리)"),
                (1.0 / GRAVITY,  "body", "g 단위 + 무회전"),
            ]
            for scale, frame, desc in combos:
                print(f"\n  >> {desc}")
                ad = load_android(args.android, calib_sec=args.calib_sec,
                                  linacc_scale=scale)
                res = evaluate(net, ad, mean, std, frame=frame)
                print_result(res)
                results.append(res)
        else:
            ad = load_android(args.android, calib_sec=args.calib_sec,
                              linacc_scale=args.linacc_scale)
            res = evaluate(net, ad, mean, std, frame=args.frame)
            print_result(res)
            results.append(res)

    if not results:
        print("\n[!] --oxiod 또는 --android 중 최소 하나 필요")
        sys.exit(1)

    # 종합 판정
    print(f"\n{'='*72}\n  종합\n{'='*72}")
    ox_res = next((r for r in results if r["src"].startswith("OxIOD")), None)
    if ox_res and ox_res["rmse"] is not None:
        ok = ox_res["rmse"] < 2.5
        print(f"  하니스 검증: OxIOD RMSE {ox_res['rmse']:.2f}m "
              f"→ {'정상 (모델+norm+전처리 신뢰 가능)' if ok else '비정상 (점검 필요)'}")
    for r in results:
        if r["src"].startswith("Android"):
            med = np.median(r["disp_xy"]) if len(r["disp_xy"]) else 0.0
            verdict = ("정상 범위" if 0.3 <= med <= 1.8
                       else "과대" if med > 1.8 else "과소")
            print(f"  {r['src']} [{r['frame']}]: window |disp_xy| median "
                  f"{med:.2f}m → {verdict}")
    print("  -> Android median 이 OxIOD 수준(~1m)인 조합 = 올바른 단위/프레임.")
    print("    그 조합이 단말 코드의 InferenceEngine/transformWindowToWorldFrame 와")
    print("    다르면 그것이 8~13배 과대추정의 원인.")

    if args.plot:
        make_plot(results, args.plot)


if __name__ == "__main__":
    main()
