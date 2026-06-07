# -*- coding: utf-8 -*-
"""절대 GT ATE — 단말 마크(est) ↔ 알려진 웨이포인트 좌표(GT) 강체 정렬 후 RMSE.
SE(2) 정렬(회전+평행이동, 스케일=1 고정 → 스케일 오차를 흡수하지 않음).
인자 없으면 self-test(합성)로 분석코드 검증(②-a 드라이런).
  실측: python android_gt_ate.py marks_*.csv waypoints.csv
  검증: python android_gt_ate.py            (self-test)
waypoints.csv 형식: idx,gt_x_m,gt_y_m  (마크와 같은 순서)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
except Exception:
    pass
for fn in ("Malgun Gothic", "맑은 고딕", "Gulim"):
    try:
        font_manager.findfont(fn, fallback_to_default=False); plt.rcParams["font.family"] = fn; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
LOG = Path(r"D:\mobile\imu_android\logs")


def procrustes_se2(est: np.ndarray, gt: np.ndarray):
    """est→gt 로 사상하는 회전 R(2x2)+평행이동 t. 스케일=1(반사 금지). 반환 (R,t,aligned,ate,resid)."""
    assert est.shape == gt.shape and est.shape[1] == 2
    mu_e = est.mean(0); mu_g = gt.mean(0)
    E = est - mu_e; G = gt - mu_g
    W = G.T @ E                      # 2x2 cross-cov (G = R E)
    U, _S, Vt = np.linalg.svd(W)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])  # 반사 제거
    R = U @ D @ Vt
    t = mu_g - R @ mu_e
    aligned = (R @ est.T).T + t
    resid = np.linalg.norm(aligned - gt, axis=1)
    ate = float(np.sqrt((resid ** 2).mean()))
    return R, t, aligned, ate, resid


def load_csv_xy(path, xcol, ycol):
    rows = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split(",")
        if p[0].strip().lower() in ("idx", "i"):  # header
            continue
        rows.append((float(p[xcol]), float(p[ycol])))
    return np.array(rows)


def report(est, gt, title, outname):
    R, t, aligned, ate, resid = procrustes_se2(est, gt)
    theta = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    print(f"  정렬: 회전 {theta:+.1f}°, 평행이동 ({t[0]:+.2f}, {t[1]:+.2f})")
    print(f"  {'wp':>3} {'est_x':>7} {'est_y':>7} | {'gt_x':>7} {'gt_y':>7} | {'잔차(m)':>8}")
    for i in range(len(gt)):
        print(f"  {i+1:>3} {est[i,0]:>7.2f} {est[i,1]:>7.2f} | {gt[i,0]:>7.2f} {gt[i,1]:>7.2f} | {resid[i]:>8.3f}")
    print(f"  → ATE(RMSE) = {ate:.3f} m,  최대잔차 = {resid.max():.3f} m")
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(gt[:, 0], gt[:, 1], "o-", color="#2ca02c", lw=2, ms=8, label="GT 웨이포인트")
    ax.plot(aligned[:, 0], aligned[:, 1], "x--", color="#d62728", lw=1.8, ms=10, label="추정(정렬 후)")
    for i in range(len(gt)):
        ax.plot([aligned[i, 0], gt[i, 0]], [aligned[i, 1], gt[i, 1]], "-", color="#999", lw=1)
        ax.annotate(f"{i+1}", (gt[i, 0], gt[i, 1]), fontsize=9, xytext=(4, 4), textcoords="offset points")
    ax.set_aspect("equal", "datalim"); ax.grid(ls=":", alpha=0.4); ax.legend()
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"{title}\nATE(RMSE)={ate:.2f} m (n={len(gt)})", fontsize=11)
    fig.tight_layout(); out = LOG / outname; fig.savefig(out, dpi=140)
    print(f"  [OK] {out}")
    return ate


def self_test():
    print("[self-test] 합성 GT(L자 5점)에 알려진 회전·평행이동·노이즈 주입 → 정렬이 복원하는지 검증")
    gt = np.array([[0, 0], [0, 8], [16, 8], [16, 0], [8, 0]], float)  # L자형 5 웨이포인트
    th = np.deg2rad(37.0); Rt = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rng_noise = np.array([[0.3, -0.2], [-0.1, 0.25], [0.2, 0.15], [-0.25, -0.1], [0.05, 0.2]])  # 고정 ~0.3m
    est = (Rt @ gt.T).T + np.array([5.0, -3.0]) + rng_noise   # 궤적 프레임: 회전37°+평행이동+노이즈
    print(f"  주입: 회전 +37.0°, 평행이동 (관측계), 점당 노이즈 ~0.25 m")
    ate = report(est, gt, "[self-test] 합성 검증", "android_gt_ate_selftest.png")
    ok = abs(ate - 0.22) < 0.15   # 노이즈 수준이면 통과
    print(f"  [{'PASS' if ok else 'FAIL'}] 복원 회전≈37°, ATE≈노이즈수준({ate:.2f} m) → 분석코드 정상" if ok
          else f"  [FAIL] ATE {ate:.2f} m — 코드 점검 필요")
    return ok


def main():
    if len(sys.argv) >= 3:
        est = load_csv_xy(sys.argv[1], 1, 2)   # marks: idx,est_x,est_y,t_ms
        gt = load_csv_xy(sys.argv[2], 1, 2)    # waypoints: idx,gt_x,gt_y
        if len(est) != len(gt):
            print(f"[오류] 마크 {len(est)}개 ≠ 웨이포인트 {len(gt)}개 — 순서/개수 확인"); return
        print(f"[실측] 마크 {len(est)} ↔ 웨이포인트 {len(gt)}")
        report(est, gt, f"절대 GT ATE — {Path(sys.argv[1]).name}", "android_gt_ate.png")
    else:
        self_test()


if __name__ == "__main__":
    main()
