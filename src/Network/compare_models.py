"""
compare_models.py
-----------------
ResMLP128 / ResNet1D-Small / ResNet1D-Full 3개 모델의
경로 예측을 Oxford 카테고리별로 비교 시각화.

실행:
  cd src/Network
  python compare_models.py

출력: doc/compare/{category}.png
  Left  : X-Y 평면 경로  (GT + 3 models)
  Right : Z 방향 누적 변위 (GT + 3 models)
"""

import os, sys, json, importlib.util
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# ── 경로 설정 ──────────────────────────────────────────────────────────
_HERE  = Path(__file__).parent          # src/Network
_TRANS = _HERE.parent / "Trans"         # src/Trans
_ROOT  = _HERE.parent.parent            # 프로젝트 루트
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_TRANS))

from model_twolayer     import TwoLayerModel
from model_resnet_small import ResNet1DSmall

# TLIO 원본 ResNet1D (importlib — sys.path 오염 없이)
_TLIO_MODEL = Path(r"C:\Users\hs091\Desktop\TLIO-master\TLIO-master\src\network\model_resnet.py")
if _TLIO_MODEL.exists():
    _spec = importlib.util.spec_from_file_location("model_resnet", _TLIO_MODEL)
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BasicBlock1D = _mod.BasicBlock1D
    ResNet1D     = _mod.ResNet1D
    _TLIO_AVAIL  = True
    print(f"[INFO] TLIO ResNet1D 로드: {_TLIO_MODEL}")
else:
    _TLIO_AVAIL = False
    print("[WARN] TLIO model_resnet.py 없음 → ResNet1DSmall(base_plane=64) 대체 사용")

# ── 설정 ──────────────────────────────────────────────────────────────
OXFORD_SPLIT_DIR = _HERE / "oxford_split"
OUTPUT_DIR       = _ROOT / "doc" / "compare"
WINDOW_LEN       = 100
_INTER_DIM       = 4   # window=100, 100Hz: 100→50→25→13→7→4

MODEL_DIRS = {
    "ResMLP128":      _HERE / "out_regression",
    "ResNet1D-Small": _HERE / "out_resnet_small",
    "ResNet1D-Full":  _HERE / "out_resnet",
}

CATEGORIES = [
    "handbag", "handheld", "large_scale",
    "multi_devices", "multi_users", "pocket",
    "running", "slow_walking", "trolley",
]

# 색상: (color, linestyle, marker, linewidth)
MODEL_STYLE = {
    "GT":             ("#2166ac", "-",  "o",  2.2),
    "ResMLP128":      ("#d7191c", "--", "s",  1.8),
    "ResNet1D-Small": ("#1a9641", "-.", "^",  1.8),
    "ResNet1D-Full":  ("#f77f00", ":",  "D",  1.8),
}

# ──────────────────────────────────────────────────────────────────────
#  모델 로드 헬퍼
# ──────────────────────────────────────────────────────────────────────
def _load_ckpt(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def load_resmlp(model_dir: Path, device):
    """ResMLP128 (TwoLayerModel)"""
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    model = _load_ckpt(TwoLayerModel(cfg["model"]).to(device),
                       model_dir / "checkpoints/best.pth", device)
    mean = np.load(model_dir / "norm_mean.npy")
    std  = np.load(model_dir / "norm_std.npy")
    return model, mean, std


def load_resnet_small(model_dir: Path, device):
    """ResNet1D-Small (ResNet1DSmall, base_plane=19, fc_dim=166)"""
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    bp = cfg.get("resnet_base_plane", 19)
    fd = cfg.get("resnet_fc_dim",     166)
    model = ResNet1DSmall(in_dim=6, out_dim=3, group_sizes=[2, 2, 2, 2],
                          inter_dim=_INTER_DIM, base_plane=bp, fc_dim=fd).to(device)
    model = _load_ckpt(model, model_dir / "checkpoints/best.pth", device)
    mean  = np.load(model_dir / "norm_mean.npy")
    std   = np.load(model_dir / "norm_std.npy")
    return model, mean, std


def load_resnet_full(model_dir: Path, device):
    """ResNet1D-Full (TLIO 원본 or ResNet1DSmall 대체)"""
    if _TLIO_AVAIL:
        model = ResNet1D(block_type=BasicBlock1D, in_dim=6, out_dim=3,
                         group_sizes=[2, 2, 2, 2], inter_dim=_INTER_DIM).to(device)
    else:
        model = ResNet1DSmall(in_dim=6, out_dim=3, group_sizes=[2, 2, 2, 2],
                              inter_dim=_INTER_DIM, base_plane=64, fc_dim=512).to(device)
    model = _load_ckpt(model, model_dir / "checkpoints/best.pth", device)
    mean  = np.load(model_dir / "norm_mean.npy")
    std   = np.load(model_dir / "norm_std.npy")
    return model, mean, std


# ──────────────────────────────────────────────────────────────────────
#  IMU 전처리 & 궤적 재구성
# ──────────────────────────────────────────────────────────────────────
def window_to_gravity_aligned(acc_body, gyr_body, quat_data, start, end):
    """body frame → gravity-aligned local frame, 시작 yaw 반환."""
    R_all     = R.from_quat(quat_data[start:end]).as_matrix().astype(np.float32)
    r_start   = R.from_quat(quat_data[start])
    yaw0      = r_start.as_euler("zyx", degrees=False)[0]
    R_yaw_inv = R.from_euler("z", yaw0).inv().as_matrix().astype(np.float32)

    acc_w  = np.einsum("tij,tj->ti", R_all, acc_body[start:end])
    gyr_w  = np.einsum("tij,tj->ti", R_all, gyr_body[start:end])
    acc_ga = (R_yaw_inv @ acc_w.T).T
    gyr_ga = (R_yaw_inv @ gyr_w.T).T
    return np.concatenate([acc_ga, gyr_ga], axis=1).astype(np.float32), yaw0


@torch.no_grad()
def reconstruct(npy_path, model, mean, std, device, is_resnet: bool):
    """비겹침 윈도우로 displacement 예측 → 누적 경로."""
    data      = np.load(npy_path)
    gyr_body  = data[:, 1:4].astype(np.float32)
    acc_body  = data[:, 4:7].astype(np.float32)
    quat_data = data[:, 14:18].astype(np.float32)
    pos_data  = data[:, 18:21].astype(np.float32)
    T = len(acc_body)

    gt_pts   = [np.zeros(3, dtype=np.float32)]
    pred_pts = [np.zeros(3, dtype=np.float32)]

    for start in range(0, T - WINDOW_LEN, WINDOW_LEN):
        end = start + WINDOW_LEN

        # GT 누적
        gt_pts.append(gt_pts[-1] + (pos_data[end] - pos_data[start]))

        # 예측
        imu_np, yaw0 = window_to_gravity_aligned(acc_body, gyr_body, quat_data, start, end)
        imu_t = torch.tensor(((imu_np - mean) / std).T[None], dtype=torch.float32, device=device)

        if is_resnet:
            y_hat, _ = model(imu_t)
            pred_local = y_hat[0].cpu().numpy()
        else:
            pred_local, _, _ = model(imu_t)
            pred_local = pred_local[0].cpu().numpy()

        R_yaw = R.from_euler("z", yaw0).as_matrix().astype(np.float32)
        pred_pts.append(pred_pts[-1] + R_yaw @ pred_local)

    return np.array(gt_pts, dtype=np.float32), np.array(pred_pts, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
#  데이터 탐색 (test → val → train 우선순위)
# ──────────────────────────────────────────────────────────────────────
def find_sequence(category):
    for split in ["test", "val", "train"]:
        split_dir = OXFORD_SPLIT_DIR / split
        if not split_dir.exists():
            continue
        for d in sorted(split_dir.iterdir()):
            if d.is_dir() and category in d.name:
                npy = d / "imu0_resampled.npy"
                if npy.exists():
                    return npy, d.name, split
    return None, None, None


# ──────────────────────────────────────────────────────────────────────
#  시각화
# ──────────────────────────────────────────────────────────────────────
def rmse_of(gt, pred):
    err = gt - pred
    return float(np.sqrt((err**2).sum(1).mean())), \
           float(np.sqrt((err[:, :2]**2).sum(1).mean()))


def plot_comparison(category, seq_name, split, trajs, output_dir):
    """
    trajs: OrderedDict  { "GT": arr, "ResMLP128": arr, ... }
    """
    gt   = trajs["GT"]
    t_ax = np.arange(len(gt))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("#f8f8f8")

    rmse_summary = []

    # ── Left: X-Y 경로 ──────────────────────────────────────────────
    ax = axes[0]
    for name, traj in trajs.items():
        c, ls, mk, lw = MODEL_STYLE[name]
        if name == "GT":
            lbl = "GT"
        else:
            r, rxy = rmse_of(gt, traj)
            lbl = f"{name}  (RMSE {r:.3f} m | xy {rxy:.3f} m)"
            rmse_summary.append((name, r, rxy))
        ax.plot(traj[:, 0], traj[:, 1],
                color=c, linestyle=ls, linewidth=lw,
                marker=mk, markersize=2.5, label=lbl, zorder=3)

    ax.plot(*gt[0, :2],  "k^", markersize=9,  zorder=6, label="Start")
    ax.plot(*gt[-1, :2], "b*", markersize=11, zorder=6, label="End (GT)")
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(f"X–Y  |  {seq_name}", fontsize=10)
    ax.set_aspect("equal")
    ax.legend(fontsize=8.5, loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f2f2f2")

    # ── Right: Z 방향 ───────────────────────────────────────────────
    ax2 = axes[1]
    for name, traj in trajs.items():
        c, ls, mk, lw = MODEL_STYLE[name]
        ax2.plot(t_ax, traj[:, 2],
                 color=c, linestyle=ls, linewidth=lw,
                 marker=mk, markersize=2.5, label=name, zorder=3)
    ax2.set_xlabel("Window index (×1 sec)", fontsize=11)
    ax2.set_ylabel("Cumulative Z displacement (m)", fontsize=11)
    ax2.set_title("Z Direction", fontsize=10)
    ax2.legend(fontsize=8.5, framealpha=0.85)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#f2f2f2")

    # ── 상단 제목 (카테고리 + RMSE 요약)
    rmse_str = "   |   ".join(f"{n}: {r:.3f} m" for n, r, _ in rmse_summary)
    fig.suptitle(
        f"Oxford  [{category}]  ({split} set)\n{rmse_str}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    out_path = output_dir / f"{category}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 저장: {out_path}")
    return rmse_summary


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    print(">> 모델 로드 중...")
    resmlp,       mn1, sd1 = load_resmlp(      MODEL_DIRS["ResMLP128"],      device)
    resnet_small, mn2, sd2 = load_resnet_small(MODEL_DIRS["ResNet1D-Small"], device)
    resnet_full,  mn3, sd3 = load_resnet_full( MODEL_DIRS["ResNet1D-Full"],  device)
    print("  [OK] 3개 모델 로드 완료\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for cat in CATEGORIES:
        npy_path, seq_name, split = find_sequence(cat)
        if npy_path is None:
            print(f"[{cat}] ⚠ 시퀀스 없음, 스킵")
            continue

        print(f"[{cat}]  {seq_name}  ({split} set)")
        gt,  p1 = reconstruct(npy_path, resmlp,       mn1, sd1, device, is_resnet=False)
        _,   p2 = reconstruct(npy_path, resnet_small, mn2, sd2, device, is_resnet=True)
        _,   p3 = reconstruct(npy_path, resnet_full,  mn3, sd3, device, is_resnet=True)

        trajs = {
            "GT":             gt,
            "ResMLP128":      p1,
            "ResNet1D-Small": p2,
            "ResNet1D-Full":  p3,
        }
        summary = plot_comparison(cat, seq_name, split, trajs, OUTPUT_DIR)
        all_results.append((cat, seq_name, split, summary))

    # ── 결과 요약 테이블
    print("\n" + "=" * 80)
    print(f"{'Category':<18} {'Split':<7} {'ResMLP128':>14} {'ResNet1D-S':>14} {'ResNet1D-F':>14}")
    print("-" * 80)
    for cat, seq, split, rl in all_results:
        vals = {n: r for n, r, _ in rl}
        r1 = vals.get("ResMLP128",      float("nan"))
        r2 = vals.get("ResNet1D-Small", float("nan"))
        r3 = vals.get("ResNet1D-Full",  float("nan"))
        best = min(r1, r2, r3)
        def fmt(v):
            mark = " ◀" if abs(v - best) < 1e-9 else "  "
            return f"{v:>8.4f} m{mark}"
        print(f"{cat:<18} {split:<7} {fmt(r1)} {fmt(r2)} {fmt(r3)}")
    print("=" * 80)
    print(f"\n[완료] 저장 위치: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
