"""
train_resnet.py
---------------
TLIO ResNet1D 를 ResMLP128 과 동일한 환경에서 학습.

  · 데이터셋   : Oxford (oxford_split) + TLIO golden 증강 (동일)
  · 증강       : bias_noise + gravity_perturb + yaw_rotation (동일)
  · 손실       : Two-stage  MSE(epoch ≤30) → Gaussian NLL(epoch >30) (동일)
  · 옵티마이저 : AdamW, CosineAnnealingLR, warmup (동일)
  · 네트워크   : TLIO ResNet1D (full / small)  ← 유일한 차이점

model_variant:
  "full"  → 원본 TLIO ResNet1D (base_plane=64, fc_dim=512, ~5M params)
  "small" → 소형 ResNet1D      (base_plane/fc_dim 설정 가능, ~460K params)

출력 디렉터리: out_resnet/ 또는 out_resnet_small/
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────────────
_HERE      = Path(__file__).parent                      # src/Network
_TRANS     = _HERE.parent / "Trans"                     # src/Trans
_TLIO_MODEL = Path(r"C:\Users\hs091\Desktop\TLIO-master\TLIO-master\src\network\model_resnet.py")

# TLIO network 폴더를 sys.path 에 넣으면 TLIO 자체 losses.py 와 충돌하므로
# model_resnet.py 만 importlib 로 직접 로드한다.
sys.path.insert(0, str(_HERE))   # src/Network (losses.py, dataset.py)
sys.path.insert(0, str(_TRANS))  # src/Trans
# ──────────────────────────────────────────────────────────────────

import argparse, importlib.util, json, logging, time
import numpy as np
import torch
import torch.nn as nn

from dataset            import build_dataloaders
from losses             import CombinedLoss       # ResMLP128 의 손실 함수 재사용
from model_resnet_small import ResNet1DSmall      # 소형 ResNet1D (base_plane 가변)

# TLIO 원본 ResNet1D 직접 로드 (sys.path 오염 없이)
_spec = importlib.util.spec_from_file_location("model_resnet", _TLIO_MODEL)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BasicBlock1D = _mod.BasicBlock1D
ResNet1D     = _mod.ResNet1D


# ── window_len=100 / 100Hz 기준 inter_dim 계산 ─────────────────────
# Input [B,6,100]
# Conv1d(k=7,s=2,p=3) → 50,  MaxPool(k=3,s=2,p=1) → 25
# ResGroup stride(1,2,2,2) → 25→13→7→4
_INTER_DIM = 4
# ──────────────────────────────────────────────────────────────────


DEFAULT_CONFIG = {
    # ── 모델 변형 선택 ───────────────────────────────────────────────
    # "full"  : 원본 TLIO ResNet1D (base_plane=64, fc_dim=512, ~5M params)
    # "small" : 소형 ResNet1D      (base_plane=19, fc_dim=166, ~460K params)
    "model_variant": "full",
    "resnet_base_plane": 64,
    "resnet_fc_dim":     512,

    # ── 데이터 (ResMLP128 과 동일) ──────────────────────────────────
    "train_dir":       "oxford_split/train",
    "val_dir":         "oxford_split/val",
    "test_dir":        "oxford_split/test",
    "fmt":             "oxford",
    "with_label":      False,
    "window_len":      100,
    "train_stride":    10,
    "eval_stride":     10,
    "batch_size":      128,
    "num_workers":     4,

    # ── TLIO golden 증강 (ResMLP128 과 동일) ─────────────────────────
    "tlio_aug_dir":    r"C:\Users\hs091\Desktop\TLIO-master\golden-new-format-cc-by-nc-with-imus",
    "tlio_aug_stride": 10,

    # ── 손실 (ResMLP128 과 동일) ─────────────────────────────────────
    "loss": {
        "cls_weight":    0.0,
        "dir_weight":    0.0,
        "label_smoothing": 0.0,
        "mse_epochs":    30,       # epoch 1~30: MSE, epoch 31~100: NLL
    },

    # ── 최적화 (ResMLP128 과 동일) ────────────────────────────────────
    "optimizer":  {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-4},
    "scheduler":  {"name": "CosineAnnealingLR", "T_max": 100, "eta_min": 1e-6},
    "warmup_epochs": 5,
    "epochs":     100,
    "grad_clip":  1.0,

    # ── 로깅 / 저장 ────────────────────────────────────────────────
    "log_interval": 50,
    "save_every":   10,
    "output_dir":   "out_resnet",
}


# ── 유틸 ────────────────────────────────────────────────────────────

def setup_logger(log_path):
    logger = logging.getLogger("train_resnet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, path, logger):
    torch.save(state, path)
    logger.info(f"  체크포인트 저장: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler and "scheduler" in ck and ck["scheduler"]:
        scheduler.load_state_dict(ck["scheduler"])
    return ck


class AverageMeter:
    def __init__(self): self.n = 0; self.sum = 0.0
    def update(self, v, k=1): self.n += k; self.sum += v * k
    @property
    def avg(self): return self.sum / max(self.n, 1)


# ── 학습 루프 ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, cfg, logger, epoch):
    """
    ResNet1D 출력: (mean [B,3], logstd [B,3])
    CombinedLoss 는 log_var 를 받으므로  log_var = 2 * logstd  변환 필요.
    (var = exp(log_var) = exp(2*logstd) = sigma^2, sigma = exp(logstd))
    """
    model.train()
    m = {k: AverageMeter() for k in ["reg", "rmse"]}

    for step, batch in enumerate(loader):
        imu, target = batch[0], batch[1]
        imu, target = imu.to(device), target.to(device)

        y_hat, logstd = model(imu)
        log_var = 2.0 * logstd          # logstd → log_var 변환

        # CombinedLoss: pose_logits=None, pose_labels=None → 분류 손실 0
        loss, d = criterion(y_hat, log_var, target, None, None)

        optimizer.zero_grad()
        loss.backward()
        if cfg.get("grad_clip"):
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        B = imu.size(0)
        m["reg"].update(d["reg"], B)
        with torch.no_grad():
            rmse = torch.sqrt(((y_hat - target) ** 2).sum(1).mean()).item()
        m["rmse"].update(rmse, B)

        if (step + 1) % cfg.get("log_interval", 50) == 0:
            logger.info(f"  Epoch {epoch:03d} [{step+1:4d}/{len(loader)}]"
                        f"  reg={d['reg']:.4f}  rmse={m['rmse'].avg:.4f}")

    return {k: v.avg for k, v in m.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    reg_m = AverageMeter()
    errs = []

    for batch in loader:
        imu, target = batch[0], batch[1]
        imu, target = imu.to(device), target.to(device)

        y_hat, logstd = model(imu)
        log_var = 2.0 * logstd

        _, d = criterion(y_hat, log_var, target, None, None)
        reg_m.update(d["reg"], imu.size(0))
        errs.append((y_hat - target).cpu().numpy())

    errs = np.concatenate(errs, 0)
    rmse_xyz = np.sqrt((errs ** 2).mean(0))
    rmse     = float(np.sqrt((errs ** 2).sum(1).mean()))
    return {
        "reg":    reg_m.avg,
        "rmse":   rmse,
        "rmse_x": float(rmse_xyz[0]),
        "rmse_y": float(rmse_xyz[1]),
        "rmse_z": float(rmse_xyz[2]),
    }


# ── 메인 학습 함수 ────────────────────────────────────────────────────

def train(cfg):
    out_dir  = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    logger   = setup_logger(out_dir / "train.log")

    logger.info("=" * 60)
    logger.info("TLIO ResNet1D  (ResMLP128 과 동일 환경)")
    logger.info(f"출력: {out_dir}")

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    device = get_device()
    logger.info(f"디바이스: {device}")

    # ── 데이터 로더 ──────────────────────────────────────────────────
    logger.info("데이터셋 로딩... (Oxford + TLIO golden 증강)")
    loaders = build_dataloaders(
        train_paths  = cfg["train_dir"],
        val_paths    = cfg["val_dir"],
        test_paths   = cfg.get("test_dir"),
        window_len   = cfg["window_len"],
        train_stride = cfg["train_stride"],
        eval_stride  = cfg["eval_stride"],
        batch_size   = cfg["batch_size"],
        num_workers  = cfg["num_workers"],
        fmt          = cfg.get("fmt", "oxford"),
        with_label   = False,
        tlio_aug_dir    = cfg.get("tlio_aug_dir"),
        tlio_aug_stride = cfg.get("tlio_aug_stride", 10),
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    test_loader  = loaders.get("test")

    mean, std = loaders["stats"]
    np.save(out_dir / "norm_mean.npy", mean)
    np.save(out_dir / "norm_std.npy",  std)
    logger.info(f"Train samples: {len(train_loader.dataset):,} | "
                f"Val samples: {len(val_loader.dataset):,}")

    # ── 모델 ─────────────────────────────────────────────────────────
    variant    = cfg.get("model_variant", "full")
    base_plane = cfg.get("resnet_base_plane", 64)
    fc_dim     = cfg.get("resnet_fc_dim", 512)

    if variant == "small":
        model = ResNet1DSmall(
            in_dim      = 6,
            out_dim     = 3,
            group_sizes = [2, 2, 2, 2],
            inter_dim   = _INTER_DIM,
            base_plane  = base_plane,
            fc_dim      = fc_dim,
        ).to(device)
    else:
        model = ResNet1D(
            block_type  = BasicBlock1D,
            in_dim      = 6,
            out_dim     = 3,
            group_sizes = [2, 2, 2, 2],
            inter_dim   = _INTER_DIM,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"ResNet1D ({variant}) 파라미터 수: {n_params:,}")

    # ── 손실 ─────────────────────────────────────────────────────────
    mse_epochs = cfg["loss"].get("mse_epochs", 30)
    criterion = CombinedLoss(
        cls_weight  = 0.0,
        dir_weight  = 0.0,
        use_nll     = (mse_epochs <= 0),
    ).to(device)
    if mse_epochs > 0:
        logger.info(f"[two-stage] Stage 1: MSE (epoch 1~{mse_epochs})")

    # ── 옵티마이저 / 스케줄러 ─────────────────────────────────────────
    opt_cfg   = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
    )
    sch_cfg   = cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = sch_cfg.get("T_max", cfg["epochs"]),
        eta_min = sch_cfg.get("eta_min", 1e-6),
    )

    # ── Resume: 최신 체크포인트 자동 탐지 ────────────────────────────
    best_val    = float("inf")
    start_epoch = 1
    warmup_epochs = cfg.get("warmup_epochs", 5)
    base_lr       = opt_cfg["lr"]

    # epoch_XXX.pth 중 가장 최신 것을 자동으로 찾아 이어서 학습
    saved = sorted(ckpt_dir.glob("epoch_*.pth"))
    if saved:
        latest = saved[-1]
        ck = load_checkpoint(latest, model, optimizer, scheduler, device)
        start_epoch = ck.get("epoch", 0) + 1
        best_val    = ck.get("val_rmse", float("inf"))
        # NLL 구간이면 criterion 복원
        if mse_epochs > 0 and start_epoch > mse_epochs + 1:
            criterion.use_nll = True
        logger.info(f"[resume] {latest.name} 에서 재개 → epoch {start_epoch} 부터")
        logger.info(f"[resume] best_val={best_val:.4f}, use_nll={criterion.use_nll}")
    else:
        logger.info(f"[새 학습] 체크포인트 없음 → epoch 1 부터 시작")

    logger.info(f"총 {cfg['epochs']} epoch (시작: {start_epoch})")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        # 선형 워밍업
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_lr = 1e-5 + (base_lr - 1e-5) * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # MSE → NLL 전환
        if mse_epochs > 0 and epoch == mse_epochs + 1 and not criterion.use_nll:
            criterion.use_nll = True
            best_val = float("inf")      # NLL 스케일 변환으로 best 리셋
            logger.info(f"[two-stage] Stage 2: NLL (epoch {epoch}) — best_val 리셋")

        tr = train_one_epoch(model, train_loader, criterion,
                             optimizer, device, cfg, logger, epoch)
        vl = evaluate(model, val_loader, criterion, device)

        if epoch > warmup_epochs:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:03d}/{cfg['epochs']}  lr={lr_now:.2e}  "
            f"train[reg={tr['reg']:.4f} rmse={tr['rmse']:.4f}]  "
            f"val[rmse={vl['rmse']:.4f} "
            f"xyz=({vl['rmse_x']:.3f},{vl['rmse_y']:.3f},{vl['rmse_z']:.3f})]"
            f"  ({time.time()-t0:.1f}s)"
        )

        # best 저장
        if vl["rmse"] < best_val:
            best_val = vl["rmse"]
            save_checkpoint({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_rmse": vl["rmse"],
                "config":   cfg,
            }, ckpt_dir / "best.pth", logger)

        # 주기적 저장
        if epoch % cfg.get("save_every", 10) == 0:
            save_checkpoint({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_rmse": vl["rmse"],
                "config":   cfg,
            }, ckpt_dir / f"epoch_{epoch:03d}.pth", logger)

    logger.info(f"학습 완료. Best val RMSE: {best_val:.4f} m")

    # ── 테스트 평가 ──────────────────────────────────────────────────
    if test_loader is not None:
        logger.info("=" * 60)
        logger.info("테스트 평가 (best.pth 로드)")
        load_checkpoint(ckpt_dir / "best.pth", model, device=device)
        tm = evaluate(model, test_loader, criterion, device)
        logger.info(f"Test  rmse={tm['rmse']:.4f}  "
                    f"xyz=({tm['rmse_x']:.3f},{tm['rmse_y']:.3f},{tm['rmse_z']:.3f})")


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     type=str,   default=None)
    ap.add_argument("--output_dir", type=str,   default=None)
    ap.add_argument("--epochs",     type=int,   default=None)
    ap.add_argument("--batch_size", type=int,   default=None)
    ap.add_argument("--lr",         type=float, default=None)
    return ap.parse_args()


def merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            merge(base[k], v)
        else:
            base[k] = v


def main():
    args = parse_args()
    cfg  = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.config:
        with open(args.config) as f:
            merge(cfg, json.load(f))
    over = {}
    if args.output_dir: over["output_dir"] = args.output_dir
    if args.epochs:     over["epochs"]     = args.epochs
    if args.batch_size: over["batch_size"] = args.batch_size
    if args.lr:         over["optimizer"]  = {"lr": args.lr}
    merge(cfg, over)
    train(cfg)


if __name__ == "__main__":
    main()
