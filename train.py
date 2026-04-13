"""
train.py  (Phase 2: pose-conditioned regression)
------------------------------------------------
- Classifier 복원 + regression 에 pose 정보 직접 주입
- 학습 시 teacher forcing (GT 라벨), 평가 시 predicted softmax
- Best.pth 기준: val rmse_hard (handbag + pocket)
- Two-stage: MSE → NLL
"""

# Windows에서 DataLoader worker가 spawn될 때 OpenMP 라이브러리 중복 로드로
# 인한 충돌을 방지. torch import 전에 반드시 설정해야 함.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse, json, logging, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import build_dataloaders, CLASS_NAMES, NUM_CLASSES
from losses import CombinedLoss, compute_class_weights
from model_twolayer import TwoLayerModel


DEFAULT_CONFIG = {
    "train_dir": "dataset_split/train",
    "val_dir":   "dataset_split/val",
    "test_dir":  "dataset_split/test",
    "window_len":   100,
    "train_stride": 10,
    "eval_stride":  10,
    "batch_size":   256,
    "num_workers":  0,
    "model": {
        # 14ch (gravity+attitude) + 논문 ResMLP128 설정: feature_dim=128, layer_num=6, dropout=0.2
        "input_len": 100, "input_channel": 14, "patch_len": 25,
        "feature_dim": 128, "out_dim": 3, "active_func": "GELU",
        "extractor": {"name": "ResMLP", "layer_num": 6,
                      "expansion": 2, "dropout": 0.2},
        "reg": {"name": "SimpleMean", "layer_num": 3, "dropout": 0.2},
        "use_classifier": False,
    },
    "loss": {
        "cls_weight": 0.0,
        "dir_weight": 0.0,
        "label_smoothing": 0.0,
        "use_class_weights": False,
        "mse_epochs": 60,   # 논문: MSE로 수렴 후 NLL로 전환 (2단계)
    },
    # 논문: Adam, lr=5e-4 (AdamW X, weight_decay X)
    "optimizer": {"name": "Adam", "lr": 5e-4},
    "scheduler": {"name": "CosineAnnealingLR", "T_max": 150, "eta_min": 1e-6},
    "warmup_epochs": 5,
    "epochs": 200,
    "grad_clip": 1.0,
    "log_interval": 50,
    "save_every": 10,
    "early_stopping": 60,
    "output_dir": "out_14ch_128",
    "hard_classes": ["handbag", "pocket"],
}


def setup_logger(log_path):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler();       ch.setFormatter(fmt); logger.addHandler(ch)
    return logger


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, path, logger):
    torch.save(state, path)
    logger.info(f"  ckpt 저장: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"])
    if optimizer and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler and "scheduler" in ck and ck["scheduler"]:
        scheduler.load_state_dict(ck["scheduler"])
    return ck


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class AverageMeter:
    def __init__(self): self.n = 0; self.sum = 0.0
    def update(self, v, k=1): self.n += k; self.sum += v * k
    @property
    def avg(self): return self.sum / max(self.n, 1)


def hard_class_indices(names):
    return [CLASS_NAMES.index(n) for n in names if n in CLASS_NAMES]


def collect_train_labels(loader):
    lbls = []
    for _, _, lb in loader:
        lbls.append(lb)
    return torch.cat(lbls, dim=0)


def compute_tf_ratio(epoch, total_epochs, tf_start=1.0, tf_end=0.0, warmup=40):
    """Scheduled sampling: epoch 1~warmup은 1.0, 이후 선형 감소 → tf_end."""
    if epoch <= warmup:
        return tf_start
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return max(tf_end, tf_start - progress * (tf_start - tf_end))


def train_one_epoch(model, loader, criterion, optimizer, device, cfg, logger, epoch):
    model.train()
    use_cls = cfg["model"].get("use_classifier", False)
    tf_ratio = compute_tf_ratio(epoch, cfg["epochs"]) if use_cls else 0.0
    m = {k: AverageMeter() for k in ["reg", "dir", "cls", "rmse"]}
    for step, (imu, target, label) in enumerate(loader):
        imu, target, label = imu.to(device), target.to(device), label.to(device)

        y_hat, log_var, pose_logits = model(imu,
                                            pose_labels=label if use_cls else None,
                                            tf_ratio=tf_ratio)
        loss, d = criterion(y_hat, log_var, target, pose_logits, label)

        optimizer.zero_grad()
        loss.backward()
        if cfg.get("grad_clip"):
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        B = imu.size(0)
        m["reg"].update(d["reg"], B)
        m["dir"].update(d["dir"], B)
        m["cls"].update(d["cls"], B)
        with torch.no_grad():
            rmse = torch.sqrt(((y_hat - target) ** 2).sum(1).mean()).item()
        m["rmse"].update(rmse, B)

        if (step + 1) % cfg.get("log_interval", 50) == 0:
            logger.info(f"  Epoch {epoch:03d} [{step+1:4d}/{len(loader)}]  "
                        f"reg={d['reg']:.4f}  dir={d['dir']:.4f}  rmse={m['rmse'].avg:.4f}")

    return {k: v.avg for k, v in m.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device, hard_idx):
    model.eval()
    reg_m = AverageMeter(); cls_m = AverageMeter()
    errs, labels, preds = [], [], []
    for imu, target, label in loader:
        imu, target, label = imu.to(device), target.to(device), label.to(device)
        y_hat, log_var, pose_logits = model(imu)
        _, d = criterion(y_hat, log_var, target, pose_logits, label)
        B = imu.size(0)
        reg_m.update(d["reg"], B); cls_m.update(d["cls"], B)
        errs.append((y_hat - target).cpu().numpy())
        labels.append(label.cpu().numpy())
        # 분류기 없으면 pred를 label과 같다고 처리 (acc=1.0으로 고정, 의미 없음)
        if pose_logits is not None:
            preds.append(pose_logits.argmax(1).cpu().numpy())
        else:
            preds.append(label.cpu().numpy())

    errs = np.concatenate(errs, 0)
    labels = np.concatenate(labels, 0)
    preds = np.concatenate(preds, 0)
    rmse_xyz = np.sqrt((errs ** 2).mean(0))
    rmse = float(np.sqrt((errs ** 2).sum(1).mean()))
    acc = float((preds == labels).mean())
    hard_mask = np.isin(labels, hard_idx)
    rmse_hard = float(np.sqrt((errs[hard_mask] ** 2).sum(1).mean())) \
        if hard_mask.any() else float("nan")

    return {"reg": reg_m.avg, "cls": cls_m.avg, "rmse": rmse,
            "rmse_x": float(rmse_xyz[0]), "rmse_y": float(rmse_xyz[1]),
            "rmse_z": float(rmse_xyz[2]), "rmse_hard": rmse_hard, "acc": acc}


def train(cfg):
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    logger = setup_logger(out_dir / "train.log")
    logger.info("=" * 60)
    logger.info("Phase 2: pose-conditioned regression")
    logger.info(f"출력: {out_dir}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    device = get_device()
    logger.info(f"디바이스: {device}")

    logger.info("데이터셋 로딩...")
    # input_channel에 따라 입력 컬럼 자동 선택
    from dataset import IMU_COLS_6, IMU_COLS_14
    _ch = cfg["model"]["input_channel"]
    _input_cols = IMU_COLS_6 if _ch == 6 else (IMU_COLS_14 if _ch == 14 else None)
    logger.info(f"입력 채널: {_ch}ch → {len(_input_cols) if _input_cols else '기본값'}개 컬럼")
    _aug = cfg.get("augment", {})
    loaders = build_dataloaders(
        train_paths=cfg["train_dir"], val_paths=cfg["val_dir"],
        test_paths=cfg.get("test_dir"),
        window_len=cfg["window_len"],
        train_stride=cfg["train_stride"], eval_stride=cfg["eval_stride"],
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        normalize=True,
        input_cols=_input_cols,
        augment_noise=_aug.get("noise", False),
        noise_std=_aug.get("noise_std", 0.02),
        augment_scale=_aug.get("scale", False),
        scale_range=tuple(_aug.get("scale_range", [0.8, 1.25])),
    )
    train_loader = loaders["train"]; val_loader = loaders["val"]
    test_loader = loaders.get("test")

    mean, std = loaders["stats"]
    np.save(out_dir / "norm_mean.npy", mean)
    np.save(out_dir / "norm_std.npy",  std)

    model = TwoLayerModel(cfg["model"]).to(device)
    logger.info(f"파라미터 수: {count_parameters(model):,}")

    # class weights (분류기 사용 시에만)
    use_classifier = cfg["model"].get("use_classifier", False)
    class_weights = None
    if use_classifier and cfg["loss"].get("use_class_weights", True):
        logger.info("train set 클래스 가중치 계산...")
        all_lbls = collect_train_labels(train_loader)
        class_weights = compute_class_weights(all_lbls, NUM_CLASSES).to(device)
        logger.info(f"  weights: {class_weights.cpu().numpy().round(3)}")

    mse_epochs = cfg["loss"].get("mse_epochs", 0)
    criterion = CombinedLoss(
        cls_weight=cfg["loss"].get("cls_weight", 0.0),
        dir_weight=cfg["loss"].get("dir_weight", 0.0),
        class_weights=class_weights,
        label_smoothing=cfg["loss"].get("label_smoothing", 0.0),
        use_nll=(mse_epochs <= 0),
    ).to(device)

    if mse_epochs > 0:
        logger.info(f"[two-stage] Stage 1: MSE (epoch 1~{mse_epochs})")
    if not use_classifier:
        logger.info("[LLIO mode] Pose classifier OFF - pure regression")

    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opt_cfg["lr"],
                                  weight_decay=opt_cfg.get("weight_decay", 1e-4)) \
        if opt_cfg["name"] == "AdamW" else \
        torch.optim.Adam(model.parameters(), lr=opt_cfg["lr"])

    sch_cfg = cfg.get("scheduler", {})
    scheduler = None
    if sch_cfg.get("name") == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sch_cfg.get("T_max", cfg["epochs"]),
            eta_min=sch_cfg.get("eta_min", 1e-6))
    elif sch_cfg.get("name") == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=sch_cfg.get("patience", 10))

    hard_idx = hard_class_indices(cfg.get("hard_classes", ["handbag", "pocket"]))
    logger.info(f"Hard classes: {[CLASS_NAMES[i] for i in hard_idx]} → {hard_idx}")

    best_val = float("inf")
    early_cnt = 0
    start_epoch = 1

    if cfg.get("resume"):
        p = Path(cfg["resume"])
        if not p.exists(): raise FileNotFoundError(p)
        logger.info(f"[resume] {p}")
        ck = load_checkpoint(p, model, optimizer, scheduler, device)
        start_epoch = ck.get("epoch", 0) + 1
        best_val = ck.get("val_rmse", float("inf"))
        if mse_epochs > 0 and start_epoch > mse_epochs and not criterion.use_nll:
            criterion.use_nll = True
            logger.info("[resume] NLL 자동 활성화")
        logger.info(f"[resume] start_epoch={start_epoch}, best={best_val:.4f}")

    logger.info(f"총 {cfg['epochs']} epoch (시작 {start_epoch})")

    warmup_epochs = cfg.get("warmup_epochs", 0)
    base_lr = cfg["optimizer"]["lr"]

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        # 선형 워밍업: epoch 1~warmup_epochs 동안 lr을 1e-5 → base_lr
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_lr = 1e-5 + (base_lr - 1e-5) * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        if mse_epochs > 0 and epoch == mse_epochs + 1 and not criterion.use_nll:
            criterion.use_nll = True
            logger.info(f"[two-stage] Stage 2: NLL (epoch {epoch})")
            best_val = float("inf"); early_cnt = 0
            logger.info("  best_val 리셋 (스케일 변경)")

        tr = train_one_epoch(model, train_loader, criterion, optimizer,
                             device, cfg, logger, epoch)
        vl = evaluate(model, val_loader, criterion, device, hard_idx)

        if scheduler is not None:
            if epoch > warmup_epochs:  # 워밍업 중에는 scheduler.step() 스킵
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(vl["rmse_hard"])
                else:
                    scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        use_cls = cfg["model"].get("use_classifier", False)
        logger.info(
            f"Epoch {epoch:03d}/{cfg['epochs']}  lr={lr_now:.2e}  "
            f"train[reg={tr['reg']:.4f} dir={tr['dir']:.4f} rmse={tr['rmse']:.4f}]  "
            f"val[rmse={vl['rmse']:.4f} rmse_hard={vl['rmse_hard']:.4f} "
            f"xyz=({vl['rmse_x']:.3f},{vl['rmse_y']:.3f},{vl['rmse_z']:.3f})"
            + (f" acc={vl['acc']:.3f}" if use_cls else "") +
            f"]  ({time.time()-t0:.1f}s)"
        )

        # best model 기준: 전체 val rmse (rmse_hard는 handbag/pocket만 반영해
        # underfitted E1 모델을 계속 선택하는 문제 발생)
        val_metric = vl["rmse"]
        if val_metric < best_val:
            best_val = val_metric
            early_cnt = 0
            save_checkpoint({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "val_rmse_hard": vl["rmse_hard"], "val_rmse": vl["rmse"], "config": cfg,
            }, ckpt_dir / "best.pth", logger)
        else:
            early_cnt += 1

        if epoch % cfg.get("save_every", 10) == 0:
            save_checkpoint({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "val_rmse_hard": val_metric, "val_rmse": vl["rmse"], "config": cfg,
            }, ckpt_dir / f"epoch_{epoch:03d}.pth", logger)

        if cfg.get("early_stopping") and early_cnt >= cfg["early_stopping"]:
            logger.info(f"Early stopping: {cfg['early_stopping']} epoch 개선 없음")
            break

    logger.info(f"학습 완료. Best val rmse_hard: {best_val:.4f}")

    if test_loader is not None:
        logger.info("=" * 60)
        logger.info("테스트 평가 (best.pth 로드)")
        load_checkpoint(ckpt_dir / "best.pth", model, device=device)
        tm = evaluate(model, test_loader, criterion, device, hard_idx)
        logger.info(f"Test  rmse={tm['rmse']:.4f}  rmse_hard={tm['rmse_hard']:.4f}  "
                    f"xyz=({tm['rmse_x']:.3f},{tm['rmse_y']:.3f},{tm['rmse_z']:.3f})  "
                    f"acc={tm['acc']:.3f}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--train_dir", type=str, default=None)
    ap.add_argument("--val_dir",   type=str, default=None)
    ap.add_argument("--test_dir",  type=str, default=None)
    ap.add_argument("--output_dir",type=str, default=None)
    ap.add_argument("--epochs",    type=int, default=None)
    ap.add_argument("--batch_size",type=int, default=None)
    ap.add_argument("--lr",        type=float, default=None)
    ap.add_argument("--resume",    type=str, default=None)
    return ap.parse_args()


def merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            merge(base[k], v)
        else:
            base[k] = v


def main():
    args = parse_args()
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.config:
        with open(args.config) as f:
            merge(cfg, json.load(f))
    over = {}
    if args.train_dir:  over["train_dir"]  = args.train_dir
    if args.val_dir:    over["val_dir"]    = args.val_dir
    if args.test_dir:   over["test_dir"]   = args.test_dir
    if args.output_dir: over["output_dir"] = args.output_dir
    if args.epochs:     over["epochs"]     = args.epochs
    if args.batch_size: over["batch_size"] = args.batch_size
    if args.lr:         over["optimizer"]  = {"lr": args.lr}
    if args.resume:     over["resume"]     = args.resume
    merge(cfg, over)
    train(cfg)


if __name__ == "__main__":
    main()
