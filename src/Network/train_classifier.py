# Windows에서 DataLoader worker + OpenMP 충돌 방지
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Trans"))

import argparse
import json
import logging
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from classification_dataset import (
    CLASS_NAMES,
    build_classification_dataloaders,
    compute_class_weights_from_counts,
)
from model_twolayer import PoseClassifier, ResMLPExtractor


DEFAULT_CONFIG = {
    # --- 데이터셋 설정 (회귀 모델과 동일: oxford_split, window_len=100, stride=10) ---
    "train_dir": "oxford_split/train",
    "val_dir": "oxford_split/val",
    "test_dir": "oxford_split/test",
    "window_len": 100,      # 100Hz × 1s (회귀 모델과 동일, EKF 호환)
    "train_stride": 10,     # 회귀 모델과 동일
    "eval_stride": 10,
    "batch_size": 128,
    "num_workers": 4,
    "epochs": 100,          # 회귀 모델과 동일
    "warmup_epochs": 5,
    "grad_clip": 1.0,
    "log_interval": 50,
    "early_stopping": 20,
    "output_dir": "out_cls_oxford",
    # 증강: 회귀 모델의 TLIONpySingleDataset is_train=True 시 bias_noise+gravity_perturb+yaw가
    # 이미 적용됨. 여기서는 추가 noise/scale 증강만 설정.
    "augment": {
        "noise": True,
        "noise_std": 0.01,
        "scale": True,
        "scale_range": [0.95, 1.05],
    },
    "model": {
        "input_len": 100,       # window_len과 일치 (회귀 모델과 동일)
        "input_channel": 6,
        "patch_len": 10,        # 패치 수 10개 유지 (100/10, 회귀 모델과 동일)
        "feature_dim": 128,     # 회귀 모델과 동일
        "active_func": "GELU",
        "extractor": {
            "layer_num": 6,
            "expansion": 2,
            "dropout": 0.2,
        },
        "classifier": {
            "num_classes": 7,
            "layer_num": 2,
            "dropout": 0.3,
            "pooling_type": "mean",
        },
    },
    "loss": {
        # label_smoothing과 극단적 class weight(noise=6.79)를 동시 사용하면
        # 모든 샘플에 noise logit 강화 gradient가 발생 → collapse.
        # noise 샘플이 0개이므로 label_smoothing=0으로 설정.
        "label_smoothing": 0.0,
        "use_class_weights": True,
    },
    "optimizer": {
        "name": "AdamW",
        "lr": 3e-4,
        "weight_decay": 1e-4,
    },
    "scheduler": {
        "name": "CosineAnnealingLR",
        "T_max": 100,
        "eta_min": 1e-6,
    },
}


class AverageMeter:
    def __init__(self):
        self.n = 0
        self.sum = 0.0

    def update(self, value: float, k: int = 1):
        self.n += k
        self.sum += float(value) * k

    @property
    def avg(self) -> float:
        return self.sum / max(self.n, 1)


class ResMLPClassifier(nn.Module):
    def __init__(self, model_cfg: Dict):
        super().__init__()
        act = nn.ReLU() if model_cfg["active_func"] == "ReLU" else nn.GELU()
        patch_num = int(model_cfg["input_len"] / model_cfg["patch_len"])
        self.extractor = ResMLPExtractor(
            patch_num=patch_num,
            patch_len=model_cfg["patch_len"],
            input_channel=model_cfg["input_channel"],
            mlp_in_dim=model_cfg["feature_dim"],
            expansion=model_cfg["extractor"]["expansion"],
            active_func=act,
            layer_num=model_cfg["extractor"]["layer_num"],
            dropout=model_cfg["extractor"]["dropout"],
        )
        self.classifier = PoseClassifier(
            feature_len=model_cfg["feature_dim"],
            num_classes=model_cfg["classifier"]["num_classes"],
            layer_num=model_cfg["classifier"].get("layer_num", 2),
            active_fun=act,
            dropout=model_cfg["classifier"].get("dropout", 0.3),
            pooling_type=model_cfg["classifier"].get("pooling_type", "mean"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.extractor(x)
        logits = self.classifier(feature)
        return logits



def setup_logger(log_path: Path):
    logger = logging.getLogger("train_classifier")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger



def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def save_checkpoint(state: Dict, path: Path, logger):
    torch.save(state, path)
    logger.info(f"  ckpt 저장: {path}")



def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt



def compute_confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm



def compute_macro_f1(cm: np.ndarray) -> float:
    f1_list: List[float] = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = (2 * tp + fp + fn)
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        f1_list.append(float(f1))
    return float(np.mean(f1_list))



def train_one_epoch(model, loader, criterion, optimizer, device, cfg, logger, epoch):
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    for step, (imu, label) in enumerate(loader):
        imu = imu.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logits = model(imu)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        if cfg.get("grad_clip"):
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == label).float().mean().item()

        B = imu.size(0)
        loss_m.update(loss.item(), B)
        acc_m.update(acc, B)

        if (step + 1) % cfg.get("log_interval", 50) == 0:
            logger.info(
                f"  Epoch {epoch:03d} [{step+1:4d}/{len(loader)}] "
                f"loss={loss_m.avg:.4f} acc={acc_m.avg:.4f}"
            )

    return {"loss": loss_m.avg, "acc": acc_m.avg}


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names: List[str]):
    model.eval()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    all_preds = []
    all_labels = []

    for imu, label in loader:
        imu = imu.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logits = model(imu)
        loss = criterion(logits, label)
        pred = logits.argmax(dim=1)
        acc = (pred == label).float().mean().item()

        B = imu.size(0)
        loss_m.update(loss.item(), B)
        acc_m.update(acc, B)

        all_preds.append(pred.cpu().numpy())
        all_labels.append(label.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    cm = compute_confusion_matrix(preds, labels, num_classes=len(class_names))
    macro_f1 = compute_macro_f1(cm)
    per_class_acc = {}
    for i, name in enumerate(class_names):
        row_sum = cm[i].sum()
        per_class_acc[name] = 0.0 if row_sum == 0 else float(cm[i, i] / row_sum)

    return {
        "loss": loss_m.avg,
        "acc": acc_m.avg,
        "macro_f1": macro_f1,
        "cm": cm,
        "per_class_acc": per_class_acc,
    }



def merge(base: Dict, over: Dict):
    for k, v in over.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            merge(base[k], v)
        else:
            base[k] = v



def train(cfg: Dict):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    logger = setup_logger(out_dir / "train_classifier.log")

    logger.info("=" * 60)
    logger.info("7-way classifier training (oxford_split, window_len=100, stride=10)")
    logger.info(f"출력: {out_dir}")

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    device = get_device()
    logger.info(f"디바이스: {device}")

    aug = cfg.get("augment", {})
    loaders = build_classification_dataloaders(
        train_dir=cfg["train_dir"],
        val_dir=cfg["val_dir"],
        test_dir=cfg.get("test_dir"),
        window_len=cfg["window_len"],
        train_stride=cfg["train_stride"],
        eval_stride=cfg["eval_stride"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        normalize=True,
        augment_noise=aug.get("noise", False),
        noise_std=aug.get("noise_std", 0.01),
        augment_scale=aug.get("scale", False),
        scale_range=tuple(aug.get("scale_range", [0.95, 1.05])),
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders.get("test")
    class_names = loaders["class_names"]

    mean, std = loaders["stats"]
    np.save(out_dir / "norm_mean.npy", mean)
    np.save(out_dir / "norm_std.npy", std)

    logger.info(f"class names: {class_names}")
    logger.info(f"train window counts: {loaders['train_window_class_counts']}")
    logger.info(f"val   window counts: {loaders['val_window_class_counts']}")
    if "test_window_class_counts" in loaders:
        logger.info(f"test  window counts: {loaders['test_window_class_counts']}")

    model = ResMLPClassifier(cfg["model"]).to(device)
    logger.info(f"파라미터 수: {count_parameters(model):,}")

    class_weights = None
    if cfg["loss"].get("use_class_weights", True):
        class_weights = compute_class_weights_from_counts(loaders["train_window_class_counts"]).to(device)
        logger.info(f"class weights: {class_weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["loss"].get("label_smoothing", 0.0),
    )

    opt_cfg = cfg["optimizer"]
    if opt_cfg["name"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg["lr"])

    scheduler = None
    sch_cfg = cfg.get("scheduler", {})
    if sch_cfg.get("name") == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sch_cfg.get("T_max", cfg["epochs"]),
            eta_min=sch_cfg.get("eta_min", 1e-6),
        )

    best_val_acc = -1.0
    early_cnt = 0
    start_epoch = 1

    if cfg.get("resume"):
        ckpt = load_checkpoint(Path(cfg["resume"]), model, optimizer, scheduler, device=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_acc", -1.0)
        logger.info(f"[resume] start_epoch={start_epoch}, best_val_acc={best_val_acc:.4f}")

    warmup_epochs = cfg.get("warmup_epochs", 0)
    base_lr = cfg["optimizer"]["lr"]

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_lr = 1e-5 + (base_lr - 1e-5) * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg, logger, epoch)
        vl = evaluate(model, val_loader, criterion, device, class_names)

        if scheduler is not None and epoch > warmup_epochs:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:03d}/{cfg['epochs']} lr={lr_now:.2e} "
            f"train[loss={tr['loss']:.4f} acc={tr['acc']:.4f}] "
            f"val[loss={vl['loss']:.4f} acc={vl['acc']:.4f} macro_f1={vl['macro_f1']:.4f}] "
            f"({time.time()-t0:.1f}s)"
        )
        logger.info(f"  per-class acc: {json.dumps(vl['per_class_acc'], ensure_ascii=False)}")

        if vl["acc"] > best_val_acc:
            best_val_acc = vl["acc"]
            early_cnt = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "val_acc": vl["acc"],
                    "config": cfg,
                    "class_names": class_names,
                },
                ckpt_dir / "best.pth",
                logger,
            )
            np.save(out_dir / "best_val_confusion_matrix.npy", vl["cm"])
        else:
            early_cnt += 1

        if epoch % 10 == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "val_acc": vl["acc"],
                    "config": cfg,
                    "class_names": class_names,
                },
                ckpt_dir / f"epoch_{epoch:03d}.pth",
                logger,
            )

        if cfg.get("early_stopping") and early_cnt >= cfg["early_stopping"]:
            logger.info(f"Early stopping: {cfg['early_stopping']} epoch 개선 없음")
            break

    logger.info(f"학습 완료. best val acc={best_val_acc:.4f}")

    if test_loader is not None:
        logger.info("=" * 60)
        logger.info("테스트 평가 (best.pth 로드)")
        load_checkpoint(ckpt_dir / "best.pth", model, device=device)
        tm = evaluate(model, test_loader, criterion, device, class_names)
        logger.info(
            f"Test loss={tm['loss']:.4f} acc={tm['acc']:.4f} macro_f1={tm['macro_f1']:.4f}"
        )
        logger.info(f"Test per-class acc: {json.dumps(tm['per_class_acc'], ensure_ascii=False)}")
        np.save(out_dir / "test_confusion_matrix.npy", tm["cm"])

        with open(out_dir / "test_report.txt", "w", encoding="utf-8") as f:
            f.write(f"acc={tm['acc']:.6f}\n")
            f.write(f"macro_f1={tm['macro_f1']:.6f}\n")
            f.write("per_class_acc\n")
            for name in class_names:
                f.write(f"  {name}: {tm['per_class_acc'][name]:.6f}\n")
            f.write("\nconfusion_matrix\n")
            f.write(np.array2string(tm["cm"]))



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--train_dir", type=str, default=None)
    ap.add_argument("--val_dir", type=str, default=None)
    ap.add_argument("--test_dir", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resume", type=str, default=None)
    return ap.parse_args()



def main():
    args = parse_args()
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            merge(cfg, json.load(f))

    over = {}
    if args.train_dir:
        over["train_dir"] = args.train_dir
    if args.val_dir:
        over["val_dir"] = args.val_dir
    if args.test_dir:
        over["test_dir"] = args.test_dir
    if args.output_dir:
        over["output_dir"] = args.output_dir
    if args.epochs:
        over["epochs"] = args.epochs
    if args.batch_size:
        over["batch_size"] = args.batch_size
    if args.lr:
        over["optimizer"] = {"lr": args.lr}
    if args.resume:
        over["resume"] = args.resume
    merge(cfg, over)

    # model.input_len과 window_len은 첫 baseline에서는 맞춰두는 편이 안전함
    cfg["model"]["input_len"] = cfg["window_len"]
    train(cfg)


if __name__ == "__main__":
    main()
