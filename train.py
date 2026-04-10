"""
train.py
--------
LLIO + PoseClassifier 학습 루프

실행 예시:
    # 기본 설정으로 실행
    python train.py

    # config 파일 지정
    python train.py --config config.json

    # 인자 직접 전달
    python train.py --train_dir data/train --val_dir data/val --epochs 100

파일 구조:
    dataset.py          ← OXIODDataset, build_dataloaders
    losses.py           ← CombinedLoss, compute_class_weights
    model_twolayer.py   ← TwoLayerModel
    train.py            ← (현재 파일)
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import OXIODDataset, build_dataloaders, CLASS_NAMES, NUM_CLASSES
from losses import CombinedLoss, GaussianNLLLoss, compute_class_weights
from model_twolayer import TwoLayerModel


# =============================================================================
# 기본 설정 (config.json 또는 --config 인자로 덮어쓸 수 있음)
# =============================================================================

DEFAULT_CONFIG = {
    # ---- 데이터 ----
    #  train/val/test 디렉터리를 직접 지정
    "train_dir"  : "dataset",
    "val_dir"    : "val_dataset",
    "test_dir"   : "test_dataset",
    "window_len"   : 100,
    "train_stride" : 10,
    "eval_stride"  : 50,
    "batch_size"   : 256,
    "num_workers"  : 4,

    # ---- 모델 ----
    "model": {
        "input_len"    : 100,
        "input_channel": 6,
        "patch_len"    : 25,
        "feature_dim"  : 512,
        "out_dim"      : 3,
        "active_func"  : "GELU",
        "extractor": {
            "name"     : "ResMLP",
            "layer_num": 6,
            "expansion": 2,
            "dropout"  : 0.2,
        },
        "reg": {
            "name"     : "MeanMLP",
            "layer_num": 3,
        },
        "classifier": {
            "num_classes" : 7,
            "layer_num"   : 2,
            "dropout"     : 0.3,
            "pooling_type": "mean",
        },
    },

    # ---- 손실 ----
    "loss": {
        "cls_weight"        : 1.0,    # reg와 cls를 동등하게 (0.3은 cls 신호 부족)
        "label_smoothing"   : 0.1,
        "ignore_noise_class": False,
        "use_class_weights" : True,
    },

    # ---- 최적화 ----
    "optimizer": {
        "name"        : "AdamW",
        "lr"          : 1e-3,
        "weight_decay": 1e-3,   # overfitting 방지 (1e-4 → 1e-3)
    },
    "scheduler": {
        "name"      : "CosineAnnealingLR",
        "T_max"     : 100,
        "eta_min"   : 1e-6,
        "patience"  : 10,
        "step_size" : 30,
        "gamma"     : 0.5,
    },

    # ---- 학습 ----
    "epochs"        : 100,
    "early_stopping": 20,
    "grad_clip"     : 1.0,

    # ---- 저장 / 로깅 ----
    "output_dir"    : "runs/exp02",
    "save_every"    : 10,       # N epoch마다 체크포인트 저장
    "log_interval"  : 50,       # N step마다 train loss 출력
}

# =============================================================================
# 유틸리티
# =============================================================================

def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    # 콘솔
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 파일
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    state: dict,
    path: Path,
    logger: logging.Logger,
):
    torch.save(state, path)
    logger.info(f"  체크포인트 저장: {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 클래스 가중치 수집 (전체 train DataLoader 스캔)
# =============================================================================

def collect_train_labels(loader: DataLoader) -> torch.Tensor:
    """train DataLoader 전체를 순회해 레이블 텐서 반환 (클래스 가중치 계산용)."""
    all_labels = []
    for _, _, labels in loader:
        all_labels.append(labels)
    return torch.cat(all_labels)


# =============================================================================
# 메트릭 계산
# =============================================================================

class AverageMeter:
    """배치별 평균값 추적."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum   += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def compute_metrics(
    y_hat: torch.Tensor,
    target: torch.Tensor,
    pose_logits: Optional[torch.Tensor],
    pose_labels: Optional[torch.Tensor],
) -> dict:
    """
    배치 단위 메트릭 계산.

    Returns:
        rmse      : 위치 예측 RMSE [m]
        rmse_x/y/z: 축별 RMSE
        cls_acc   : 분류 정확도 (classifier 있을 때만)
    """
    with torch.no_grad():
        err   = (y_hat - target) ** 2           # [B, 3]
        rmse  = err.mean().sqrt().item()
        rmse_x = err[:, 0].mean().sqrt().item()
        rmse_y = err[:, 1].mean().sqrt().item()
        rmse_z = err[:, 2].mean().sqrt().item()

    metrics = {
        "rmse"  : rmse,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_z": rmse_z,
    }

    if pose_logits is not None and pose_labels is not None:
        with torch.no_grad():
            pred   = pose_logits.argmax(dim=1)
            acc    = (pred == pose_labels).float().mean().item()
        metrics["cls_acc"] = acc

    return metrics


# =============================================================================
# 학습 / 검증 스텝
# =============================================================================

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    device    : torch.device,
    cfg       : dict,
    logger    : logging.Logger,
    epoch     : int,
) -> dict:
    model.train()
    use_cls = model.use_classifier

    meters = {k: AverageMeter() for k in
              ["total", "reg", "cls", "rmse", "cls_acc"]}

    t0 = time.time()
    for step, (imu, target, labels) in enumerate(loader):
        imu    = imu.to(device)
        target = target.to(device)
        labels = labels.to(device)

        # ---- Forward ----
        if use_cls:
            y_hat, log_var, pose_logits = model(imu)
            loss, detail = criterion(y_hat, log_var, target, pose_logits, labels)
        else:
            y_hat, log_var = model(imu)
            loss = criterion(y_hat, log_var, target)
            detail = {"reg": loss.item(), "cls": 0.0, "total": loss.item()}
            pose_logits = None

        # ---- Backward ----
        optimizer.zero_grad()
        loss.backward()
        if cfg.get("grad_clip"):
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        # ---- 메트릭 업데이트 ----
        B = imu.size(0)
        meters["total"].update(detail["total"], B)
        meters["reg"].update(detail["reg"], B)
        meters["cls"].update(detail["cls"], B)

        m = compute_metrics(y_hat.detach(), target.detach(),
                            pose_logits.detach() if pose_logits is not None else None,
                            labels.detach() if use_cls else None)
        meters["rmse"].update(m["rmse"], B)
        if "cls_acc" in m:
            meters["cls_acc"].update(m["cls_acc"], B)

        # ---- 중간 로그 ----
        if (step + 1) % cfg.get("log_interval", 50) == 0:
            elapsed = time.time() - t0
            logger.info(
                f"  Epoch {epoch:03d} [{step+1:4d}/{len(loader)}]"
                f"  loss={detail['total']:.4f}"
                f"  reg={detail['reg']:.4f}"
                f"  cls={detail['cls']:.4f}"
                f"  rmse={m['rmse']:.4f}"
                + (f"  acc={m.get('cls_acc', 0):.3f}" if use_cls else "")
                + f"  ({elapsed:.1f}s)"
            )
            t0 = time.time()

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def evaluate(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    device    : torch.device,
) -> dict:
    model.eval()
    use_cls = model.use_classifier

    meters = {k: AverageMeter() for k in
              ["total", "reg", "cls", "rmse", "rmse_x", "rmse_y", "rmse_z", "cls_acc"]}

    # 클래스별 정확도 집계
    cls_correct = torch.zeros(NUM_CLASSES)
    cls_total   = torch.zeros(NUM_CLASSES)

    for imu, target, labels in loader:
        imu    = imu.to(device)
        target = target.to(device)
        labels = labels.to(device)

        if use_cls:
            y_hat, log_var, pose_logits = model(imu)
            loss, detail = criterion(y_hat, log_var, target, pose_logits, labels)
        else:
            y_hat, log_var = model(imu)
            loss = criterion(y_hat, log_var, target)
            detail = {"reg": loss.item(), "cls": 0.0, "total": loss.item()}
            pose_logits = None

        B = imu.size(0)
        meters["total"].update(detail["total"], B)
        meters["reg"].update(detail["reg"], B)
        meters["cls"].update(detail["cls"], B)

        m = compute_metrics(y_hat, target,
                            pose_logits if pose_logits is not None else None,
                            labels if use_cls else None)
        for k in ["rmse", "rmse_x", "rmse_y", "rmse_z"]:
            meters[k].update(m[k], B)
        if "cls_acc" in m:
            meters["cls_acc"].update(m["cls_acc"], B)

        # 클래스별 정확도
        if use_cls and pose_logits is not None:
            pred = pose_logits.argmax(dim=1).cpu()
            lbl  = labels.cpu()
            for c in range(NUM_CLASSES):
                mask = lbl == c
                cls_correct[c] += (pred[mask] == c).sum()
                cls_total[c]   += mask.sum()

    result = {k: v.avg for k, v in meters.items()}

    # 클래스별 정확도 추가
    if use_cls:
        per_cls = {}
        for c in range(NUM_CLASSES):
            if cls_total[c] > 0:
                per_cls[CLASS_NAMES[c]] = (cls_correct[c] / cls_total[c]).item()
            else:
                per_cls[CLASS_NAMES[c]] = float("nan")
        result["per_class_acc"] = per_cls

    return result


# =============================================================================
# 메인 학습 루프
# =============================================================================

def train(cfg: dict):
    # ---- 출력 디렉터리 ----
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    logger = setup_logger(out_dir / "train.log")
    logger.info("=" * 60)
    logger.info("학습 시작")
    logger.info(f"출력 디렉터리: {out_dir}")

    # 설정 저장
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    device = get_device()
    logger.info(f"디바이스: {device}")

    # ---- 파일 분리 & 데이터셋 ----
    logger.info("데이터셋 로딩 중...")

    train_files = cfg.get("train_dir", "dataset")
    val_files   = cfg.get("val_dir",   "val_dataset")
    test_files  = cfg.get("test_dir",  None)

    loaders = build_dataloaders(
        train_paths  = train_files,
        val_paths    = val_files,
        test_paths   = test_files,
        window_len   = cfg["window_len"],
        train_stride = cfg["train_stride"],
        eval_stride  = cfg["eval_stride"],
        batch_size   = cfg["batch_size"],
        num_workers  = cfg["num_workers"],
        normalize    = True,
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    test_loader  = loaders.get("test", None)

    # 정규화 통계 저장 (추론 시 재사용)
    mean, std = loaders["stats"]
    np.save(out_dir / "norm_mean.npy", mean)
    np.save(out_dir / "norm_std.npy",  std)
    logger.info(f"정규화 통계 저장 완료: {out_dir}/norm_mean.npy, norm_std.npy")

    # ---- 모델 ----
    model = TwoLayerModel(cfg["model"]).to(device)
    logger.info(f"파라미터 수: {count_parameters(model):,}")

    # ---- 클래스 가중치 ----
    class_weights = None
    if cfg["loss"].get("use_class_weights", True) and cfg["model"].get("classifier"):
        logger.info("클래스 가중치 계산 중 (train set 전체 스캔)...")
        all_labels     = collect_train_labels(train_loader)
        class_weights  = compute_class_weights(all_labels, NUM_CLASSES).to(device)
        logger.info(f"클래스 가중치: { {CLASS_NAMES[i]: f'{class_weights[i].item():.3f}' for i in range(NUM_CLASSES)} }")

    # ---- 손실 함수 ----
    loss_cfg = cfg["loss"]
    if cfg["model"].get("classifier"):
        criterion = CombinedLoss(
            cls_weight         = loss_cfg["cls_weight"],
            class_weights      = class_weights,
            label_smoothing    = loss_cfg.get("label_smoothing", 0.0),
            ignore_noise_class = loss_cfg.get("ignore_noise_class", False),
        ).to(device)
    else:
        # classifier 없으면 회귀 손실만
        from losses import GaussianNLLLoss
        criterion = GaussianNLLLoss().to(device)

    # ---- 옵티마이저 ----
    opt_cfg = cfg["optimizer"]
    if opt_cfg["name"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = opt_cfg["lr"],
            weight_decay = opt_cfg.get("weight_decay", 1e-4),
        )
    elif opt_cfg["name"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr           = opt_cfg["lr"],
            weight_decay = opt_cfg.get("weight_decay", 0),
        )
    elif opt_cfg["name"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr       = opt_cfg["lr"],
            momentum = opt_cfg.get("momentum", 0.9),
            weight_decay = opt_cfg.get("weight_decay", 1e-4),
        )
    else:
        raise ValueError(f"알 수 없는 optimizer: {opt_cfg['name']}")

    # ---- 스케줄러 ----
    sch_cfg = cfg["scheduler"]
    if sch_cfg["name"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = sch_cfg.get("T_max", cfg["epochs"]),
            eta_min = sch_cfg.get("eta_min", 1e-6),
        )
    elif sch_cfg["name"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode     = "min",
            patience = sch_cfg.get("patience", 10),
            factor   = sch_cfg.get("gamma", 0.5),
        )
    elif sch_cfg["name"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = sch_cfg.get("step_size", 30),
            gamma     = sch_cfg.get("gamma", 0.5),
        )
    else:
        scheduler = None

    # ---- 학습 루프 ----
    best_val_loss    = float("inf")
    early_stop_count = 0
    history          = []

    logger.info("=" * 60)
    logger.info(f"총 {cfg['epochs']} epoch 학습 시작")

    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, cfg, logger, epoch,
        )

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)

        # 스케줄러 업데이트
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["total"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - epoch_start

        # ---- 에폭 요약 로그 ----
        log_line = (
            f"Epoch {epoch:03d}/{cfg['epochs']}"
            f"  lr={current_lr:.2e}"
            f"  train[loss={train_metrics['total']:.4f}"
            f"  reg={train_metrics['reg']:.4f}"
            f"  cls={train_metrics['cls']:.4f}"
            f"  rmse={train_metrics['rmse']:.4f}]"
            f"  val[loss={val_metrics['total']:.4f}"
            f"  reg={val_metrics['reg']:.4f}"
            f"  rmse={val_metrics['rmse']:.4f}"
            f"  rmse_xyz=({val_metrics['rmse_x']:.4f},{val_metrics['rmse_y']:.4f},{val_metrics['rmse_z']:.4f})"
        )
        if model.use_classifier:
            log_line += f"  acc={val_metrics['cls_acc']:.3f}"
        log_line += f"]  ({elapsed:.1f}s)"
        logger.info(log_line)

        # 클래스별 정확도 로그
        if model.use_classifier and epoch % 5 == 0:
            per_cls = val_metrics.get("per_class_acc", {})
            acc_str = "  ".join(
                f"{name}={acc:.3f}" if not np.isnan(acc) else f"{name}=N/A"
                for name, acc in per_cls.items()
            )
            logger.info(f"  per-class acc: {acc_str}")

        # ---- 히스토리 기록 ----
        record = {"epoch": epoch, "lr": current_lr,
                  **{f"train_{k}": v for k, v in train_metrics.items()
                     if not isinstance(v, dict)},
                  **{f"val_{k}": v for k, v in val_metrics.items()
                     if not isinstance(v, dict)}}
        history.append(record)

        # ---- Best 체크포인트 ----
        val_loss = val_metrics["total"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            save_checkpoint(
                {
                    "epoch"    : epoch,
                    "model"    : model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "val_loss" : best_val_loss,
                    "config"   : cfg,
                },
                ckpt_dir / "best.pth",
                logger,
            )
        else:
            early_stop_count += 1

        # ---- 주기적 체크포인트 ----
        if epoch % cfg.get("save_every", 10) == 0:
            save_checkpoint(
                {
                    "epoch"    : epoch,
                    "model"    : model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "val_loss" : val_loss,
                    "config"   : cfg,
                },
                ckpt_dir / f"epoch_{epoch:03d}.pth",
                logger,
            )

        # ---- Early Stopping ----
        if cfg.get("early_stopping") and early_stop_count >= cfg["early_stopping"]:
            logger.info(
                f"Early stopping: val loss가 {cfg['early_stopping']} epoch 동안 개선 없음."
            )
            break

    # ---- 히스토리 저장 ----
    import csv
    if history:
        keys = [k for k in history[0].keys() if not isinstance(history[0][k], dict)]
        with open(out_dir / "history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in history:
                writer.writerow({k: row[k] for k in keys})
        logger.info(f"학습 히스토리 저장: {out_dir}/history.csv")

    # ---- 최종 테스트 평가 ----
    if test_loader is not None:
        logger.info("=" * 60)
        logger.info("테스트 세트 평가 (best.pth 로드)")
        load_checkpoint(ckpt_dir / "best.pth", model, device=device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        logger.info(
            f"Test  loss={test_metrics['total']:.4f}"
            f"  reg={test_metrics['reg']:.4f}"
            f"  rmse={test_metrics['rmse']:.4f}"
            + (f"  acc={test_metrics['cls_acc']:.3f}" if model.use_classifier else "")
        )
        if model.use_classifier:
            per_cls = test_metrics.get("per_class_acc", {})
            acc_str = "  ".join(
                f"{name}={acc:.3f}" if not np.isnan(acc) else f"{name}=N/A"
                for name, acc in per_cls.items()
            )
            logger.info(f"  per-class acc: {acc_str}")

    logger.info("=" * 60)
    logger.info(f"학습 완료. Best val loss: {best_val_loss:.4f}")
    logger.info(f"결과 저장 위치: {out_dir}")


# =============================================================================
# 진입점
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="LLIO + PoseClassifier 학습")
    parser.add_argument("--config",      type=str,   default=None,
                        help="JSON config 파일 경로. 없으면 DEFAULT_CONFIG 사용.")
    parser.add_argument("--train_dir",   type=str,   default=None)
    parser.add_argument("--val_dir",     type=str,   default=None)
    parser.add_argument("--test_dir",    type=str,   default=None)
    parser.add_argument("--output_dir",  type=str,   default=None)
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--cls_weight",  type=float, default=None,
                        help="분류 손실 가중치")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="이어서 학습할 체크포인트 경로")
    return parser.parse_args()


def merge_config(base: dict, overrides: dict) -> dict:
    """중첩 dict 병합 (overrides가 base를 덮어씀)."""
    result = base.copy()
    for k, v in overrides.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = v
    return result


if __name__ == "__main__":
    args = parse_args()

    # 1) 기본 설정
    cfg = DEFAULT_CONFIG.copy()

    # 2) config 파일 덮어쓰기
    if args.config:
        with open(args.config) as f:
            file_cfg = json.load(f)
        cfg = merge_config(cfg, file_cfg)

    # 3) 커맨드라인 인자 덮어쓰기
    cli_overrides = {}
    if args.train_dir  : cli_overrides["train_dir"]   = args.train_dir
    if args.val_dir    : cli_overrides["val_dir"]      = args.val_dir
    if args.test_dir   : cli_overrides["test_dir"]     = args.test_dir
    if args.output_dir : cli_overrides["output_dir"]   = args.output_dir
    if args.epochs     : cli_overrides["epochs"]       = args.epochs
    if args.batch_size : cli_overrides["batch_size"]   = args.batch_size
    if args.lr         : cli_overrides["optimizer"]    = {"lr": args.lr}
    if args.cls_weight : cli_overrides["loss"]         = {"cls_weight": args.cls_weight}
    cfg = merge_config(cfg, cli_overrides)

    # 4) 체크포인트에서 이어 학습 (resume)은 train() 안에서 처리하도록 cfg에 포함
    if args.resume:
        cfg["resume"] = args.resume

    train(cfg)
