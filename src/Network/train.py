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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Trans"))

import argparse, json, logging, time

import numpy as np
import torch
import torch.nn as nn

from dataset import build_dataloaders
from losses import CombinedLoss, compute_class_weights
from model_twolayer import TwoLayerModel

# Oxford raw label → 0-indexed class id
# raw: -1=noise, 1=handbag, 2=handheld, 3=pocket, 4=running, 5=slow, 6=trolley
LABEL_REMAP = {-1: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
NUM_CLASSES  = 7


DEFAULT_CONFIG = {
    # --- 데이터셋 설정 ---
    "train_dir": "oxford_split/train",
    "val_dir":   "oxford_split/val",
    "test_dir":  "oxford_split/test",
    "fmt":        "oxford",   # "tlio" | "oxford"
    "with_label": False,      # regression only (분류 레이블 불필요)

    "window_len":   100,   # Oxford 100Hz × 1s = TLIO 200Hz × 1s 와 동일 시간
    "train_stride": 10,
    "eval_stride":  10,

    "batch_size":   128,
    "num_workers":  4,

    # --- 모델 설정 ---
    "model": {
        "input_len":     100,
        "input_channel": 6,
        "patch_len":     10,  # 패치 수 10개 유지 (100/10)
        "feature_dim":   128,
        "out_dim":       3,
        "active_func":   "GELU",

        "extractor": {"name": "ResMLP", "layer_num": 6, "expansion": 2, "dropout": 0.2},
        "reg":       {"name": "SimpleMean", "layer_num": 3, "dropout": 0.2},
        "classifier": {"num_classes": 7, "layer_num": 2, "dropout": 0.3, "pooling_type": "mean"},
        "use_classifier": False,
    },

    # --- TLIO golden 증강 데이터 ---
    # TLIO-master golden-new-format 데이터를 학습 시 Oxford 데이터와 혼합.
    # None 으로 설정하면 Oxford 데이터만 사용.
    "tlio_aug_dir":    r"C:\Users\hs091\Desktop\TLIO-master\golden-new-format-cc-by-nc-with-imus",
    "tlio_aug_stride": 10,

    # --- 사전학습 가중치 ---
    # extractor: 회귀 모델에서 이식 / cls_head: standalone 분류기에서 이식
    "pretrained_reg": "outputs/out_tlio_6ch_128/checkpoints/best.pth",
    "pretrained_cls": "outputs/out_classifier_7way/checkpoints/best.pth",

    # --- 손실 함수 설정 ---
    "loss": {
        "cls_weight":       0.0,   # regression only
        "dir_weight":       0.0,
        "label_smoothing":  0.0,
        "use_class_weights": False,
        "mse_epochs":       30,    # 초반 MSE로 회귀 안정화 후 NLL 전환
    },

    # --- 최적화 설정 ---
    "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-4},
    "scheduler": {"name": "CosineAnnealingLR", "T_max": 100, "eta_min": 1e-6},
    "warmup_epochs": 5,
    "epochs": 100,
    "grad_clip": 1.0,
    "log_interval": 50,
    "save_every": 10,
    "early_stopping": 30,
    "output_dir": "out_regression",
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

def collect_train_labels(loader):
    lbls = []
    for batch in loader:
        lbls.append(batch[2])
    return torch.cat(lbls, dim=0)


def remap_labels(labels: torch.Tensor) -> torch.Tensor:
    """Oxford raw label(-1,1-6) → 0-indexed(0-6)."""
    out = labels.clone()
    for raw, idx in LABEL_REMAP.items():
        out[labels == raw] = idx
    return out


def load_pretrained_weights(model: TwoLayerModel, reg_ckpt: str, cls_ckpt: str, logger):
    """
    extractor → 회귀 모델 체크포인트에서 이식
    pose_classifier → standalone 분류기 체크포인트에서 이식 (키 매칭 시)
    reg head는 구조 변경(PoseConditioned)으로 인해 랜덤 초기화 유지
    """
    device = next(model.parameters()).device

    if reg_ckpt and Path(reg_ckpt).exists():
        ck = torch.load(reg_ckpt, map_location=device, weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        # extractor 가중치만 추출
        ext_sd_raw = {k[len("extractor."):]: v for k, v in sd.items() if k.startswith("extractor.")}
        # 크기 불일치 레이어 제외 (patch_len 변경 시 첫 Linear 차원 달라짐)
        cur_sd = model.extractor.state_dict()
        ext_sd = {k: v for k, v in ext_sd_raw.items()
                  if k in cur_sd and v.shape == cur_sd[k].shape}
        skipped = [k for k in ext_sd_raw if k not in ext_sd]
        missing, unexpected = model.extractor.load_state_dict(ext_sd, strict=False)
        logger.info(f"[pretrain] extractor 이식: {len(ext_sd)}개 레이어 완료, {len(skipped)}개 크기 불일치 → 랜덤 초기화")
        if skipped:
            logger.info(f"  → 랜덤 초기화 레이어: {skipped}")
    else:
        logger.info(f"[pretrain] reg 체크포인트 없음, extractor 랜덤 초기화")

    if cls_ckpt and Path(cls_ckpt).exists() and model.use_classifier:
        ck = torch.load(cls_ckpt, map_location=device)
        sd = ck["model"] if "model" in ck else ck
        # pose_classifier 키 시도
        cls_sd = {k[len("pose_classifier."):]: v for k, v in sd.items() if k.startswith("pose_classifier.")}
        if cls_sd:
            missing, unexpected = model.pose_classifier.load_state_dict(cls_sd, strict=False)
            logger.info(f"[pretrain] pose_classifier 이식 완료 (missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            # standalone 분류기는 다른 구조일 수 있음 → 랜덤 초기화 유지
            logger.info(f"[pretrain] cls 체크포인트 키 불일치, pose_classifier 랜덤 초기화")
    else:
        logger.info(f"[pretrain] cls 체크포인트 없음 또는 분류기 미사용")


def compute_tf_ratio(epoch, total_epochs, tf_start=1.0, tf_end=0.0, warmup=40):
    """Scheduled sampling: epoch 1~warmup은 1.0, 이후 선형 감소 → tf_end."""
    if epoch <= warmup:
        return tf_start
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return max(tf_end, tf_start - progress * (tf_start - tf_end))


def train_one_epoch(model, loader, criterion, optimizer, device, cfg, logger, epoch):
    model.train()
    use_cls = cfg["model"].get("use_classifier", False)
    with_label = cfg.get("with_label", False)
    tf_ratio = compute_tf_ratio(epoch, cfg["epochs"]) if use_cls else 0.0
    m = {k: AverageMeter() for k in ["reg", "dir", "cls", "rmse"]}

    for step, batch in enumerate(loader):
        if with_label:
            imu, target, label_raw = batch
            label = remap_labels(label_raw).to(device)
        else:
            imu, target = batch
            label = None

        imu, target = imu.to(device), target.to(device)

        y_hat, log_var, pose_logits = model(imu, pose_labels=label, tf_ratio=tf_ratio)
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
                        f"reg={d['reg']:.4f}  cls={d['cls']:.4f}  rmse={m['rmse'].avg:.4f}")

    return {k: v.avg for k, v in m.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device, with_label=False):
    model.eval()
    reg_m = AverageMeter(); cls_m = AverageMeter()
    errs = []
    correct = 0; total = 0

    for batch in loader:
        if with_label:
            imu, target, label_raw = batch
            label = remap_labels(label_raw).to(device)
        else:
            imu, target = batch
            label = None

        imu, target = imu.to(device), target.to(device)

        y_hat, log_var, pose_logits = model(imu)
        _, d = criterion(y_hat, log_var, target, pose_logits, label)

        B = imu.size(0)
        reg_m.update(d["reg"], B)
        cls_m.update(d["cls"], B)
        errs.append((y_hat - target).cpu().numpy())

        if pose_logits is not None and label is not None:
            pred_cls = pose_logits.argmax(dim=1)
            correct += (pred_cls == label).sum().item()
            total   += B

    errs = np.concatenate(errs, 0)
    rmse_xyz = np.sqrt((errs ** 2).mean(0))
    rmse = float(np.sqrt((errs ** 2).sum(1).mean()))
    acc  = correct / total if total > 0 else 0.0

    return {"reg": reg_m.avg, "cls": cls_m.avg, "rmse": rmse,
            "rmse_x": float(rmse_xyz[0]), "rmse_y": float(rmse_xyz[1]),
            "rmse_z": float(rmse_xyz[2]), "acc": acc}


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

    fmt        = cfg.get("fmt", "tlio")
    with_label = cfg.get("with_label", False)
    logger.info(f"데이터셋 로딩... fmt={fmt}  with_label={with_label}")
    loaders = build_dataloaders(
        train_paths=cfg["train_dir"], val_paths=cfg["val_dir"],
        test_paths=cfg.get("test_dir"),
        window_len=cfg["window_len"],
        train_stride=cfg["train_stride"], eval_stride=cfg["eval_stride"],
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        fmt=fmt, with_label=with_label,
        tlio_aug_dir=cfg.get("tlio_aug_dir"),
        tlio_aug_stride=cfg.get("tlio_aug_stride", 10),
    )
    train_loader = loaders["train"]; val_loader = loaders["val"]
    test_loader = loaders.get("test")

    mean, std = loaders["stats"]
    np.save(out_dir / "norm_mean.npy", mean)
    np.save(out_dir / "norm_std.npy",  std)

    model = TwoLayerModel(cfg["model"]).to(device)
    logger.info(f"파라미터 수: {count_parameters(model):,}")

    # 사전학습 가중치 이식
    load_pretrained_weights(
        model,
        reg_ckpt=cfg.get("pretrained_reg", ""),
        cls_ckpt=cfg.get("pretrained_cls", ""),
        logger=logger,
    )

    # class weights (joint training 시 label 불균형 보정)
    use_classifier = cfg["model"].get("use_classifier", False)
    class_weights = None
    if use_classifier and with_label and cfg["loss"].get("use_class_weights", False):
        logger.info("train set 클래스 가중치 계산...")
        all_lbls = remap_labels(collect_train_labels(train_loader))
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

    #hard_idx = hard_class_indices(cfg.get("hard_classes", ["handbag", "pocket"]))
    #logger.info(f"Hard classes: {[CLASS_NAMES[i] for i in hard_idx]} → {hard_idx}")

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
        vl = evaluate(model, val_loader, criterion, device, with_label=with_label)

        if scheduler is not None:
            if epoch > warmup_epochs:  # 워밍업 중에는 scheduler.step() 스킵
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(vl["rmse_hard"])
                else:
                    scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        acc_str = f" acc={vl['acc']:.3f}" if use_classifier else ""
        logger.info(
            f"Epoch {epoch:03d}/{cfg['epochs']}  lr={lr_now:.2e}  "
            f"train[reg={tr['reg']:.4f} cls={tr['cls']:.4f} rmse={tr['rmse']:.4f}]  "
            f"val[rmse={vl['rmse']:.4f} cls={vl['cls']:.4f}"
            f" xyz=({vl['rmse_x']:.3f},{vl['rmse_y']:.3f},{vl['rmse_z']:.3f})"
            f"{acc_str}]  ({time.time()-t0:.1f}s)"
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
                "val_rmse": vl["rmse"], "config": cfg,
            }, ckpt_dir / "best.pth", logger)
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
        tm = evaluate(model, test_loader, criterion, device, with_label=with_label)
        logger.info(f"Test  rmse={tm['rmse']:.4f}  "
                    f"xyz=({tm['rmse_x']:.3f},{tm['rmse_y']:.3f},{tm['rmse_z']:.3f})  "
                    f"cls={tm['cls']:.4f}  acc={tm['acc']:.3f}")


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
