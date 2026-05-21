"""
losses.py
---------
Phase 2: pose-conditioned regression + classifier auxiliary task.

포함:
    - GaussianNLLLoss       : uncertainty 포함 회귀 손실
    - PoseLoss              : CrossEntropy (class weighted)
    - CombinedLoss          : L_reg + cls_weight * L_cls, two-stage (MSE/NLL) 지원
    - compute_class_weights : sqrt_inv 기본
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class GaussianNLLLoss(nn.Module):
    def __init__(self, reduction="mean", log_var_min=-2.0, log_var_max=2.0):
        super().__init__()
        self.reduction = reduction
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

    def forward(self, y_hat, log_var, target):
        log_var = torch.clamp(log_var, self.log_var_min, self.log_var_max)
        loss = 0.5 * torch.exp(-log_var) * (target - y_hat) ** 2 + 0.5 * log_var
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class PoseLoss(nn.Module):
    def __init__(self, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights,
                                      label_smoothing=label_smoothing)

    def forward(self, logits, labels):
        return self.ce(logits, labels)


class CombinedLoss(nn.Module):
    """
    L_total = L_reg + dir_weight * L_dir + cls_weight * L_cls

    two-stage:
      use_nll=False → regression 부분은 MSE
      use_nll=True  → regression 부분은 Gaussian NLL

    direction loss (L_dir):
      1 - cosine_similarity(y_hat_xy, target_xy)
      방향 오차가 지배적일 때 cosine loss로 직접 방향 학습 유도.
      xy 평면만 사용 (z는 거의 0이므로 제외).
      target norm이 작은 샘플(정지 구간)은 마스킹.
    """

    def __init__(self,
                 cls_weight: float = 1.0,
                 dir_weight: float = 0.0,
                 class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0,
                 log_var_min: float = -2.0,
                 log_var_max: float = 2.0,
                 use_nll: bool = False,
                 dir_norm_min: float = 0.01):
        super().__init__()
        self.cls_weight = cls_weight
        self.dir_weight = dir_weight
        self.use_nll = use_nll
        self.dir_norm_min = dir_norm_min
        self.nll = GaussianNLLLoss(log_var_min=log_var_min, log_var_max=log_var_max)
        self.cls = PoseLoss(class_weights=class_weights,
                            label_smoothing=label_smoothing)

    def forward(self, y_hat, log_var, target, pose_logits, pose_labels):
        if self.use_nll:
            l_reg = self.nll(y_hat, log_var, target)
        else:
            l_reg = ((y_hat - target) ** 2).mean()

        # direction loss: 1 - cos(pred_xy, target_xy), target가 작으면 마스킹
        if self.dir_weight > 0.0:
            pred_xy   = y_hat[:, :2]
            tgt_xy    = target[:, :2]
            tgt_norm  = tgt_xy.norm(dim=1, keepdim=True).clamp(min=1e-9)
            pred_norm = pred_xy.norm(dim=1, keepdim=True).clamp(min=1e-9)
            cos_sim   = (pred_xy / pred_norm * tgt_xy / tgt_norm).sum(dim=1)  # [B]
            moving    = (tgt_xy.norm(dim=1) > self.dir_norm_min)              # mask
            if moving.any():
                l_dir = (1.0 - cos_sim[moving]).mean()
            else:
                l_dir = torch.zeros(1, device=y_hat.device)[0]
        else:
            l_dir = torch.zeros(1, device=y_hat.device)[0]

        # 분류 손실 (pose_logits=None이면 skip — LLIO 논문 원본 모드)
        # label < 0 인 샘플(TLIO golden 등 레이블 없는 데이터)은 마스킹
        if pose_logits is not None and self.cls_weight > 0.0:
            valid = pose_labels >= 0
            if valid.any():
                l_cls = self.cls(pose_logits[valid], pose_labels[valid])
            else:
                l_cls = torch.zeros(1, device=y_hat.device)[0]
        else:
            l_cls = torch.zeros(1, device=y_hat.device)[0]

        total = l_reg + self.dir_weight * l_dir + self.cls_weight * l_cls

        detail = {
            "reg": l_reg.item(),
            "dir": l_dir.item(),
            "cls": l_cls.item(),
            "total": total.item(),
            "log_var_mean": log_var.mean().item(),
        }
        return total, detail



def compute_class_weights(labels, num_classes=7, method="sqrt_inv"):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()
    counts_safe = counts.clamp(min=1.0)
    if method == "sqrt_inv":
        w = 1.0 / counts_safe.sqrt()
    elif method == "inv_freq":
        w = 1.0 / counts_safe
    else:
        raise ValueError(method)
    w[counts == 0] = 0.0
    present = counts > 0
    if present.any():
        w[present] = w[present] / w[present].mean()
    return w
