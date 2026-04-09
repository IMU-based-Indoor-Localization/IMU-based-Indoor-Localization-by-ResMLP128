"""
losses.py
---------
LLIO + Pose Classification 학습에 사용하는 손실 함수 모음

포함된 손실:
    1. GaussianNLLLoss  — 불확실성 추정 포함 회귀 손실 (TLIO 계열 논문 표준)
    2. PoseLoss         — CrossEntropy 기반 자세 분류 손실
    3. CombinedLoss     — 두 손실을 가중합하는 메인 손실

수식 (Gaussian NLL):
    L_reg = mean( 0.5 * exp(-log_var) * ||y - y_hat||^2 + 0.5 * log_var )
    where log_var = model의 y_cov 출력 (log σ²를 직접 예측)

    → y_cov가 작을수록(확실할수록) 회귀 오차에 더 큰 패널티
    → y_cov가 너무 커지면 log_var 항이 패널티를 부여 → 자동 균형
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# ---------------------------------------------------------------------------
# 1. Gaussian NLL Regression Loss
# ---------------------------------------------------------------------------

class GaussianNLLLoss(nn.Module):
    """
    불확실성 추정을 포함한 회귀 손실.
    모델이 예측값(y_hat)과 log-variance(log_var)를 동시에 출력할 때 사용.

    Args:
        reduction : 'mean' | 'sum' | 'none'
        eps       : log_var 클리핑 하한 (수치 안정성)

    Input:
        y_hat  : [B, D]  — 모델 위치 예측 (TwoLayerModel의 첫 번째 출력)
        log_var: [B, D]  — 모델 불확실성 출력 (TwoLayerModel의 두 번째 출력 = y_cov)
        target : [B, D]  — 정답 변위

    Note:
        log_var = log(σ²) 를 직접 예측하는 방식 사용.
        σ² 대신 log(σ²)를 예측하면 항상 양수가 보장되고 학습이 안정적.
    """

    def __init__(self, reduction: str = "mean", eps: float = -10.0):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.eps = eps  # log_var 최솟값 클리핑

    def forward(
        self,
        y_hat:   torch.Tensor,
        log_var: torch.Tensor,
        target:  torch.Tensor,
    ) -> torch.Tensor:
        """
        L = 0.5 * exp(-log_var) * (y - y_hat)² + 0.5 * log_var
        """
        log_var = torch.clamp(log_var, min=self.eps)

        loss = 0.5 * torch.exp(-log_var) * (target - y_hat) ** 2 \
             + 0.5 * log_var              # [B, D]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # [B, D]
# ---------------------------------------------------------------------------
# 2. Pose Classification Loss
# ---------------------------------------------------------------------------

class PoseLoss(nn.Module):
    """
    자세 분류 CrossEntropy 손실.

    Args:
        class_weights : 클래스 불균형 보정용 가중치 텐서 [num_classes].
                        None이면 균등 가중치.
                        compute_class_weights() 헬퍼로 자동 계산 가능.
        label_smoothing: 레이블 스무딩 (0.0~0.1 권장)
        ignore_noise_class: True이면 class 0(noise/-1)을 손실 계산에서 제외.
                            noise 레이블이 섞인 배치에서 유용.

    Input:
        logits: [B, num_classes]  — PoseClassifier 출력 (raw logits)
        labels: [B]               — 클래스 인덱스 (0~6)
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        ignore_noise_class: bool = False,
    ):
        super().__init__()
        ignore_index = 0 if ignore_noise_class else -100  # -100 = PyTorch 기본값 (무시 없음)
        self.ce = nn.CrossEntropyLoss(
            weight          = class_weights,
            label_smoothing = label_smoothing,
            ignore_index    = ignore_index,
        )
        self.ignore_noise_class = ignore_noise_class

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.ce(logits, labels)


# ---------------------------------------------------------------------------
# 3. Combined Loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    회귀 손실 + 분류 손실의 가중합.

        L_total = L_reg + cls_weight * L_cls

    Args:
        cls_weight          : 분류 손실 가중치 (기본 0.3).
                              처음엔 낮게(0.1~0.2) 시작 후 증가 권장.
        class_weights       : PoseLoss에 전달할 클래스 가중치
        label_smoothing     : PoseLoss 레이블 스무딩
        ignore_noise_class  : noise 클래스(0) 분류 손실 무시 여부
        nll_eps             : GaussianNLLLoss 클리핑 하한

    Input:
        y_hat      : [B, D]          — 회귀 예측
        log_var    : [B, D]          — 불확실성 예측 (y_cov)
        target     : [B, D]          — 회귀 정답
        pose_logits: [B, num_classes] — 분류 logits
        pose_labels: [B]             — 분류 정답 (클래스 인덱스 0~6)

    Returns:
        total_loss : scalar
        detail     : dict {"reg": float, "cls": float}  ← 로깅용
    """

    def __init__(
        self,
        cls_weight:          float = 0.3,
        class_weights:       Optional[torch.Tensor] = None,
        label_smoothing:     float = 0.0,
        ignore_noise_class:  bool  = False,
        nll_eps:             float = -10.0,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_loss   = GaussianNLLLoss(eps=nll_eps)
        self.cls_loss   = PoseLoss(
            class_weights      = class_weights,
            label_smoothing    = label_smoothing,
            ignore_noise_class = ignore_noise_class,
        )

    def forward(
        self,
        y_hat:       torch.Tensor,
        log_var:     torch.Tensor,
        target:      torch.Tensor,
        pose_logits: torch.Tensor,
        pose_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:

        l_reg = self.reg_loss(y_hat, log_var, target)
        l_cls = self.cls_loss(pose_logits, pose_labels)
        total = l_reg + self.cls_weight * l_cls

        detail = {
            "reg" : l_reg.item(),
            "cls" : l_cls.item(),
            "total": total.item(),
        }
        return total, detail


# ---------------------------------------------------------------------------
# 헬퍼: 클래스 불균형 보정 가중치 계산
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int = 7,
    method: str = "inv_freq",
) -> torch.Tensor:
    """
    레이블 텐서로부터 클래스 가중치를 계산합니다.

    Args:
        labels      : 전체 데이터셋 레이블 [N] (클래스 인덱스)
        num_classes : 클래스 수 (OXIOD: 7)
        method      : 'inv_freq'  — 빈도 역수 가중치 (희귀 클래스 ↑)
                      'sqrt_inv'  — sqrt(빈도 역수) (완만한 보정)
                      'effective' — Effective Number of Samples 방법

    Returns:
        weights: FloatTensor [num_classes]

    Example:
        all_labels = torch.cat([label for _, _, label in train_dataset])
        weights = compute_class_weights(all_labels, num_classes=7)
        criterion = CombinedLoss(class_weights=weights.to(device))
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()

    # 0인 클래스 처리 (데이터에 없는 클래스)
    counts = counts.clamp(min=1.0)

    if method == "inv_freq":
        weights = 1.0 / counts
    elif method == "sqrt_inv":
        weights = 1.0 / counts.sqrt()
    elif method == "effective":
        beta = 0.9999
        weights = (1 - beta) / (1 - beta ** counts)
    else:
        raise ValueError(f"알 수 없는 method: {method}")

    # 평균이 1이 되도록 정규화
    weights = weights / weights.mean()
    return weights


# ---------------------------------------------------------------------------
# 빠른 검증
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, D, C = 32, 3, 7

    y_hat   = torch.randn(B, D)
    log_var = torch.randn(B, D) * 0.5
    target  = torch.randn(B, D)
    logits  = torch.randn(B, C)
    labels  = torch.randint(0, C, (B,))

    # --- 개별 손실 ---
    nll = GaussianNLLLoss()
    print(f"GaussianNLL : {nll(y_hat, log_var, target):.4f}")

    ce = PoseLoss()
    print(f"PoseLoss    : {ce(logits, labels):.4f}")

    # --- 합산 손실 ---
    combined = CombinedLoss(cls_weight=0.3)
    total, detail = combined(y_hat, log_var, target, logits, labels)
    print(f"CombinedLoss: {total:.4f}  detail={detail}")

    # --- 클래스 가중치 계산 ---
    all_labels = torch.tensor([0]*10 + [1]*50 + [2]*80 + [3]*30 +
                               [4]*100 + [5]*60 + [6]*20)
    weights = compute_class_weights(all_labels, num_classes=7)
    print(f"\n클래스 가중치 (inv_freq): {weights.numpy().round(3)}")
    weights_sqrt = compute_class_weights(all_labels, num_classes=7, method="sqrt_inv")
    print(f"클래스 가중치 (sqrt_inv): {weights_sqrt.numpy().round(3)}")
