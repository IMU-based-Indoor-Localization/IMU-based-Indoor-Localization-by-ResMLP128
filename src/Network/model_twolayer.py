"""
model_twolayer.py
-----------------
Phase 2: pose-conditioned regression.

핵심 설계:
    1. extractor → feature [B, P, F]
    2. classifier(feature) → pose_logits [B, C]
    3. regression head 는 (feature ⊕ pose_info) 를 입력받음
         - 학습 시:  pose_info = GT label 의 one-hot (teacher forcing)
         - 평가 시:  pose_info = softmax(pose_logits)
    4. 회귀 head 의 첫 Linear 는 입력 차원이 feature_dim + num_classes 로 확장됨

이로써 classifier 출력이 regression 경로에 직접 영향을 미친다
(이전 병렬 구조에서는 auxiliary task 였을 뿐이었음).
"""

import os, sys
sys.path.append(os.path.dirname(__file__))

import torch
from torch import nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange, Reduce
from model_MLP import PreAffinePostLayerScale, Affine


class ResMLPExtractor(nn.Module):
    def __init__(self, patch_num=10, patch_len=10, input_channel=6,
                 mlp_in_dim=15, expansion=4, active_func=nn.GELU(),
                 layer_num=4, dropout=0.5):
        super().__init__()
        def wrapper(i, fn):
            return PreAffinePostLayerScale(mlp_in_dim, i, fn)
        self.net = nn.Sequential(
            Rearrange('b c (l w) -> b l (w c)', w=patch_len),
            nn.Linear(int(patch_len * input_channel), mlp_in_dim),
            *[
                nn.Sequential(
                    wrapper(i, nn.Conv1d(patch_num, patch_num, 1, bias=False)),
                    wrapper(i, nn.Sequential(
                        nn.Linear(mlp_in_dim, mlp_in_dim * expansion, bias=False),
                        active_func,
                        nn.Dropout(p=dropout, inplace=False),
                        nn.Linear(mlp_in_dim * expansion, mlp_in_dim, bias=False),
                    )),
                ) for i in range(layer_num)
            ],
            Affine(mlp_in_dim),
        )

    def forward(self, x):
        return self.net(x)


class PoseClassifier(nn.Module):
    """feature [B, P, F] → logits [B, num_classes]"""

    def __init__(self, feature_len, num_classes=7, layer_num=2,
                 active_fun=nn.GELU(), dropout=0.5, pooling_type='mean'):
        super().__init__()
        self.net = nn.Sequential(
            Reduce('b l f -> b f', pooling_type),
            nn.LayerNorm(feature_len),
            *[nn.Sequential(
                nn.Linear(feature_len, feature_len),
                nn.LayerNorm(feature_len),
                active_fun,
                nn.Dropout(p=dropout),
            ) for _ in range(layer_num)],
        )
        self.classifier = nn.Linear(feature_len, num_classes)
        nn.init.zeros_(self.classifier.bias)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        return self.classifier(self.net(x))


class PoseConditionedPoolingReg(nn.Module):
    """
    INPUT: feature [B, P, F], pose_info [B, num_classes]
    OUTPUT: y [B, 3], y_cov [B, 3]

    feature 를 pooling → pose_info 와 concat → MLP → (y, y_cov).
    첫 Linear 의 입력 차원이 feature_len + num_classes 로 확장됨.
    """

    def __init__(self, feature_len, num_classes, out_dim=3, layer_num=4,
                 active_fun=nn.GELU(), dropout=0.5, pooling_type='mean'):
        super().__init__()
        self.in_dim = feature_len + num_classes
        self.pool = Reduce('b l f -> b f', pooling_type)

        layers = []
        # 첫 레이어: (feature+pose) → feature_len
        layers.append(nn.Sequential(
            nn.Linear(self.in_dim, feature_len),
            active_fun,
            nn.Dropout(p=dropout),
        ))
        for _ in range(layer_num - 1):
            layers.append(nn.Sequential(
                nn.Linear(feature_len, feature_len),
                active_fun,
                nn.Dropout(p=dropout),
            ))
        self.net = nn.Sequential(*layers)
        self.out_linear  = nn.Linear(feature_len, out_dim)
        self.out2_linear = nn.Linear(feature_len, out_dim)

    def forward(self, feature, pose_info):
        pooled = self.pool(feature)                    # [B, F]
        x = torch.cat([pooled, pose_info], dim=-1)     # [B, F+C]
        out = self.net(x)
        return self.out_linear(out), self.out2_linear(out)


class SimplePoolingReg(nn.Module):
    """LLIO 논문 원본 회귀 헤드: feature pooling → MLP → (y, y_cov).
    Pose 정보 주입 없이 순수 회귀만 수행.
    """

    def __init__(self, feature_len, out_dim=3, layer_num=3,
                 active_fun=nn.GELU(), dropout=0.2, pooling_type='mean'):
        super().__init__()
        self.pool = Reduce('b l f -> b f', pooling_type)
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Sequential(
                nn.Linear(feature_len, feature_len),
                active_fun,
                nn.Dropout(p=dropout),
            ))
        self.net = nn.Sequential(*layers)
        self.out_linear  = nn.Linear(feature_len, out_dim)
        self.out2_linear = nn.Linear(feature_len, out_dim)

    def forward(self, feature):
        x = self.pool(feature)
        out = self.net(x)
        return self.out_linear(out), self.out2_linear(out)


class TwoLayerModel(nn.Module):
    """LLIO 기반 회귀 모델.

    use_classifier=False (기본, 논문 원본):
        feature → 회귀 헤드 → (y, y_cov, None)
        Pose 분류기 없음. 순수 regression.

    use_classifier=True (확장):
        feature → 분류기 → pose 정보 주입 → 회귀 헤드
    """

    def __init__(self, model_para=None):
        super().__init__()

        if model_para is None:
            model_para = {
                "input_len": 100, "input_channel": 6, "patch_len": 25,
                "feature_dim": 256, "out_dim": 3, "active_func": "GELU",
                "extractor": {"name": "ResMLP", "layer_num": 4,
                              "expansion": 2, "dropout": 0.2},
                "reg": {"name": "SimpleMean", "layer_num": 3, "dropout": 0.2},
                "use_classifier": False,
            }

        act = nn.ReLU() if model_para["active_func"] == "ReLU" else nn.GELU()
        self.active_function = act
        self.use_classifier = model_para.get("use_classifier", False)

        patch_num = int(model_para["input_len"] / model_para["patch_len"])
        self.extractor = ResMLPExtractor(
            patch_num=patch_num,
            patch_len=model_para["patch_len"],
            input_channel=model_para["input_channel"],
            mlp_in_dim=model_para["feature_dim"],
            expansion=model_para["extractor"]["expansion"],
            active_func=act,
            layer_num=model_para["extractor"]["layer_num"],
            dropout=model_para["extractor"]["dropout"],
        )

        reg_cfg = model_para["reg"]

        if self.use_classifier:
            cls_cfg = model_para["classifier"]
            self.num_classes = cls_cfg.get("num_classes", 7)
            self.pose_classifier = PoseClassifier(
                feature_len=model_para["feature_dim"],
                num_classes=self.num_classes,
                layer_num=cls_cfg.get("layer_num", 2),
                active_fun=act,
                dropout=cls_cfg.get("dropout", 0.3),
                pooling_type=cls_cfg.get("pooling_type", "mean"),
            )
            pool = "max" if reg_cfg.get("name") == "PoseCondMax" else "mean"
            self.reg = PoseConditionedPoolingReg(
                feature_len=model_para["feature_dim"],
                num_classes=self.num_classes,
                out_dim=model_para["out_dim"],
                layer_num=reg_cfg.get("layer_num", 3),
                active_fun=act,
                dropout=reg_cfg.get("dropout", 0.2),
                pooling_type=pool,
            )
        else:
            # LLIO 논문 원본: 분류기 없는 단순 회귀
            self.reg = SimplePoolingReg(
                feature_len=model_para["feature_dim"],
                out_dim=model_para["out_dim"],
                layer_num=reg_cfg.get("layer_num", 3),
                active_fun=act,
                dropout=reg_cfg.get("dropout", 0.2),
            )

    def forward(self, x, pose_labels=None, tf_ratio: float = 1.0):
        """
        x: [B, input_channel, input_len]
        pose_labels, tf_ratio: use_classifier=True 일 때만 사용.
        반환: (y, y_cov, pose_logits)  — pose_logits은 분류기 없으면 None.
        """
        feature = self.extractor(x)                        # [B, P, F]

        if self.use_classifier:
            pose_logits = self.pose_classifier(feature)    # [B, C]
            pred_info = F.softmax(pose_logits, dim=-1)
            if pose_labels is not None and self.training and tf_ratio > 0.0:
                gt_info = F.one_hot(pose_labels,
                                    num_classes=self.num_classes).float()
                if tf_ratio >= 1.0:
                    pose_info = gt_info
                else:
                    use_gt = (torch.rand(x.size(0), 1, device=x.device) < tf_ratio)
                    pose_info = torch.where(use_gt, gt_info, pred_info)
            else:
                pose_info = pred_info
            y, y_cov = self.reg(feature, pose_info)
        else:
            pose_logits = None
            y, y_cov = self.reg(feature)

        return y, y_cov, pose_logits


if __name__ == '__main__':
    model_para = {
        "input_len": 100, "input_channel": 14, "patch_len": 25,
        "feature_dim": 512, "out_dim": 3, "active_func": "GELU",
        "extractor": {"name": "ResMLP", "layer_num": 6,
                      "expansion": 2, "dropout": 0.2},
        "reg": {"name": "PoseCondMean", "layer_num": 3},
        "classifier": {"num_classes": 7, "layer_num": 2,
                       "dropout": 0.3, "pooling_type": "mean"},
    }
    net = TwoLayerModel(model_para)
    x = torch.rand([8, 14, 100])
    lbl = torch.randint(0, 7, (8,))

    net.train()
    y, yc, pl = net(x, lbl)
    print("train:", y.shape, yc.shape, pl.shape)

    net.eval()
    y, yc, pl = net(x)
    print("eval :", y.shape, yc.shape, pl.shape)
