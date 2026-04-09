from math import exp
import os
import sys

sys.path.append(os.path.dirname(__file__))

import einops
import torch
from torch import nn

from einops.layers.torch import *
from torch.nn.modules.dropout import Dropout

from model_MLP import *


# ===== 원본 코드 (수정 없음) =============================================

class ResMLPExtractor(nn.Module):
    def __init__(self, patch_num=10,
                 patch_len=10,
                 input_channel=6,
                 mlp_in_dim=15,
                 expansion=4,
                 active_func=nn.GELU(),
                 layer_num=4,
                 dropout=0.5):
        '''
        INPUT: [Batch_num, input_dim(6 for 6-axis IMU), measurement_length]
        OUTPUT: [Batch_num, patch_num, feature_length]
        '''
        super(ResMLPExtractor, self).__init__()

        def wrapper(i, fn): return PreAffinePostLayerScale(mlp_in_dim, i, fn)

        self.net = nn.Sequential(
            ##### Feature Convert
            Rearrange('b c (l w) -> b l (w c)', w=patch_len),
            nn.Linear(int(patch_len * input_channel), mlp_in_dim),
            #### ResMLP Module
            *[
                nn.Sequential(
                    wrapper(i, nn.Conv1d(patch_num, patch_num, 1, bias=False)),
                    wrapper(i, nn.Sequential(
                        nn.Linear(mlp_in_dim, mlp_in_dim *
                                  expansion, bias=False),
                        active_func,
                        # [변경 사유] 원본은 inplace=True 이나,
                        # extractor 출력을 reg/classifier 두 곳에서 동시에
                        # 사용할 경우 in-place 연산이 autograd graph를 손상시킴.
                        # 분류 헤드 추가에 따른 최소 필수 변경.
                        nn.Dropout(p=dropout, inplace=False),
                        nn.Linear(mlp_in_dim * expansion,
                                  mlp_in_dim, bias=False)
                    ))
                ) for i in range(layer_num)
            ],
            Affine(mlp_in_dim)
        )

    def forward(self, x):
        return self.net(x)


class ResLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SimpleMLPReg(nn.Module):
    def __init__(self, patch_num=10, feature_len=10, out_dim=3, layer_num=4, active_fun=nn.GELU(), dropout=0.5):
        '''
        INPUT: [Batch_size Patch_num Feature_lenght]
        OUTPUT: [Batch_size, out_dim] [Batch_size, out_dim] (y, y_cov)
        '''

        super(SimpleMLPReg, self).__init__()
        self.input_len = int(patch_num * feature_len)
        self.dropout = nn.Dropout(p=dropout)
        self.net = nn.Sequential(
            Rearrange('b l f -> b (l f)'),
            *[nn.Sequential(
                nn.Linear(self.input_len, self.input_len),
                active_fun,
                self.dropout
            )
                for i in range(layer_num)
            ]
        )
        self.out_linear = nn.Linear(self.input_len, out_dim)
        self.out2_linear = nn.Linear(self.input_len, out_dim)

    def forward(self, x):
        out = self.net(x)
        return self.out_linear(out), self.out2_linear(out)


class PoolingMLPReg(nn.Module):
    def __init__(self, patch_num=10,
                 feature_len=10,
                 out_dim=3,
                 layer_num=4,
                 active_fun=nn.GELU(),
                 dropout=0.5,
                 pooling_type='mean'):
        '''
        INPUT: [Batch_size Patch_num Feature_lenght]
        OUTPUT: [Batch_size, out_dim] [Batch_size, out_dim] (y, y_cov)
        '''

        super(PoolingMLPReg, self).__init__()
        self.input_len = int(feature_len)
        self.dropout = nn.Dropout(p=dropout)
        self.net = nn.Sequential(
            Reduce('b l f-> b f', pooling_type),
            *[nn.Sequential(
                nn.Linear(self.input_len, self.input_len),
                active_fun,
                self.dropout
            )
                for i in range(layer_num)
            ]
        )
        self.out_linear = nn.Linear(self.input_len, out_dim)
        self.out2_linear = nn.Linear(self.input_len, out_dim)

    def forward(self, x):
        out = self.net(x)
        return self.out_linear(out), self.out2_linear(out)


# ===== 추가된 코드 ========================================================

class PoseClassifier(nn.Module):
    """
    자세 분류 헤드. extractor 출력을 공유해 분기 처리.

    INPUT : [B, patch_num, feature_dim]   (ResMLPExtractor 출력)
    OUTPUT: [B, num_classes]              (raw logits)

    Args:
        feature_len  : feature_dim (ResMLPExtractor의 mlp_in_dim)
        num_classes  : 7 (OXIOD: -1→0, 1~6→1~6)
        layer_num    : FC 레이어 수
        active_fun   : 활성화 함수 (backbone과 통일 권장)
        dropout      : 드롭아웃 비율
        pooling_type : patch 축 풀링 방식 ('mean' | 'max')
    """

    def __init__(self,
                 feature_len: int,
                 num_classes: int = 7,
                 layer_num: int = 2,
                 active_fun: nn.Module = nn.GELU(),
                 dropout: float = 0.5,
                 pooling_type: str = 'mean'):
        super(PoseClassifier, self).__init__()
        self.net = nn.Sequential(
            Reduce('b l f -> b f', pooling_type),
            *[nn.Sequential(
                nn.Linear(feature_len, feature_len),
                nn.LayerNorm(feature_len),
                active_fun,
                nn.Dropout(p=dropout)
            ) for _ in range(layer_num)]
        )
        self.classifier = nn.Linear(feature_len, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.net(x))


# ===== 원본 코드 (TwoLayerModel에 classifier 블록만 추가) =================

class TwoLayerModel(nn.Module):
    def __init__(self, model_para=None):
        super(TwoLayerModel, self).__init__()

        if model_para is None:
            model_para = {
                "input_len": 100,
                "input_channel": 6,
                "patch_len": 10,
                "feature_dim": 20,
                "out_dim": 3,
                "active_func": "GELU",
                "extractor": {
                    "name": "ResMLP",
                    "layer_num": 4,
                    "expansion": 4,
                },
                "reg": {
                    "name": "MLP",
                    "layer_num": 3,
                }
                # classifier 키가 없으면 분류 헤드 비활성화 → 원본과 동일 동작
            }

        self.active_function = None
        if model_para["active_func"] == "GELU":
            self.active_function = nn.GELU()
        elif model_para["active_func"] == "ReLU":
            self.active_function = nn.ReLU()
        else:
            self.active_function = nn.GELU()

        patch_num = int(model_para["input_len"] / model_para["patch_len"])
        if model_para["extractor"]["name"] == "ResMLP":
            self.extractor = ResMLPExtractor(patch_num=patch_num, patch_len=model_para["patch_len"],
                                             input_channel=model_para["input_channel"],
                                             mlp_in_dim=model_para["feature_dim"],
                                             expansion=model_para["extractor"]["expansion"],
                                             active_func=self.active_function,
                                             layer_num=model_para["extractor"]["layer_num"],
                                             dropout=model_para["extractor"]["dropout"]
                                             )

        else:
            print("unknown extractor name:", model_para["extractor"]["name"])

        if model_para["reg"]["name"] == "MLP":
            self.reg = SimpleMLPReg(patch_num=patch_num, feature_len=model_para["feature_dim"],
                                    layer_num=model_para["reg"]["layer_num"], active_fun=self.active_function)
        elif model_para['reg']['name'] == "MaxMLP":
            self.reg = PoolingMLPReg(patch_num=patch_num,

                                     out_dim=model_para['out_dim'],
                                     feature_len=model_para['feature_dim'],
                                     layer_num=model_para['reg']['layer_num'],
                                     active_fun=self.active_function,
                                     pooling_type='max')
        elif model_para['reg']['name'] == 'MeanMLP':
            self.reg = PoolingMLPReg(patch_num=patch_num, feature_len=model_para['feature_dim'],
                                     out_dim=model_para['out_dim'],
                                     layer_num=model_para['reg']['layer_num'], active_fun=self.active_function,
                                     pooling_type='mean')
        else:
            print("unknown reg name: ", model_para['reg']['name'])

        # ---- [추가] 분류 헤드 (classifier 키가 없으면 비활성화) ----
        cls_cfg = model_para.get("classifier", None)
        self.use_classifier = cls_cfg is not None
        if self.use_classifier:
            self.pose_classifier = PoseClassifier(
                feature_len  = model_para["feature_dim"],
                num_classes  = cls_cfg.get("num_classes", 7),
                layer_num    = cls_cfg.get("layer_num", 2),
                active_fun   = self.active_function,
                dropout      = cls_cfg.get("dropout", 0.5),
                pooling_type = cls_cfg.get("pooling_type", "mean"),
            )

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feature = self.extractor(x)
        out1, out2 = self.reg(feature)

        # ---- [추가] 분류 헤드 분기 ----
        if self.use_classifier:
            pose_logits = self.pose_classifier(feature)
            return out1, out2, pose_logits

        return out1, out2


# ===== 원본 __main__ (수정 없음) =========================================

if __name__ == '__main__':
    model_para = {
        "input_len": 100,
        "input_channel": 6,
        "patch_len": 25,
        "feature_dim": 512,
        "out_dim": 3,
        "active_func": "GELU",
        "extractor": {
            "name": "ResMLP",
            "layer_num": 6,
            "expansion": 2,
            "dropout": 0.2,
        },
        "reg": {
            "name": "MeanMLP",
            "layer_num": 3,
        }
    }

    net = TwoLayerModel(model_para)
    x = torch.rand([512, 6, 100])
    y, y_cov = net(x)
    print('x:', x.shape, 'y', y.shape, 'y_cov', y_cov.shape)

    # ---- 분류 헤드 추가 예시 ----
    model_para["classifier"] = {
        "num_classes" : 7,
        "layer_num"   : 2,
        "dropout"     : 0.3,
        "pooling_type": "mean",
    }
    net_cls = TwoLayerModel(model_para)
    x = torch.rand([512, 6, 100])
    y, y_cov, pose_logits = net_cls(x)
    print('x:', x.shape, 'y', y.shape, 'y_cov', y_cov.shape,
          'pose_logits', pose_logits.shape)
