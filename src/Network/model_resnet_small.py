"""
model_resnet_small.py
---------------------
TLIO ResNet1D 의 base_plane / fc_dim 을 파라미터화한 소형 버전.
원본(base_plane=64, fc_dim=512) 대비 채널 수만 줄이며 구조는 동일하게 유지.
"""

import torch.nn as nn


def conv3x1(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x1(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2   = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class FcBlock(nn.Module):
    def __init__(self, in_channel, out_channel, in_dim, prep_channel, fc_dim):
        super().__init__()
        self.prep1 = nn.Conv1d(in_channel, prep_channel, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm1d(prep_channel)
        self.fc1   = nn.Linear(prep_channel * in_dim, fc_dim)
        self.fc2   = nn.Linear(fc_dim, fc_dim)
        self.fc3   = nn.Linear(fc_dim, out_channel)
        self.relu  = nn.ReLU(True)
        self.drop  = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.prep1(x)))
        x = self.relu(self.drop(self.fc1(x.view(x.size(0), -1))))
        x = self.relu(self.drop(self.fc2(x)))
        return self.fc3(x)


class ResNet1DSmall(nn.Module):
    """
    TLIO ResNet1D 와 동일한 구조, base_plane / fc_dim 만 조정 가능.

    base_plane=64, fc_dim=512 → 원본 TLIO 와 동일 (5M params)
    base_plane=16, fc_dim=128 → ~460K params (ResMLP128 과 유사)
    """

    def __init__(self, in_dim=6, out_dim=3, group_sizes=None,
                 inter_dim=4, base_plane=64, fc_dim=512):
        super().__init__()
        if group_sizes is None:
            group_sizes = [2, 2, 2, 2]

        self.base_plane = base_plane
        self.inplanes   = base_plane

        self.input_block = nn.Sequential(
            nn.Conv1d(in_dim, base_plane, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_plane),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.residual_groups = nn.Sequential(
            self._make_group(base_plane * 1, group_sizes[0], stride=1),
            self._make_group(base_plane * 2, group_sizes[1], stride=2),
            self._make_group(base_plane * 4, group_sizes[2], stride=2),
            self._make_group(base_plane * 8, group_sizes[3], stride=2),
        )

        final_ch   = base_plane * 8
        prep_ch    = max(final_ch // 4, 8)   # 원본: 512//4=128
        self.output_block1 = FcBlock(final_ch, out_dim, inter_dim, prep_ch, fc_dim)
        self.output_block2 = FcBlock(final_ch, out_dim, inter_dim, prep_ch, fc_dim)

        self._initialize()

    def _make_group(self, planes, group_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride=stride),
                nn.BatchNorm1d(planes),
            )
        layers = [BasicBlock1D(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, group_size):
            layers.append(BasicBlock1D(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.input_block(x)
        x = self.residual_groups(x)
        return self.output_block1(x), self.output_block2(x)
