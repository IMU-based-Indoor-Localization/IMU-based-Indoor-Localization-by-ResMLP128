"""
export_resnet_small.py
----------------------
src/Network/out_resnet_small/checkpoints/best.pth 를
PyTorch Mobile Lite (.ptl) 로 익스포트한다.

실행:
    anaconda3\\python.exe export_resnet_small.py
"""

import os
import shutil
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

# ── 모델 정의 (EKF 브랜치 model_resnet_small.py 인라인) ──────

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

        final_ch = base_plane * 8
        prep_ch  = max(final_ch // 4, 8)
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
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_block(x)
        x = self.residual_groups(x)
        return self.output_block1(x), self.output_block2(x)


# ── 경로 설정 ─────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH  = os.path.join(ROOT, "src", "Network", "out_resnet_small", "checkpoints", "best.pth")
OUT_PTL    = os.path.join(ROOT, "mobile_assets", "imu_model.ptl")
ASSETS_PTL = os.path.join(ROOT, "android", "app", "src", "main", "assets", "imu_model.ptl")

# config.json 파라미터: base_plane=19, fc_dim=166, window_len=100
BASE_PLANE = 19
FC_DIM     = 166
WINDOW_LEN = 100

# ── 모델 생성 및 가중치 로드 ──────────────────────────────────
print(f"[1/4] 모델 생성: ResNet1DSmall(base_plane={BASE_PLANE}, fc_dim={FC_DIM})")
model = ResNet1DSmall(base_plane=BASE_PLANE, fc_dim=FC_DIM)
model.eval()

print(f"[2/4] 체크포인트 로드: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
print(f"      체크포인트 키: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

if isinstance(ckpt, dict):
    if "model" in ckpt:
        state = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
else:
    state = ckpt

model.load_state_dict(state)
print(f"      파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# ── 형상 검증 ─────────────────────────────────────────────────
print(f"[3/4] 형상 검증: 입력 [1, 6, {WINDOW_LEN}]")
dummy = torch.zeros(1, 6, WINDOW_LEN)
with torch.no_grad():
    disp, cov = model(dummy)
print(f"      disp shape: {disp.shape}  (기대: [1, 3])")
print(f"      cov  shape: {cov.shape}   (기대: [1, 3])")

# ── TorchScript → Mobile 변환 ─────────────────────────────────
print(f"[4/4] TorchScript 변환 및 .ptl 저장 (optimize_for_mobile 제외): {OUT_PTL}")
with torch.no_grad():
    traced = torch.jit.trace(model, dummy)
    # optimize_for_mobile 생략 - PyTorch 2.x 에서 호환성 문제 발생 가능
    traced_eval = torch.jit.freeze(traced)
os.makedirs(os.path.dirname(OUT_PTL), exist_ok=True)
traced_eval._save_for_lite_interpreter(OUT_PTL)

size_mb = os.path.getsize(OUT_PTL) / 1024 / 1024
print(f"      저장 완료: {size_mb:.2f} MB")

# assets 폴더에도 복사
shutil.copy2(OUT_PTL, ASSETS_PTL)
print(f"      assets 복사: {ASSETS_PTL}")

print("\n✅ 완료! Android Studio 에서 Rebuild 후 테스트하세요.")
