"""
Network 단독 궤적 / EKF 필터 궤적 / Ground Truth 궤적 3가지를 한 화면에 시각화.

실행 예시 (src/ 폴더에서):
    python View/visualize_comparison.py \
        --data_path  TLIO_Oxford_Dataset/oxford_handbag_1/imu0_resampled.npy \
        --model_path outputs/out_classifier2/checkpoints/best.pth \
        --norm_mean  outputs/out_classifier2/norm_mean.npy \
        --norm_std   outputs/out_classifier2/norm_std.npy
"""

import sys
from pathlib import Path

# ---------- 경로 설정 ----------
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "Network"))
sys.path.insert(0, str(_SRC / "Trans"))
sys.path.insert(0, str(_SRC))           # tracker/, utils/ 검색

import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from dataset import TLIONpySingleDataset
from model_twolayer import TwoLayerModel
from tracker.scekf import ImuMSCKF
from tracker.imu_buffer import ImuBuffer
from utils.from_scipy import compute_euler_from_matrix
from utils.math_utils import mat_exp

G_NORM = 9.81  # m/s²

# ---------------------------------------------------------------------------
# 상태별 EKF 파라미터
#   meascov_scale : 클수록 네트워크 측정값 불신 (공분산 확대)
#   sigma_na      : 가속도계 노이즈 밀도 (None → 전역 기본값 사용)
#   sigma_ng      : 자이로 노이즈 밀도   (None → 전역 기본값 사용)
#   ita_ba        : 가속도 바이어스 불안정성  (None → 전역 기본값 사용)
#   ita_bg        : 자이로 바이어스 불안정성  (None → 전역 기본값 사용)
# 상태 ID: -1=unknown, 1=handbag, 2=handheld, 3=pocket,
#           4=running,  5=slow-walking, 6=trolley
# ---------------------------------------------------------------------------
STATE_EKF_PARAMS = {
    # state_id: dict(meascov_scale, sigma_na, sigma_ng, ita_ba, ita_bg)
    # None 값은 run_ekf_imutracker 의 전역 파라미터를 그대로 사용함
    # meascov_scale 은 ekf_tune.py 그리드 서치로 자동 튜닝된 값 (2026-05-05)
    -1: dict(meascov_scale=0.001,  sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # unknown
    1:  dict(meascov_scale=0.001,  sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # handbag
    2:  dict(meascov_scale=0.001,  sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # handheld
    3:  dict(meascov_scale=0.001,  sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # pocket
    4:  dict(meascov_scale=0.01,   sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # running
    5:  dict(meascov_scale=1.0,    sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # slow-walking
    6:  dict(meascov_scale=0.001,  sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # trolley
    7:  dict(meascov_scale=0.001,  sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # multi_devices
    8:  dict(meascov_scale=0.01,   sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # multi_users
    9:  dict(meascov_scale=0.1,    sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # large_scale
}


# ---------------------------------------------------------------------------
# 모델 로딩
# ---------------------------------------------------------------------------
def load_model(model_path: str, device: torch.device,
               window_len: int = 100, patch_len: int = 10) -> TwoLayerModel:
    model_para = {
        "input_len": window_len, "input_channel": 6, "patch_len": patch_len,
        "feature_dim": 128, "out_dim": 3, "active_func": "GELU",
        "extractor": {"name": "ResMLP", "layer_num": 6, "expansion": 2, "dropout": 0.0},
        "reg":       {"name": "SimpleMean", "layer_num": 3, "dropout": 0.0},
        "use_classifier": False,
    }
    model = TwoLayerModel(model_para).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# npy 파일에서 원시 IMU 복원
# Oxford 포맷: acc(4:7)은 선형가속도(중력 제거), gravity(7:10)는 body 좌표 중력 단위벡터
# raw_acc = linear_acc + gravity_unit * G_NORM
# ---------------------------------------------------------------------------
def load_npy(data_path: str):
    data = np.load(data_path)
    ts_us    = data[:, 0].astype(np.int64)
    gyr      = data[:, 1:4].astype(np.float64)   # [T, 3] rad/s
    acc_lin  = data[:, 4:7].astype(np.float64)   # [T, 3] linear acc
    grav_b   = data[:, 7:10].astype(np.float64)  # [T, 3] gravity unit vec (body)
    quat     = data[:, 14:18].astype(np.float64) # [T, 4] xyzw world←device
    pos_gt   = data[:, 18:21].astype(np.float64) # [T, 3]
    vel_gt   = data[:, 21:24].astype(np.float64) # [T, 3]

    acc_raw = acc_lin - grav_b * G_NORM           # [T, 3] specific force (accelerometer reading)
    return ts_us, gyr, acc_raw, quat, pos_gt, vel_gt


# ---------------------------------------------------------------------------
# TLIO 17컬럼 포맷 로더 (200Hz world 프레임 → 100Hz body 프레임 변환)
# 컬럼: ts(1) gyr_w(3) acc_lin_w(3) qxyzw(4) pos(3) vel(3)
# ---------------------------------------------------------------------------
def load_npy_tlio(data_path: str):
    data = np.load(data_path)
    # 200Hz → 100Hz 다운샘플링
    data = data[::2]
    T = len(data)

    ts_us      = data[:, 0].astype(np.int64)
    gyr_w      = data[:, 1:4].astype(np.float64)    # world 프레임 자이로
    acc_lin_w  = data[:, 4:7].astype(np.float64)    # world 프레임 선형가속도 (중력 제거됨)
    quat       = data[:, 7:11].astype(np.float64)   # xyzw
    pos_gt     = data[:, 11:14].astype(np.float64)
    vel_gt     = data[:, 14:17].astype(np.float64)

    # world → body 프레임 변환 (EKF 입력용)
    Rs = R.from_quat(quat).as_matrix()              # [T, 3, 3] R_world←body
    # gyr_body = R^T @ gyr_world
    gyr_body = np.einsum("tij,tj->ti", Rs.transpose(0, 2, 1), gyr_w)
    # specific force (body) = R^T @ (acc_lin_world - g_world)
    #   g_world = [0, 0, -9.81] → acc_lin_world - g = acc_lin_world + [0,0,9.81]
    acc_lin_w_no_grav = acc_lin_w.copy()
    acc_lin_w_no_grav[:, 2] += G_NORM               # acc_lin_w - [0,0,-9.81]
    acc_raw_body = np.einsum("tij,tj->ti", Rs.transpose(0, 2, 1), acc_lin_w_no_grav)

    return ts_us, gyr_body, acc_raw_body, quat, pos_gt, vel_gt, gyr_w, acc_lin_w


# ---------------------------------------------------------------------------
# TLIO 17컬럼 포맷 Network 추론 (world 프레임 IMU 직접 사용)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_network_tlio(model: TwoLayerModel, gyr_w, acc_lin_w, quat, pos_gt,
                     device: torch.device, norm_mean: np.ndarray, norm_std: np.ndarray,
                     window_len: int = 100, stride: int = 100):
    """
    world 프레임 gyr/acc_lin 로 직접 추론.
    yaw 정규화 → 정규화 → 모델 → 결과를 world 프레임으로 복원.
    """
    T = len(pos_gt)
    indices = list(range(0, T - window_len, stride))

    all_features = []
    R_yaws = []
    for start in indices:
        end = start + window_len
        r_s  = R.from_quat(quat[start])
        yaw  = r_s.as_euler("zyx", degrees=False)[0]
        Ry   = R.from_euler("z", yaw).as_matrix().astype(np.float32)
        Ry_inv = Ry.T
        acc_yaw = (Ry_inv @ acc_lin_w[start:end].T).T.astype(np.float32)
        gyr_yaw = (Ry_inv @ gyr_w[start:end].T).T.astype(np.float32)
        feat = np.concatenate([acc_yaw, gyr_yaw], axis=1)  # [100, 6] acc 먼저
        feat = (feat - norm_mean) / norm_std
        all_features.append(feat)
        R_yaws.append(Ry)

    # 배치 추론
    batch = torch.from_numpy(
        np.stack(all_features, axis=0).transpose(0, 2, 1)  # [K, 6, 100]
    ).to(device)
    y_hat, _, _ = model(batch)
    preds_local = y_hat.cpu().numpy()  # [K, 3]

    pred_world_steps = np.array([
        R_yaws[k] @ preds_local[k] for k in range(len(indices))
    ], dtype=np.float32)

    net_pos = [pos_gt[0].copy()]
    for step in pred_world_steps:
        net_pos.append(net_pos[-1] + step)
    net_pos = np.array(net_pos, dtype=np.float32)

    return pos_gt.astype(np.float32), net_pos, pred_world_steps, np.array(indices)


# ---------------------------------------------------------------------------
# Network 단독 궤적 (dead-reckoning)
# inference_plot.py 와 동일한 로직
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_network(model: TwoLayerModel, data_path: str, device: torch.device,
                norm_mean: np.ndarray, norm_std: np.ndarray,
                window_len: int, stride: int):
    dataset = TLIONpySingleDataset(
        npy_path=data_path, window_len=window_len, stride=stride,
        normalize=True, precomputed_stats=(norm_mean, norm_std),
        is_train=False, fmt="oxford", with_label=False,
    )

    data    = np.load(data_path)
    quat    = data[:, 14:18].astype(np.float32)
    pos_gt  = data[:, 18:21].astype(np.float32)

    preds_local = []
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    for imu, _ in loader:
        y_hat, _, _ = model(imu.to(device))
        preds_local.append(y_hat.cpu().numpy())
    preds_local = np.concatenate(preds_local, axis=0)  # [K, 3]

    anchor_gt, pred_world_steps = [], []
    for k, start in enumerate(dataset.indices):
        end    = start + window_len
        r_s    = R.from_quat(quat[start])
        yaw    = r_s.as_euler("zyx", degrees=False)[0]
        R_yaw  = R.from_euler("z", yaw).as_matrix().astype(np.float32)
        pred_world_steps.append(R_yaw @ preds_local[k])
        anchor_gt.append(pos_gt[end])

    pred_world_steps = np.array(pred_world_steps)
    anchor_gt        = np.array(anchor_gt)

    net_pos = [pos_gt[0].copy()]
    for step in pred_world_steps:
        net_pos.append(net_pos[-1] + step)
    net_pos = np.array(net_pos, dtype=np.float32)

    return pos_gt, net_pos, pred_world_steps, anchor_gt, dataset.indices


# ---------------------------------------------------------------------------
# TwoLayerMeasSource: ImuTracker 용 측정 소스 (TorchScript 없이 직접 사용)
# ---------------------------------------------------------------------------
class TwoLayerMeasSource:
    """ImuTracker._process_update 가 호출하는 get_displacement_measurement 인터페이스."""

    def __init__(self, model: TwoLayerModel, norm_mean: np.ndarray,
                 norm_std: np.ndarray, device: torch.device,
                 forced_state_id: int = None,
                 meascov_override: float = None,
                 use_soft_switching: bool = True):
        self.model              = model
        self.norm_mean          = norm_mean.astype(np.float32)  # [6]
        self.norm_std           = norm_std.astype(np.float32)   # [6]
        self.device             = device
        self.last_state_id      = -1   # 마지막 추론에서 예측한 상태 ID
        # Method-B: 외부에서 상태 ID 주입 (classifier 없는 모델 대응)
        self.forced_state_id    = forced_state_id   # None → 모델 classifier 결과 사용
        self.meascov_override   = meascov_override  # None → STATE_EKF_PARAMS 값 사용
        # Context-Aware Soft Switching: True → softmax 확률 가중 보간,
        #                               False → argmax hard switching
        self.use_soft_switching = use_soft_switching

    @torch.no_grad()
    def get_displacement_measurement(self, net_gyr_w: np.ndarray,
                                     net_acc_w: np.ndarray):
        """
        net_gyr_w : [N, 3] yaw-정규화 월드 프레임 각속도
        net_acc_w : [N, 3] yaw-정규화 월드 프레임 원시 가속도 (중력 포함)
        returns   : (meas [3,1], meas_cov [3,3]) — yaw-정규화 프레임 변위
        """
        # 중력 제거: net_acc_w = R @ specific_force = acc_lin_world + [0,0,+9.81]
        # → acc_lin_world = net_acc_w - [0, 0, +9.81]
        acc_lin_w = net_acc_w.copy()
        acc_lin_w[:, 2] -= G_NORM

        # [acc | gyr] 순서 (dataset.py 와 동일)
        features = np.concatenate([acc_lin_w, net_gyr_w], axis=1).astype(np.float32)  # [N,6]
        features = (features - self.norm_mean) / self.norm_std

        x = torch.from_numpy(features.T).unsqueeze(0).to(self.device)  # [1,6,N]
        y_hat, log_var, class_logits = self.model(x)

        meas = y_hat[0].cpu().numpy().reshape(3, 1)
        lv   = np.clip(log_var[0].cpu().numpy(), -4.0, 4.0)

        # ── meascov_scale 결정 (우선순위: override > forced > soft/hard switching) ──
        if self.meascov_override is not None:
            # 그리드 서치 등 외부 고정 스케일
            meascov_scale = self.meascov_override
            if self.forced_state_id is not None:
                self.last_state_id = self.forced_state_id
            elif class_logits is not None:
                pred_class = int(class_logits[0].argmax().item()) + 1
                self.last_state_id = pred_class if pred_class in STATE_EKF_PARAMS else -1
            else:
                self.last_state_id = -1
        elif self.forced_state_id is not None:
            # Method-B: 시퀀스 파일명 기반 고정 state_id (classifier 없는 모델 대응)
            self.last_state_id = self.forced_state_id
            meascov_scale = STATE_EKF_PARAMS.get(
                self.forced_state_id, STATE_EKF_PARAMS[-1])["meascov_scale"]
        elif class_logits is not None and self.use_soft_switching:
            # ── Context-Aware Soft Switching ──────────────────────────────
            # softmax 확률을 가중치로 각 상태의 meascov_scale 을 선형 보간:
            #   meascov_scale = Σ_k  p_k * scale_k
            probs = torch.softmax(class_logits[0], dim=0).cpu().numpy()   # [K]
            meascov_scale = float(sum(
                probs[k] * STATE_EKF_PARAMS.get(k + 1, STATE_EKF_PARAMS[-1])["meascov_scale"]
                for k in range(len(probs))
            ))
            pred_class = int(class_logits[0].argmax().item()) + 1
            self.last_state_id = pred_class if pred_class in STATE_EKF_PARAMS else -1
        elif class_logits is not None:
            # Hard Switching: argmax 클래스에 해당하는 고정 scale 사용
            pred_class = int(class_logits[0].argmax().item()) + 1
            state_id   = pred_class if pred_class in STATE_EKF_PARAMS else -1
            self.last_state_id = state_id
            meascov_scale = STATE_EKF_PARAMS.get(state_id, STATE_EKF_PARAMS[-1])["meascov_scale"]
        else:
            # classifier 없는 모델 (state_id=-1, 기본 스케일 사용)
            self.last_state_id = -1
            meascov_scale = STATE_EKF_PARAMS[-1]["meascov_scale"]
        meas_cov = meascov_scale * np.diag(np.exp(lv).astype(np.float64))

        return meas, meas_cov


# ---------------------------------------------------------------------------
# TwoLayerImuTracker: ImuTracker 독립 구현 (import 체인 없음)
# ---------------------------------------------------------------------------
class TwoLayerImuTracker:
    """ImuTracker 핵심 로직을 복제한 독립 클래스 (TorchScript 불필요)."""

    def __init__(self, model: TwoLayerModel, norm_mean: np.ndarray,
                 norm_std: np.ndarray, filter_tuning_cfg,
                 update_freq: float = 1.0, device: torch.device = None,
                 forced_state_id: int = None,
                 meascov_override: float = None,
                 gt_quat: np.ndarray = None,
                 gt_ts_us: np.ndarray = None,
                 use_soft_switching: bool = True):
        if device is None:
            device = next(model.parameters()).device

        imu_freq_net = 100
        window_time  = 1.0

        self.imu_freq_net                = imu_freq_net
        self.past_data_size              = 0
        self.disp_window_size            = int(window_time * imu_freq_net)   # 100
        self.net_input_size              = self.disp_window_size              # 100
        self.update_freq                 = update_freq
        self.clone_every_n_netimu_sample = int(imu_freq_net / update_freq)   # 100
        self.update_distance_num_clone   = int(window_time * update_freq)    # 1
        self.dt_interp_us                = int(1.0 / imu_freq_net * 1e6)     # 10000 μs
        self.dt_update_us                = int(1.0 / update_freq * 1e6)      # 1000000 μs

        # 전역 기본 파라미터 보존 (상태별 override 가 None 일 때 fallback)
        self._default_sigma_na = filter_tuning_cfg.sigma_na
        self._default_sigma_ng = filter_tuning_cfg.sigma_ng
        self._default_ita_ba   = filter_tuning_cfg.ita_ba
        self._default_ita_bg   = filter_tuning_cfg.ita_bg

        # Method-E: GT quaternion yaw 를 측정 프레임 정규화에 사용 (진단 전용)
        self.gt_quat   = gt_quat    # [T, 4] xyzw  None → EKF 추정 rotation 사용 (기본)
        self.gt_ts_us  = gt_ts_us   # [T]   타임스탬프 (μs)

        self.filter      = ImuMSCKF(filter_tuning_cfg)
        self.meas_source = TwoLayerMeasSource(model, norm_mean, norm_std, device,
                                              forced_state_id=forced_state_id,
                                              meascov_override=meascov_override,
                                              use_soft_switching=use_soft_switching)
        self.imu_buffer  = ImuBuffer()

        self.callback_first_update    = None
        self.debug_callback_get_meas  = None
        self.has_done_first_update    = False

        self.last_t_us                        = -1
        self.t_us_before_next_interpolation   = -1
        self.last_acc_before_next_interp_time = None
        self.last_gyr_before_next_interp_time = None
        self.next_interp_t_us = None
        self.next_aug_t_us    = None

    def _add_interpolated_imu_to_buffer(self, acc, gyr, t_us):
        self.imu_buffer.add_data_interpolated(
            self.t_us_before_next_interpolation, t_us,
            self.last_gyr_before_next_interp_time, gyr,
            self.last_acc_before_next_interp_time, acc,
            self.next_interp_t_us,
        )
        self.next_interp_t_us += self.dt_interp_us

    def init_with_state_at_time(self, t_us, Rot, v, p, gyr_raw, acc_raw):
        self.filter.initialize_with_state(t_us, Rot, v, p,
                                          np.zeros((3, 1)), np.zeros((3, 1)))
        self.next_interp_t_us = t_us
        self.next_aug_t_us    = t_us
        self._add_interpolated_imu_to_buffer(acc_raw, gyr_raw, t_us)
        self.next_aug_t_us  = t_us + self.dt_update_us
        self.last_t_us      = t_us
        self.t_us_before_next_interpolation   = t_us
        self.last_acc_before_next_interp_time = acc_raw
        self.last_gyr_before_next_interp_time = gyr_raw

    def on_imu_measurement(self, t_us, gyr_raw, acc_raw):
        do_interp  = t_us >= self.next_interp_t_us
        do_aug_upd = t_us >= self.next_aug_t_us

        t_aug = self.next_aug_t_us if do_aug_upd else None

        if do_interp:
            self._add_interpolated_imu_to_buffer(acc_raw, gyr_raw, t_us)

        self.filter.propagate(acc_raw, gyr_raw, t_us, t_augmentation_us=t_aug)

        if do_aug_upd:
            self._process_update(t_us)
            self.next_aug_t_us += self.dt_update_us

        self.last_t_us = t_us
        if t_us < self.t_us_before_next_interpolation:
            self.t_us_before_next_interpolation   = t_us
            self.last_acc_before_next_interp_time = acc_raw
            self.last_gyr_before_next_interp_time = gyr_raw

    def _get_imu_samples_for_network(self, t_begin_us, t_oldest_state_us, t_end_us):
        net_tus_begin = t_begin_us
        net_tus_end   = t_end_us - self.dt_interp_us
        net_acc, net_gyr, net_tus = self.imu_buffer.get_data_from_to(
            net_tus_begin, net_tus_end
        )
        R_oldest_wfb, _ = self.filter.get_past_state(t_oldest_state_us)
        if self.gt_quat is not None and self.gt_ts_us is not None:
            # Method-E: GT quaternion 의 yaw 로 정규화 (EKF yaw drift 무력화)
            idx_old = int(np.searchsorted(self.gt_ts_us, t_oldest_state_us))
            idx_old = min(idx_old, len(self.gt_quat) - 1)
            ri_z = R.from_quat(self.gt_quat[idx_old]).as_euler("zyx", degrees=False)[0]
        else:
            ri_z = compute_euler_from_matrix(R_oldest_wfb, "xyz", extrinsic=True)[0, 2]
        Ri_z = np.array([
            [ np.cos(ri_z), -np.sin(ri_z), 0],
            [ np.sin(ri_z),  np.cos(ri_z), 0],
            [0, 0, 1],
        ])
        R_oldest_wfb = Ri_z.T @ R_oldest_wfb

        bg = self.filter.state.s_bg
        Rs_bofbi = np.zeros((net_tus.shape[0], 3, 3))
        Rs_bofbi[0] = np.eye(3)
        for j in range(1, net_tus.shape[0]):
            dt_us = net_tus[j] - net_tus[j - 1]
            dR = mat_exp((net_gyr[j].reshape(3, 1) - bg) * dt_us * 1e-6)
            Rs_bofbi[j] = Rs_bofbi[j - 1] @ dR

        oldest_idx = np.where(net_tus == t_oldest_state_us)[0][0]
        R_bofboldstate = R_oldest_wfb @ Rs_bofbi[oldest_idx].T
        Rs_net_wfb = np.einsum("ip,tpj->tij", R_bofboldstate, Rs_bofbi)
        net_acc_w  = np.einsum("tij,tj->ti", Rs_net_wfb, net_acc)
        net_gyr_w  = np.einsum("tij,tj->ti", Rs_net_wfb, net_gyr)
        return net_gyr_w, net_acc_w

    def _process_update(self, t_us):
        if self.filter.state.N <= self.update_distance_num_clone:
            return False
        t_oldest = self.filter.state.si_timestamps_us[
            self.filter.state.N - self.update_distance_num_clone - 1
        ]
        t_begin_us = t_oldest - self.dt_interp_us * self.past_data_size
        t_end_us   = self.filter.state.si_timestamps_us[-1]

        if t_begin_us < self.imu_buffer.net_t_us[0]:
            return False

        if self.debug_callback_get_meas:
            meas, meas_cov = self.debug_callback_get_meas(t_oldest, t_end_us)
        else:
            net_gyr_w, net_acc_w = self._get_imu_samples_for_network(
                t_begin_us, t_oldest, t_end_us
            )
            meas, meas_cov = self.meas_source.get_displacement_measurement(
                net_gyr_w, net_acc_w
            )
            # ── 상태별 W / Q 동적 갱신 ──────────────────────────────────
            state_id    = self.meas_source.last_state_id
            sp          = STATE_EKF_PARAMS.get(state_id, STATE_EKF_PARAMS[-1])
            sigma_na = sp["sigma_na"] if sp["sigma_na"] is not None else self._default_sigma_na
            sigma_ng = sp["sigma_ng"] if sp["sigma_ng"] is not None else self._default_sigma_ng
            ita_ba   = sp["ita_ba"]   if sp["ita_ba"]   is not None else self._default_ita_ba
            ita_bg   = sp["ita_bg"]   if sp["ita_bg"]   is not None else self._default_ita_bg
            var_a  = sigma_na ** 2;  var_g  = sigma_ng ** 2
            var_ba = ita_ba   ** 2;  var_bg = ita_bg   ** 2
            self.filter.W = np.diag([var_g, var_g, var_g, var_a, var_a, var_a])
            self.filter.Q = np.diag([var_bg, var_bg, var_bg, var_ba, var_ba, var_ba])
            # ────────────────────────────────────────────────────────────
        self.filter.update(meas, meas_cov, t_oldest, t_end_us)
        self.has_done_first_update = True

        oldest_idx = self.filter.state.si_timestamps_us.index(t_oldest)
        self.filter.marginalize(oldest_idx)
        self.imu_buffer.throw_data_before(t_begin_us)
        return True


# ---------------------------------------------------------------------------
# EKF 실행 (ImuTracker 기반)
# ---------------------------------------------------------------------------
def _correct_yaw_to_gt(filter_state, ts_us, quat, t_idx):
    """EKF 의 evolving state + 모든 augmented state 의 yaw 를 GT 쿼터니언으로 교정.
    pitch/roll 은 EKF 추정값 유지, yaw 만 GT 로 덮어씀.
    """
    def _yaw_of_R(Rmat):
        return R.from_matrix(Rmat).as_euler("zyx", degrees=False)[0]

    def _apply_delta_yaw(Rmat, delta):
        cd, sd = np.cos(delta), np.sin(delta)
        Rz = np.array([[cd, -sd, 0.], [sd, cd, 0.], [0., 0., 1.]])
        return Rz @ Rmat

    # ── evolving state 교정
    yaw_gt_now = R.from_quat(quat[min(t_idx, len(quat) - 1)]).as_euler("zyx", degrees=False)[0]
    delta = yaw_gt_now - _yaw_of_R(filter_state.s_R)
    filter_state.s_R = _apply_delta_yaw(filter_state.s_R, delta)

    # ── augmented states 교정
    for k in range(len(filter_state.si_Rs)):
        t_k   = filter_state.si_timestamps_us[k]
        idx_k = int(np.searchsorted(ts_us, t_k))
        idx_k = min(idx_k, len(quat) - 1)
        yaw_gt_k = R.from_quat(quat[idx_k]).as_euler("zyx", degrees=False)[0]
        dk = yaw_gt_k - _yaw_of_R(filter_state.si_Rs[k])
        filter_state.si_Rs[k] = _apply_delta_yaw(filter_state.si_Rs[k], dk)


def run_ekf_imutracker(ts_us, gyr, acc_raw, quat, pos_gt, vel_gt,
                       model, norm_mean, norm_std, window_len=100,
                       use_gt_meas=False,
                       meascov_scale=10.0, sigma_na=np.sqrt(1e-3),
                       sigma_ng=np.sqrt(1e-4), init_vel_sigma=1.0,
                       ita_ba=1e-4, ita_bg=1e-6,
                       forced_state_id=None, meascov_override=None,
                       mahalanobis_fail_scale=10.0,
                       gt_yaw_inject=False,
                       use_gt_yaw_norm=False,
                       use_soft_switching: bool = True):
    """ImuTracker 파이프라인으로 EKF 궤적을 반환.
    use_gt_meas=True     : 네트워크 대신 GT 변위를 측정값으로 사용 (디버그)
    use_gt_yaw_norm=True : 측정 프레임 yaw 정규화에 GT quat 사용 (Method-E 진단)
    """
    from types import SimpleNamespace
    from utils.from_scipy import compute_euler_from_matrix

    ekf_cfg = SimpleNamespace(
        sigma_na   = sigma_na,
        sigma_ng   = sigma_ng,
        ita_ba     = ita_ba,
        ita_bg     = ita_bg,
        init_attitude_sigma = 1.0 / 180.0 * np.pi,
        init_yaw_sigma      = 0.1 / 180.0 * np.pi,
        init_vel_sigma      = init_vel_sigma,
        init_pos_sigma      = 0.001,
        init_bg_sigma       = 1e-4,
        init_ba_sigma       = 0.02,
        g_norm              = 9.81,
        meascov_scale       = 1.0,   # 상태별 스케일은 TwoLayerMeasSource 내부에서 적용
        mahalanobis_fail_scale = mahalanobis_fail_scale,
    )

    device  = next(model.parameters()).device
    tracker = TwoLayerImuTracker(
        model, norm_mean, norm_std, ekf_cfg,
        update_freq=1.0, device=device,
        forced_state_id=forced_state_id,
        meascov_override=meascov_override,
        gt_quat=(quat  if use_gt_yaw_norm else None),
        gt_ts_us=(ts_us if use_gt_yaw_norm else None),
        use_soft_switching=use_soft_switching,
    )

    # GT 측정값 콜백 (디버그)
    if use_gt_meas:
        def gt_callback(t_oldest_us, t_end_us):
            idx_old = int(np.searchsorted(ts_us, t_oldest_us))
            idx_end = int(np.searchsorted(ts_us, t_end_us))
            dp_world = (pos_gt[idx_end] - pos_gt[idx_old]).reshape(3, 1)
            if use_gt_yaw_norm:
                # GT yaw 로 정규화 (Method-E)
                idx_clamped = min(idx_old, len(quat) - 1)
                ri_z = R.from_quat(quat[idx_clamped]).as_euler("zyx", degrees=False)[0]
            else:
                R_old, _ = tracker.filter.get_past_state(t_oldest_us)
                ri_z = compute_euler_from_matrix(R_old, "xyz", extrinsic=True)[0, 2]
            Ri_z = np.array([
                [np.cos(ri_z), -np.sin(ri_z), 0],
                [np.sin(ri_z),  np.cos(ri_z), 0],
                [0, 0, 1],
            ])
            meas = Ri_z.T @ dp_world          # yaw 정규화 프레임
            meas_cov = 1e-4 * np.eye(3)       # GT → 완전 신뢰
            return meas, meas_cov
        tracker.debug_callback_get_meas = gt_callback
        print("  [디버그] GT 측정값 사용")

    R0 = R.from_quat(quat[0]).as_matrix()
    v0 = vel_gt[0].reshape(3, 1)
    p0 = pos_gt[0].reshape(3, 1)
    tracker.init_with_state_at_time(
        int(ts_us[0]), R0, v0, p0,
        gyr[0].reshape(3, 1),
        acc_raw[0].reshape(3, 1),
    )

    T = len(ts_us)
    ekf_positions = []
    for i in range(1, T):
        tracker.on_imu_measurement(
            int(ts_us[i]),
            gyr[i].reshape(3, 1),
            acc_raw[i].reshape(3, 1),
        )
        # ── GT yaw 주입 (방안 C): IMU 적분 yaw 오차를 GT 로 교정
        if gt_yaw_inject:
            _correct_yaw_to_gt(tracker.filter.state, ts_us, quat, i)
        _, _, p_ekf, _, _ = tracker.filter.get_evolving_state()
        ekf_positions.append(p_ekf.flatten().copy())

    return np.array(ekf_positions, dtype=np.float32)


# ---------------------------------------------------------------------------
# EKF 실행 (ImuMSCKF 직접 구동)
# ---------------------------------------------------------------------------
def run_ekf(ts_us, gyr, acc_raw, quat, pos_gt, vel_gt,
            pred_world_steps, window_indices, log_vars, window_len):
    from types import SimpleNamespace
    ekf_cfg = SimpleNamespace(
        sigma_na            = np.sqrt(1e-3),
        sigma_ng            = np.sqrt(1e-4),
        ita_ba              = 1e-4,
        ita_bg              = 1e-6,
        init_attitude_sigma = 1.0 / 180.0 * np.pi,
        init_yaw_sigma      = 0.1 / 180.0 * np.pi,
        init_vel_sigma      = 1.0,
        init_pos_sigma      = 0.001,
        init_bg_sigma       = 1e-4,
        init_ba_sigma       = 0.02,
        g_norm              = 9.81,
        meascov_scale       = 10.0,
        mahalanobis_fail_scale = 0,
    )
    ekf = ImuMSCKF(config=ekf_cfg)

    R0 = R.from_quat(quat[0]).as_matrix()
    ekf.initialize_with_state(
        int(ts_us[0]), R0,
        vel_gt[0].reshape(3, 1),
        pos_gt[0].reshape(3, 1),
        np.zeros((3, 1)), np.zeros((3, 1)),
    )

    T = len(ts_us)
    boundary_indices = set()
    for idx in window_indices:
        boundary_indices.add(idx)
        if idx + window_len < T:
            boundary_indices.add(idx + window_len)

    aug_at_next_step = 0 in boundary_indices
    aug_at_self      = boundary_indices - {0}
    win_end = {idx + window_len: k for k, idx in enumerate(window_indices)
               if idx + window_len < T}

    ekf_positions = []
    for i in range(1, T):
        t_us_i = int(ts_us[i])

        t_aug = None
        if i == 1 and aug_at_next_step:
            t_aug = int(ts_us[0])
        elif i in aug_at_self:
            t_aug = t_us_i

        ekf.propagate(
            acc_raw[i].reshape(3, 1),
            gyr[i].reshape(3, 1),
            t_us_i,
            t_augmentation_us=t_aug,
        )

        if i in win_end:
            k          = win_end[i]
            start_idx  = window_indices[k]
            t_begin_us = int(ts_us[start_idx])
            t_end_us   = t_us_i

            meas = pred_world_steps[k].reshape(3, 1).astype(np.float64)
            var_local = np.exp(log_vars[k])
            cov_local = np.diag(var_local.astype(np.float64))
            r_s   = R.from_quat(quat[start_idx])
            yaw   = r_s.as_euler("zyx", degrees=False)[0]
            R_yaw = R.from_euler("z", yaw).as_matrix()
            meas_cov = R_yaw @ cov_local @ R_yaw.T

            try:
                ekf.update(meas, meas_cov, t_begin_us, t_end_us)
                if ekf.state.N > 1:
                    ekf.marginalize(0)
            except Exception:
                pass

        _, _, p_ekf, _, _ = ekf.get_evolving_state()
        ekf_positions.append(p_ekf.flatten().copy())

    return np.array(ekf_positions, dtype=np.float32)


# ---------------------------------------------------------------------------
# 로그 분산 수집 (EKF 공분산 계산용)
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_log_vars(model: TwoLayerModel, data_path: str, device: torch.device,
                     norm_mean: np.ndarray, norm_std: np.ndarray,
                     window_len: int, stride: int):
    dataset = TLIONpySingleDataset(
        npy_path=data_path, window_len=window_len, stride=stride,
        normalize=True, precomputed_stats=(norm_mean, norm_std),
        is_train=False, fmt="oxford", with_label=False,
    )
    from torch.utils.data import DataLoader
    loader    = DataLoader(dataset, batch_size=256, shuffle=False)
    log_vars  = []
    for imu, _ in loader:
        _, lv, _ = model(imu.to(device))
        log_vars.append(lv.cpu().numpy())
    return np.concatenate(log_vars, axis=0)  # [K, 3]


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def plot_comparison(gt_pos, net_pos, ekf_pos, win_indices, window_len,
                    title: str, save_path: str = None):
    """
    gt_pos   : [T, 3]  GT (100Hz 전체)
    net_pos  : [K+1, 3]  Network dead-reckoning (윈도우 단위)
    ekf_pos  : [T-1, 3]  EKF (100Hz, ts[1]~ts[-1])
    win_indices : 각 윈도우 시작 샘플 인덱스 (길이 K)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    origin = gt_pos[0].copy()

    # --- XY 궤적: 각자 자연스러운 샘플링 그대로 표시 ---
    gt_xy  = gt_pos[:, :2]  - origin[:2]
    net_xy = net_pos[:, :2] - origin[:2]
    ekf_xy = ekf_pos[:, :2] - origin[:2]

    ax = axes[0]
    ax.plot(gt_xy[:,0],  gt_xy[:,1],  lw=2.0, label="GT",     alpha=0.8,  color="C0")
    ax.plot(net_xy[:,0], net_xy[:,1], lw=1.8, label="Network", alpha=0.85, color="C1", linestyle="--")
    ax.plot(ekf_xy[:,0], ekf_xy[:,1], lw=1.8, label="EKF",     alpha=0.85, color="C2", linestyle="-.")
    ax.scatter(0, 0, s=50, marker="o", zorder=5, color="k", label="Start")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(); ax.grid(alpha=0.3)

    # --- 오차: 윈도우 앵커(윈도우 끝 샘플)에서만 비교 ---
    # anchor_idxs[k] = win_indices[k] + window_len (윈도우 끝 샘플 인덱스)
    anchor_idxs = np.array([idx + window_len for idx in win_indices])
    valid = anchor_idxs < len(gt_pos)
    anchor_idxs = anchor_idxs[valid]
    K = len(anchor_idxs)

    gt_anchor  = gt_pos[anchor_idxs, :2]  - origin[:2]   # [K, 2]
    net_anchor = net_pos[1:K+1,     :2]   - origin[:2]   # [K, 2]
    # ekf_pos[i] = state at ts[i+1], so ekf_pos[anchor_idx-1] = state at ts[anchor_idx]
    ekf_anchor_idxs = np.clip(anchor_idxs - 1, 0, len(ekf_pos) - 1)
    ekf_anchor = ekf_pos[ekf_anchor_idxs, :2] - origin[:2]  # [K, 2]

    err_net_xy = np.sqrt(np.sum((net_anchor - gt_anchor)**2, axis=1))
    err_ekf_xy = np.sqrt(np.sum((ekf_anchor - gt_anchor)**2, axis=1))
    t_anchors  = anchor_idxs / 100.0  # 100Hz → 초

    ax = axes[1]
    ax.plot(t_anchors, err_net_xy, lw=1.5, label="Network XY err", color="C1", linestyle="--")
    ax.plot(t_anchors, err_ekf_xy, lw=1.5, label="EKF XY err",     color="C2", linestyle="-.")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("XY Error (m)")
    ax.set_title("Positional Error over Time (window anchors)")
    ax.legend(); ax.grid(alpha=0.3)

    rmse_net = float(np.sqrt(np.mean(err_net_xy**2))) if K > 0 else float("nan")
    rmse_ekf = float(np.sqrt(np.mean(err_ekf_xy**2))) if K > 0 else float("nan")
    fig.suptitle(
        f"{title}\n"
        f"RMSE_XY — Network: {rmse_net:.3f} m | EKF: {rmse_ekf:.3f} m",
        fontsize=12
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"저장: {save_path}")
    plt.show()

    return rmse_net, rmse_ekf


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path",  required=True,
                    help="imu0_resampled.npy 경로")
    ap.add_argument("--model_path", required=True,
                    help="체크포인트 .pth 경로")
    ap.add_argument("--norm_mean",  required=True,
                    help="norm_mean.npy 경로")
    ap.add_argument("--norm_std",   required=True,
                    help="norm_std.npy 경로")
    ap.add_argument("--window_len", type=int, default=100)
    ap.add_argument("--stride",     type=int, default=100)
    ap.add_argument("--save",            type=str,   default=None)
    ap.add_argument("--gt_meas",         action="store_true",
                    help="디버그: GT 변위를 EKF 측정값으로 사용")
    # EKF 튜닝 파라미터 (기본값 = TLIO 원본과 동일)
    ap.add_argument("--meascov_scale",   type=float, default=10.0)
    ap.add_argument("--sigma_na",        type=float, default=np.sqrt(1e-3))  # ≈ 0.03162
    ap.add_argument("--sigma_ng",        type=float, default=np.sqrt(1e-4))  # ≈ 0.01
    ap.add_argument("--init_vel_sigma",  type=float, default=1.0)
    ap.add_argument("--ita_ba",          type=float, default=1e-4)
    ap.add_argument("--ita_bg",          type=float, default=1e-6)
    args = ap.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_mean = np.load(args.norm_mean)
    norm_std  = np.load(args.norm_std)

    print(f"모델 로드: {args.model_path}")
    model = load_model(args.model_path, device, args.window_len)

    print(f"데이터 로드: {args.data_path}")
    ncols = np.load(args.data_path).shape[1]
    print(f"  → 컬럼 수: {ncols}  ", end="")

    if ncols == 24:
        # ── Oxford 24컬럼 포맷 ──────────────────────────────────────
        print("(Oxford 포맷)")
        ts_us, gyr, acc_raw, quat, pos_gt, vel_gt = load_npy(args.data_path)

        print("Network 추론 중...")
        gt_pos, net_pos, pred_steps, _, win_indices = run_network(
            model, args.data_path, device, norm_mean, norm_std,
            args.window_len, args.stride,
        )
    elif ncols == 17:
        # ── TLIO 17컬럼 포맷 (200Hz world 프레임) ────────────────────
        print("(TLIO 17컬럼 포맷, 200Hz→100Hz 다운샘플)")
        ts_us, gyr, acc_raw, quat, pos_gt, vel_gt, gyr_w, acc_lin_w = load_npy_tlio(args.data_path)

        print("Network 추론 중...")
        gt_pos, net_pos, pred_steps, win_indices = run_network_tlio(
            model, gyr_w, acc_lin_w, quat, pos_gt,
            device, norm_mean, norm_std,
            args.window_len, args.stride,
        )
    else:
        raise ValueError(f"지원하지 않는 npy 컬럼 수: {ncols} (지원: 17, 24)")

    print("EKF 실행 중 (ImuTracker 파이프라인)...")
    ekf_pos = run_ekf_imutracker(
        ts_us, gyr, acc_raw, quat, pos_gt, vel_gt,
        model, norm_mean, norm_std, args.window_len,
        use_gt_meas=args.gt_meas,
        meascov_scale=args.meascov_scale,
        sigma_na=args.sigma_na,
        sigma_ng=args.sigma_ng,
        init_vel_sigma=args.init_vel_sigma,
        ita_ba=args.ita_ba,
        ita_bg=args.ita_bg,
    )

    title = Path(args.data_path).parent.name
    save_path = args.save or f"comparison_{title}.png"
    rmse_net, rmse_ekf = plot_comparison(
        gt_pos, net_pos, ekf_pos,
        win_indices, args.window_len,
        title, save_path,
    )
    print(f"\nRMSE_XY  Network: {rmse_net:.3f} m  |  EKF: {rmse_ekf:.3f} m")


if __name__ == "__main__":
    main()
