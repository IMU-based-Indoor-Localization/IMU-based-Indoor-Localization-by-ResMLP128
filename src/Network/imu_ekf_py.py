"""
imu_ekf_py.py — Stochastic-Cloning EKF (Python self-contained)
==============================================================
C++ android/app/src/main/cpp/ekf/imu_ekf.cpp 의 SC-EKF 를 numpy 만으로
재구현한다. 본 파일 단독으로 동작하도록 외부 의존성을 최소화한다.

지원하는 식은 단말 EKF 의 핵심 경로만:
  - propagate(acc, gyr, t_us, t_aug_us=None)
      · IMU 100Hz strapdown 적분
      · t_aug_us 가 주어지면 클론 추가 (state augmentation)
  - update(meas, meas_cov, t_begin_us, t_end_us)
      · 두 클론 간 변위를 yaw-anchored gravity-aligned frame 으로 관측
      · 절대 innov gate (MAX_INNOV_NORM) + Mahalanobis χ² gate (11.345)
      · Joseph form 공분산 갱신

마진alization / ZUPT / position hold / freeze 등 단말 전용 헬퍼는 포함
하지 않는다 (오프라인 비교 도구 compare_tlio_ekf.py 의 비교 범위 밖).

용도:
  같은 IMU 시퀀스에 대해 *두 cfg 변형* (현재 앱 cfg vs TLIO 논문 cfg) 으로
  같은 식의 EKF 를 돌려 궤적 차이만 격리 측정한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SO(3) 유틸리티 — scipy 의존 회피 위해 직접 구현
# ─────────────────────────────────────────────────────────────────────────────
def hat(v: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric."""
    return np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],  0.0,  -v[0]],
        [-v[1], v[0],  0.0],
    ])


def mat_exp(omega: np.ndarray) -> np.ndarray:
    """Rodrigues SO(3) exponential.  omega ∈ R^3 → 3×3 rotation."""
    angle = float(np.linalg.norm(omega))
    if angle < 1e-10:
        return np.eye(3)
    axis = omega / angle
    K = hat(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def Jr_exp(omega: np.ndarray) -> np.ndarray:
    """SO(3) right Jacobian of exp(omega).  C++ imu_ekf.cpp 동일 식."""
    angle = float(np.linalg.norm(omega))
    if angle < 1e-10:
        return np.eye(3)
    K = hat(omega) / angle
    s = np.sin(angle)
    c = np.cos(angle)
    return (np.eye(3)
            - ((1.0 - c) / angle) * K
            + ((angle - s) / angle) * (K @ K))


def rot_from_gravity(acc: np.ndarray) -> np.ndarray:
    """Initial R from a single static acc reading.  C++ get_rotation_from_gravity 동일."""
    ig_w = np.array([0.0, 0.0, 1.0])
    a = acc / max(float(np.linalg.norm(acc)), 1e-12)
    axis = np.cross(a, ig_w)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(a, ig_w))
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])  # 180° about X
    K = hat(axis / s)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


# ─────────────────────────────────────────────────────────────────────────────
# cfg
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FilterConfig:
    """C++ FilterConfig (imu_ekf.h:144~) 와 1:1 대응.

    기본값은 *현재 단말 cfg* 그대로다 (배포 default).  TLIO 비교를 위해선
    아래 4 개를 덮어쓰면 된다:
      init_vel_sigma : 1.0  → 0.1
      init_ba_sigma  : 0.02 → 0.2
      meascov_scale  : 1.0  → 10.0
      (init_pos_sigma 는 TLIO 가 'strong prior' 라고만 기술 — 본 비교에선
       gauge fix 용 작은 값을 유지)
    """
    g_norm: float = 9.81
    # process noise (continuous-time std)
    sigma_na: float = np.sqrt(1e-3)
    sigma_ng: float = np.sqrt(1e-4)
    ita_ba:   float = 1e-4
    ita_bg:   float = 1e-6
    # init covariance (std)
    init_attitude_sigma: float = 10.0 / 180.0 * np.pi
    init_yaw_sigma:      float = 0.1  / 180.0 * np.pi
    init_vel_sigma:      float = 1.0
    init_pos_sigma:      float = 0.001
    init_bg_sigma:       float = 1e-4
    init_ba_sigma:       float = 0.02
    # measurement
    meascov_scale:         float = 1.0
    mahalanobis_fail_scale: float = 0.0   # 0 이면 NSE>11.345 시 update skip
    # gates
    #   max_innov_norm: 단말 imu_ekf.cpp 의 절대 게이트(=3.0m) — TLIO 논문 식엔
    #     없는 *단말 전용 안전망*. 비교 도구 default 는 비활성(1e9)로 두고,
    #     필요 시 호출자가 명시적으로 3.0 등으로 설정한다.
    #   chi2_thresh: TLIO §V-D 와 동일 (3-DOF χ² 99%).
    max_innov_norm: float = 1e9
    chi2_thresh:    float = 11.345


# 두 비교용 cfg
def current_cfg() -> FilterConfig:
    """현재 단말 앱의 imu_ekf.cpp default — 그대로."""
    return FilterConfig()


def tlio_cfg() -> FilterConfig:
    """TLIO 논문 §V-D / §V-E 값으로 EKF 계수만 교체.

    - σ_v          0.1 m/s   (논문 §V-E)
    - σ_ba         0.2 m/s²
    - σ_bg         1e-4 rad/s   (현재와 동일)
    - σ_θ (roll/pitch) 10°       (현재와 동일)
    - σ_θ (yaw)        0.1°      (현재와 동일)
    - meascov_scale ×10          (논문 §V-D 끝: temporal correlation 보정)
    - χ² gate 11.345 (3-DOF 99%) (현재와 동일)
    process noise (sigma_na/ng, ita_ba/bg) 는 BMI055 와 다른 단말 IMU 라 그대로 둔다.
    """
    c = FilterConfig()
    c.init_vel_sigma  = 0.1
    c.init_ba_sigma   = 0.2
    c.meascov_scale   = 10.0
    return c


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class State:
    R:  np.ndarray = field(default_factory=lambda: np.eye(3))   # IMU→world
    v:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    p:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    t_us: int = -1
    # past clones
    si_Rs: List[np.ndarray] = field(default_factory=list)
    si_ps: List[np.ndarray] = field(default_factory=list)
    si_timestamps_us: List[int] = field(default_factory=list)

    @property
    def N(self) -> int:
        return len(self.si_Rs)


# ─────────────────────────────────────────────────────────────────────────────
# 핵심: propagate_rvt_and_jac — C++ 와 동일
# ─────────────────────────────────────────────────────────────────────────────
def _propagate_rvt_and_jac(R_k, v_k, p_k, bg_k, ba_k, gyr, acc, g, dt):
    """반환: (R_new, v_new, p_new, A_15x15)."""
    omega_c = gyr - bg_k
    acc_c   = acc - ba_k

    dtheta = omega_c * dt
    dR     = mat_exp(dtheta)
    Rd     = R_k @ dR

    dv_w = R_k @ acc_c * dt
    dp_w = 0.5 * dv_w * dt
    vd   = v_k + dv_w + g * dt
    pd   = p_k + v_k * dt + dp_w + g * 0.5 * dt * dt

    A = np.eye(15)
    A[3:6,  0:3]  = -hat(dv_w)
    A[6:9,  0:3]  = -hat(dp_w)
    A[6:9,  3:6]  = np.eye(3) * dt
    A[0:3,  9:12] = -Rd @ Jr_exp(dtheta) * dt
    A[3:6, 12:15] = -R_k * dt
    A[6:9, 12:15] = -0.5 * R_k * dt * dt
    return Rd, vd, pd, A


# ─────────────────────────────────────────────────────────────────────────────
# ScEkf
# ─────────────────────────────────────────────────────────────────────────────
class ScEkf:
    def __init__(self, cfg: FilterConfig):
        self.cfg     = cfg
        self.state   = State()
        self.Sigma   = np.zeros((15, 15))
        self.initialized = False
        self._build_noise()

    # ─── noise ────────────────────────────────────────────────────────────
    def _build_noise(self):
        c = self.cfg
        # W (6×6): [n_g; n_a] 입력 노이즈 PSD
        self.W = np.zeros((6, 6))
        self.W[0:3, 0:3] = np.eye(3) * (c.sigma_ng ** 2)
        self.W[3:6, 3:6] = np.eye(3) * (c.sigma_na ** 2)
        # Q (15×15): bias random walk
        self.Q15 = np.zeros((15, 15))
        self.Q15[9:12,  9:12]  = np.eye(3) * (c.ita_bg ** 2)
        self.Q15[12:15, 12:15] = np.eye(3) * (c.ita_ba ** 2)

    def _reset_cov(self):
        c = self.cfg
        d = np.array([
            c.init_attitude_sigma**2, c.init_attitude_sigma**2, c.init_yaw_sigma**2,
            c.init_vel_sigma**2,      c.init_vel_sigma**2,      c.init_vel_sigma**2,
            c.init_pos_sigma**2,      c.init_pos_sigma**2,      c.init_pos_sigma**2,
            c.init_bg_sigma**2,       c.init_bg_sigma**2,       c.init_bg_sigma**2,
            c.init_ba_sigma**2,       c.init_ba_sigma**2,       c.init_ba_sigma**2,
        ])
        self.Sigma = np.diag(d)

    # ─── init ─────────────────────────────────────────────────────────────
    def initialize(self, t_us: int, acc0: np.ndarray,
                   ba_init: Optional[np.ndarray] = None,
                   bg_init: Optional[np.ndarray] = None,
                   R0: Optional[np.ndarray] = None):
        self.state = State()
        self.state.R    = R0 if R0 is not None else rot_from_gravity(acc0)
        self.state.ba   = np.zeros(3) if ba_init is None else ba_init.copy()
        self.state.bg   = np.zeros(3) if bg_init is None else bg_init.copy()
        self.state.t_us = int(t_us)
        self._reset_cov()
        self.initialized = True

    # ─── propagate ────────────────────────────────────────────────────────
    def propagate(self, acc: np.ndarray, gyr: np.ndarray,
                  t_us: int, t_aug_us: Optional[int] = None):
        if not self.initialized:
            return
        c = self.cfg
        g = np.array([0.0, 0.0, -c.g_norm])
        dt_us = float(t_us - self.state.t_us)
        dt    = dt_us * 1e-6
        if dt <= 0.0:
            return

        Rd, vd, pd, A = _propagate_rvt_and_jac(
            self.state.R, self.state.v, self.state.p,
            self.state.bg, self.state.ba, gyr, acc, g, dt)

        # B (15×6): gyro→θ, acc→v, acc→p
        B = np.zeros((15, 6))
        B[0:3, 0:3] = -A[0:3,  9:12]
        B[3:6, 3:6] = -A[3:6, 12:15]
        B[6:9, 3:6] = -A[6:9, 12:15]

        N = self.state.N
        do_aug = (t_aug_us is not None and t_aug_us >= 0)

        if do_aug:
            dtd = float(t_aug_us - self.state.t_us) * 1e-6
            Rd_a, vd_a, pd_a, A_a = _propagate_rvt_and_jac(
                self.state.R, self.state.v, self.state.p,
                self.state.bg, self.state.ba, gyr, acc, g, dtd)

            # JA (6×15): clone (θ, p) 를 현재 오차 상태로 선형화
            JA = np.zeros((6, 15))
            JA[0:3, :] = A_a[0:3, :]
            JA[3:6, :] = A_a[6:9, :]

            old_sz = 15 + 6 * N
            new_sz = old_sz + 6

            A_aug = np.zeros((new_sz, old_sz))
            if N > 0:
                A_aug[0:6*N, 0:6*N] = np.eye(6*N)
            A_aug[6*N:6*N+6, old_sz-15:old_sz] = JA
            A_aug[6*N+6:,    old_sz-15:old_sz] = A

            BJ = np.zeros((6, 6))
            BJ[0:3, 0:3] = -A_a[0:3,  9:12]
            BJ[3:6, 3:6] = -A_a[6:9, 12:15]

            B_aug = np.zeros((new_sz, 6))
            B_aug[6*N:6*N+6, :] = BJ
            B_aug[6*N+6:,    :] = B

            Q_aug = np.zeros((new_sz, new_sz))
            Q_aug[new_sz-15:, new_sz-15:] = self.Q15

            self.Sigma = (A_aug @ self.Sigma @ A_aug.T
                          + B_aug @ self.W @ B_aug.T * dt
                          + Q_aug * dt)
            self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)

            # clone 추가 (부분 적분 결과)
            self.state.si_Rs.append(Rd_a.copy())
            self.state.si_ps.append(pd_a.copy())
            self.state.si_timestamps_us.append(int(t_aug_us))
        else:
            sz = 15 + 6 * N
            A_aug = np.eye(sz)
            A_aug[sz-15:, sz-15:] = A
            B_aug = np.zeros((sz, 6))
            B_aug[sz-15:, :] = B
            Q_aug = np.zeros((sz, sz))
            Q_aug[sz-15:, sz-15:] = self.Q15

            self.Sigma = (A_aug @ self.Sigma @ A_aug.T
                          + B_aug @ self.W @ B_aug.T * dt
                          + Q_aug * dt)
            self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)

        # 상태 갱신
        self.state.R    = Rd
        self.state.v    = vd
        self.state.p    = pd
        self.state.t_us = int(t_us)

        # 속도 클램핑 (C++ 와 동일 5 m/s)
        spd = float(np.linalg.norm(self.state.v))
        if spd > 5.0:
            self.state.v *= 5.0 / spd

    # ─── update ───────────────────────────────────────────────────────────
    def update(self, meas: np.ndarray, meas_cov: np.ndarray,
               t_begin_us: int, t_end_us: int) -> dict:
        """반환: {'applied': bool, 'reason': str, 'innov_norm': float, 'nse': float}."""
        out = {"applied": False, "reason": "", "innov_norm": 0.0, "nse": 0.0}
        if not self.initialized:
            out["reason"] = "not_initialized"; return out
        ts = self.state.si_timestamps_us
        if t_begin_us not in ts or t_end_us not in ts:
            out["reason"] = "clone_ts_missing"; return out
        bi = ts.index(t_begin_us)
        ei = ts.index(t_end_us)
        if bi >= ei:
            out["reason"] = "bi_>=_ei"; return out

        c = self.cfg
        R_meas = c.meascov_scale * meas_cov
        R_meas = 0.5 * (R_meas + R_meas.T)

        Ri = self.state.si_Rs[bi]
        # ZYX (extrinsic XYZ) 오일러 직접 (C++ 와 동일)
        ri_z = float(np.arctan2(Ri[1, 0], Ri[0, 0]))
        ri_y = float(np.arctan2(-Ri[2, 0], np.sqrt(Ri[0,0]**2 + Ri[1,0]**2)))
        if abs(np.cos(ri_y)) < 1e-5:
            out["reason"] = "singular"; return out

        cz, sz_ = np.cos(ri_z), np.sin(ri_z)
        Ri_z = np.array([[cz, -sz_, 0.0],
                         [sz_,  cz, 0.0],
                         [0.0, 0.0, 1.0]])
        pred = Ri_z.T @ (self.state.si_ps[ei] - self.state.si_ps[bi])

        N = self.state.N
        sz = 15 + 6 * N
        H = np.zeros((3, sz))
        H[:, 6*bi + 3 : 6*bi + 6] = -Ri_z.T
        H[:, 6*ei + 3 : 6*ei + 6] =  Ri_z.T

        cy = np.cos(ri_y); sy = np.sin(ri_y)
        tan_y = sy / cy
        Hz = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [cz * tan_y, sz_ * tan_y, 1.0]])
        dp_eb = self.state.si_ps[ei] - self.state.si_ps[bi]
        H[:, 6*bi : 6*bi + 3] = Ri_z.T @ hat(dp_eb) @ Hz

        S = H @ self.Sigma @ H.T + R_meas
        innov = meas - pred
        innov_norm = float(np.linalg.norm(innov))
        out["innov_norm"] = innov_norm

        if innov_norm > c.max_innov_norm:
            out["reason"] = f"innov_gate({innov_norm:.2f}>{c.max_innov_norm})"
            return out

        S_inv = np.linalg.inv(S)
        nse = float(innov @ S_inv @ innov)
        out["nse"] = nse
        if nse > c.chi2_thresh:
            if c.mahalanobis_fail_scale <= 0.0:
                out["reason"] = f"chi2_gate({nse:.2f}>{c.chi2_thresh})"
                return out
            R_meas = c.mahalanobis_fail_scale * R_meas
            S      = H @ self.Sigma @ H.T + R_meas
            S_inv  = np.linalg.inv(S)

        # Kalman + Joseph form
        K = self.Sigma @ H.T @ S_inv
        dX = (K @ innov).ravel()
        self._apply_correction(dX)
        I = np.eye(sz)
        IKH = I - K @ H
        self.Sigma = IKH @ self.Sigma @ IKH.T + K @ R_meas @ K.T
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
        out["applied"] = True
        return out

    # ─── correction ───────────────────────────────────────────────────────
    def _apply_correction(self, dX: np.ndarray):
        N = self.state.N
        # past
        for i in range(N):
            dtheta = dX[6*i     : 6*i + 3]
            dp     = dX[6*i + 3 : 6*i + 6]
            dR = mat_exp(dtheta)
            self.state.si_Rs[i] = dR @ self.state.si_Rs[i]
            self.state.si_ps[i] = self.state.si_ps[i] + dp
        # current (last 15)
        d  = dX[6*N:]
        dR = mat_exp(d[0:3])
        self.state.R  = dR @ self.state.R
        self.state.v  = self.state.v + d[3:6]
        self.state.p  = self.state.p + d[6:9]
        self.state.bg = self.state.bg + d[9:12]
        self.state.ba = self.state.ba + d[12:15]

    # ─── marginalize old clones (FIFO) ────────────────────────────────────
    def marginalize_until(self, keep_after_ts_us: int):
        """keep_after_ts_us 이전 모든 clone 제거 (corresponding cov rows/cols 도)."""
        N = self.state.N
        if N == 0:
            return
        # 제거할 인덱스
        ts = self.state.si_timestamps_us
        keep = [i for i, t in enumerate(ts) if t >= keep_after_ts_us]
        if len(keep) == N:
            return
        drop = [i for i in range(N) if i not in keep]
        # 새 covariance: 6*keep + 15 evol
        new_idx = []
        for i in keep:
            new_idx += [6*i, 6*i+1, 6*i+2, 6*i+3, 6*i+4, 6*i+5]
        new_idx += list(range(6*N, 6*N + 15))
        self.Sigma = self.Sigma[np.ix_(new_idx, new_idx)]
        self.state.si_Rs              = [self.state.si_Rs[i] for i in keep]
        self.state.si_ps              = [self.state.si_ps[i] for i in keep]
        self.state.si_timestamps_us   = [ts[i] for i in keep]
