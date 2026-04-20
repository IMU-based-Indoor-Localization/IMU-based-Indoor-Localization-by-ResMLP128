#!/usr/bin/env python3
"""
ImuTracker — feeds IMU measurements into the EKF and drives network updates.

Adapted from TLIO's imu_tracker.py.
Changes vs. original:
  - Accepts model/norm_mean/norm_std/net_config directly (no JSON model_param_path).
  - Uses MeasSourcePyTorch instead of MeasSourceTorchScript.
  - Classifier removed (net output tuple is (y, y_cov, _)).
"""

from typing import Optional

import numpy as np
from numba import jit
from scipy.interpolate import interp1d
from tracker.imu_buffer import ImuBuffer
from tracker.imu_calib import ImuCalib
from tracker.meas_source_pytorch import MeasSourcePyTorch
from tracker.scekf import ImuMSCKF
from utils.dotdict import dotdict
from utils.from_scipy import compute_euler_from_matrix
from utils.logging import logging
from utils.math_utils import mat_exp


class ImuTracker:
    """
    Drives the EKF with IMU measurements and periodic network updates.

    net_config keys (required):
        imu_freq_net  : int   — IMU frequency fed to the network (Hz), e.g. 200
        past_time     : float — seconds of context before the displacement window, e.g. 0.0
        window_time   : float — seconds of the displacement window, e.g. 1.0
    """

    def __init__(
        self,
        model,
        norm_mean,
        norm_std,
        net_config: dict,
        update_freq: int,
        filter_tuning_cfg,
        imu_calib: Optional[ImuCalib] = None,
        device=None,
    ):
        cfg = dotdict(net_config)

        if not (cfg.past_time * cfg.imu_freq_net).is_integer():
            raise ValueError("past_time × imu_freq_net must be an integer.")
        if not (cfg.window_time * cfg.imu_freq_net).is_integer():
            raise ValueError("window_time × imu_freq_net must be an integer.")
        if not (cfg.imu_freq_net / update_freq).is_integer():
            raise ValueError("imu_freq_net must be divisible by update_freq.")
        if not (cfg.window_time * update_freq).is_integer():
            raise ValueError("window_time × update_freq must be an integer.")

        self.imu_freq_net = int(cfg.imu_freq_net)
        self.past_data_size = int(cfg.past_time * cfg.imu_freq_net)
        self.disp_window_size = int(cfg.window_time * cfg.imu_freq_net)
        self.net_input_size = self.disp_window_size + self.past_data_size

        self.update_freq = update_freq
        self.clone_every_n_netimu_sample = int(cfg.imu_freq_net / update_freq)
        self.update_distance_num_clone = int(cfg.window_time * update_freq)

        self.dt_interp_us = int(1.0 / self.imu_freq_net * 1e6)
        self.dt_update_us = int(1.0 / self.update_freq * 1e6)

        logging.info(
            f"Net input: {cfg.past_time + cfg.window_time}s "
            f"= {cfg.past_time}s past + {cfg.window_time}s window "
            f"({self.net_input_size} samples)"
        )
        logging.info(f"IMU interp freq: {self.imu_freq_net} Hz  |  Update freq: {self.update_freq} Hz")
        logging.info(f"Update stride: {self.update_distance_num_clone} clones")

        self.icalib = imu_calib
        self.filter = ImuMSCKF(filter_tuning_cfg)
        self.meas_source = MeasSourcePyTorch(model, norm_mean, norm_std, device)
        self.imu_buffer = ImuBuffer()

        self.callback_first_update = None
        self.debug_callback_get_meas = None

        self.last_t_us = -1
        self.t_us_before_next_interpolation = -1
        self.last_acc_before_next_interp_time = None
        self.last_gyr_before_next_interp_time = None

        self.next_interp_t_us = None
        self.next_aug_t_us = None
        self.has_done_first_update = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_imu_measurement(self, t_us: int, gyr_raw, acc_raw) -> bool:
        """Process one IMU sample. Returns True if a filter update occurred."""
        assert isinstance(t_us, int)
        gyr_raw = np.ascontiguousarray(gyr_raw).reshape(3, 1)
        acc_raw = np.ascontiguousarray(acc_raw).reshape(3, 1)
        if self.filter.initialized:
            return self._on_imu_measurement_after_init(t_us, gyr_raw, acc_raw)
        else:
            self._init_without_state_at_time(t_us, gyr_raw, acc_raw)
            return False

    def init_with_state_at_time(self, t_us: int, R, v, p, gyr_raw, acc_raw):
        """Initialize filter with a known state (e.g. from GT at t=0)."""
        gyr_raw = np.ascontiguousarray(gyr_raw).reshape(3, 1)
        acc_raw = np.ascontiguousarray(acc_raw).reshape(3, 1)
        assert R.shape == (3, 3)
        assert v.shape == (3, 1)
        assert p.shape == (3, 1)
        logging.info(f"Initializing filter with known state at t={t_us * 1e-6:.3f}s")
        gyr_b, acc_b, init_bg, init_ba = self._compensate(gyr_raw, acc_raw)
        self.filter.initialize_with_state(t_us, R, v, p, init_ba, init_bg)
        self._after_init_setup(t_us, gyr_b, acc_b)
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compensate(self, gyr_raw, acc_raw):
        if self.icalib:
            init_ba = self.icalib.accelBias
            init_bg = self.icalib.gyroBias
            acc_b, gyr_b = self.icalib.calibrate_raw(acc_raw, gyr_raw)
        else:
            init_ba = np.zeros((3, 1))
            init_bg = np.zeros((3, 1))
            acc_b, gyr_b = acc_raw, gyr_raw
        return gyr_b, acc_b, init_bg, init_ba

    def _after_init_setup(self, t_us: int, gyr_b, acc_b):
        self.next_interp_t_us = t_us
        self.next_aug_t_us = t_us
        self._add_interpolated_imu_to_buffer(acc_b, gyr_b, t_us)
        self.next_aug_t_us = t_us + self.dt_update_us
        self.last_t_us = t_us
        self.t_us_before_next_interpolation = t_us
        self.last_acc_before_next_interp_time = acc_b
        self.last_gyr_before_next_interp_time = gyr_b

    def _init_without_state_at_time(self, t_us: int, gyr_raw, acc_raw):
        assert isinstance(t_us, int)
        logging.info(f"Initializing filter from gravity at t={t_us * 1e-6:.3f}s")
        gyr_b, acc_b, init_bg, init_ba = self._compensate(gyr_raw, acc_raw)
        self.filter.initialize(t_us, acc_b, init_ba, init_bg)
        self._after_init_setup(t_us, gyr_b, acc_b)

    @jit(forceobj=True, parallel=False, cache=False)
    def _get_imu_samples_for_network(self, t_begin_us, t_oldest_state_us, t_end_us):
        net_acc, net_gyr, net_tus = self.imu_buffer.get_data_from_to(
            t_begin_us, t_end_us - self.dt_interp_us
        )
        assert net_gyr.shape[0] == self.net_input_size
        assert net_acc.shape[0] == self.net_input_size

        R_oldest_state_wfb, _ = self.filter.get_past_state(t_oldest_state_us)
        ri_z = compute_euler_from_matrix(R_oldest_state_wfb, "xyz", extrinsic=True)[0, 2]
        Ri_z = np.array(
            [
                [np.cos(ri_z), -(np.sin(ri_z)), 0],
                [np.sin(ri_z), np.cos(ri_z), 0],
                [0, 0, 1],
            ]
        )
        R_oldest_state_wfb = Ri_z.T @ R_oldest_state_wfb

        bg = self.filter.state.s_bg
        Rs_bofbi = np.zeros((net_tus.shape[0], 3, 3))
        Rs_bofbi[0, :, :] = np.eye(3)
        for j in range(1, net_tus.shape[0]):
            dt_us = net_tus[j] - net_tus[j - 1]
            dR = mat_exp((net_gyr[j, :].reshape((3, 1)) - bg) * dt_us * 1e-6)
            Rs_bofbi[j, :, :] = Rs_bofbi[j - 1, :, :].dot(dR)

        oldest_state_idx_in_net = np.where(net_tus == t_oldest_state_us)[0][0]
        R_bofboldstate = R_oldest_state_wfb @ Rs_bofbi[oldest_state_idx_in_net, :, :].T
        Rs_net_wfb = np.einsum("ip,tpj->tij", R_bofboldstate, Rs_bofbi)

        net_acc_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_acc)
        net_gyr_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_gyr)

        return net_gyr_w, net_acc_w

    def _on_imu_measurement_after_init(self, t_us: int, gyr_raw, acc_raw) -> bool:
        assert isinstance(t_us, int)

        if self.icalib:
            acc_b, gyr_b = self.icalib.calibrate_raw(acc_raw, gyr_raw)
            acc_raw, gyr_raw = self.icalib.scale_raw(acc_raw, gyr_raw)
        else:
            acc_b = acc_raw
            gyr_b = gyr_raw

        do_interpolation = t_us >= self.next_interp_t_us
        do_aug_and_update = t_us >= self.next_aug_t_us

        assert (do_aug_and_update and do_interpolation) or not do_aug_and_update

        t_augmentation_us = self.next_aug_t_us if do_aug_and_update else None

        if do_interpolation:
            self._add_interpolated_imu_to_buffer(acc_b, gyr_b, t_us)

        self.filter.propagate(acc_raw, gyr_raw, t_us, t_augmentation_us=t_augmentation_us)

        did_update = False
        if do_aug_and_update:
            did_update = self._process_update(t_us)
            self.next_aug_t_us += self.dt_update_us

        self.last_t_us = t_us

        if t_us < self.t_us_before_next_interpolation:
            self.t_us_before_next_interpolation = t_us
            self.last_acc_before_next_interp_time = acc_b
            self.last_gyr_before_next_interp_time = gyr_b

        return did_update

    def _process_update(self, t_us: int) -> bool:
        logging.debug(f"Update @ {t_us * 1e-6:.3f}s  | N_states={self.filter.state.N}")
        if self.filter.state.N <= self.update_distance_num_clone:
            return False

        t_oldest_state_us = self.filter.state.si_timestamps_us[
            self.filter.state.N - self.update_distance_num_clone - 1
        ]
        t_begin_us = t_oldest_state_us - self.dt_interp_us * self.past_data_size
        t_end_us = self.filter.state.si_timestamps_us[-1]

        if t_begin_us < self.imu_buffer.net_t_us[0]:
            return False

        if not self.has_done_first_update and self.callback_first_update:
            self.callback_first_update(self)

        assert t_begin_us <= t_oldest_state_us

        if self.debug_callback_get_meas:
            meas, meas_cov = self.debug_callback_get_meas(t_oldest_state_us, t_end_us)
        else:
            net_gyr_w, net_acc_w = self._get_imu_samples_for_network(
                t_begin_us, t_oldest_state_us, t_end_us
            )
            meas, meas_cov = self.meas_source.get_displacement_measurement(net_gyr_w, net_acc_w)

        self.filter.update(meas, meas_cov, t_oldest_state_us, t_end_us)
        self.has_done_first_update = True

        oldest_idx = self.filter.state.si_timestamps_us.index(t_oldest_state_us)
        self.filter.marginalize(oldest_idx)
        self.imu_buffer.throw_data_before(t_begin_us)
        return True

    def _add_interpolated_imu_to_buffer(self, acc_b, gyr_b, t_us: int):
        self.imu_buffer.add_data_interpolated(
            self.t_us_before_next_interpolation,
            t_us,
            self.last_gyr_before_next_interp_time,
            gyr_b,
            self.last_acc_before_next_interp_time,
            acc_b,
            self.next_interp_t_us,
        )
        self.next_interp_t_us += self.dt_interp_us
