import numpy as np
import torch
from utils.logging import logging


class MeasSourcePyTorch:
    """
    Measurement source that wraps TwoLayerModel for EKF updates.

    The tracker calls get_displacement_measurement(net_gyr_w, net_acc_w)
    where both arrays are in world/local frame, shape [N, 3].

    Key differences from TLIO's TorchScript source:
    - Model loaded directly (not TorchScript)
    - Input channel order: [acc, gyr] (our training convention)
    - Input normalized with precomputed mean/std
    - y_cov output is log(sigma^2); converted to diagonal covariance sigma^2 = exp(y_cov)
    """

    def __init__(self, model, norm_mean, norm_std, device=None):
        """
        Args:
            model: TwoLayerModel instance (already moved to device, in eval mode)
            norm_mean: np.ndarray [6], channel-wise mean from training
            norm_std:  np.ndarray [6], channel-wise std from training
            device:    torch.device (defaults to model's device)
        """
        self.model = model
        self.norm_mean = norm_mean.astype(np.float32)   # [6]
        self.norm_std = norm_std.astype(np.float32)     # [6]
        self.device = device if device is not None else next(model.parameters()).device
        logging.info(f"MeasSourcePyTorch ready on device {self.device}")

    def get_displacement_measurement(self, net_gyr_w, net_acc_w, clip_small_disp=False):
        """
        Args:
            net_gyr_w: [N, 3] gyroscope in local gravity-aligned frame
            net_acc_w: [N, 3] accelerometer in local gravity-aligned frame
        Returns:
            meas:     [3, 1] predicted displacement (local frame, metres)
            meas_cov: [3, 3] diagonal covariance matrix
        """
        # --- build input [N, 6] with acc first, gyr second (training convention) ---
        features = np.concatenate([net_acc_w, net_gyr_w], axis=1).astype(np.float32)  # [N, 6]

        # --- normalize ---
        features = (features - self.norm_mean) / self.norm_std  # [N, 6]

        # --- convert to [1, 6, N] tensor ---
        features_t = torch.from_numpy(features.T).float().unsqueeze(0).to(self.device)  # [1, 6, N]

        with torch.no_grad():
            y, y_cov, _ = self.model(features_t)   # y: [1,3],  y_cov: [1,3]

        # --- displacement prediction ---
        meas = y.cpu().numpy().reshape(3, 1)        # [3, 1]

        # --- covariance conversion ---
        # y_cov = log(sigma^2)  (from GaussianNLLLoss convention)
        # Actual variance: sigma^2 = exp(y_cov)
        log_var = y_cov.cpu().numpy()[0]            # [3]
        log_var = np.clip(log_var, -8.0, 8.0)       # numerical guard
        var = np.exp(log_var)                        # [3]  sigma^2
        meas_cov = np.diag(var)                     # [3, 3] diagonal

        if clip_small_disp and np.linalg.norm(meas) < 0.001:
            meas = np.zeros_like(meas)

        return meas, meas_cov
