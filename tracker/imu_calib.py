import numpy as np


class ImuCalib:
    """IMU calibration parameters (scale, bias)."""

    def __init__(self):
        self.accelScaleInv = np.eye(3)
        self.gyroScaleInv = np.eye(3)
        self.gyroGSense = np.zeros((3, 3))
        self.accelBias = np.zeros((3, 1))
        self.gyroBias = np.zeros((3, 1))

    def calibrate_raw(self, acc, gyr):
        """Apply scale and bias correction. acc/gyr shape: (3, 1)."""
        acc_cal = self.accelScaleInv @ acc - self.accelBias
        gyr_cal = self.gyroScaleInv @ gyr - self.gyroGSense @ acc - self.gyroBias
        return acc_cal, gyr_cal

    def scale_raw(self, acc, gyr):
        acc_cal = self.accelScaleInv @ acc
        gyr_cal = self.gyroScaleInv @ gyr - self.gyroGSense @ acc
        return acc_cal, gyr_cal
