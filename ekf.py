import numpy as np
from scipy.spatial.transform import Rotation as R

def hat(v):
    """Skew-symmetric matrix for a vector."""
    v = v.flatten()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def Jr_exp(phi):
    """Right Jacobian for the exponential map of SO(3)."""
    phi = phi.flatten()
    theta = np.linalg.norm(phi)
    phi_hat = hat(phi)
    if theta < 1e-4:
        return np.eye(3) - 0.5 * phi_hat + (1.0 / 6.0) * (phi_hat @ phi_hat)
    else:
        return (np.eye(3) - ((1 - np.cos(theta)) / (theta**2)) * phi_hat + 
                ((theta - np.sin(theta)) / (theta**3)) * (phi_hat @ phi_hat))

class State:
    def __init__(self):
        self.R = np.eye(3)
        self.v = np.zeros((3, 1))
        self.p = np.zeros((3, 1))
        self.ba = np.zeros((3, 1))
        self.bg = np.zeros((3, 1))
        self.timestamp_us = -1
        
        # For stochastic cloning
        self.N = 0
        self.past_Rs = []
        self.past_ps = []
        self.past_timestamps_us = []

    def initialize(self, t_us, R_init, v_init, p_init, ba_init, bg_init):
        self.R = R_init
        self.v = v_init
        self.p = p_init
        self.ba = ba_init
        self.bg = bg_init
        self.timestamp_us = int(t_us)
        self.past_Rs = []
        self.past_ps = []
        self.past_timestamps_us = []
        self.N = 0

    def apply_correction(self, dX):
        # dX is [6*N + 15, 1]
        # Past states correction
        if self.N > 0:
            dX_past = dX[:6*self.N].reshape((self.N, 6))
            for i in range(self.N):
                dtheta = dX_past[i, 0:3]
                dp = dX_past[i, 3:6].reshape((3, 1))
                dR = R.from_rotvec(dtheta).as_matrix()
                self.past_Rs[i] = dR @ self.past_Rs[i]
                self.past_ps[i] += dp

        # Current state correction (last 15 elements)
        dX_curr = dX[-15:].flatten()
        dtheta = dX_curr[0:3]
        dv = dX_curr[3:6].reshape((3, 1))
        dp = dX_curr[6:9].reshape((3, 1))
        dbg = dX_curr[9:12].reshape((3, 1))
        dba = dX_curr[12:15].reshape((3, 1))

        dR = R.from_rotvec(dtheta).as_matrix()
        self.R = dR @ self.R
        self.v += dv
        self.p += dp
        self.bg += dbg
        self.ba += dba

class ImuEKF:
    def __init__(self, sigma_na=1e-2, sigma_ng=1e-3, sigma_ba=1e-4, sigma_bg=1e-5):
        self.state = State()
        self.initialized = False
        
        # Noise parameters
        self.sigma_na = sigma_na # Accel noise
        self.sigma_ng = sigma_ng # Gyro noise
        self.sigma_ba = sigma_ba # Accel bias random walk
        self.sigma_bg = sigma_bg # Gyro bias random walk
        
        self.g = np.array([[0], [0], [-9.81]])
        
        self.Sigma = None # Full covariance matrix
        
    def initialize(self, t_us, acc_init, R_init=None, v_init=None, p_init=None, ba_init=None, bg_init=None):
        if R_init is None:
            # Simple gravity alignment for tilt initialization
            acc_n = acc_init / np.linalg.norm(acc_init)
            z_axis = np.array([0, 0, 1]).reshape((3, 1))
            v = np.cross(acc_n.flatten(), z_axis.flatten())
            c = np.dot(acc_n.flatten(), z_axis.flatten())
            s = np.linalg.norm(v)
            if s < 1e-6:
                R_init = np.eye(3)
            else:
                kmat = hat(v)
                R_init = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
        
        v_init = v_init if v_init is not None else np.zeros((3, 1))
        p_init = p_init if p_init is not None else np.zeros((3, 1))
        ba_init = ba_init if ba_init is not None else np.zeros((3, 1))
        bg_init = bg_init if bg_init is not None else np.zeros((3, 1))
        
        self.state.initialize(t_us, R_init, v_init, p_init, ba_init, bg_init)
        
        # Initialize covariance (15x15)
        self.Sigma = np.eye(15) * 1e-4
        self.Sigma[0:3, 0:3] *= 0.1 # Attitude
        self.Sigma[3:6, 3:6] *= 0.1 # Velocity
        self.Sigma[6:9, 6:9] *= 0.01 # Position
        self.Sigma[9:12, 9:12] *= 1e-6 # Gyro bias
        self.Sigma[12:15, 12:15] *= 1e-4 # Accel bias
        
        self.initialized = True

    def propagate(self, t_us, acc, gyr, augment=False):
        dt = (t_us - self.state.timestamp_us) * 1e-6
        if dt <= 0: return
        
        R_k = self.state.R
        v_k = self.state.v
        p_k = self.state.p
        bg_k = self.state.bg
        ba_k = self.state.ba
        
        # Correct measurements
        omega = gyr - bg_k
        acc_b = acc - ba_k
        
        # Propagation (Euler integration for simplicity, can be upgraded to RK4)
        dtheta = omega * dt
        dR = R.from_rotvec(dtheta.flatten()).as_matrix()
        R_next = R_k @ dR
        
        acc_w = R_k @ acc_b + self.g
        v_next = v_k + acc_w * dt
        p_next = p_k + v_k * dt + 0.5 * acc_w * (dt**2)
        
        # Error state transition matrix A (15x15)
        A = np.eye(15)
        A[0:3, 0:3] = dR.T
        A[0:3, 9:12] = -Jr_exp(dtheta) * dt
        A[3:6, 0:3] = -R_k @ hat(acc_b) * dt
        A[3:6, 12:15] = -R_k * dt
        A[6:9, 0:3] = -0.5 * R_k @ hat(acc_b) * (dt**2)
        A[6:9, 3:6] = np.eye(3) * dt
        A[6:9, 12:15] = -0.5 * R_k * (dt**2)

        # Process noise covariance Q
        # Q_imu = diag(sigma_ng^2, sigma_na^2, sigma_bg^2, sigma_ba^2)
        Q = np.zeros((15, 15))
        Q[0:3, 0:3] = (self.sigma_ng**2) * dt * np.eye(3)
        Q[3:6, 3:6] = (self.sigma_na**2) * dt * np.eye(3)
        Q[9:12, 9:12] = (self.sigma_bg**2) * dt * np.eye(3)
        Q[12:15, 12:15] = (self.sigma_ba**2) * dt * np.eye(3)
        
        # State augmentation
        if augment:
            # J_aug is [6, 15] - Jacobian of [R_k, p_k] w.r.t full state
            J_aug = np.zeros((6, 15))
            J_aug[0:3, 0:3] = np.eye(3)
            J_aug[3:6, 6:9] = np.eye(3)
            
            # Augment covariance
            num_past = self.state.N
            new_Sigma = np.zeros((15 + 6*(num_past+1), 15 + 6*(num_past+1)))
            new_Sigma[:6*num_past, :6*num_past] = self.Sigma[:6*num_past, :6*num_past]
            new_Sigma[:6*num_past, 6*num_past:6*num_past+6] = self.Sigma[:6*num_past, -15:] @ J_aug.T
            new_Sigma[6*num_past:6*num_past+6, :6*num_past] = J_aug @ self.Sigma[-15:, :6*num_past]
            new_Sigma[6*num_past:6*num_past+6, 6*num_past:6*num_past+6] = J_aug @ self.Sigma[-15:, -15:] @ J_aug.T
            
            # Propagate current state part
            new_Sigma[-15:, -15:] = A @ self.Sigma[-15:, -15:] @ A.T + Q
            # Cross terms
            new_Sigma[:6*(num_past+1), -15:] = new_Sigma[:6*(num_past+1), 6*num_past:6*num_past+6] @ np.zeros((6, 15)) # Simplified
            # Correct cross terms between past states and new evolving state
            new_Sigma[:6*(num_past+1), -15:] = self.Sigma[:6*(num_past+1), -15:] @ A.T
            new_Sigma[-15:, :6*(num_past+1)] = new_Sigma[:6*(num_past+1), -15:].T
            
            self.Sigma = new_Sigma
            self.state.past_Rs.append(R_k.copy())
            self.state.past_ps.append(p_k.copy())
            self.state.past_timestamps_us.append(self.state.timestamp_us)
            self.state.N += 1
        else:
            # Standard propagation
            last_15 = self.Sigma[-15:, -15:]
            self.Sigma[-15:, -15:] = A @ last_15 @ A.T + Q
            if self.state.N > 0:
                self.Sigma[:-15, -15:] = self.Sigma[:-15, -15:] @ A.T
                self.Sigma[-15:, :-15] = self.Sigma[:-15, -15:].T

        # Update state
        self.state.R = R_next
        self.state.v = v_next
        self.state.p = p_next
        self.state.timestamp_us = t_us

    def update_displacement(self, meas_dp, meas_cov, t_start_us, t_end_us):
        """
        Update with relative displacement predicted by model.
        meas_dp: displacement in local frame (usually start of window)
        """
        try:
            start_idx = self.state.past_timestamps_us.index(t_start_us)
            end_idx = self.state.past_timestamps_us.index(t_end_us) if t_end_us in self.state.past_timestamps_us else -1
        except ValueError:
            return # Timestamps not found
            
        R_i = self.state.past_Rs[start_idx]
        p_i = self.state.past_ps[start_idx]
        
        if end_idx == -1:
            R_j = self.state.R
            p_j = self.state.p
        else:
            R_j = self.state.past_Rs[end_idx]
            p_j = self.state.past_ps[end_idx]
            
        # Predicted displacement in start frame
        # Actually in TLIO, it's often in a gravity-aligned frame, but let's stick to start frame first
        pred_dp = R_i.T @ (p_j - p_i)
        
        # Innovation
        y = meas_dp.reshape((3, 1)) - pred_dp
        
        # Jacobian H
        H = np.zeros((3, self.Sigma.shape[0]))
        
        # H w.r.t p_i
        H_pi = -R_i.T
        H[:, 6*start_idx+3 : 6*start_idx+6] = H_pi
        
        # H w.r.t p_j
        H_pj = R_i.T
        if end_idx == -1:
            H[:, -9:-6] = H_pj
        else:
            H[:, 6*end_idx+3 : 6*end_idx+6] = H_pj
            
        # H w.r.t R_i (rotation error)
        H_theta_i = hat(R_i.T @ (p_j - p_i))
        H[:, 6*start_idx : 6*start_idx+3] = H_theta_i
        
        # Kalman gain
        S = H @ self.Sigma @ H.T + meas_cov
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        
        # Correction
        dX = K @ y
        self.state.apply_correction(dX)
        
        # Covariance update
        I = np.eye(self.Sigma.shape[0])
        self.Sigma = (I - K @ H) @ self.Sigma @ (I - K @ H).T + K @ meas_cov @ K.T
        
    def marginalize(self, keep_from_idx):
        """Remove past states before keep_from_idx."""
        if keep_from_idx <= 0: return
        
        remove_size = 6 * keep_from_idx
        self.Sigma = self.Sigma[remove_size:, remove_size:]
        self.state.past_Rs = self.state.past_Rs[keep_from_idx:]
        self.state.past_ps = self.state.past_ps[keep_from_idx:]
        self.state.past_timestamps_us = self.state.past_timestamps_us[keep_from_idx:]
        self.state.N -= keep_from_idx
