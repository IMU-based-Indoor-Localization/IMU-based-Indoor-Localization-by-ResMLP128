import numpy as np
from .from_scipy import compute_q_from_matrix
from .quiet_numba import jit


def get_rotation_from_gravity(acc):
    ig_w = np.array([0, 0, 1.0]).reshape((3, 1))
    return rot_2vec(acc, ig_w)


def hat(v):
    v = np.squeeze(v)
    R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return R


@jit(nopython=True, parallel=False, cache=True)
def rot_2vec(a, b):
    assert a.shape == (3, 1)
    assert b.shape == (3, 1)

    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    a_n = np.linalg.norm(a)
    b_n = np.linalg.norm(b)
    a_hat = a / a_n
    b_hat = b / b_n
    omega = np.cross(a_hat.T, b_hat.T).T
    c = 1.0 / (1 + np.dot(a_hat.T, b_hat))
    R_ba = np.eye(3) + hat(omega) + c * hat(omega) @ hat(omega)
    return R_ba


@jit(nopython=True, parallel=False, cache=True)
def mat_exp(omega):
    if len(omega) != 3:
        raise ValueError("tangent vector must have length 3")

    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    angle = np.linalg.norm(omega)

    if angle < 1e-10:
        return np.identity(3) + hat(omega)

    axis = omega / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * hat(axis)


mat_exp_vec = np.vectorize(mat_exp, signature="(3)->(3,3)")


def mat_log(R):
    q = compute_q_from_matrix(R)
    w = q[3]
    vec = q[0:3]
    n = np.linalg.norm(vec)
    epsilon = 1e-7

    if n < epsilon:
        w2 = w * w
        n2 = n * n
        atn = 2.0 / w - (2.0 * n2) / (w * w2)
    else:
        if np.absolute(w) < epsilon:
            if w > 0:
                atn = np.pi / n
            else:
                atn = -np.pi / n
        else:
            atn = 2.0 * np.arctan2(n, w) / n
    tangent = atn * vec
    return tangent


@jit(nopython=True, parallel=False, cache=True)
def Jr_exp(phi):
    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    theta = np.linalg.norm(phi)
    if theta < 1e-3:
        J = np.eye(3) - 0.5 * hat(phi) + 1.0 / 6.0 * (hat(phi) @ hat(phi))
    else:
        J = (
            np.eye(3)
            - (1 - np.cos(theta)) / np.power(theta, 2.0) * hat(phi)
            + (theta - np.sin(theta)) / np.power(theta, 3.0) * (hat(phi) @ hat(phi))
        )
    return J
