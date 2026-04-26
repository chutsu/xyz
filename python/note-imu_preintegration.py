#!/usr/bin/env python3
import numpy as np
from numpy import sqrt
from numpy import eye
from numpy import zeros
from numpy import sin
from numpy import cos
from numpy import sinc
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.linalg import cholesky as chol


def sec2ts(time_s):
  """Convert time in seconds to timestamp"""
  return np.int64(time_s * 1e9)


def ts2sec(ts):
  """Convert timestamp to seconds"""
  return np.float64(ts) * 1e-9


def skew(vec):
  """Form skew-symmetric matrix from vector `vec`"""
  assert vec.shape == (3,) or vec.shape == (3, 1)

  if vec.shape == (3,):
    x = vec[0]
    y = vec[1]
    z = vec[2]
  else:
    x = vec[0][0]
    y = vec[1][0]
    z = vec[2][0]

  return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_vec(q):
  """Returns vector part of a quaternion"""
  _, qx, qy, qz = q
  return np.array([qx, qy, qz])


def quat_norm(q):
  """Returns norm of a quaternion"""
  qw, qx, qy, qz = q
  return sqrt(qw**2 + qx**2 + qy**2 + qz**2)


def quat_normalize(q):
  """Normalize quaternion"""
  n = quat_norm(q)
  qw, qx, qy, qz = q
  return np.array([qw / n, qx / n, qy / n, qz / n])


def quat_conj(q):
  """Return conjugate quaternion"""
  qw, qx, qy, qz = q
  q_conj = np.array([qw, -qx, -qy, -qz])
  return q_conj


def quat_inv(q):
  """Invert quaternion"""
  return quat_conj(q)


def quat_left(q):
  """Quaternion left product matrix"""
  qw, qx, qy, qz = q
  row0 = [qw, -qx, -qy, -qz]
  row1 = [qx, qw, -qz, qy]
  row2 = [qy, qz, qw, -qx]
  row3 = [qz, -qy, qx, qw]
  return np.array([row0, row1, row2, row3])


def quat_right(q):
  """Quaternion right product matrix"""
  qw, qx, qy, qz = q
  row0 = [qw, -qx, -qy, -qz]
  row1 = [qx, qw, qz, -qy]
  row2 = [qy, -qz, qw, qx]
  row3 = [qz, qy, -qx, qw]
  return np.array([row0, row1, row2, row3])


def quat_lmul(p, q):
  """Quaternion left multiply"""
  assert len(p) == 4
  assert len(q) == 4
  lprod = quat_left(p)
  return lprod @ q


def quat_rmul(p, q):
  """Quaternion right multiply"""
  assert len(p) == 4
  assert len(q) == 4
  rprod = quat_right(q)
  return rprod @ p


def quat_mul(p, q):
  """Quaternion multiply p * q"""
  return quat_lmul(p, q)


def quat_rot(q, x):
  """Rotate vector x of size 3 by Quaternion q"""
  # y = q * p * q_conj
  q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
  p = np.array([0.0, x[0], x[1], x[2]])
  p_new = quat_mul(quat_mul(q, p), q_conj)
  return np.array([p_new[1], p_new[2], p_new[3]])


def quat_omega(w):
  """Quaternion omega matrix"""
  Omega = np.zeros((4, 4))
  Omega[0, 1:4] = -w.T
  Omega[1:4, 0] = w
  Omega[1:4, 1:4] = -skew(w)
  return Omega


def quat_delta(dalpha):
  """Form quaternion from small angle rotation vector dalpha"""
  half_norm = 0.5 * norm(dalpha)
  scalar = cos(half_norm)
  vector = sinc(half_norm) * 0.5 * dalpha

  dqw = scalar
  dqx, dqy, dqz = vector
  dq = np.array([dqw, dqx, dqy, dqz])

  return dq


def quat_integrate(q_k, w, dt):
  """
  Sola, Joan. "Quaternion kinematics for the error-state Kalman filter." arXiv
  preprint arXiv:1711.02508 (2017).
  [Section 4.6.1 Zeroth-order integration, p.47]
  """
  w_norm = norm(w)
  q_scalar = 0.0
  q_vec = np.array([0.0, 0.0, 0.0])

  if w_norm > 1e-5:
    q_scalar = cos(w_norm * dt * 0.5)
    q_vec = w / w_norm * sin(w_norm * dt * 0.5)
  else:
    q_scalar = 1.0
    q_vec = [0.0, 0.0, 0.0]

  q_kp1 = quat_mul(q_k, np.array([q_scalar, q_vec]))
  return q_kp1


def quat2rot(q):
  """
  Convert quaternion to 3x3 rotation matrix.

  Source:
  Blanco, Jose-Luis. "A tutorial on se (3) transformation parameterizations
  and on-manifold optimization." University of Malaga, Tech. Rep 3 (2010): 6.
  [Page 18, Equation (2.20)]
  """
  assert len(q) == 4
  qw, qx, qy, qz = q

  qx2 = qx**2
  qy2 = qy**2
  qz2 = qz**2
  qw2 = qw**2

  # Homogeneous form
  C11 = qw2 + qx2 - qy2 - qz2
  C12 = 2.0 * (qx * qy - qw * qz)
  C13 = 2.0 * (qx * qz + qw * qy)

  C21 = 2.0 * (qx * qy + qw * qz)
  C22 = qw2 - qx2 + qy2 - qz2
  C23 = 2.0 * (qy * qz - qw * qx)

  C31 = 2.0 * (qx * qz - qw * qy)
  C32 = 2.0 * (qy * qz + qw * qx)
  C33 = qw2 - qx2 - qy2 + qz2

  return np.array([[C11, C12, C13], [C21, C22, C23], [C31, C32, C33]])


class ImuBuffer:
  """IMU buffer"""
  def __init__(self, ts=None, acc=None, gyr=None):
    self.ts = ts if ts is not None else []
    self.acc = acc if acc is not None else []
    self.gyr = gyr if gyr is not None else []

  def add(self, ts, acc, gyr):
    """Add imu measurement"""
    self.ts.append(ts)
    self.acc.append(acc)
    self.gyr.append(gyr)

  def length(self):
    """Return length of imu buffer"""
    return len(self.ts)


class ImuPreintegrator:
  def __init__(self, imu_buf, imu_params, ba_i, bg_i):
    self.Dt = 0.0
    self.g = np.array([0.0, 0.0, 9.81])
    self.state_F = np.eye(15)  # State jacobian
    self.state_P = np.zeros((15, 15))  # State covariance
    self.ba_i = ba_i
    self.bg_i = bg_i

    # Noise matrix Q
    Q = zeros((12, 12))
    Q[0:3, 0:3] = imu_params.noise_acc**2 * eye(3)
    Q[3:6, 3:6] = imu_params.noise_gyr**2 * eye(3)
    Q[6:9, 6:9] = imu_params.noise_ba**2 * eye(3)
    Q[9:12, 9:12] = imu_params.noise_bg**2 * eye(3)

    # Pre-integrate relative position, velocity, rotation and biases
    dr = np.array([0.0, 0.0, 0.0])  # Relative position
    dv = np.array([0.0, 0.0, 0.0])  # Relative velocity
    dq = np.array([1.0, 0.0, 0.0, 0.0])  # Relative rotation

    # Pre-integrate imu measuremenets
    for k in range(len(imu_buf.ts) - 1):
      # Timestep
      ts_i = imu_buf.ts[k]
      ts_j = imu_buf.ts[k + 1]
      dt = ts2sec(ts_j - ts_i)
      dt_sq = dt * dt

      # Accelerometer and gyroscope measurements
      acc_i = imu_buf.acc[k]
      gyr_i = imu_buf.gyr[k]

      # Propagate IMU state using Euler method
      dr = dr + (dv * dt) + (0.5 * quat_rot(dq, acc_i - ba_i) * dt_sq)
      dv = dv + quat_rot(dq, acc_i - ba_i) * dt
      dq = quat_integrate(dq, gyr_i - bg_i, dt)
      dC = quat2rot(dq)

      # Continuous time transition matrix F
      F = zeros((15, 15))
      F[0:3, 3:6] = eye(3)
      F[3:6, 6:9] = -1.0 * dC @ skew(acc_i - ba_i)
      F[3:6, 9:12] = -1.0 * dC
      F[6:9, 6:9] = -1.0 * skew(gyr_i - bg_i)
      F[6:9, 12:15] = -eye(3)

      # Continuous time input jacobian G
      G = zeros((15, 12))
      G[3:6, 0:3] = -1.0 * dC
      G[6:9, 3:6] = -eye(3)
      G[9:12, 6:9] = eye(3)
      G[12:15, 9:12] = eye(3)

      # Update
      G_dt = G * dt
      I_F_dt = eye(15) + F * dt
      self.state_F = I_F_dt @ self.state_F
      self.state_P = I_F_dt @ self.state_P @ I_F_dt.T + G_dt @ Q @ G_dt.T
      self.Dt += dt

    self.dr = dr
    self.dv = dv
    self.dq = dq
    self.state_P = (self.state_P + self.state_P.T) / 2.0
    self.sqrt_info = chol(inv(self.state_P)).T


class Frame:
  def __init__(self):
    self.p_sfm_cam = np.array([0.0, 0.0, 0.0])
    self.q_sfm_cam = np.array([1.0, 0.0, 0.0, 0.0])


def solve_gyro_bias(vision_data, imu_data, imu_params):
  """
  Linearized least-squares solve for gyroscope bias using the rotation
  consistency constraint between:

  - Vision-estimated relative rotation (typically from mono-VO or sfm)
  - IMU preintegrated rotation

  """
  A = zeros((3, 3))
  b = zeros((3, 1))
  ba = zeros(3)
  bg = zeros(3)

  for k in range(1, len(vision_data)):
    # Get relative rotation between i and j from sfm pose pair
    frame_i = vision_data[k - 1]
    frame_j = vision_data[k]
    q_ij_sfm = quat_mul(quat_inv(frame_i.q_sfm_cam), frame_j.q_sfm_cam)

    # Get relative rotation between i and j from imu data
    q_ij_imu = np.array([1.0, 0.0, 0.0, 0.0])
    imu_buf = []
    imu_pint = ImuPreintegrator(imu_buf, imu_params, ba, bg)

    A_k = imu_pint.state_F[6:9, 12:15]  # dtheta_dbg
    b_k = 2.0 * quat_vec(quat_mul(quat_inv(q_ij_imu), q_ij_sfm))
    A += A_k.T @ A_k
    b += A_k.T @ b_k

  # Estimate gyro bias correction
  dbg = np.linalg.solve(A, b)

  return dbg


def estimate_scale(vision_data, imu_data, imu_params):
  """Estimate scale"""
  num_frames = len(vision_data)
  n_state = num_frames * 3 + 3 + 1
  A = zeros((n_state, n_state))
  b = zeros(n_state)

  for k in range(1, num_frames):
    frame_i = vision_data[k - 1]
    frame_j = vision_data[k]
    A_k = zeros((6, 10))
    b_k = zeros(6)

    # dt = frame_j->second.pre_integration->sum_dt;
    Dt = 0.0

    RiT = frame_i.R.transpose()
    Rj = frame_j.R
    pi = frame_i.p
    pj = frame_j.p

    A_k[0:3, 0:3] = -eye(3) * Dt
    A_k[0:3, 6:9] = RiT * Dt * Dt / 2 * eye(3)
    A_k[0:3, 9] = RiT * (pj - pi) / 100.0

    A_k[3:6, 0:3] = -eye(3)
    A_k[3:6, 3:6] = RiT * Rj
    A_k[3:6, 6:9] = RiT * Dt * eye(3)

    b_k[0:3, 0] = pint_i.dp + RiT * Rj * TIC[0] - TIC[0]
    b_k[3:6, 0] = pint_j.dv

    H_k = A_k.T @ A_k
    b_k = A_k.T @ b_k

    # Velocity
    # A[i*3:i*3+3, i*3:i*3+3] += H_k[0:6, 0:6]
    # b[i*3:i*3+3] += b_k

    # Gravity and scale

    # Off-diagonal

    # A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
    # b.tail<4>() += r_b.tail<4>();
    #
    # A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
    # A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();

  # Solve for scale
  x = np.linalg.solve(A, b)
  scale = x[-1] / 100.0

  return scale
