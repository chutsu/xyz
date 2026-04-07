import unittest
from math import atan

import numpy as np
from numpy import eye
from numpy import pi
from numpy import sqrt
import matplotlib.pyplot as plt


def compl_filter(gyro, accel, dt, roll, pitch):
  """
  A simple complementary filter that uses `gyro` and `accel` measurements to
  estimate the attitude in `roll` and `pitch`. Where `dt` is the update
  rate of the `gyro` measurements in seconds.
  """
  # Calculate pitch and roll using gyroscope
  wx, wy, _ = gyro
  gyro_roll = (wx * dt) + roll
  gyro_pitch = (wy * dt) + pitch

  # Calculate pitch and roll using accelerometer
  ax, ay, az = accel
  accel_roll = (atan(ay / sqrt(ax * ay + az * az))) * 180.0 / pi
  accel_pitch = (atan(ax / sqrt(ay * ay + az * az))) * 180.0 / pi

  # Complimentary filter
  pitch = (0.98 * gyro_pitch) + (0.02 * accel_pitch)
  roll = (0.98 * gyro_roll) + (0.02 * accel_roll)

  return (roll, pitch)


class KalmanFilter:
  """Kalman Filter"""
  def __init__(self, **kwargs):
    self.x = kwargs["x0"]
    self.F = kwargs["F"]
    self.H = kwargs["H"]
    self.B = kwargs.get("B", np.array([0]))
    self.Q = kwargs.get("Q", eye(self.F.shape[1]))
    self.R = kwargs.get("R", eye(self.H.shape[0]))
    self.P = kwargs.get("P", eye(self.F.shape[1]))

  def predict(self, u=np.array([0.0])):
    """Predict"""
    self.x = self.F @ self.x + self.B @ u
    self.P = self.F @ self.P @ self.F.T + self.Q
    return self.x

  def update(self, z):
    """Measurement Update"""
    I = eye(self.F.shape[1])
    y = z - self.H @ self.x
    S = self.R + self.H @ self.P @ self.H.T
    K = self.P @ self.H.T @ np.linalg.inv(S)
    self.x = self.x + K @ y
    self.P = (I - K @ self.H) @ self.P
    return self.x


class TestKalmanFilter(unittest.TestCase):
  """Test Kalman Filter"""
  def test_constant_acceleration_example(self):
    # Simulation parameters
    dt = 0.01
    dt_sq = dt * dt
    t = 0.0
    t_end = 5.0

    # -- Initial state
    rx = 0.0
    ry = 0.0
    vx = 9.0
    vy = 30.0
    ax = 0.0
    ay = -12.0
    x0 = np.array([rx, ry, vx, vy, ax, ay])

    # -- Setup Kalman Filter
    # yapf:disable
    # ---- Transition Matrix
    F = np.array([[1.0, 0.0, dt, 0.0, 0.5 * dt**2, 0.0],
                  [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt**2],
                  [0.0, 0.0, 1.0, 0.0, dt, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    # ---- Measurement Matrix
    H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    # ---- Input Matrix
    B = np.array([0])
    # ---- Process Noise Matrix
    Q = 0.1 * eye(6)
    # ---- Measurement Noise Matrix
    R = 10.0 * eye(2)
    # yapf:enable
    # ---- Kalman Filter
    kwargs = {"x0": x0, "F": F, "H": H, "B": B, "Q": Q, "R": R}
    kf = KalmanFilter(**kwargs)

    # Simulate
    time = []
    gnd_rx = []
    gnd_ry = []
    gnd_vx = []
    gnd_vy = []
    gnd_ax = []
    gnd_ay = []
    meas_zx = []
    meas_zy = []
    est_rx = []
    est_ry = []

    while t <= t_end:
      # Simulate Ground-truth
      rx += (vx * dt) + (0.5 * ax * dt_sq)
      ry += (vy * dt) + (0.5 * ay * dt_sq)
      vx += ax * dt
      vy += ay * dt

      # Simulate input and noisy measurements
      u = np.array([0.0])
      noise_zx = np.random.normal(0.0, 1.0)
      noise_zy = np.random.normal(0.0, 1.0)
      z = np.array([rx + noise_zx, ry + noise_zy])
      meas_zx.append(z[0])
      meas_zy.append(z[1])

      # Kalman filter prediction and update
      kf.predict(u)
      kf.update(z)
      est_rx.append(kf.x[0])
      est_ry.append(kf.x[1])

      # Record and update
      time.append(t)
      gnd_rx.append(rx)
      gnd_ry.append(ry)
      gnd_vx.append(vx)
      gnd_vy.append(vy)
      gnd_ax.append(ax)
      gnd_ay.append(ay)
      t += dt

    # Plot X-Y
    debug = False
    if debug:
      plt.plot(gnd_rx, gnd_ry, "k--", label="Ground-Truth")
      plt.plot(meas_zx, meas_zy, "r.", label="Measurement")
      plt.plot(est_rx, est_ry, "b-", label="Estimate")
      plt.axis("equal")
      plt.legend(loc=0)
      plt.xlabel("x [m]")
      plt.ylabel("y [m]")
      plt.show()
