import unittest

import numpy as np
from numpy import rad2deg
from numpy import deg2rad
from numpy.linalg import norm
import matplotlib.pyplot as plt

from xyz import euler321
from xyz import tf
from xyz import wrap_pi
from xyz import clip_value
from xyz import plot_mav


class PID:
  """PID controller"""
  def __init__(self, k_p, k_i, k_d):
    self.k_p = k_p
    self.k_i = k_i
    self.k_d = k_d

    self.error_p = 0.0
    self.error_i = 0.0
    self.error_d = 0.0
    self.error_prev = 0.0
    self.error_sum = 0.0

  def update(self, setpoint, actual, dt):
    """Update"""
    # Calculate errors
    error = setpoint - actual
    self.error_sum += error * dt

    # Calculate output
    self.error_p = self.k_p * error
    self.error_i = self.k_i * self.error_sum
    self.error_d = self.k_d * (error - self.error_prev) / dt
    output = self.error_p + self.error_i + self.error_d

    # Keep track of error
    self.error_prev = error

    return output

  def reset(self):
    """Reset"""
    self.error_prev = 0
    self.error_sum = 0
    self.error_p = 0
    self.error_i = 0
    self.error_d = 0


class CarrotController:
  """Carrot Controller"""
  def __init__(self):
    self.waypoints = []
    self.wp_start = None
    self.wp_end = None
    self.wp_index = None
    self.look_ahead_dist = 0.0

  def _calculate_closest_point(self, pos):
    """Calculate closest point"""
    assert self.wp_start
    assert self.wp_end
    v1 = pos - self.wp_start
    v2 = self.wp_end - self.wp_start
    t = v1 @ v2 / v2.squaredNorm()
    pt = self.wp_start + t * v2

    return (t, pt)

  def _calculate_carrot_point(self, pos):
    """Calculate carrot point"""
    assert len(pos) == 3
    assert self.wp_start
    assert self.wp_end

    t, closest_pt = self._calculate_closest_point(pos)
    carrot_pt = None

    if t == -1:
      # Closest point is before wp_start
      carrot_pt = self.wp_start

    elif t == 0:
      # Closest point is between wp_start wp_end
      u = self.wp_end - self.wp_start
      v = u / norm(u)
      carrot_pt = closest_pt + self.look_ahead_dist * v

    elif t == 1:
      # Closest point is after wp_end
      carrot_pt = self.wp_end

    return (t, carrot_pt)

  def update(self, pos):
    """Update"""
    assert len(pos) == 3
    assert self.wp_start
    assert self.wp_end
    assert self.wp_index

    # Calculate new carot point
    status, carrot_pt = self._calculate_carrot_point(pos)

    # Check if there are more waypoints
    if (self.wp_index + 1) == len(self.waypoints):
      return None

    # Update waypoints
    if status == 1:
      self.wp_index += 1
      self.wp_start = self.wp_end
      self.wp_end = self.waypoints[self.wp_index]

    return carrot_pt


class MavModel:
  def __init__(self, **kwargs):
    self.x = np.zeros((12,))
    self.x[0] = kwargs.get("roll", 0.0)
    self.x[1] = kwargs.get("pitch", 0.0)
    self.x[2] = kwargs.get("yaw", 0.0)
    self.x[3] = kwargs.get("wx", 0.0)
    self.x[4] = kwargs.get("wy", 0.0)
    self.x[5] = kwargs.get("wz", 0.0)
    self.x[6] = kwargs.get("rx", 0.0)
    self.x[7] = kwargs.get("ry", 0.0)
    self.x[8] = kwargs.get("rz", 0.0)
    self.x[9] = kwargs.get("vx", 0.0)
    self.x[10] = kwargs.get("vy", 0.0)
    self.x[11] = kwargs.get("vz", 0.0)

    self.inertia = [0.0963, 0.0963, 0.1927]
    self.kr = 0.1  # Rotation drag constant
    self.kt = 0.2  # Translation drag constant
    self.l = 0.9  # Arm length
    self.d = 1.0  # Drag co-efficient
    self.m = 1.0  # Mass
    self.g = 9.81  # Gravitational constant

  def set_attitude(self, rpy):
    """Set attitude"""
    self.x[0] = rpy[0]
    self.x[1] = rpy[1]
    self.x[2] = rpy[2]

  def set_angular_velocity(self, vel):
    """Set angular velocity"""
    self.x[3] = vel[0]
    self.x[4] = vel[1]
    self.x[5] = vel[2]

  def set_position(self, pos):
    """Set position"""
    self.x[6] = pos[0]
    self.x[7] = pos[1]
    self.x[8] = pos[2]

  def set_velocity(self, vel):
    """Set velocity"""
    self.x[9] = vel[0]
    self.x[10] = vel[1]
    self.x[11] = vel[2]

  def get_attitude(self):
    """Get attitude"""
    return np.array([self.x[0], self.x[1], self.x[2]])

  def get_angular_velocity(self):
    """Get angular velocity"""
    return np.array([self.x[3], self.x[4], self.x[5]])

  def get_position(self):
    """Get position"""
    return np.array([self.x[6], self.x[7], self.x[8]])

  def get_velocity(self):
    """Get velocity"""
    return np.array([self.x[9], self.x[10], self.x[11]])

  def get_pose(self):
    """Get Pose"""
    C_WB = euler321(self.x[2], self.x[1], self.x[0])
    r_WB = np.array([self.x[6], self.x[7], self.x[8]])
    T_WB = tf(C_WB, r_WB)
    return T_WB

  def update(self, u, dt):
    """Update mav model"""
    # -- Attitude
    ph = self.x[0]
    th = self.x[1]
    ps = self.x[2]
    # -- Angular velocity
    p = self.x[3]
    q = self.x[4]
    r = self.x[5]
    # -- Velocity
    vx = self.x[9]
    vy = self.x[10]
    vz = self.x[11]

    # Map out constants
    Ix = self.inertia[0]
    Iy = self.inertia[1]
    Iz = self.inertia[2]
    kr = self.kr
    kt = self.kt
    m = self.m
    mr = 1.0 / m
    g = self.g

    # Convert motor inputs to angular p, q, r and total thrust
    # yapf:disable
    A = np.array([
      1.0, 1.0, 1.0, 1.0,
      0.0, -self.l, 0.0, self.l,
      -self.l, 0.0, self.l, 0.0,
      -self.d, self.d, -self.d, self.d
    ]).reshape((4, 4))
    # yapf:enable

    # tau = A * u
    mt = 5.0  # Max-thrust
    s = np.array([mt * u[0], mt * u[1], mt * u[2], mt * u[3]])
    tauf, taup, tauq, taur = A @ s

    # Update state
    cph = np.cos(ph)
    sph = np.sin(ph)
    cth = np.cos(th)
    sth = np.sin(th)
    tth = np.tan(th)
    cps = np.cos(ps)
    sps = np.sin(ps)

    # yapf:disable
    # -- Attitude
    self.x[0] += (p + q * sph * tth + r * cph * tth) * dt
    self.x[1] += (q * cph - r * sph) * dt
    self.x[2] += ((1 / cth) * (q * sph + r * cph)) * dt
    # -- Angular velocity
    self.x[3] += (-((Iz - Iy) / Ix) * q * r - (kr * p / Ix) + (1 / Ix) * taup) * dt
    self.x[4] += (-((Ix - Iz) / Iy) * p * r - (kr * q / Iy) + (1 / Iy) * tauq) * dt
    self.x[5] += (-((Iy - Ix) / Iz) * p * q - (kr * r / Iz) + (1 / Iz) * taur) * dt
    # -- Position
    self.x[6] += vx * dt
    self.x[7] += vy * dt
    self.x[8] += vz * dt
    # -- Linear velocity
    self.x[9] += ((-kt * vx / m) + mr * (cph * sth * cps + sph * sps) * tauf) * dt
    self.x[10] += ((-kt * vy / m) + mr * (cph * sth * sps - sph * cps) * tauf) * dt
    self.x[11] += (-(kt * vz / m) + mr * (cph * cth) * tauf - g) * dt
    # yapf:enable

    # Wrap yaw
    if self.x[2] > np.pi:
      self.x[2] -= 2.0 * np.pi
    elif self.x[2] < -np.pi:
      self.x[2] += 2.0 * np.pi


class MavAttitudeControl:
  def __init__(self):
    self.dt = 0
    self.pid_roll = PID(10.0, 0.0, 5.0)
    self.pid_pitch = PID(10.0, 0.0, 5.0)
    self.pid_yaw = PID(10.0, 0.0, 1.0)
    self.u = np.array([0.0, 0.0, 0.0, 0.0])

  def update(self, sp, pv, dt):
    """Update"""
    # Check rate
    self.dt += dt
    if self.dt < 0.001:
      return self.u  # Return previous command

    # Roll, pitch, yaw and thrust
    error_yaw = wrap_pi(sp[2] - pv[2])
    r = self.pid_roll.update(sp[0], pv[0], self.dt)
    p = self.pid_pitch.update(sp[1], pv[1], self.dt)
    y = self.pid_yaw.update(error_yaw, 0.0, self.dt)
    t = clip_value(sp[3], 0.0, 1.0)

    # Map roll, pitch, yaw and thrust to motor outputs
    self.u[0] = clip_value(-p - y + t, 0.0, 1.0)
    self.u[1] = clip_value(-r + y + t, 0.0, 1.0)
    self.u[2] = clip_value(p - y + t, 0.0, 1.0)
    self.u[3] = clip_value(r + y + t, 0.0, 1.0)

    # Keep track of control action
    self.dt = 0.0  # Reset dt

    return self.u

  def reset(self):
    """Reset"""
    self.dt = 0.0
    self.pid_roll.reset()
    self.pid_pitch.reset()
    self.pid_yaw.reset()
    self.u = np.array([0.0, 0.0, 0.0, 0.0])


class MavVelocityControl:
  def __init__(self):
    self.period = 0.0011
    self.roll_min = deg2rad(-35.0)
    self.roll_max = deg2rad(35.0)
    self.pitch_min = deg2rad(-35.0)
    self.pitch_max = deg2rad(35.0)

    self.dt = 0
    self.pid_vx = PID(10.0, 0.0, 0.5)
    self.pid_vy = PID(10.0, 0.0, 0.5)
    self.pid_vz = PID(10.0, 0.0, 0.5)
    self.u = np.array([0.0, 0.0, 0.0, 0.0])

  def update(self, sp, pv, dt):
    """Update"""
    # Check rate
    self.dt += dt
    if self.dt < self.period:
      return self.u  # Return previous command

    # Calculate transform velocity commands in world frame to body frame
    errors_W = np.array([sp[0] - pv[0], sp[1] - pv[1], sp[2] - pv[2]])
    C_WS = euler321(pv[3], 0.0, 0.0)
    errors = C_WS.T @ errors_W

    # Roll, pitch, yaw and thrust
    r = -self.pid_vy.update(errors[1], 0.0, dt)
    p = self.pid_vx.update(errors[0], 0.0, dt)
    y = sp[3]
    t = 0.5 + self.pid_vz.update(errors[2], 0.0, dt)

    self.u[0] = clip_value(r, self.roll_min, self.roll_max)
    self.u[1] = clip_value(p, self.pitch_min, self.pitch_max)
    self.u[2] = y
    self.u[3] = clip_value(t, 0.0, 1.0)

    # Keep track of control action
    self.dt = 0.0  # Reset dt

    return self.u

  def reset(self):
    """Reset"""
    self.dt = 0.0
    self.pid_vx.reset()
    self.pid_vy.reset()
    self.pid_vz.reset()
    self.u = np.array([0.0, 0.0, 0.0, 0.0])


class MavPositionControl:
  def __init__(self, output_mode="VELOCITY"):
    self.output_mode = output_mode
    self.dt = 0
    self.u = [0.0, 0.0, 0.0, 0.0]

    if self.output_mode == "VELOCITY":
      self.period = 0.011
      self.vx_min = -5.0
      self.vx_max = 5.0
      self.vy_min = -5.0
      self.vy_max = 5.0
      self.vz_min = -5.0
      self.vz_max = 5.0

      self.pid_x = PID(0.5, 0.0, 0.05)
      self.pid_y = PID(0.5, 0.0, 0.05)
      self.pid_z = PID(1.0, 0.0, 0.1)

    elif self.output_mode == "ATTITUDE":
      self.period = 0.011
      self.roll_min = deg2rad(-35.0)
      self.roll_max = deg2rad(35.0)
      self.pitch_min = deg2rad(-35.0)
      self.pitch_max = deg2rad(35.0)
      self.hover_thrust = 0.5

      self.pid_x = PID(5.0, 0.0, 0.1)
      self.pid_y = PID(5.0, 0.0, 0.1)
      self.pid_z = PID(5.0, 0.0, 0.1)

    else:
      raise NotImplementedError()

  def update(self, sp, pv, dt):
    """Update"""
    # Check rate
    self.dt += dt
    if self.dt < self.period:
      return self.u  # Return previous command

    if self.output_mode == "VELOCITY":
      # Calculate position errors in world frame
      errors = np.array([sp[0] - pv[0], sp[1] - pv[1], sp[2] - pv[2]])

      # Velocity commands
      vx = self.pid_x.update(errors[0], 0.0, self.dt)
      vy = self.pid_y.update(errors[1], 0.0, self.dt)
      vz = self.pid_z.update(errors[2], 0.0, self.dt)
      yaw = sp[3]

      # Velocity command (vx, vy, vz, yaw)
      self.u[0] = clip_value(vx, self.vx_min, self.vx_max)
      self.u[1] = clip_value(vy, self.vy_min, self.vy_max)
      self.u[2] = clip_value(vz, self.vz_min, self.vz_max)
      self.u[3] = yaw

    elif self.output_mode == "ATTITUDE":
      # Calculate position errors in mav frame
      errors = euler321(pv[3], 0.0, 0.0).T @ (sp[0:3] - pv[0:3])

      # Attitude commands
      roll = -self.pid_y.update(errors[1], 0.0, dt)
      pitch = self.pid_x.update(errors[0], 0.0, dt)
      thrust = self.hover_thrust + self.pid_z.update(errors[2], 0.0, dt)

      # Attitude command (roll, pitch, yaw, thrust)
      self.u[0] = clip_value(roll, self.roll_min, self.roll_max)
      self.u[1] = clip_value(pitch, self.pitch_min, self.pitch_max)
      self.u[2] = sp[3]
      self.u[3] = clip_value(thrust, 0.0, 1.0)

    else:
      raise NotImplementedError()

    # Reset dt
    self.dt = 0.0

    return self.u

  def reset(self):
    """Reset"""
    assert self.dt is not None

    self.dt = 0.0
    self.pid_x.reset()
    self.pid_y.reset()
    self.pid_z.reset()
    self.u = [0.0, 0.0, 0.0, 0.0]


class MavTrajectoryControl:
  def __init__(self, **kwargs):
    self.A = kwargs.get("A", 2.0)
    self.B = kwargs.get("B", 2.0)
    self.a = kwargs.get("a", 3.0)
    self.b = kwargs.get("b", 2.0)
    self.z = kwargs["z"]
    self.T = kwargs["T"]
    self.f = 1.0 / self.T
    self.delta = kwargs.get("delta", np.pi)
    self.hover_thrust = kwargs.get("hover_thrust", 0.5)

    # Position and velocity controller
    self.last_ts = None
    self.pos_ctrl = MavPositionControl("ATTITUDE")
    self.vel_ctrl = MavVelocityControl()

  # def symdiff_velocity(self):
  #   import sympy
  #
  #   f, t = sympy.symbols("f t")
  #   a, A, delta = sympy.symbols("a A delta")
  #   b, B = sympy.symbols("b B")
  #
  #   w = 2.0 * sympy.pi * f
  #   theta = sympy.Pow(sympy.sin(0.25 * w * t), 2)
  #
  #   ka = 2.0 * sympy.pi * a
  #   kb = 2.0 * sympy.pi * b
  #
  #   x = A * sympy.sin(ka * theta + delta)
  #   y = B * sympy.sin(kb * theta)
  #
  #   vx = sympy.diff(x, t)
  #   vy = sympy.diff(y, t)
  #
  #   # print(vx)
  #   # print(vy)

  def get_traj(self):
    """Return trajectory"""
    pos_data = np.zeros((3, 1000))
    time = np.linspace(0.0, self.T, 1000)
    for i, t in enumerate(time):
      pos_data[:, i] = self.get_position(t).T
    return pos_data.T

  def get_position(self, t):
    """Get position"""
    w = 2.0 * np.pi * self.f
    theta = np.sin(0.25 * w * t)**2

    ka = 2.0 * np.pi * self.a
    kb = 2.0 * np.pi * self.b

    x = self.A * np.sin(ka * theta + self.delta)
    y = self.B * np.sin(kb * theta)
    z = self.z

    return np.array([x, y, z])

  def get_yaw(self, t):
    """Get yaw"""
    p0 = self.get_position(t)
    p1 = self.get_position(t + 0.1)
    dx, dy, _ = p1 - p0

    heading = np.arctan2(dy, dx)
    if heading > np.pi:
      heading -= 2.0 * np.pi
    elif heading < -np.pi:
      heading += 2.0 * np.pi

    return heading

  def get_velocity(self, t):
    # w = 2.0 * np.pi * self.f
    # theta = np.sin(0.25 * w * t)**2

    ka = 2.0 * np.pi * self.a
    kb = 2.0 * np.pi * self.b
    kpift = 0.5 * np.pi * self.f * t
    kx = 2.0 * np.pi**2 * self.A * self.a * self.f
    ky = 2.0 * np.pi**2 * self.B * self.b * self.f
    ksincos = np.sin(kpift) * np.cos(kpift)

    vx = kx * ksincos * np.cos(ka * np.sin(kpift)**2 + self.delta)
    vy = ky * ksincos * np.cos(kb * np.sin(kpift)**2)
    vz = 0.0

    return np.array([vx, vy, vz])

  def update(self, pos_pv, vel_pv, t):
    # Pre-check
    if self.last_ts is None:
      self.last_ts = t
      return np.array([0.0, 0.0, 0.0, 0.0])
    dt = t - self.last_ts

    # Get trajectory position, velocity and yaw
    traj_pos = self.get_position(t)
    traj_vel = self.get_velocity(t)
    traj_yaw = self.get_yaw(t)

    # Form position and velocity setpoints
    pos_sp = np.array([traj_pos[0], traj_pos[1], traj_pos[2], traj_yaw])
    vel_sp = [traj_vel[0], traj_vel[1], traj_vel[2], traj_yaw]

    # Position control
    att_pos_sp = self.pos_ctrl.update(pos_sp, pos_pv, dt)

    # Velocity control
    att_vel_sp = self.vel_ctrl.update(vel_sp, vel_pv, dt)

    # Mix both position and velocity control into a single attitude setpoint
    att_sp = np.array([0.0, 0.0, 0.0, 0.0])
    att_sp[0] = att_vel_sp[0] + att_pos_sp[0]
    att_sp[1] = att_vel_sp[1] + att_pos_sp[1]
    att_sp[2] = traj_yaw
    att_sp[3] = att_vel_sp[3] + att_pos_sp[3]

    att_sp[0] = clip_value(att_sp[0], deg2rad(-35.0), deg2rad(35.0))
    att_sp[1] = clip_value(att_sp[1], deg2rad(-35.0), deg2rad(35.0))
    att_sp[2] = att_sp[2]
    att_sp[3] = clip_value(att_sp[3], 0.0, 1.0)

    # Update
    self.last_ts = t

    return att_sp

  def plot(self):
    """Plot"""
    pos_data = np.zeros((3, 1000))
    vel_data = np.zeros((3, 1000))
    time = np.linspace(0.0, self.T, 1000)
    for i, t in enumerate(time):
      pos_data[:, i] = self.get_position(t).T
      vel_data[:, i] = self.get_velocity(t).T

    plt.subplot(311)
    plt.plot(pos_data[0, :], pos_data[1, :])
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    plt.subplot(312)
    plt.plot(time, pos_data[0, :], "r-", label="Position - x")
    plt.plot(time, pos_data[1, :], "g-", label="Position - y")
    plt.plot(time, pos_data[2, :], "b-", label="Position - z")
    plt.xlabel("Time [s]")
    plt.ylabel("Positions [m]")
    plt.legend(loc=0)

    plt.subplot(313)
    plt.plot(time, vel_data[0, :], "r-", label="Velocity - x")
    plt.plot(time, vel_data[1, :], "g-", label="Velocity - y")
    plt.plot(time, vel_data[2, :], "b-", label="Velocity - z")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [ms^-1]")
    plt.legend(loc=0)

    plt.show()


class TestMav(unittest.TestCase):
  """Test Mav"""

  # def test_symdiff_velocity(self):
  #   """Test symbolic differentiate velocity"""
  #   traj_ctrl = MavTrajectoryControl(z=2.0, T=10.0)
  #   traj_ctrl.symdiff_velocity()

  # def test_plot(self):
  #   """ Test Plot """
  #   traj_ctrl = MavTrajectoryControl(z=2.0, T=20.0)
  #   # traj_ctrl.plot()

  def test_mav_attitude_control(self):
    # Simulation parameters
    debug = False
    dt = 0.001
    t_end = 0.5
    t = 0.0
    att_sp = np.array([0.1, 0.2, -0.2, 0.0])  # roll, pitch, yaw, thrust

    # Setup model and controllers
    idx = 0
    N = t_end / dt
    mav = MavModel()
    att_ctrl = MavAttitudeControl()

    # Simulate
    time_data = []
    att_data = []
    pos_data = []
    vel_data = []

    while idx < N:
      att_pv = [mav.x[0], mav.x[1], mav.x[2]]
      u = att_ctrl.update(att_sp, att_pv, dt)
      mav.update(u, dt)

      time_data.append(t)
      att_data.append(mav.get_attitude())
      pos_data.append(mav.get_position())
      vel_data.append(mav.get_velocity())

      t += dt
      idx += 1

    # Plot results
    if debug:
      time_data = np.array(time_data)
      att_data = np.array(att_data)
      pos_data = np.array(pos_data)
      vel_data = np.array(vel_data)
      plt.plot(time_data, rad2deg(att_data[:, 0]), "r-", label="Roll")
      plt.plot(time_data, rad2deg(att_data[:, 1]), "g-", label="Pitch")
      plt.plot(time_data, rad2deg(att_data[:, 2]), "b-", label="Yaw")
      plt.xlabel("Time [s]")
      plt.ylabel("Attitude [deg]")
      # plt.show()

  def test_mav_velocity_control(self):
    # Simulation parameters
    debug = False
    dt = 0.001
    t = 0.0
    t_end = 10.0
    vel_sp = np.array([0.1, 0.2, 1.0, 0.0])  # vx, vy, vz, yaw

    # Setup model and controllers
    idx = 0
    N = t_end / dt
    mav = MavModel()
    att_ctrl = MavAttitudeControl()
    vel_ctrl = MavVelocityControl()

    # Simulate
    time_data = []
    att_data = []
    pos_data = []
    vel_data = []

    while idx < N:
      vel_pv = [mav.x[9], mav.x[10], mav.x[11], mav.x[2]]
      att_pv = [mav.x[0], mav.x[1], mav.x[2]]

      att_sp = vel_ctrl.update(vel_sp, vel_pv, dt)
      u = att_ctrl.update(att_sp, att_pv, dt)
      mav.update(u, dt)

      time_data.append(t)
      att_data.append(mav.get_attitude())
      pos_data.append(mav.get_position())
      vel_data.append(mav.get_velocity())

      t += dt
      idx += 1

    # Plot results
    if debug:
      time_data = np.array(time_data)
      att_data = np.array(att_data)
      pos_data = np.array(pos_data)
      vel_data = np.array(vel_data)

      # -- Plot attitude
      plt.subplot(211)
      plt.plot(time_data, rad2deg(att_data[:, 0]), "r-", label="Roll")
      plt.plot(time_data, rad2deg(att_data[:, 1]), "g-", label="Pitch")
      plt.plot(time_data, rad2deg(att_data[:, 2]), "b-", label="Yaw")
      plt.xlabel("Time [s]")
      plt.ylabel("Attitude [deg]")

      # -- Plot velocity
      plt.subplot(212)
      plt.plot(time_data, vel_data[:, 0], "r-", label="vx")
      plt.plot(time_data, vel_data[:, 1], "g-", label="vy")
      plt.plot(time_data, vel_data[:, 2], "b-", label="vz")
      plt.xlabel("Time [s]")
      plt.ylabel("Velocity [ms^-1]")

      plt.show()

  def test_mav_position_control(self):
    # Simulation parameters
    debug = False
    dt = 0.001
    t = 0.0
    t_end = 10.0
    pos_sp = np.array([2.0, 1.0, 5.0, np.deg2rad(135)])  # x, y, z, yaw
    plot_anim = False
    self.keep_plotting = True

    # Setup models and controller
    idx = 0
    N = t_end / dt
    mav = MavModel()
    att_ctrl = MavAttitudeControl()
    vel_ctrl = MavVelocityControl()
    pos_ctrl = MavPositionControl()

    # Setup plot
    ax_3d = None
    ax_xy = None
    fig = None
    cid = None
    if debug:
      fig = plt.figure()
      ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
      ax_xy = fig.add_subplot(1, 2, 2)

      def on_key(event, fig):
        if event.key == "escape" or event.key == "q":
          self.keep_plotting = False
          plt.close(fig)

      cid = fig.canvas.mpl_connect(
        "key_press_event", lambda event: on_key(event, self.keep_plotting))

    # Simulate
    time_data = []
    att_data = []
    pos_data = []
    vel_data = []

    t = 0.0
    idx = 0
    while idx < N and self.keep_plotting:
      if ax_3d and plot_anim and idx % 50 == 0:
        ax_3d.cla()
        # T_WB = mav.get_pose()
        # tf_data = plot_tf(ax_3d, T_WB, size=0.5)
        ax_3d.set_xlim([-5.0, 5.0])
        ax_3d.set_ylim([-5.0, 5.0])
        ax_3d.set_zlim([0.0, 10.0])
        ax_3d.set_xlabel("x [m]")
        ax_3d.set_ylabel("y [m]")
        ax_3d.set_zlabel("z [m]")
        plt.draw()
        plt.pause(0.05)

        if ax_xy and len(pos_data) > 100:
          pos = np.array(pos_data).T
          ax_xy.cla()
          ax_xy.plot(pos[0, -1], pos[1, -1], "rx")
          ax_xy.set_xlim(-5.0, 5.0)
          ax_xy.set_ylim(-5.0, 5.0)
          ax_xy.set_xlabel("x [m]")
          ax_xy.set_ylabel("y [m]")

      # Position, velocity and attitude process variables
      pos_pv = [mav.x[6], mav.x[7], mav.x[8], mav.x[2]]
      vel_pv = [mav.x[9], mav.x[10], mav.x[11], mav.x[2]]
      att_pv = [mav.x[0], mav.x[1], mav.x[2]]

      # Update controllers and model
      vel_sp = pos_ctrl.update(pos_sp, pos_pv, dt)
      att_sp = vel_ctrl.update(vel_sp, vel_pv, dt)
      u = att_ctrl.update(att_sp, att_pv, dt)
      mav.update(u, dt)

      # Record
      time_data.append(t)
      att_data.append(mav.get_attitude())
      pos_data.append(mav.get_position())
      vel_data.append(mav.get_velocity())

      # Update
      t += dt
      idx += 1

    # Disconnect figure event callback
    if debug:
      assert fig and cid
      fig.canvas.mpl_disconnect(cid)

    # Plot results
    if debug:
      time_data = np.array(time_data)
      att_data = np.array(att_data)
      pos_data = np.array(pos_data)
      vel_data = np.array(vel_data)

      # -- Plot attitude
      plt.subplot(311)
      plt.plot(time_data, rad2deg(att_data[:, 0]), "r-", label="Roll")
      plt.plot(time_data, rad2deg(att_data[:, 1]), "g-", label="Pitch")
      plt.plot(time_data, rad2deg(att_data[:, 2]), "b-", label="Yaw")
      plt.xlabel("Time [s]")
      plt.ylabel("Attitude [deg]")

      # -- Plot velocity
      plt.subplot(312)
      plt.plot(time_data, vel_data[:, 0], "r-", label="vx")
      plt.plot(time_data, vel_data[:, 1], "g-", label="vy")
      plt.plot(time_data, vel_data[:, 2], "b-", label="vz")
      plt.xlabel("Time [s]")
      plt.ylabel("Velocity [ms^-1]")

      # -- Plot position
      plt.subplot(313)
      plt.plot(time_data, pos_data[:, 0], "r-", label="x")
      plt.plot(time_data, pos_data[:, 1], "g-", label="y")
      plt.plot(time_data, pos_data[:, 2], "b-", label="z")
      plt.xlabel("Time [s]")
      plt.ylabel("Position [m]")

      plt.show()

  def test_mav_trajectory_control(self):
    # Simulation parameters
    debug = False
    dt = 0.001
    z_sp = 5.0
    t_end = 30.0
    N = t_end / dt

    # Setup models and controller
    att_ctrl = MavAttitudeControl()
    traj_ctrl = MavTrajectoryControl(a=3, b=2, z=z_sp, T=t_end, delta=np.pi / 2)
    yaw0 = traj_ctrl.get_yaw(0.0)
    r0 = traj_ctrl.get_position(0.0)
    v0 = traj_ctrl.get_velocity(0.0)
    mav = MavModel(
      rx=r0[0] + 0.5,
      ry=r0[1] - 0.5,
      rz=z_sp,
      vx=v0[0],
      vy=v0[1],
      vz=v0[2],
      yaw=yaw0,
    )

    # Setup plot
    plot_anim = False
    self.keep_plotting = True
    fig = None
    cid = None
    ax_3d = None
    ax_xy = None
    if debug:
      fig = plt.figure()
      ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
      ax_xy = fig.add_subplot(1, 2, 2)

      def on_key(event, fig):
        if event.key == "escape" or event.key == "q":
          self.keep_plotting = False
          plt.close(fig)

      cid = fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: on_key(event, self.keep_plotting),
      )

    # Simulate
    time_data = []
    att_data = []
    pos_data = []
    vel_data = []

    t = 0.0
    idx = 0
    while idx < N and self.keep_plotting:
      if ax_3d and plot_anim and idx % 50 == 0:
        ax_3d.cla()
        T_WB = mav.get_pose()
        plot_mav(ax_3d, T_WB, size=0.5)
        ax_3d.set_xlim([-5.0, 5.0])
        ax_3d.set_ylim([-5.0, 5.0])
        ax_3d.set_zlim([0.0, 10.0])
        ax_3d.set_xlabel("x [m]")
        ax_3d.set_ylabel("y [m]")
        ax_3d.set_zlabel("z [m]")

        if ax_xy and len(pos_data) > 100:
          pos = np.array(pos_data).T
          ax_xy.plot(pos[0, ::100], pos[1, ::100], "r-")
          ax_xy.set_xlim((-5.0, 5.0))
          ax_xy.set_ylim((-5.0, 5.0))
          ax_xy.set_xlabel("x [m]")
          ax_xy.set_ylabel("y [m]")

        plt.draw()
        plt.pause(0.01)

      # Velocity and attitude process variables
      pos_pv = [mav.x[6], mav.x[7], mav.x[8], mav.x[2]]
      vel_pv = [mav.x[9], mav.x[10], mav.x[11], mav.x[2]]
      att_pv = [mav.x[0], mav.x[1], mav.x[2]]

      # Update controllers and model
      att_sp = traj_ctrl.update(pos_pv, vel_pv, t)
      u = att_ctrl.update(att_sp, att_pv, dt)
      mav.update(u, dt)

      # Record
      time_data.append(t)
      att_data.append(mav.get_attitude())
      pos_data.append(mav.get_position())
      vel_data.append(mav.get_velocity())

      # Update
      t += dt
      idx += 1

    # Disconnect figure event callback
    if debug:
      assert fig and cid
      fig.canvas.mpl_disconnect(cid)

    # Plot results
    if debug:
      time_data = np.array(time_data)
      att_data = np.array(att_data)
      pos_data = np.array(pos_data)
      vel_data = np.array(vel_data)
      traj_data = traj_ctrl.get_traj()

      # -- Plot actual vs planned trajectory
      plt.subplot(311)
      plt.plot(pos_data[:, 0], pos_data[:, 1], "r-", label="Actual")
      plt.plot(traj_data[:, 0], traj_data[:, 1], "k--", label="Trajectory")
      plt.xlabel("x [m]")
      plt.ylabel("y [m]")
      plt.axis("equal")
      plt.legend(loc=0)

      # -- Plot velocity
      plt.subplot(312)
      plt.plot(time_data, vel_data[:, 0], "r-", label="vx")
      plt.plot(time_data, vel_data[:, 1], "g-", label="vy")
      plt.plot(time_data, vel_data[:, 2], "b-", label="vz")
      plt.xlabel("Time [s]")
      plt.ylabel("Velocity [ms^-1]")
      plt.legend(loc=0)

      # -- Plot attitude
      plt.subplot(313)
      plt.plot(time_data, rad2deg(att_data[:, 0]), "r-", label="Roll")
      plt.plot(time_data, rad2deg(att_data[:, 1]), "g-", label="Pitch")
      plt.plot(time_data, rad2deg(att_data[:, 2]), "b-", label="Yaw")
      plt.xlabel("Time [s]")
      plt.ylabel("Attitude [deg]")
      plt.legend(loc=0)

      plt.show()
