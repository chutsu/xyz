import numpy as np
from numpy import eye

from xyz import hat
from xyz import so3_exp


def twistSE3(twist):
  """Twist to SE3(3)

  Let twist:

    twist = s * theta = [w, v] * theta

  Example Input:

    s = np.array([1, 2, 3, 4, 5, 6])

  Example Output:

    np.array([[ 0, -3,  2, 4],
              [ 3,  0, -1, 5],
              [-2,  1,  0, 6],
              [ 0,  0,  0, 0]])

  """
  w = twist[0:3]
  v = twist[3:]
  return np.block([[hat(w), v.reshape((3, 1))], [np.zeros((1, 4))]])


def poe(screw_axis, theta, tol = 1e-6):
  """Matrix exponential of se(3) to SE(3)"""
  s = screw_axis * theta
  aa = s[0:3]  # Axis-angle (w * theta)
  # v = s[3:]  # Linear velocity
  se3mat = twistSE3(s)

  if np.linalg.norm(aa) < tol:
    C = eye(3)
    r = theta * screw_axis[3:].reshape((3, 1))
    return np.block([[C, r], [0.0, 0.0, 0.0, 1.0]])

  I3 = eye(3)
  c_th = np.cos(theta)
  s_th = np.sin(theta)
  w_skew = se3mat[0:3, 0:3] / theta
  w_skew_sq = w_skew @ w_skew

  A = so3_exp(se3mat[0:3, 0:3])
  B = (I3 * theta + (1.0 - c_th) * w_skew +
       (theta - s_th) * w_skew_sq) @ screw_axis[3:]

  return np.block([[A, B.reshape((3, 1))], [0.0, 0.0, 0.0, 1.0]])


def fwdkinspace(M, S_list, theta_list):
  """
  Computes the forward kinematics in space frame for an open chain manipulator.

  Args:

    M: Home configuration (position and orientation of the end-effector
    S_list: The joint screw axes in the space frame whene the manipulator is at
            the home position, in the format of a matrix with axes as the columns.
    theta_list: A list of joint coordinates

  Returns:

    Homogeneous 4x4 transformation from base to end-effector frame


  Example input:

    M = np.array([[-1, 0,  0, 0],
                  [ 0, 1,  0, 6],
                  [ 0, 0, -1, 2],
                  [ 0, 0,  0, 1]])
    S_list = np.array([[0, 0,  1,  4, 0,    0],
                       [0, 0,  0,  0, 1,    0],
                       [0, 0, -1, -6, 0, -0.1]])
    theta_list = np.array([np.pi / 2.0, 3, np.pi])

  Example output:

    np.array([[0, 1,  0,         -5],
              [1, 0,  0,          4],
              [0, 0, -1, 1.68584073],
              [0, 0,  0,          1]])

  """
  T = np.array(M)
  for S, theta in reversed(list(zip(S_list, theta_list))):
    T = poe(S, theta) @ T

  return T
