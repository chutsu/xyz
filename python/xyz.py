"""
xyz.py

Contains the following library code useful for prototyping robotic algorithms:

- IO
- YAML
- TIME
- PROFILING
- MATHS
- LINEAR ALGEBRA
- GEOMETRY
- LIE
- TRANSFORM
- QUATERNION
- TF
- MATPLOTLIB
- CV
- DATASET
- MANIPULATOR
- STATE ESTIMATION
- CALIBRATION
- SIMULATION

"""

import os
import sys
import glob
import math
import copy
import random
import json
import tarfile
import unittest
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from dataclasses import field
from collections import namedtuple

import typing
from typing import TypeVar
from typing import Annotated
from typing import Literal
from typing import Any
from typing import Callable

import cv2
import yaml
import numpy as np
import scipy.signal
import scipy.sparse
import scipy.sparse.linalg
import scipy.ndimage

import cProfile
from pstats import Stats

SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
EUROC_DATA_PATH = "/data/euroc/V1_01"

from numpy.typing import NDArray

DType = TypeVar("DType", bound=np.generic)
Vec2 = Annotated[NDArray[DType], Literal[2]]
Vec3 = Annotated[NDArray[DType], Literal[3]]
Vec4 = Annotated[NDArray[DType], Literal[4]]
Vec5 = Annotated[NDArray[DType], Literal[5]]
Vec6 = Annotated[NDArray[DType], Literal[6]]
Vec7 = Annotated[NDArray[DType], Literal[7]]
VecN = Annotated[NDArray[DType], Literal["N"]]
Mat2 = Annotated[NDArray[DType], Literal[2, 2]]
Mat3 = Annotated[NDArray[DType], Literal[3, 3]]
Mat34 = Annotated[NDArray[DType], Literal[3, 4]]
Mat4 = Annotated[NDArray[DType], Literal[4, 4]]
MatN = Annotated[NDArray[DType], Literal["N", "N"]]
MatNx2 = Annotated[NDArray[DType], Literal["N", "2"]]
MatNx3 = Annotated[NDArray[DType], Literal["N", "3"]]
MatNx4 = Annotated[NDArray[DType], Literal["N", "4"]]
Mat2xN = Annotated[NDArray[DType], Literal["2", "N"]]
Mat2x3 = Annotated[NDArray[DType], Literal["2", "3"]]
Mat3x4 = Annotated[NDArray[DType], Literal["3", "4"]]
Mat3xN = Annotated[NDArray[DType], Literal["3", "N"]]
Mat4xN = Annotated[NDArray[DType], Literal["4", "N"]]
Image = Annotated[NDArray[DType], Literal["N", "N"]]

###############################################################################
# I/O
###############################################################################


def extract_tar_gz(file_path, extract_path):
  with tarfile.open(str(file_path), "r:gz") as tar:
    tar.extractall(path=str(extract_path))

  return True


###############################################################################
# YAML
###############################################################################


def load_yaml(yaml_path):
  """Load YAML and return a named tuple"""
  assert yaml_path is not None
  assert yaml_path != ""

  # Load yaml_file
  yaml_data = None
  with open(yaml_path, "r") as stream:
    yaml_data = yaml.safe_load(stream)

  # Convert dict to named tuple
  data = json.dumps(yaml_data)  # Python dict to json
  data = json.loads(data,
                    object_hook=lambda d: namedtuple("X", d.keys())
                    (*d.values()))

  return data


###############################################################################
# TIME
###############################################################################


def sec2ts(time_s):
  """Convert time in seconds to timestamp"""
  return np.int64(time_s * 1e9)


def ts2sec(ts):
  """Convert timestamp to seconds"""
  return np.float64(ts) * 1e-9


###############################################################################
# PROFILING
###############################################################################


def profile_start():
  """Start profile"""
  prof = cProfile.Profile()
  prof.enable()
  return prof


def profile_stop(prof, **kwargs):
  """Stop profile"""
  key = kwargs.get("key", "cumtime")
  N = kwargs.get("N", 10)

  stats = Stats(prof)
  stats.strip_dirs()
  stats.sort_stats(key).print_stats(N)


###############################################################################
# MATHS
###############################################################################

from numpy import deg2rad
from numpy import rad2deg
from math import pi
from math import isclose
from math import sqrt
from math import ceil
from math import floor
from math import cos
from math import sin
from math import tan
from math import acos
from math import atan
from math import atan2


def rmse(errors):
  """Root Mean Squared Error"""
  return np.sqrt(np.mean(errors**2))


def clip_value(x, vmin, vmax):
  """Clip"""
  x_tmp = x
  x_tmp = vmax if (x_tmp > vmax) else x_tmp
  x_tmp = vmin if (x_tmp < vmin) else x_tmp
  return x_tmp


def wrap_180(d):
  x = np.fmod(d + 180, 360)
  if x < 0:
    x += 360.0

  return x - 180.0


def wrap_360(d):
  """Wrap angle `d` in degrees to 0 to 360 degrees"""
  x = np.fmod(d, 360)
  if x < 0:
    x += 360.0
  return x


def wrap_pi(r):
  """Wrap angle `r` in radians to +- pi radians."""
  return deg2rad(wrap_180(rad2deg(r)))


###############################################################################
# LINEAR ALGEBRA
###############################################################################

from numpy import deg2rad
from numpy import rad2deg
from numpy import sinc
from numpy import zeros
from numpy import ones
from numpy import eye
from numpy import trace
from numpy import diagonal as diag
from numpy import cross
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import matrix_rank as rank
from numpy.linalg import eig
from numpy.linalg import svd
from numpy.linalg import cholesky as chol


def pprint_matrix(mat, fmt = "g"):
  """Pretty Print matrix"""
  col_maxes = [
    max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T
  ]
  for x in mat:
    for i, y in enumerate(x):
      print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
    print("")


def normalize(v):
  """Normalize vector v"""
  n = np.linalg.norm(v)
  if n == 0:
    return v
  return v / n


def full_rank(A):
  """Check if matrix A is full rank"""
  return rank(A) == A.shape[0]


def hat(vec):
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


def vee(A):
  """Form skew symmetric matrix vector"""
  assert A.shape == (3, 3)
  return np.array([A[2, 1], A[0, 2], A[1, 0]])


def fwdsubs(L, b):
  """
  Solving a lower triangular system by forward-substitution
  Input matrix L is an n by n lower triangular matrix
  Input vector b is n by 1
  Output vector x is the solution to the linear system
  L x = b
  """
  assert L.shape[1] == b.shape[0]
  n = b.shape[0]

  x = zeros((n, 1))
  for j in range(n):
    if L[j, j] == 0:
      raise RuntimeError("Matrix is singular!")
    x[j] = b[j] / L[j, j]
    b[j:n] = b[j:n] - L[j:n, j] * x[j]


def bwdsubs(U, b):
  """
  Solving an upper triangular system by back-substitution
  Input matrix U is an n by n upper triangular matrix
  Input vector b is n by 1
  Output vector x is the solution to the linear system
  U x = b
  """
  assert U.shape[1] == b.shape[0]
  n = b.shape[0]

  x = zeros((n, 1))
  for j in range(n):
    if U[j, j] == 0:
      raise RuntimeError("Matrix is singular!")
    x[j] = b[j] / U[j, j]
    b[0:j] = b[0:j] - U[0:j, j] * x[j]


def solve_svd(A, b):
  """
  Solve Ax = b with SVD
  """
  # First decompose A with SVD to get U, s, Vh
  #
  #   A x = b
  #   (U diag(s) Vh) x = b
  #
  # Moving U to the R.H.S
  #
  #   diag(s) Vh x = U.T b
  #
  # Let c = U.T b
  U, s, Vh = svd(A)
  c = np.dot(U.T, b)

  # Now lets move diag(s) to the R.H.S
  #
  #   diag(s) Vh x = c
  #   Vh x = diag(1/s) c
  #
  # Let w = diag(1/s) c
  w = np.dot(np.diag(1 / s), c)

  # Finally to solve for x we move Vh to the R.H.S
  #
  #   Vh x = w
  #   x = Vh.H w
  #
  # where .H stands for hermitian = conjugate transpose)
  x = Vh.conj().T @ w

  return x


def schurs_complement(
  H,
  g,
  m,
  r,
  precond = False,
):
  """Shurs-complement"""
  assert H.shape[0] == (m + r)

  # H = [Hmm, Hmr
  #      Hrm, Hrr];
  Hmm = H[0:m, 0:m]
  Hmr = H[0:m, m:]
  Hrm = Hmr.T
  Hrr = H[m:, m:]

  # g = [gmm, grr]
  gmm = g[:m]
  grr = g[m:]

  # Precondition Hmm
  if precond:
    Hmm = 0.5 * (Hmm + Hmm.T)

  # Invert Hmm
  assert rank(Hmm) == Hmm.shape[0]
  (w, V) = eig(Hmm)
  W_inv = diag(1.0 / w)
  Hmm_inv = V @ W_inv @ V.T

  # Schurs complement
  H_marg = Hrr - Hrm @ Hmm_inv @ Hmr
  g_marg = grr - Hrm @ Hmm_inv @ gmm

  return (H_marg, g_marg)


def is_pd(B):
  """Returns true when input is positive-definite, via Cholesky"""
  try:
    _ = chol(B)
    return True
  except np.linalg.LinAlgError:
    return False


def nearest_pd(A):
  """Find the nearest positive-definite matrix to input

  A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
  credits [2].

  [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

  [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
  matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
  """
  B = (A + A.T) / 2
  _, s, V = svd(B)
  H = np.dot(V.T, np.dot(np.diag(s), V))
  A2 = (B + H) / 2
  A3 = (A2 + A2.T) / 2

  if is_pd(A3):
    return A3

  spacing = np.spacing(np.linalg.norm(A))
  # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
  # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
  # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
  # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
  # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
  # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
  # `spacing` will, for Gaussian random matrixes of small dimension, be on
  # othe order of 1e-16. In practice, both ways converge, as the unit test
  # below suggests.
  I = eye(A.shape[0])
  k = 1
  while not is_pd(A3):
    mineig = np.min(np.real(np.linalg.eigvals(A3)))
    A3 += I * (-mineig * k**2 + spacing)
    k += 1

  return A3


def matrix_equal(
  A,
  B,
  tol = 1e-8,
  verbose = False,
):
  """Compare matrices `A` and `B`"""
  diff = A - B

  if len(diff.shape) == 1:
    for i in range(diff.shape[0]):
      if abs(diff[i]) > tol:
        if verbose:
          print("A - B:")
          print(diff)

  elif len(diff.shape) == 2:
    for i in range(diff.shape[0]):
      for j in range(diff.shape[1]):
        if abs(diff[i, j]) > tol:
          if verbose:
            print("A - B:")
            print(diff)
          return False

  return True


def plot_compare_matrices(title_A, A, title_B, B):
  """Plot compare matrices"""
  plt.matshow(A)
  plt.colorbar()
  plt.title(title_A)

  plt.matshow(B)
  plt.colorbar()
  plt.title(title_B)

  diff = A - B
  plt.matshow(diff)
  plt.colorbar()
  plt.title(f"{title_A} - {title_B}")

  print(f"max_coeff({title_A}): {np.max(np.max(A))}")
  print(f"max_coeff({title_B}): {np.max(np.max(B))}")
  print(f"min_coeff({title_A}): {np.min(np.min(A))}")
  print(f"min_coeff({title_B}): {np.min(np.min(B))}")
  print(f"max_diff: {np.max(np.max(np.abs(diff)))}")

  plt.show()


def check_jacobian(
  jac_name,
  fdiff,
  jac,
  threshold,
  verbose = False,
):
  """Check jacobians"""

  # Check if numerical diff is same as analytical jacobian
  if matrix_equal(fdiff, jac, threshold):
    if verbose:
      print(f"Check [{jac_name}] passed!")
      print("-" * 60)

      print("J_fdiff:")
      pprint_matrix(fdiff)
      print()

      print("J:")
      pprint_matrix(jac)
      print()

      print("J_fdiff - J:")
      pprint_matrix(fdiff - jac)
      print()

      print("-" * 60)
    return True

  # Failed - print differences
  if verbose:
    fdiff_minus_jac = fdiff - jac

    print(f"Check [{jac_name}] failed!")
    print("-" * 60)

    print("J_fdiff:")
    pprint_matrix(fdiff)
    print()

    print("J:")
    pprint_matrix(jac)
    print()

    print("J_fdiff - J:")
    pprint_matrix(fdiff_minus_jac)
    print()

    print("-" * 60)

  return False


class TestLinearAlgebra(unittest.TestCase):
  """Test Linear Algebra"""
  def test_normalize(self):
    """Test normalize()"""
    x = np.array([1.0, 2.0, 3.0])
    x_prime = normalize(x)
    self.assertTrue(isclose(norm(x_prime), 1.0))

  def test_hat(self):
    """Test hat()"""
    x = np.array([1.0, 2.0, 3.0])
    S = np.array([[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])
    self.assertTrue(matrix_equal(S, hat(x)))

  def test_vee(self):
    """Test vee()"""
    x = np.array([1.0, 2.0, 3.0])
    S = np.array([[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])
    self.assertTrue(matrix_equal(x, vee(S)))

  def test_matrix_equal(self):
    """Test matrix_equal()"""
    A = ones((3, 3))
    B = ones((3, 3))
    self.assertTrue(matrix_equal(A, B))

    C = 2.0 * ones((3, 3))
    self.assertFalse(matrix_equal(A, C))

  # def test_check_jacobian(self):
  #   step_size = 1e-6
  #   threshold = 1e-5
  #
  #   x = 2
  #   y0 = x**2
  #   y1 = (x + step_size)**2
  #   jac = 2 * x
  #   fdiff = y1 - y0
  #
  #   jac_name = "jac"
  #   fdiff = (y1 - y0) / step_size
  #   self.assertTrue(check_jacobian(jac_name, fdiff, jac, threshold))


###############################################################################
# GEOMETRY
###############################################################################


def lerp(x0, x1, t):
  """Linear interpolation"""
  return (1.0 - t) * x0 + t * x1


def lerp2d(p0, p1, t):
  """Linear interpolation 2D"""
  assert len(p0) == 2
  assert len(p1) == 2
  assert 0.0 <= t <= 1.0
  x = lerp(p0[0], p1[0], t)
  y = lerp(p0[1], p1[1], t)
  return np.array([x, y])


def lerp3d(p0, p1, t):
  """Linear interpolation 3D"""
  assert len(p0) == 3
  assert len(p1) == 3
  assert 0.0 <= t <= 1.0
  x = lerp(p0[0], p1[0], t)
  y = lerp(p0[1], p1[1], t)
  z = lerp(p0[2], p1[2], t)
  return np.array([x, y, z])


def circle(r, theta):
  """Circle"""
  x = r * cos(theta)
  y = r * sin(theta)
  return np.array([x, y])


def sphere(rho, theta, phi):
  """
  Sphere

  Args:

    rho (float): Sphere radius
    theta (float): longitude [rad]
    phi (float): Latitude [rad]

  Returns:

    Point on sphere

  """
  x = rho * sin(theta) * cos(phi)
  y = rho * sin(theta) * sin(phi)
  z = rho * cos(theta)
  return np.array([x, y, z])


def circle_loss(c, x, y):
  """
  Calculate the algebraic distance between the data points and the mean
  circle centered at c=(xc, yc)
  """
  xc, yc = c
  # Euclidean dist from center (xc, yc)
  Ri = np.sqrt((x - xc)**2 + (y - yc)**2)
  return Ri - Ri.mean()


def find_circle(x, y):
  """
  Find the circle center and radius given (x, y) data points using least
  squares. Returns `(circle_center, circle_radius, residual)`
  """
  x_m = np.mean(x)
  y_m = np.mean(y)
  center_init = x_m, y_m
  center, _ = scipy.optimize.leastsq(
    circle_loss,  # pyright: ignore
    center_init,
    args=(x, y),
  )

  xc, yc = center
  radii = np.sqrt((x - xc)**2 + (y - yc)**2)
  radius = radii.mean()
  residual = np.sum((radii - radius)**2)

  return (center, radius, residual)


def bresenham(p0, p1):
  """
  Bresenham's line algorithm is a line drawing algorithm that determines the
  points of an n-dimensional raster that should be selected in order to form
  a close approximation to a straight line between two points. It is commonly
  used to draw line primitives in a bitmap image (e.g. on a computer screen),
  as it uses only integer addition, subtraction and bit shifting, all of
  which are very cheap operations in standard computer architectures.

  Args:

    p0 (np.array): Starting point (x, y)
    p1 (np.array): End point (x, y)

  Returns:

    A list of (x, y) intermediate points from p0 to p1.

  """
  x0, y0 = p0
  x1, y1 = p1
  dx = abs(x1 - x0)
  dy = abs(y1 - y0)
  sx = 1.0 if x0 < x1 else -1.0
  sy = 1.0 if y0 < y1 else -1.0
  err = dx - dy

  line = []
  while True:
    line.append([x0, y0])
    if x0 == x1 and y0 == y1:
      return line

    e2 = 2 * err
    if e2 > -dy:
      # overshot in the y direction
      err = err - dy
      x0 = x0 + sx
    if e2 < dx:
      # overshot in the x direction
      err = err + dx
      y0 = y0 + sy


def find_intersection(
  p1,
  p2,
  q1,
  q2,
):
  """
  Find the intersection between two lines formed by points p1, p2 and q1, q2
  for line 1 and line 2 respectively.

  Args:

    p1 (np.array): Starting point for line 1
    p2 (np.array): End point for line 1
    q1 (np.array): Starting point for line 2
    q2 (np.array): End point for line 2

  Returns:

    status (bool): To denote whether there is an intersection
    intersection (np.array): Intersection point from line 1

  """
  # Direction vectors for line 1 and line 2
  d1 = p2 - p1
  d2 = q2 - q1

  # Form Ax = b
  A = np.array([d1, -d2]).T  # A 3x2 matrix
  b = q1 - p1  # A 3x1 vector

  # Use least squares to solve (since A is not square)
  t_s, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
  t, s = t_s

  # Check if they intersect
  line1_intersect = p1 + t * d1
  line2_intersect = q1 + s * d2
  status = np.allclose(line1_intersect, line2_intersect)

  return status, p1 + t * d1, residuals, rank


def find_planes_intersect(
  p1,
  p2,
  p3,
):
  """Given 3 planes, find their intersect if it exists"""
  A = np.zeros((3, 3))
  A[0, 0:3] = p1[0:3]
  A[1, 0:3] = p2[0:3]
  A[2, 0:3] = p3[0:3]

  b = np.zeros((3, 1))
  b[0] = p1[3]
  b[1] = p2[3]
  b[2] = p3[3]

  try:
    return np.linalg.solve(A, b)
  except np.linalg.LinAlgError:
    return None


def fix_rotation_matrix(R):
  """Ensure R is a valid rotation matrix by enforcing det(R) = 1."""
  U, _, Vt = np.linalg.svd(R)
  R_fixed = U @ Vt  # Project onto SO(3)

  # Ensure det(R) = 1 (corrects reflections)
  if np.linalg.det(R_fixed) < 0:
    U[:, -1] *= -1  # Flip last column of U
    R_fixed = U @ Vt

  return R_fixed


###############################################################################
# LIE
###############################################################################


def Exp(phi):
  """Exponential Map"""
  assert phi.shape == (3,) or phi.shape == (3, 1)
  if norm(phi) < 1e-3:
    C = eye(3) + hat(phi)
    return C

  phi_norm = norm(phi)
  phi_skew = hat(phi)
  phi_skew_sq = phi_skew @ phi_skew

  C = eye(3)
  C += (sin(phi_norm) / phi_norm) * phi_skew
  C += ((1 - cos(phi_norm)) / phi_norm**2) * phi_skew_sq
  return C


def Log(C):
  """Logarithmic Map"""
  assert C.shape == (3, 3)
  # phi = acos((trace(C) - 1) / 2);
  # u = vee(C - C') / (2 * sin(phi));
  # rvec = phi * u;

  C00, C01, C02 = C[0, :]
  C10, C11, C12 = C[1, :]
  C20, C21, C22 = C[2, :]

  tr = np.trace(C)
  rvec = None
  if tr + 1.0 < 1e-10:
    if abs(C22 + 1.0) > 1.0e-5:
      x = np.array([C02, C12, 1.0 + C22])
      rvec = (pi / np.sqrt(2.0 + 2.0 * C22)) @ x
    elif abs(C11 + 1.0) > 1.0e-5:
      x = np.array([C01, 1.0 + C11, C21])
      rvec = (pi / np.sqrt(2.0 + 2.0 * C11)) @ x
    else:
      x = np.array([1.0 + C00, C10, C20])
      rvec = (pi / np.sqrt(2.0 + 2.0 * C00)) @ x

  else:
    tr_3 = tr - 3.0  # always negative
    if tr_3 < -1e-7:
      theta = acos((tr - 1.0) / 2.0)
      magnitude = theta / (2.0 * sin(theta))
    else:
      # when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
      # use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
      # see https://github.com/borglab/gtsam/issues/746 for details
      magnitude = 0.5 - tr_3 / 12.0
    rvec = magnitude @ np.array([C21 - C12, C02 - C20, C10 - C01])

  return rvec


def Jr(theta):
  """
  Right jacobian

  Forster, Christian, et al. "IMU preintegration on manifold for efficient
  visual-inertial maximum-a-posteriori estimation." Georgia Institute of
  Technology, 2015.
  [Page 2, Equation (8)]
  """
  theta_norm = norm(theta)
  theta_norm_sq = theta_norm * theta_norm
  theta_norm_cube = theta_norm_sq * theta_norm
  theta_skew = hat(theta)
  theta_skew_sq = theta_skew @ theta_skew

  J = eye(3)
  J -= ((1 - cos(theta_norm)) / theta_norm_sq) * theta_skew
  J += (theta_norm - sin(theta_norm)) / (theta_norm_cube) * theta_skew_sq
  return J


def Jr_inv(theta):
  """Inverse right jacobian"""
  theta_norm = norm(theta)
  theta_norm_sq = theta_norm * theta_norm
  theta_skew = hat(theta)
  theta_skew_sq = theta_skew @ theta_skew

  A = 1.0 / theta_norm_sq
  B = (1 + cos(theta_norm)) / (2 * theta_norm * sin(theta_norm))

  J = eye(3)
  J += 0.5 * theta_skew
  J += (A - B) * theta_skew_sq
  return J


def SO3_boxplus(C, alpha):
  """Box plus"""
  assert C.shape == (3, 3)
  # C_updated = C [+] alpha
  C_updated = C @ Exp(alpha)
  return C_updated


def SO3_boxminus(C_a, C_b):
  """Box minus"""
  assert C_a.shape == (3, 3)
  assert C_b.shape == (3, 3)
  # alpha = C_a [-] C_b
  alpha = Log(C_b.T @ C_a)
  return alpha


def so3_exp(so3mat, tol=1e-6):
  """Computes the matrix exponential of a matrix in so(3)

  Example Input:
    so3mat = np.array([[ 0, -3,  2],
                       [ 3,  0, -1],
                       [-2,  1,  0]])

  Output:

    np.array([[-0.69492056,  0.71352099,  0.08929286],
              [-0.19200697, -0.30378504,  0.93319235],
              [ 0.69297817,  0.6313497 ,  0.34810748]])

  """
  aa = vee(so3mat)
  if np.linalg.norm(aa) < tol:
    return eye(3)

  _, theta = aa_decomp(aa)
  omgmat = so3mat / theta

  I3 = eye(3)
  s_theta = np.sin(theta)
  c_theta = np.cos(theta)

  return I3 + s_theta * omgmat + (1.0 - c_theta) * np.dot(omgmat, omgmat)


def so3_Exp(w):
  """Exponential Map R3 to so3"""
  return so3_exp(hat(w))


class TestLie(unittest.TestCase):
  """Test Lie algebra functions"""
  def test_Exp_Log(self):
    """Test Exp() and Log()"""
    pass

  def test_sandbox(self):
    """Test sandbox"""
    step_size = 1e-8
    threshold = 1e-4

    # Test Jacobian w.r.t C_10 in p_1 = T_10 * p_0
    C_10 = euler321(0.1, 0.2, 0.3)
    r_10 = np.array([0.1, 0.2, 0.3])
    T_10 = tf(C_10, r_10)
    p_0 = np.random.uniform(-1.0, 1.0, size=(3,))
    p_1 = tf_point(T_10, p_0)

    J_fdiff = np.zeros((3, 3))
    for i in range(3):
      T_fwd = tf_perturb(T_10, 3 + i, step_size)
      p_1_fwd = tf_point(T_fwd, p_0)
      J_fdiff[:, i] = (p_1_fwd - p_1) / step_size

    J = np.zeros((3, 3))
    J[0:3, 0:3] = -tf_rot(T_10) @ hat(p_0)
    check_jacobian("J_rot", J_fdiff, J, threshold, verbose=False)

    # Test Jacobian w.r.t C_10 in p_2 = T_21 * T_10 * p_0
    C_10 = euler321(0.1, 0.2, 0.3)
    r_10 = np.array([0.1, 0.2, 0.3])
    T_10 = tf(C_10, r_10)

    C_21 = euler321(0.1, 0.2, 0.3)
    r_21 = np.array([0.1, 0.2, 0.3])
    T_21 = tf(C_21, r_21)

    p_0 = np.random.uniform(-1.0, 1.0, size=(3,))
    p_2 = tf_point(T_21 @ T_10, p_0)

    J_fdiff = np.zeros((3, 3))
    for i in range(3):
      T_10_fwd = tf_perturb(T_10, 3 + i, step_size)
      p_2_fwd = tf_point(T_21 @ T_10_fwd, p_0)
      J_fdiff[:, i] = (p_2_fwd - p_2) / step_size

    J = np.zeros((3, 3))
    J[0:3, 0:3] = C_21 @ -tf_rot(T_10) @ hat(p_0)
    check_jacobian("J_rot", J_fdiff, J, threshold, verbose=False)

    # Test Jacobian w.r.t C_21 in p_3 = T_32 * inv(T_21) * T_10 * p_0
    C_10 = euler321(*np.random.uniform(-1.0, 1.0, size=(3,)))
    r_10 = np.random.uniform(-1.0, 1.0, size=(3,))
    T_10 = tf(C_10, r_10)

    C_21 = euler321(*np.random.uniform(-1.0, 1.0, size=(3,)))
    r_21 = np.random.uniform(-1.0, 1.0, size=(3,))
    T_21 = tf(C_21, r_21)

    C_32 = euler321(*np.random.uniform(-1.0, 1.0, size=(3,)))
    r_32 = np.random.uniform(-1.0, 1.0, size=(3,))
    T_32 = tf(C_32, r_32)

    p_0 = np.random.uniform(-1.0, 1.0, size=(3,))
    p_3 = tf_point(T_32 @ inv(T_21) @ T_10, p_0)

    J_fdiff = np.zeros((3, 3))
    for i in range(3):
      T_21_fwd = tf_perturb(T_21, 3 + i, step_size)
      p_3_fwd = tf_point(T_32 @ inv(T_21_fwd) @ T_10, p_0)
      J_fdiff[:, i] = (p_3_fwd - p_3) / step_size

    J = np.zeros((3, 3))
    p_1 = tf_point(T_10, p_0)
    J[0:3, 0:3] = C_32 @ -C_21.T @ hat(p_1 - r_21) @ -C_21
    check_jacobian("J_rot", J_fdiff, J, threshold, verbose=False)

    # Test Jacobian w.r.t C_21 in p_3 = inv(T_32) * inv(T_21) * T_10 * p_0
    C_10 = euler321(*np.random.uniform(-1.0, 1.0, size=(3,)))
    r_10 = np.random.uniform(-1.0, 1.0, size=(3,))
    T_10 = tf(C_10, r_10)

    C_21 = euler321(*np.random.uniform(-1.0, 1.0, size=(3,)))
    r_21 = np.random.uniform(-1.0, 1.0, size=(3,))
    T_21 = tf(C_21, r_21)

    C_32 = euler321(*np.random.uniform(-1.0, 1.0, size=(3,)))
    r_32 = np.random.uniform(-1.0, 1.0, size=(3,))
    T_32 = tf(C_32, r_32)

    p_0 = np.random.uniform(-1.0, 1.0, size=(3,))
    p_3 = tf_point(inv(T_32) @ inv(T_21) @ T_10, p_0)

    J_fdiff = np.zeros((3, 3))
    for i in range(3):
      T_21_fwd = tf_perturb(T_21, 3 + i, step_size)
      p_3_fwd = tf_point(inv(T_32) @ inv(T_21_fwd) @ T_10, p_0)
      J_fdiff[:, i] = (p_3_fwd - p_3) / step_size

    J = np.zeros((3, 3))
    p_1 = tf_point(T_10, p_0)
    J[0:3, 0:3] = C_32.T @ -C_21.T @ hat(p_1 - r_21) @ -C_21
    check_jacobian("J_rot", J_fdiff, J, threshold, verbose=False)

    # Test Jacobian w.r.t C_10 in p_1 = inv(T_10) * p_0
    C_10 = euler321(0.1, 0.2, 0.3)
    r_10 = np.array([0.1, 0.2, 0.3])
    T_10 = tf(C_10, r_10)
    T_01 = inv(T_10)
    p_1 = np.random.uniform(-1.0, 1.0, size=(3,))
    p_0 = tf_point(T_01, p_1)

    J_fdiff = np.zeros((3, 3))
    for i in range(3):
      C_10_fwd = rot_perturb(C_10, i, step_size)
      T_10_fwd = tf(C_10_fwd, r_10)
      T_01_fwd = inv(T_10_fwd)
      p_0_fwd = tf_point(T_01_fwd, p_1)
      J_fdiff[:, i] = (p_0_fwd - p_0) / step_size

    J = np.zeros((3, 3))
    J[0:3, 0:3] = -C_10.T @ hat(p_1 - r_10) @ -C_10
    check_jacobian("J_rot", J_fdiff, J, threshold, verbose=False)


###############################################################################
# TRANSFORM
###############################################################################


def homogeneous(p):
  """Turn point `p` into its homogeneous form"""
  return np.array([*p, 1.0])


def dehomogeneous(hp):
  """De-homogenize point `hp` into `p`"""
  return hp[0:3]


def rotx(theta):
  """Form rotation matrix around x axis"""
  row0 = [1.0, 0.0, 0.0]
  row1 = [0.0, cos(theta), -sin(theta)]
  row2 = [0.0, sin(theta), cos(theta)]
  return np.array([row0, row1, row2])


def roty(theta):
  """Form rotation matrix around y axis"""
  row0 = [cos(theta), 0.0, sin(theta)]
  row1 = [0.0, 1.0, 0.0]
  row2 = [-sin(theta), 0.0, cos(theta)]
  return np.array([row0, row1, row2])


def rotz(theta):
  """Form rotation matrix around z axis"""
  row0 = [cos(theta), -sin(theta), 0.0]
  row1 = [sin(theta), cos(theta), 0.0]
  row2 = [0.0, 0.0, 1.0]
  return np.array([row0, row1, row2])


def aa2quat(axis, angle):
  """
  Convert Axis-angle to quaternion

  Source:
  Sola, Joan. "Quaternion kinematics for the error-state Kalman filter." arXiv
  preprint arXiv:1711.02508 (2017).
  [Page 22, eq (101), "Quaternion and rotation vector"]
  """
  ax, ay, az = axis
  qw = cos(angle / 2.0)
  qx = ax * sin(angle / 2.0)
  qy = ay * sin(angle / 2.0)
  qz = az * sin(angle / 2.0)
  return np.array([qw, qx, qy, qz])


def aa2rot(aa):
  """Axis-angle to rotation matrix"""
  # If small rotation
  theta = sqrt(aa @ aa)  # = norm(aa), but faster
  eps = 1e-8
  if theta < eps:
    return hat(aa)

  # Convert aa to rotation matrix
  aa = aa / theta
  x, y, z = aa

  c = cos(theta)
  s = sin(theta)
  C = 1 - c

  xs = x * s
  ys = y * s
  zs = z * s

  xC = x * C
  yC = y * C
  zC = z * C

  xyC = x * yC
  yzC = y * zC
  zxC = z * xC

  row0 = [x * xC + c, xyC - zs, zxC + ys]
  row1 = [xyC + zs, y * yC + c, yzC - xs]
  row2 = [zxC - ys, yzC + xs, z * zC + c]
  return np.array([row0, row1, row2])


def aa_vec(axis, angle):
  """Form Axis-Angle Vector"""
  assert axis.shape[0] == 3
  return axis * angle


def aa_decomp(aa):
  """Decompose an axis-angle into its components"""
  w = aa / np.linalg.norm(aa)
  theta = np.linalg.norm(aa)
  return w, theta


def vecs2aa(u, v):
  """From 2 vectors form an axis-angle vector"""
  angle = math.acos(u.T * v)
  ax = normalize(np.cross(u, v))
  return ax * angle


def euler321(
  yaw,
  pitch,
  roll,
):
  """
  Convert yaw, pitch, roll in radians to a 3x3 rotation matrix.

  Source:
  Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
  Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
  Princeton University Press, 1999. Print.
  [Page 85-86, "The Aerospace Sequence"]
  """
  psi = yaw
  theta = pitch
  phi = roll

  cpsi = cos(psi)
  spsi = sin(psi)
  ctheta = cos(theta)
  stheta = sin(theta)
  cphi = cos(phi)
  sphi = sin(phi)

  C11 = cpsi * ctheta
  C21 = spsi * ctheta
  C31 = -stheta

  C12 = cpsi * stheta * sphi - spsi * cphi
  C22 = spsi * stheta * sphi + cpsi * cphi
  C32 = ctheta * sphi

  C13 = cpsi * stheta * cphi + spsi * sphi
  C23 = spsi * stheta * cphi - cpsi * sphi
  C33 = ctheta * cphi

  return np.array([[C11, C12, C13], [C21, C22, C23], [C31, C32, C33]])


def euler2quat(yaw, pitch, roll):
  """
  Convert yaw, pitch, roll in radians to a quaternion.

  Source:
  Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
  Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
  Princeton University Press, 1999. Print.
  [Page 166-167, "Euler Angles to Quaternion"]
  """
  psi = yaw  # Yaw
  theta = pitch  # Pitch
  phi = roll  # Roll

  c_phi = cos(phi / 2.0)
  c_theta = cos(theta / 2.0)
  c_psi = cos(psi / 2.0)
  s_phi = sin(phi / 2.0)
  s_theta = sin(theta / 2.0)
  s_psi = sin(psi / 2.0)

  qw = c_psi * c_theta * c_phi + s_psi * s_theta * s_phi
  qx = c_psi * c_theta * s_phi - s_psi * s_theta * c_phi
  qy = c_psi * s_theta * c_phi + s_psi * c_theta * s_phi
  qz = s_psi * c_theta * c_phi - c_psi * s_theta * s_phi

  mag = sqrt(qw**2 + qx**2 + qy**2 + qz**2)
  return np.array([qw / mag, qx / mag, qy / mag, qz / mag])


def quat2euler(q):
  """
  Convert quaternion to euler angles (yaw, pitch, roll).

  Source:
  Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
  Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
  Princeton University Press, 1999. Print.
  [Page 168, "Quaternion to Euler Angles"]
  """
  qw, qx, qy, qz = q

  m11 = (2 * qw**2) + (2 * qx**2) - 1
  m12 = 2 * (qx * qy + qw * qz)
  m13 = 2 * qx * qz - 2 * qw * qy
  m23 = 2 * qy * qz + 2 * qw * qx
  m33 = (2 * qw**2) + (2 * qz**2) - 1

  psi = math.atan2(m12, m11)
  theta = math.asin(-m13)
  phi = math.atan2(m23, m33)

  ypr = np.array([psi, theta, phi])
  return ypr


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


def rot2euler(C):
  """
  Convert 3x3 rotation matrix to euler angles (yaw, pitch, roll). The result is
  also equivalent to rotation around (z, y, x) axes.
  """
  assert C.shape == (3, 3)
  q = rot2quat(C)
  return quat2euler(q)


def rot2quat(C):
  """
  Convert 3x3 rotation matrix to quaternion.
  """
  assert C.shape == (3, 3)

  m00 = C[0, 0]
  m01 = C[0, 1]
  m02 = C[0, 2]

  m10 = C[1, 0]
  m11 = C[1, 1]
  m12 = C[1, 2]

  m20 = C[2, 0]
  m21 = C[2, 1]
  m22 = C[2, 2]

  tr = m00 + m11 + m22

  if tr > 0:
    S = sqrt(tr + 1.0) * 2.0
    # S=4*qw
    qw = 0.25 * S
    qx = (m21 - m12) / S
    qy = (m02 - m20) / S
    qz = (m10 - m01) / S
  elif (m00 > m11) and (m00 > m22):
    S = sqrt(1.0 + m00 - m11 - m22) * 2.0
    # S=4*qx
    qw = (m21 - m12) / S
    qx = 0.25 * S
    qy = (m01 + m10) / S
    qz = (m02 + m20) / S
  elif m11 > m22:
    S = sqrt(1.0 + m11 - m00 - m22) * 2.0
    # S=4*qy
    qw = (m02 - m20) / S
    qx = (m01 + m10) / S
    qy = 0.25 * S
    qz = (m12 + m21) / S
  else:
    S = sqrt(1.0 + m22 - m00 - m11) * 2.0
    # S=4*qz
    qw = (m10 - m01) / S
    qx = (m02 + m20) / S
    qy = (m12 + m21) / S
    qz = 0.25 * S

  return quat_normalize(np.array([qw, qx, qy, qz]))


def rot_diff(C0, C1, tol = 1e-5):
  """Difference between two rotation matrices"""
  dC = C0.T @ C1
  tr = np.trace(dC)
  if tr < 0:
    tr *= -1

  if np.fabs(tr - 3.0) < tol:
    dtheta = 0.0
  else:
    dtheta = acos((tr - 1.0) / 2.0)

  return dtheta


# UNITESTS #####################################################################


class TestTransform(unittest.TestCase):
  """Test transform functions"""
  def test_homogeneous(self):
    """Test homogeneous()"""
    p = np.array([1.0, 2.0, 3.0])
    hp = homogeneous(p)
    self.assertTrue(hp[0] == 1.0)
    self.assertTrue(hp[1] == 2.0)
    self.assertTrue(hp[2] == 3.0)
    self.assertTrue(len(hp) == 4)

  def test_dehomogeneous(self):
    """Test dehomogeneous()"""
    p = np.array([1.0, 2.0, 3.0])
    hp = np.array([1.0, 2.0, 3.0, 1.0])
    p = dehomogeneous(hp)
    self.assertTrue(p[0] == 1.0)
    self.assertTrue(p[1] == 2.0)
    self.assertTrue(p[2] == 3.0)
    self.assertTrue(len(p) == 3)

  def test_rotx(self):
    """Test rotx()"""
    x = np.array([0.0, 1.0, 0.0])
    C = rotx(deg2rad(90.0))
    x_prime = C @ x
    self.assertTrue(np.allclose(x_prime, [0.0, 0.0, 1.0]))

  def test_roty(self):
    """Test roty()"""
    x = np.array([1.0, 0.0, 0.0])
    C = roty(deg2rad(90.0))
    x_prime = C @ x
    self.assertTrue(np.allclose(x_prime, [0.0, 0.0, -1.0]))

  def test_rotz(self):
    """Test rotz()"""
    x = np.array([1.0, 0.0, 0.0])
    C = rotz(deg2rad(90.0))
    x_prime = C @ x
    self.assertTrue(np.allclose(x_prime, [0.0, 1.0, 0.0]))

  def test_aa2quat(self):
    """Test aa2quat()"""
    pass

  def test_aa2rot(self):
    """Test rvec2quat()"""
    pass

  def test_vecs2aa(self):
    """Test vecs2aa()"""
    pass

  def test_euler321(self):
    """Test euler321()"""
    C = euler321(0.0, 0.0, 0.0)
    self.assertTrue(np.array_equal(C, eye(3)))

  def test_euler2quat_and_quat2euler(self):
    """Test euler2quat() and quat2euler()"""
    y_in = deg2rad(3.0)
    p_in = deg2rad(2.0)
    r_in = deg2rad(1.0)

    q = euler2quat(y_in, p_in, r_in)
    ypr_out = quat2euler(q)

    self.assertTrue(len(q) == 4)
    self.assertTrue(abs(y_in - ypr_out[0]) < 1e-5)
    self.assertTrue(abs(p_in - ypr_out[1]) < 1e-5)
    self.assertTrue(abs(r_in - ypr_out[2]) < 1e-5)

  def test_quat2rot(self):
    """Test quat2rot()"""
    ypr = np.array([0.1, 0.2, 0.3])
    C_i = euler321(*ypr)
    C_j = quat2rot(euler2quat(*ypr))
    self.assertTrue(np.allclose(C_i, C_j))

  def test_rot2euler(self):
    """Test rot2euler()"""
    ypr = np.array([0.1, 0.2, 0.3])
    C = euler321(*ypr)
    euler = rot2euler(C)
    self.assertTrue(np.allclose(ypr, euler))

  def test_rot2quat(self):
    """Test rot2quat()"""
    ypr = np.array([0.1, 0.2, 0.3])
    C = euler321(*ypr)
    q = rot2quat(C)
    self.assertTrue(np.allclose(quat2euler(q), ypr))


# QUATERNION ##################################################################


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
  Omega[1:4, 1:4] = -hat(w)
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


def quat_slerp(q_i, q_j, t):
  """Quaternion Slerp `q_i` and `q_j` with parameter `t`"""
  assert len(q_i) == 4
  assert len(q_j) == 4
  assert 0.0 <= t <= 1.0

  # Compute the cosine of the angle between the two vectors.
  dot_result = q_i @ q_j

  # If the dot product is negative, slerp won't take
  # the shorter path. Note that q_j and -q_j are equivalent when
  # the negation is applied to all four components. Fix by
  # reversing one quaternion.
  if dot_result < 0.0:
    q_j = -q_j
    dot_result = -1.0 * dot_result

  DOT_THRESHOLD = 0.9995
  if dot_result > DOT_THRESHOLD:
    # If the inputs are too close for comfort, linearly interpolate
    # and normalize the result.
    return q_i + t * (q_j - q_i)

  # Since dot is in range [0, DOT_THRESHOLD], acos is safe
  theta_0 = acos(dot_result)  # theta_0 = angle between input vectors
  theta = theta_0 * t  # theta = angle between q_i and result
  sin_theta = sin(theta)  # compute this value only once
  sin_theta_0 = sin(theta_0)  # compute this value only once

  # == sin(theta_0 - theta) / sin(theta_0)
  s0 = cos(theta) - dot_result * sin_theta / sin_theta_0
  s1 = sin_theta / sin_theta_0

  return (s0 * q_i) + (s1 * q_j)


# UNITESTS #####################################################################


class TestQuaternion(unittest.TestCase):
  """Test Quaternion functions"""
  def test_quat_norm(self):
    """Test quat_norm()"""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    self.assertTrue(isclose(quat_norm(q), 1.0))

  def test_quat_normalize(self):
    """Test quat_normalize()"""
    q = np.array([1.0, 0.1, 0.2, 0.3])
    q = quat_normalize(q)
    self.assertTrue(isclose(quat_norm(q), 1.0))

  def test_quat_conj(self):
    """Test quat_conj()"""
    ypr = np.array([0.1, 0.0, 0.0])
    q = rot2quat(euler321(*ypr))
    q_conj = quat_conj(q)
    self.assertTrue(np.allclose(quat2euler(q_conj), -1.0 * ypr))

  def test_quat_inv(self):
    """Test quat_inv()"""
    ypr = np.array([0.1, 0.0, 0.0])
    q = rot2quat(euler321(*ypr))
    q_inv = quat_inv(q)
    self.assertTrue(np.allclose(quat2euler(q_inv), -1.0 * ypr))

  def test_quat_mul(self):
    """Test quat_mul()"""
    p = euler2quat(deg2rad(3.0), deg2rad(2.0), deg2rad(1.0))
    q = euler2quat(deg2rad(1.0), deg2rad(2.0), deg2rad(3.0))
    r = quat_mul(p, q)
    self.assertTrue(r is not None)

  def test_quat_omega(self):
    """Test quat_omega()"""
    pass

  def test_quat_slerp(self):
    """Test quat_slerp()"""
    q_i = rot2quat(euler321(0.1, 0.0, 0.0))
    q_j = rot2quat(euler321(0.2, 0.0, 0.0))
    q_k = quat_slerp(q_i, q_j, 0.5)
    self.assertTrue(np.allclose(quat2euler(q_k), [0.15, 0.0, 0.0]))

    q_i = rot2quat(euler321(0.0, 0.1, 0.0))
    q_j = rot2quat(euler321(0.0, 0.2, 0.0))
    q_k = quat_slerp(q_i, q_j, 0.5)
    self.assertTrue(np.allclose(quat2euler(q_k), [0.0, 0.15, 0.0]))

    q_i = rot2quat(euler321(0.0, 0.0, 0.1))
    q_j = rot2quat(euler321(0.0, 0.0, 0.2))
    q_k = quat_slerp(q_i, q_j, 0.5)
    self.assertTrue(np.allclose(quat2euler(q_k), [0.0, 0.0, 0.15]))

  def test_tf(self):
    """Test tf()"""
    r = np.array([1.0, 2.0, 3.0])
    q = np.array([0.0, 0.0, 0.0, 1.0])
    T = tf(q, r)

    self.assertTrue(np.allclose(T[0:3, 0:3], quat2rot(q)))
    self.assertTrue(np.allclose(T[0:3, 3], r))


# TF ##########################################################################


def tf(rot, trans):
  """
  Form 4x4 homogeneous transformation matrix from rotation `rot` and
  translation `trans`. Where the rotation component `rot` can be a rotation
  matrix or a quaternion.
  """
  C = None
  if rot.shape == (4,) or rot.shape == (4, 1):
    C = quat2rot(rot)
  elif rot.shape == (3, 3):
    C = rot
  else:
    raise RuntimeError("Invalid rotation!")

  T = eye(4, 4)
  T[0:3, 0:3] = C
  T[0:3, 3] = trans
  return T


def tf_rot(T):
  """Return rotation matrix from 4x4 homogeneous transform"""
  assert T.shape == (4, 4)
  return T[0:3, 0:3]


def tf_quat(T):
  """Return quaternion from 4x4 homogeneous transform"""
  assert T.shape == (4, 4)
  return rot2quat(tf_rot(T))


def tf_euler(T):
  """Return Euler angles from 4x4 homogeneous transform"""
  assert T.shape == (4, 4)
  return rot2euler(tf_rot(T))


def tf2pose(T):
  """Form pose vector"""
  rx, ry, rz = tf_trans(T)
  qw, qx, qy, qz = tf_quat(T)
  return np.array([rx, ry, rz, qx, qy, qz, qw])


def pose2tf(pose_vec):
  """Convert pose vector to transformation matrix"""
  rx, ry, rz = pose_vec[0:3]
  qx, qy, qz, qw = pose_vec[3:7]
  return tf(np.array([qw, qx, qy, qz]), np.array([rx, ry, rz]))


def tf_trans(T):
  """Return translation vector from 4x4 homogeneous transform"""
  assert T.shape == (4, 4)
  return T[0:3, 3]


def tf_inv(T):
  """Invert 4x4 homogeneous transform"""
  assert T.shape == (4, 4)
  return np.linalg.inv(T)


def tf_point(T, p):
  """Transform 3d point"""
  assert T.shape == (4, 4)
  assert p.shape == (3,) or p.shape == (3, 1)
  hpoint = np.array([p[0], p[1], p[2], 1.0])
  return (T @ hpoint)[0:3]


def tf_hpoint(T, hp):
  """Transform 3d point"""
  assert T.shape == (4, 4)
  assert hp.shape == (4,) or hp.shape == (4, 1)
  return (T @ hp)[0:3]


def tf_decompose(T):
  """Decompose into rotation matrix and translation vector"""
  assert T.shape == (4, 4)
  C = tf_rot(T)
  r = tf_trans(T)
  return (C, r)


def tf_lerp(pose_i, pose_j, t):
  """Interpolate pose `pose_i` and `pose_j` with parameter `t`"""
  assert pose_i.shape == (4, 4)
  assert pose_j.shape == (4, 4)
  assert 0.0 <= t <= 1.0

  # Decompose start pose
  r_i = tf_trans(pose_i)
  q_i = tf_quat(pose_i)

  # Decompose end pose
  r_j = tf_trans(pose_j)
  q_j = tf_quat(pose_j)

  # Interpolate translation and rotation
  r_lerp = np.array(lerp(r_i, r_j, t))
  q_lerp = quat_slerp(q_i, q_j, t)

  return tf(q_lerp, r_lerp)


def rot_perturb(C, i, step_size):
  """Perturb rotation matrix"""
  # Perturb rotation
  rvec = np.array([0.0, 0.0, 0.0])
  rvec[i - 3] = step_size

  q = rot2quat(C)
  dq = quat_delta(rvec)

  q_diff = quat_mul(q, dq)
  q_diff = quat_normalize(q_diff)

  return quat2rot(q_diff)


def tf_perturb(T, i, step_size):
  """Perturb transformation matrix"""
  assert T.shape == (4, 4)
  assert i >= 0 and i <= 5

  # Setup
  C = tf_rot(T)
  r = tf_trans(T)

  if i >= 0 and i <= 2:
    # Perturb translation
    r[i] += step_size

  elif i >= 3 and i <= 5:
    # Perturb rotation
    rvec = np.array([0.0, 0.0, 0.0])
    rvec[i - 3] = step_size

    q = rot2quat(C)
    dq = quat_delta(rvec)

    q_diff = quat_mul(q, dq)
    q_diff = quat_normalize(q_diff)

    C = quat2rot(q_diff)

  return tf(C, r)


def tf_update(T, dx):
  """Update transformation matrix"""
  assert T.shape == (4, 4)

  q = tf_quat(T)
  r = tf_trans(T)

  dr = dx[0:3]
  dalpha = dx[3:6]
  dq = quat_delta(dalpha)

  return tf(quat_mul(q, dq), r + dr)


def tf_diff(T0, T1):
  """Return difference between two 4x4 homogeneous transforms"""
  r0 = tf_trans(T0)
  r1 = tf_trans(T1)
  C0 = tf_rot(T0)
  C1 = tf_rot(T1)

  # dr = r0 - r1
  dr = np.zeros((3,))
  dr[0] = r0[0] - r1[0]
  dr[1] = r0[1] - r1[1]
  dr[2] = r0[2] - r1[2]

  # dC = C0.T * C1
  # dtheta = acos((tr(dC) - 1.0) / 2.0)
  dC = C0.T @ C1
  tr = np.trace(dC)
  if tr < 0:
    tr *= -1

  if np.fabs(tr - 3.0) < 1e-5:
    dtheta = 0.0
  else:
    dtheta = acos((tr - 1.0) / 2.0)

  return (dr, dtheta)


def pose_diff(pose0, pose1):
  """Return difference between two poses"""
  # dr = r0 - r1
  dr = np.zeros((3,))
  dr[0] = pose0[0] - pose1[0]
  dr[1] = pose0[1] - pose1[1]
  dr[2] = pose0[2] - pose1[2]

  # dC = C0.T * C1
  # dtheta = acos((tr(dC) - 1.0) / 2.0)
  q0 = pose0[3:]
  q1 = pose1[3:]
  C0 = quat2rot(q0)
  C1 = quat2rot(q1)
  dC = C0.T @ C1
  tr = np.trace(dC)
  if np.fabs(tr - 3.0) < 1e-5:
    dtheta = 0.0
  else:
    dtheta = acos((tr - 1.0) / 2.0)

  return (dr, dtheta)


###############################################################################
# MATPLOTLIB
###############################################################################

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.transforms
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_bbox(ax, center, size):
  """
  Plot a 3D bounding box on the given axis.

  Parameters
  ----------

    ax: The matplotlib 3D axis.
    center: The center coordinates of the box (x, y, z).
    size: A tuple or list containing the dimensions (size_x, size_y, size_z).

  """

  # Calculate box corners
  lx, ly, lz = size
  x, y, z = center
  x1, y1, z1 = x - lx / 2, y - ly / 2, z - lz / 2
  x2, y2, z2 = x + lx / 2, y + ly / 2, z + lz / 2

  # Create box vertices
  vertices = [
    (x1, y1, z1),
    (x2, y1, z1),
    (x2, y2, z1),
    (x1, y2, z1),
    (x1, y1, z2),
    (x2, y1, z2),
    (x2, y2, z2),
    (x1, y2, z2),
  ]

  # Create edges
  edges = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],  # bottom face
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],  # top face
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],  # side edges
  ]

  # Plot edges
  for edge in edges:
    ax.plot3D(*zip(*[vertices[edge[0]], vertices[edge[1]]]), color="b")


def plot_set_axes_equal(ax):
  """
  Make axes of 3D plot have equal scale so that spheres appear as spheres,
  cubes as cubes, etc..  This is one possible solution to Matplotlib's
  ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

  Input
    ax: a matplotlib axis, e.g., as output from plt.gca().
  """
  x_limits = ax.get_xlim3d()
  y_limits = ax.get_ylim3d()
  z_limits = ax.get_zlim3d()

  x_range = abs(x_limits[1] - x_limits[0])
  x_middle = np.mean(x_limits)
  y_range = abs(y_limits[1] - y_limits[0])
  y_middle = np.mean(y_limits)
  z_range = abs(z_limits[1] - z_limits[0])
  z_middle = np.mean(z_limits)

  # The plot bounding box is a sphere in the sense of the infinity
  # norm, hence I call half the max range the plot radius.
  plot_radius = 0.5 * max([x_range, y_range, z_range])

  ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
  ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
  ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
  """
  Create a plot of the covariance confidence ellipse of *x* and *y*.

  Parameters
  ----------
  x, y : array-like, shape (n, )
      Input data.

  ax : matplotlib.axes.Axes
      The axes object to draw the ellipse into.

  n_std : float
      The number of standard deviations to determine the ellipse's radiuses.

  **kwargs
      Forwarded to `~matplotlib.patches.Ellipse`

  Returns
  -------
  matplotlib.patches.Ellipse
  """
  if x.size != y.size:
    raise ValueError("x and y must be the same size")

  cov = np.cov(x, y)
  pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
  # Using a special case to obtain the eigenvalues of this
  # two-dimensionl dataset.
  ell_radius_x = np.sqrt(1 + pearson)
  ell_radius_y = np.sqrt(1 - pearson)
  ellipse = matplotlib.patches.Ellipse(
    (0, 0),
    width=ell_radius_x * 2,
    height=ell_radius_y * 2,
    facecolor=facecolor,
    **kwargs,
  )

  # Calculating the stdandard deviation of x from
  # the squareroot of the variance and multiplying
  # with the given number of standard deviations.
  scale_x = np.sqrt(cov[0, 0]) * n_std
  mean_x = np.mean(x)

  # calculating the stdandard deviation of y ...
  scale_y = np.sqrt(cov[1, 1]) * n_std
  mean_y = np.mean(y)

  transf = matplotlib.transforms.Affine2D().rotate_deg(45).scale(
    scale_x, scale_y).translate(mean_x, mean_y)

  ellipse.set_transform(transf + ax.transData)
  return ax.add_patch(ellipse)


def plot_tf(ax, T, **kwargs):
  """
  Plot 4x4 Homogeneous Transform

  Args:

    ax (matplotlib.axes.Axes): Plot axes object
    T (np.array): 4x4 homogeneous transform (i.e. Pose in the world frame)

  Keyword args:

    size (float): Size of the coordinate-axes
    linewidth (float): Thickness of the coordinate-axes
    name (str): Frame name
    name_offset (np.array or list): Position offset for displaying the frame's name
    fontsize (float): Frame font size
    fontweight (float): Frame font weight
    colors (tuple of floats): Axes colors in x, y and z

  """
  assert T.shape == (4, 4)

  size = kwargs.get("size", 0.1)
  linewidth = kwargs.get("linewidth", 2)
  name = kwargs.get("name", None)
  nameoffset = kwargs.get("nameoffset", [0, 0, -0.01])
  fontsize = kwargs.get("fontsize", 10)
  fontweight = kwargs.get("fontweight", "bold")
  fontcolor = kwargs.get("fontcolor", "k")
  colors = kwargs.get("colors", ["r-", "g-", "b-"])

  origin = tf_trans(T)
  lx = tf_point(T, np.array([size, 0.0, 0.0]))
  ly = tf_point(T, np.array([0.0, size, 0.0]))
  lz = tf_point(T, np.array([0.0, 0.0, size]))

  # Draw x-axis
  px = [origin[0], lx[0]]
  py = [origin[1], lx[1]]
  pz = [origin[2], lx[2]]
  xaxis = ax.plot(px, py, pz, colors[0], linewidth=linewidth)[0]

  # Draw y-axis
  px = [origin[0], ly[0]]
  py = [origin[1], ly[1]]
  pz = [origin[2], ly[2]]
  yaxis = ax.plot(px, py, pz, colors[1], linewidth=linewidth)[0]

  # Draw z-axis
  px = [origin[0], lz[0]]
  py = [origin[1], lz[1]]
  pz = [origin[2], lz[2]]
  zaxis = ax.plot(px, py, pz, colors[2], linewidth=linewidth)[0]

  # Draw label
  if name is not None:
    x = origin[0] + nameoffset[0]
    y = origin[1] + nameoffset[1]
    z = origin[2] + nameoffset[2]
    text = ax.text(x,
                   y,
                   z,
                   name,
                   fontsize=fontsize,
                   fontweight=fontweight,
                   color=fontcolor)
    return (xaxis, yaxis, zaxis, text)

  return (xaxis, yaxis, zaxis)


def plot_mav(ax, T, **kwargs):
  """
  Plot MAV

  Args:

    ax (matplotlib.axes.Axes): Plot axes object
    T (np.array): 4x4 homogeneous transform (i.e. Pose in the world frame)

  Keyword args:

    size (float): Size of the coordinate-axes
    linewidth (float): Thickness of the coordinate-axes
    name (str): Frame name
    name_offset (np.array or list): Position offset for displaying the frame's name
    fontsize (float): Frame font size
    fontweight (float): Frame font weight
    colors (tuple of floats): Axes colors in x, y and z

  """
  assert T.shape == (4, 4)
  arm_length = kwargs.get("arm_length", 1.0)
  linewidth = kwargs.get("linewidth", 3)
  # name = kwargs.get('name', None)
  # name_offset = kwargs.get('name_offset', [0, 0, -0.01])
  # fontsize = kwargs.get('fontsize', 10)
  # fontweight = kwargs.get('fontweight', 'bold')
  color = kwargs.get("color", "k-")
  kwargs["size"] = arm_length / 2.0

  # Plot body frame
  plot_tf(ax, T, **kwargs)

  # Plot mav
  origin = tf_trans(T)

  fl = tf_point(T, np.array([arm_length / 2.0, arm_length / 2.0, 0.0]))
  fr = tf_point(T, np.array([arm_length / 2.0, -arm_length / 2.0, 0.0]))
  bl = tf_point(T, np.array([-arm_length / 2.0, arm_length / 2.0, 0.0]))
  br = tf_point(T, np.array([-arm_length / 2.0, -arm_length / 2.0, 0.0]))

  px = [origin[0], fl[0]]
  py = [origin[1], fl[1]]
  pz = [origin[2], fl[2]]
  fl_axis = ax.plot(px, py, pz, color, linewidth=linewidth)[0]

  px = [origin[0], fr[0]]
  py = [origin[1], fr[1]]
  pz = [origin[2], fr[2]]
  fr_axis = ax.plot(px, py, pz, color, linewidth=linewidth)[0]

  px = [origin[0], bl[0]]
  py = [origin[1], bl[1]]
  pz = [origin[2], bl[2]]
  bl_axis = ax.plot(px, py, pz, color, linewidth=linewidth)[0]

  px = [origin[0], br[0]]
  py = [origin[1], br[1]]
  pz = [origin[2], br[2]]
  br_axis = ax.plot(px, py, pz, color, linewidth=linewidth)[0]

  return (fl_axis, fr_axis, bl_axis, br_axis)


def plot_xyz(title, data, key_time, key_x, key_y, key_z, ylabel, **kwargs):
  """
  Plot XYZ plot

  Args:

    title (str): Plot title
    data (Dict[str, pandas.DataFrame]): Plot data
    key_time (str): Dictionary key for timestamps
    key_x (str): Dictionary key x-axis
    key_y (str): Dictionary key y-axis
    key_z (str): Dictionary key z-axis
    ylabel (str): Y-axis label

  """
  axis = ["x", "y", "z"]
  colors = ["r", "g", "b"]
  keys = [key_x, key_y, key_z]
  line_styles = kwargs.get("line_styles", ["--", "-", "x"])

  # Time
  time_data = {}
  for label, series_data in data.items():
    ts0 = series_data[key_time][0]
    time_data[label] = ts2sec(series_data[key_time].to_numpy() - ts0)

  # Plot subplots
  plt.figure()
  for i in range(3):
    plt.subplot(3, 1, i + 1)

    for (label, series_data), line in zip(data.items(), line_styles):
      line_style = colors[i] + line
      x_data = time_data[label]
      y_data = series_data[keys[i]].to_numpy()
      plt.plot(x_data, y_data, line_style, label=label)

    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.legend(loc=0)
    plt.title(f"{title} in {axis[i]}-axis")

  plt.subplots_adjust(hspace=0.65)


###############################################################################
# CV
###############################################################################

# UTILS #######################################################################


def z_score_normalization(image):
  """
  Z-score Normalization
  """
  mean, std = np.mean(image), np.std(image)
  return (image - mean) / (std + 1e-8)  # Avoid division by zero


def gamma_correction(image, gamma = 0.5):
  """
  Gamma correction
  """
  image = image / 255.0  # Normalize to [0,1]
  return np.power(image, gamma) * 255.0  # Apply gamma and rescale


def histogram_equalization(image):
  """
  Histogram Equalization
  """
  hist_range = (0.0, 256.0)
  hist, bins = np.histogram(image.flatten(), bins=256, range=hist_range)
  cdf = hist.cumsum()  # Cumulative distribution function
  cdf_normalized = cdf * 255 / cdf[-1]  # Normalize to [0,255]
  results = np.interp(image.flatten(), bins[:-1], cdf_normalized)
  return results.reshape(image.shape)


def find_modes_mean_shift(hist, sigma):
  """
  Efficient mean-shift approximation by histogram smoothing.

  Args:

    hist: 1D histogram.
    sigma: Standard deviation of Gaussian kernel.

  Returns:
    tuple: A tuple containing two numpy arrays:
      - modes: A 2D array where each row represents a mode,
               with columns [index, smoothed_histogram_value].
      - hist_smoothed: The smoothed histogram.
  """
  hist_len = len(hist)
  hist_smoothed = np.zeros(hist_len)

  # Compute smoothed histogram
  for i in range(hist_len):
    j = np.arange(-int(round(2 * sigma)), int(round(2 * sigma)) + 1)
    idx = (i + j) % hist_len  # Handle wraparound
    hist_smoothed[i] = np.sum(hist[idx] * scipy.stats.norm.pdf(j, 0, sigma))

  # Initialize empty array
  modes = np.array([], dtype=int).reshape(0, 2)

  # Check if all entries are nearly identical (to avoid infinite loop)
  if np.all(np.abs(hist_smoothed - hist_smoothed[0]) < 1e-5):
    return modes, hist_smoothed  # Return empty modes

  # Mode finding
  for i in range(hist_len):
    j = i
    while True:
      h0 = hist_smoothed[j]
      j1 = (j + 1) % hist_len
      j2 = (j - 1) % hist_len
      h1 = hist_smoothed[j1]
      h2 = hist_smoothed[j2]

      if h1 >= h0 and h1 >= h2:
        j = j1
      elif h2 > h0 and h2 > h1:
        j = j2
      else:
        break

    # Check if mode already found (more efficient than list search)
    if modes.size == 0 or not np.any(modes[:, 0] == j):
      modes = np.vstack((modes, [j, hist_smoothed[j]]))

  # Sort modes by smoothed histogram value (descending)
  idx = np.argsort(modes[:, 1])[::-1]  # Get indices for descending sort
  modes = modes[idx]

  return modes, hist_smoothed


def illumination_invariant_transform(image, alpha=0.9):
  """Illumination Invariant Transform

  Source:

  Maddern, Will, et al. "Illumination invariant imaging: Applications in robust
  vision-based localisation, mapping and classification for autonomous
  vehicles." Proceedings of the Visual Place Recognition in Changing
  Environments Workshop, IEEE International Conference on Robotics and
  Automation (ICRA), Hong Kong, China. Vol. 2. No. 3. 2014.

  """
  # Assert image has 3 channels and pixel values are 0-255
  assert image.shape[2] == 3
  assert image.dtype == "uint8"

  # An RGB image values are 0-255 (uint8)
  # convert from 0-255 (uint8) to 0-1 (float)
  red = image[:, :, 0] / 255.0
  green = image[:, :, 1] / 255.0
  blue = image[:, :, 2] / 255.0
  assert red.dtype == "float64"
  assert green.dtype == "float64"
  assert blue.dtype == "float64"

  # Log the red, green and blue channels
  log_red = np.log(red, where=(red != 0))
  log_green = np.log(green, where=(green != 0))
  log_blue = np.log(blue, where=(blue != 0))

  # Form illumination invariance transform image
  ii_image = 0.5 + log_green - alpha * log_blue - (1.0 - alpha) * log_red

  return ii_image


# GEOMETRY ####################################################################


def lookat(
    cam_pos,
    target_pos,
    up_axis = np.array([0.0, -1.0, 0.0]),
):
  """Form look at matrix"""
  assert len(cam_pos) == 3
  assert len(target_pos) == 3
  assert len(up_axis) == 3

  # Note: If we were using OpenGL the cam_dir would be the opposite direction,
  # since in OpenGL the camera forward is -z. In robotics, however, our camera is
  # +z forward.
  cam_z = normalize(target_pos - cam_pos)
  cam_x = normalize(cross(up_axis, cam_z))
  cam_y = cross(cam_z, cam_x)

  T_WC = zeros((4, 4))
  T_WC[0:3, 0] = cam_x.T
  T_WC[0:3, 1] = cam_y.T
  T_WC[0:3, 2] = cam_z.T
  T_WC[0:3, 3] = cam_pos
  T_WC[3, 3] = 1.0

  return T_WC


def linear_triangulation(P_i, P_j, z_i, z_j):
  """
  Linear triangulation

  This function is used to triangulate a single 3D point observed by two
  camera frames (be it in time with the same camera, or two different cameras
  with known extrinsics).

  Args:

    P_i (np.array): First camera 3x4 projection matrix
    P_j (np.array): Second camera 3x4 projection matrix
    z_i (np.array): First keypoint measurement
    z_j (np.array): Second keypoint measurement

  Returns:

    p_Ci (np.array): 3D point w.r.t first camera

  """

  # First three rows of P_i and P_j
  P1T_i = P_i[0, :]
  P2T_i = P_i[1, :]
  P3T_i = P_i[2, :]
  P1T_j = P_j[0, :]
  P2T_j = P_j[1, :]
  P3T_j = P_j[2, :]

  # Image point from the first and second frame
  x_i, y_i = z_i
  x_j, y_j = z_j

  # Form the A matrix of AX = 0
  A = zeros((4, 4))
  A[0, :] = x_i * P3T_i - P1T_i
  A[1, :] = y_i * P3T_i - P2T_i
  A[2, :] = x_j * P3T_j - P1T_j
  A[3, :] = y_j * P3T_j - P2T_j

  # Use SVD to solve AX = 0
  (_, _, Vh) = svd(A.T @ A)
  hp = Vh.T[:, -1]  # Get the best result from SVD (last column of V)
  hp = hp / hp[-1]  # Normalize the homogeneous 3D point
  p = hp[0:3]  # Return only the first three components (x, y, z)
  return p


def parallax(a, b):
  """
  Calculate the parallax between two vectors `a` and `b`.

  This is useful for example checking the parallax of two vectors before
  triangulating a feature. If the parallax is too small then the depth of the
  feature will be uncertain.

  Args:

    a (np.array): First vector
    b (np.array): Second vector

  Returns:

    Parallax between two vectors in degrees.

  """
  angle = np.arccos((a @ b) / (norm(a) * norm(b)))
  return float(np.rad2deg(angle))


def homography_find(pts_i, pts_j):
  """
  A Homography is a transformation (a 3x3 matrix) that maps the normalized
  image points from one image to the corresponding normalized image points in
  the other image. Specifically, let x and y be the n-th homogeneous points of
  pts_i and pts_j:

    x = [u_i, v_i, 1.0]
    y = [u_j, v_j, 1.0]

  The Homography is a 3x3 matrix that transforms x to y:

    y = H @ x

  **IMPORTANT**: The normalized image points `pts_i` and `pts_j` must
  correspond to points in 3D that on a plane.
  """
  assert pts_i.shape == pts_j.shape
  assert pts_i.shape[1] == 2

  num_points = pts_i.shape[0]
  A = np.zeros((num_points * 2, 9))
  for i in range(num_points):
    x, y = pts_i[i, :]
    x_, y_ = pts_j[i, :]
    A[2 * i, :] = [-x, -y, -1, 0, 0, 0, x * x_, y * x_, x_]
    A[2 * i + 1, :] = [0, 0, 0, -x, -y, -1, x * y_, y * y_, y_]

  _, _, Vt = np.linalg.svd(A)
  h = Vt[-1, :] / Vt[-1, -1]
  return h.reshape((3, 3))


def homography_pose(
  object_points,
  image_points,
  fx,
  fy,
  cx,
  cy,
):
  """
  Compute relative pose between camera and planar object T_CF.

  Source:

    Section 4.1.3: From homography to pose computation

    Marchand, Eric, Hideaki Uchiyama, and Fabien Spindler. "Pose estimation for
    augmented reality: a hands-on survey." IEEE transactions on visualization and
    computer graphics 22.12 (2015): 2633-2651.

    https://github.com/lagadic/camera_localization

  Args:

    object_points: (ndarray)
      3D object points

    image_points: (ndarray)
      2D image points in pixels

    fx: (float)
      Focal length in pixels

    fy: (float)
      Focal length in pixels

    cx: (float)
      Principal center in pixels

    cy: (float)
      Principal center in pixels

  Returns:

    4x4 Homogeneous transform T_camera_object

  """
  assert len(object_points) == len(image_points)
  # Form A to compute ||Ah|| = 0 using SVD, where A is an (N * 2) x 9 matrix
  # and h is the vectorized Homography matrix h, N is the number of points. if
  # N == 4, the matrix has more columns than rows. The solution is to add an
  # extra line with zeros.
  N = len(object_points)
  A = np.zeros((2 * N, 9))
  if N == 4:
    A = np.zeros((2 * N + 1, 9))

  for i in range(N):
    kp = image_points[i, :]
    x0 = object_points[i, :2]
    x1 = np.array([(kp[0] - cx) / fx, (kp[1] - cy) / fy])  # normalize keypoint

    A[2 * i, 3] = -x0[0]
    A[2 * i, 4] = -x0[1]
    A[2 * i, 5] = -1.0
    A[2 * i, 6] = x1[1] * x0[0]
    A[2 * i, 7] = x1[1] * x0[1]
    A[2 * i, 8] = x1[1]

    A[2 * i + 1, 0] = x0[0]
    A[2 * i + 1, 1] = x0[1]
    A[2 * i + 1, 2] = 1.0
    A[2 * i + 1, 6] = -x1[0] * x0[0]
    A[2 * i + 1, 7] = -x1[0] * x0[1]
    A[2 * i + 1, 8] = -x1[0]

  # Solve Ah = 0
  _, _, Vt = np.linalg.svd(A)

  # Extract Homography
  h = Vt[-1, :]  # Last col of V (or row of Vt) is solution to Ah = 0
  if h[-1] < 0:
    h *= -1

  H = np.zeros((3, 3))
  for i in range(3):
    for j in range(3):
      H[i, j] = h[3 * i + j]

  # Normalize H to ensure that ||c1|| = 1
  H /= sqrt(H[0, 0] * H[0, 0] + H[1, 0] * H[1, 0] + H[2, 0] * H[2, 0])

  # Form translation vector
  r = H[:, 2]

  # Form Rotation matrix
  c1 = H[:, 0]
  c2 = H[:, 1]
  c3 = np.cross(c1, c2)
  C = np.zeros((3, 3))
  for i in range(3):
    C[i, 0] = c1[i]
    C[i, 1] = c2[i]
    C[i, 2] = c3[i]

  return tf(C, r)


def dlt_pose(
  obj_pts,
  img_pts,
  fx,
  fy,
  cx,
  cy,
):
  """DLT Pose

  **IMPORTANT NOTE**: This function will not work if the object points are on a
  plane.

  Source:

    Section 3.1.2 "PnP: pose estimation from N point correspondences"

    Marchand, Eric, Hideaki Uchiyama, and Fabien Spindler. "Pose estimation for
    augmented reality: a hands-on survey." IEEE transactions on visualization and
    computer graphics 22.12 (2015): 2633-2651.

    https://github.com/lagadic/camera_localization

  Args:

    obj_pts: (ndarray)
      3D object points

    img_pts: (ndarray)
      2D image points in pixels

    fx: (float)
      Focal length in pixels

    fy: (float)
      Focal length in pixels

    cx: (float)
      Principal center in pixels

    cy: (float)
      Principal center in pixels

  Returns:

    4x4 Homogeneous transform T_camera_object

  """
  assert len(obj_pts) == len(img_pts)
  N = len(obj_pts)
  A = np.zeros((2 * N, 12))

  for i in range(N):
    pt = obj_pts[i, :]
    kp = img_pts[i, :]
    x = np.array([(kp[0] - cx) / fx, (kp[1] - cy) / fy])  # normalize keypoint

    A[2 * i, 0] = pt[0]
    A[2 * i, 1] = pt[1]
    A[2 * i, 2] = pt[2]
    A[2 * i, 3] = 1.0
    A[2 * i, 8] = -x[0] * pt[0]
    A[2 * i, 9] = -x[0] * pt[1]
    A[2 * i, 10] = -x[0] * pt[2]
    A[2 * i, 11] = -x[0]

    A[2 * i + 1, 4] = pt[0]
    A[2 * i + 1, 5] = pt[1]
    A[2 * i + 1, 6] = pt[2]
    A[2 * i + 1, 7] = 1.0
    A[2 * i + 1, 8] = -x[1] * pt[0]
    A[2 * i + 1, 9] = -x[1] * pt[1]
    A[2 * i + 1, 10] = -x[1] * pt[2]
    A[2 * i + 1, 11] = -x[1]

  # Solve Ah = 0
  _, _, Vt = np.linalg.svd(A)

  # Extract Homography
  h = Vt[-1, :]
  if h[-1] < 0:
    h *= -1

  # Normalize H to ensure that ||r3|| = 1
  h /= sqrt(h[8] * h[8] + h[9] * h[9] + h[10] * h[10])

  C = np.array([[h[0], h[1], h[2]], [h[4], h[5], h[6]], [h[8], h[9], h[10]]])
  r = np.array([h[3], h[7], h[11]])
  return tf(C, r)


def _solvepnp_cost(object_points, image_points, fx, fy, cx, cy, pose):
  """Calculate cost"""
  N = len(object_points)
  T_FC_est = pose2tf(pose)
  T_CF_est = inv(T_FC_est)
  pinhole_params = np.array([fx, fy, cx, cy])
  r = np.zeros(2 * N)
  r_idx = 0

  for n in range(N):
    # Calculate residual
    z = image_points[n, :]
    p_F = object_points[n, :]
    p_C = tf_point(T_CF_est, p_F)
    zhat = pinhole_project(pinhole_params, p_C)
    res = z - zhat

    # Form R.H.S. Gauss Newton g
    rs = r_idx
    re = r_idx + 2
    r[rs:re] = res
    r_idx = re

  return 0.5 * r.T @ r


def _solvepnp_linearize(object_points, image_points, fx, fy, cx, cy, pose):
  """Linearize Nonlinear-Least Squares Problem"""
  # Form Gauss-Newton system
  N = len(object_points)
  T_FC_est = pose2tf(pose)
  T_CF_est = inv(T_FC_est)
  pinhole_params = np.array([fx, fy, cx, cy])
  H = np.zeros((6, 6))
  g = np.zeros(6)

  for n in range(N):
    # Calculate residual
    z = image_points[n, :]
    p_F = object_points[n, :]
    p_C = tf_point(T_CF_est, p_F)
    zhat = pinhole_project(pinhole_params, p_C)
    r = z - zhat

    # Calculate Jacobian
    C_CF, _ = tf_decompose(T_CF_est)
    C_FC, r_FC = tf_decompose(T_FC_est)
    # -- Jacobian w.r.t 3D point p_C
    Jp = zeros((2, 3))
    Jp[0, :] = [1 / p_C[2], 0, -p_C[0] / p_C[2]**2]
    Jp[1, :] = [0, 1 / p_C[2], -p_C[1] / p_C[2]**2]
    # -- Jacobian w.r.t 2D point x
    Jk = zeros((2, 2))
    Jk[0, 0] = fx
    Jk[1, 1] = fy
    # -- Pinhole projection Jacobian
    Jh = -1 * Jk @ Jp
    # -- Jacobian of reprojection w.r.t. pose T_FC
    J = np.zeros((2, 6))
    J[0:2, 0:3] = Jh @ -C_CF
    J[0:2, 3:6] = Jh @ -C_CF @ hat(p_F - r_FC) @ -C_FC

    # Form Hessian
    H += J.T @ J

    # Form R.H.S. Gauss Newton g
    g += -J.T @ r

  return (H, g)


def _solvepnp_solve(lambda_k, H, g):
  """Solve for dx"""
  H_damped = H + lambda_k * eye(H.shape[0])
  c, low = scipy.linalg.cho_factor(H_damped)
  dx = scipy.linalg.cho_solve((c, low), g)
  # dx = solve_svd(H_damped, g)
  return dx


def _solvepnp_update(pose, dx):
  """Update pose estimate"""
  T_FC_est = pose2tf(pose)
  T_FC_est = tf_update(T_FC_est, dx)
  pose = tf2pose(T_FC_est)
  return pose


def solvepnp(obj_pts, img_pts, fx, fy, cx, cy, **kwargs):
  """
  Solve Perspective-N-Points

  **IMPORTANT**: This function assumes that object points lie on the plane
  because the initialization step uses DLT to estimate the homography between
  camera and planar object, then the relative pose between them is recovered.

  Args:

    obj_pts: (ndarray)
      3D object points

    img_pts: (ndarray)
      2D image points in pixels

    fx: (float)
      Focal length in pixels

    fy: (float)
      Focal length in pixels

    cx: (float)
      Principal center in pixels

    cy: (float)
      Principal center in pixels

  Returns:

    4x4 Homogeneous transform T_camera_object

  """
  assert len(obj_pts) == len(img_pts)
  assert len(obj_pts) > 6
  verbose = kwargs.get("verbose", False)
  T_CF_init = kwargs.get("T_CF_init", None)
  max_iter = kwargs.get("max_iter", 10)
  lambda_init = kwargs.get("lambda_init", 1e4)
  param_threshold = kwargs.get("param_threshold", 1e-10)
  cost_threshold = kwargs.get("cost_threshold", 1e-15)

  # Initialize pose with DLT
  T_CF = homography_pose(obj_pts, img_pts, fx, fy, cx, cy)

  # Solve Bundle Adjustment
  pose = None
  if T_CF_init is not None:
    T_FC = inv(T_CF_init)
    pose = tf2pose(T_FC)
  else:
    T_FC = inv(T_CF)
    pose = tf2pose(T_FC)

  lambda_k = lambda_init
  pose_k = copy.deepcopy(pose)
  cost_k = _solvepnp_cost(obj_pts, img_pts, fx, fy, cx, cy, pose_k)
  for i in range(max_iter):
    # Solve
    H, g = _solvepnp_linearize(obj_pts, img_pts, fx, fy, cx, cy, pose_k)
    dx = _solvepnp_solve(lambda_k, H, g)
    pose_kp1 = _solvepnp_update(pose_k, dx)
    cost_kp1 = _solvepnp_cost(obj_pts, img_pts, fx, fy, cx, cy, pose_kp1)

    # Accept or reject update
    dcost = cost_kp1 - cost_k
    if cost_kp1 < cost_k:
      # Accept update
      cost_k = cost_kp1
      pose_k = pose_kp1
      lambda_k /= 10.0

    else:
      # Reject update
      lambda_k *= 10.0

    # Display
    if verbose:
      print(f"iter: {i}, ", end="")
      print(f"lambda_k: {lambda_k:.2e}, ", end="")
      print(f"norm(dx): {norm(dx):.2e}, ", end="")
      print(f"dcost: {dcost:.2e}, ", end="")
      print(f"cost: {cost_kp1:.2e}", end="")
      print("")

    # Terminate?
    if cost_k < cost_threshold:
      break
    elif param_threshold > norm(dx):
      break

  # Return solution
  T_FC = pose2tf(pose)
  T_CF = inv(T_FC)
  return T_CF


# PINHOLE #####################################################################


def focal_length(image_width, fov_deg):
  """
  Estimated focal length based on `image_width` and field of fiew `fov_deg`
  in degrees.
  """
  return (image_width / 2.0) / tan(deg2rad(fov_deg / 2.0))


def pinhole_K(params):
  """Form camera matrix K"""
  fx, fy, cx, cy = params
  return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def pinhole_P(params, T_WC):
  """Form 3x4 projection matrix P"""
  K = pinhole_K(params)
  T_CW = inv(T_WC)
  C = tf_rot(T_CW)
  r = tf_trans(T_CW)

  P = zeros((3, 4))
  P[0:3, 0:3] = C
  P[0:3, 3] = r
  P = K @ P
  return P


def pinhole_project(proj_params, p_C):
  """Project 3D point onto image plane using pinhole camera model"""
  assert len(proj_params) == 4
  assert len(p_C) == 3

  # Project
  x = np.array([p_C[0] / p_C[2], p_C[1] / p_C[2]])

  # Scale and center
  fx, fy, cx, cy = proj_params
  z = np.array([fx * x[0] + cx, fy * x[1] + cy])

  return z


def pinhole_back_project(proj_params, z):
  """Back project image point to bearing"""
  fx, fy, cx, cy = proj_params
  x = (z[0] - cx) / fx
  y = (z[1] - cy) / fy
  return np.array([x, y])


def pinhole_params_jacobian(x):
  """Form pinhole parameter jacobian"""
  return np.array([[x[0], 0.0, 1.0, 0.0], [0.0, x[1], 0.0, 1.0]])


def pinhole_point_jacobian(proj_params):
  """Form pinhole point jacobian"""
  fx, fy, _, _ = proj_params
  return np.array([[fx, 0.0], [0.0, fy]])


# RADTAN4 #####################################################################


def radtan4_distort(dist_params, p):
  """Distort point with Radial-Tangential distortion"""
  assert len(dist_params) == 4
  assert len(p) == 2

  # Distortion parameters
  k1, k2, p1, p2 = dist_params

  # Point
  x, y = p

  # Apply radial distortion
  x2 = x * x
  y2 = y * y
  r2 = x2 + y2
  r4 = r2 * r2
  radial_factor = 1.0 + (k1 * r2) + (k2 * r4)
  x_dash = x * radial_factor
  y_dash = y * radial_factor

  # Apply tangential distortion
  xy = x * y
  x_ddash = x_dash + (2.0 * p1 * xy + p2 * (r2 + 2.0 * x2))
  y_ddash = y_dash + (p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy)
  return np.array([x_ddash, y_ddash])


def radtan4_point_jacobian(dist_params, p):
  """Radial-tangential point jacobian"""
  assert len(dist_params) == 4
  assert len(p) == 2

  # Distortion parameters
  k1, k2, p1, p2 = dist_params

  # Point
  x, y = p

  # Apply radial distortion
  x2 = x * x
  y2 = y * y
  r2 = x2 + y2
  r4 = r2 * r2

  # Point Jacobian
  # Let u = [x; y] normalized point
  # Let u' be the distorted u
  # The jacobian of u' w.r.t. u (or du'/du) is:
  J_point = zeros((2, 2))
  J_point[0, 0] = k1 * r2 + k2 * r4 + 2.0 * p1 * y + 6.0 * p2 * x
  J_point[0, 0] += x * (2.0 * k1 * x + 4.0 * k2 * x * r2) + 1.0
  J_point[1, 0] = 2.0 * p1 * x + 2.0 * p2 * y
  J_point[1, 0] += y * (2.0 * k1 * x + 4.0 * k2 * x * r2)
  J_point[0, 1] = J_point[1, 0]
  J_point[1, 1] = k1 * r2 + k2 * r4 + 6.0 * p1 * y + 2.0 * p2 * x
  J_point[1, 1] += y * (2.0 * k1 * y + 4.0 * k2 * y * r2) + 1.0
  # Above is generated using sympy

  return J_point


def radtan4_undistort(dist_params, p0):
  """Un-distort point with Radial-Tangential distortion"""
  assert len(dist_params) == 4
  assert len(p0) == 2

  # Undistort
  p = p0
  max_iter = 5

  for _ in range(max_iter):
    # Error
    p_distorted = radtan4_distort(dist_params, p)
    J = radtan4_point_jacobian(dist_params, p)
    err = p0 - p_distorted

    # Update
    # dp = inv(J' * J) * J' * err
    dp = pinv(J) @ err
    p = p + dp

    # Check threshold
    if (err.T @ err) < 1e-15:
      break

  return p


def radtan4_params_jacobian(dist_params, p):
  """Radial-Tangential distortion parameter jacobian"""
  assert len(dist_params) == 4
  assert len(p) == 2

  # Point
  x, y = p

  # Setup
  x2 = x * x
  y2 = y * y
  xy = x * y
  r2 = x2 + y2
  r4 = r2 * r2

  # Params Jacobian
  J_params = zeros((2, 4))
  J_params[0, 0] = x * r2
  J_params[0, 1] = x * r4
  J_params[0, 2] = 2.0 * xy
  J_params[0, 3] = 3.0 * x2 + y2
  J_params[1, 0] = y * r2
  J_params[1, 1] = y * r4
  J_params[1, 2] = x2 + 3.0 * y2
  J_params[1, 3] = 2.0 * xy

  return J_params


# EQUI4 #######################################################################


def equi4_distort(dist_params, p):
  """Distort point with Equi-distant distortion"""
  assert len(dist_params) == 4
  assert len(p) == 2

  # Distortion parameters
  k1, k2, k3, k4 = dist_params

  # Distort
  x, y = p
  r = sqrt(x * x + y * y)
  th = math.atan(r)
  th2 = th * th
  th4 = th2 * th2
  th6 = th4 * th2
  th8 = th4 * th4
  thd = th * (1.0 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8)
  s = thd / r
  x_dash = s * x
  y_dash = s * y
  return np.array([x_dash, y_dash])


def equi4_undistort(dist_params, p):
  """Undistort point using Equi-distant distortion"""
  thd = sqrt(p[0] * p[0] + p[1] * p[1])

  # Distortion parameters
  k1, k2, k3, k4 = dist_params

  th = thd  # Initial guess
  for _ in range(20):
    th2 = th * th
    th4 = th2 * th2
    th6 = th4 * th2
    th8 = th4 * th4
    th = thd / (1.0 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8)

  scaling = tan(th) / thd
  return np.array([p[0] * scaling, p[1] * scaling])


def equi4_params_jacobian(dist_params, p):
  """Equi-distant distortion params jacobian"""
  assert len(dist_params) == 4
  assert len(p) == 2

  # Jacobian
  x, y = p
  r = sqrt(x**2 + y**2)
  th = atan(r)

  J_params = zeros((2, 4))
  J_params[0, 0] = x * th**3 / r
  J_params[0, 1] = x * th**5 / r
  J_params[0, 2] = x * th**7 / r
  J_params[0, 3] = x * th**9 / r

  J_params[1, 0] = y * th**3 / r
  J_params[1, 1] = y * th**5 / r
  J_params[1, 2] = y * th**7 / r
  J_params[1, 3] = y * th**9 / r

  return J_params


def equi4_point_jacobian(dist_params, p):
  """Equi-distant distortion point jacobian"""
  assert len(dist_params) == 4
  assert len(p) == 2

  # Distortion parameters
  k1, k2, k3, k4 = dist_params

  # Jacobian
  x, y = p
  r = sqrt(x**2 + y**2)

  th = math.atan(r)
  th2 = th**2
  th4 = th**4
  th6 = th**6
  th8 = th**8
  thd = th * (1.0 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8)

  th_r = 1.0 / (r * r + 1.0)
  thd_th = 1.0 + 3.0 * k1 * th2
  thd_th += 5.0 * k2 * th4
  thd_th += 7.0 * k3 * th6
  thd_th += 9.0 * k4 * th8
  s = thd / r
  s_r = thd_th * th_r / r - thd / (r * r)
  r_x = 1.0 / r * x
  r_y = 1.0 / r * y

  J_point = zeros((2, 2))
  J_point[0, 0] = s + x * s_r * r_x
  J_point[0, 1] = x * s_r * r_y
  J_point[1, 0] = y * s_r * r_x
  J_point[1, 1] = s + y * s_r * r_y

  return J_point


# PINHOLE RADTAN4 #############################################################


def pinhole_radtan4_project(
  proj_params,
  dist_params,
  p_C,
):
  """Pinhole + Radial-Tangential project"""
  assert len(proj_params) == 4
  assert len(dist_params) == 4
  assert len(p_C) == 3

  # Project
  x = np.array([p_C[0] / p_C[2], p_C[1] / p_C[2]])

  # Distort
  x_dist = radtan4_distort(dist_params, x)

  # Scale and center to image plane
  fx, fy, cx, cy = proj_params
  z = np.array([fx * x_dist[0] + cx, fy * x_dist[1] + cy])
  return z


def pinhole_radtan4_backproject(
  proj_params,
  dist_params,
  z,
):
  """Pinhole + Radial-Tangential back-project"""
  assert len(proj_params) == 4
  assert len(dist_params) == 4
  assert len(z) == 2

  # Convert image pixel coordinates to normalized retinal coordintes
  fx, fy, cx, cy = proj_params
  x = np.array([(z[0] - cx) / fx, (z[1] - cy) / fy])

  # Undistort
  x = radtan4_undistort(dist_params, x)

  # 3D ray
  p = np.array([x[0], x[1], 1.0])
  return p


def pinhole_radtan4_undistort(
  proj_params,
  dist_params,
  z,
):
  """Pinhole + Radial-Tangential undistort"""
  assert len(proj_params) == 4
  assert len(dist_params) == 4
  assert len(z) == 2

  # Back project and undistort
  fx, fy, cx, cy = proj_params
  p = np.array([(z[0] - cx) / fx, (z[1] - cy) / fy])
  p_undist = radtan4_undistort(dist_params, p)

  # Project undistorted point to image plane
  return np.array([p_undist[0] * fx + cx, p_undist[1] * fy + cy])


def pinhole_radtan4_project_jacobian(
  proj_params,
  dist_params,
  p_C,
):
  """Pinhole + Radial-Tangential project jacobian"""
  assert len(proj_params) == 4
  assert len(dist_params) == 4
  assert len(p_C) == 3

  # Project 3D point
  x = np.array([p_C[0] / p_C[2], p_C[1] / p_C[2]])

  # Jacobian
  J_proj = zeros((2, 3))
  J_proj[0, :] = [1 / p_C[2], 0, -p_C[0] / p_C[2]**2]
  J_proj[1, :] = [0, 1 / p_C[2], -p_C[1] / p_C[2]**2]
  J_dist_point = radtan4_point_jacobian(dist_params, x)
  J_proj_point = pinhole_point_jacobian(proj_params)

  return J_proj_point @ J_dist_point @ J_proj


def pinhole_radtan4_params_jacobian(
  proj_params,
  dist_params,
  p_C,
):
  """Pinhole + Radial-Tangential params jacobian"""
  assert len(proj_params) == 4
  assert len(dist_params) == 4
  assert len(p_C) == 3

  x = np.array([p_C[0] / p_C[2], p_C[1] / p_C[2]])  # Project 3D point
  x_dist = radtan4_distort(dist_params, x)  # Distort point

  J_proj_point = pinhole_point_jacobian(proj_params)
  J_dist_params = radtan4_params_jacobian(dist_params, x)

  J = zeros((2, 8))
  J[0:2, 0:4] = pinhole_params_jacobian(x_dist)
  J[0:2, 4:8] = J_proj_point @ J_dist_params
  return J


# PINHOLE EQUI4 ###############################################################


def pinhole_equi4_project(
  proj_params,
  dist_params,
  p_C,
):
  """Pinhole + Equi-distant project"""
  assert len(proj_params) == 4
  assert len(dist_params) == 4
  assert len(p_C) == 3

  # Project
  x = np.array([p_C[0] / p_C[2], p_C[1] / p_C[2]])

  # Distort
  x_dist = equi4_distort(dist_params, x)

  # Scale and center to image plane
  fx, fy, cx, cy = proj_params
  z = np.array([fx * x_dist[0] + cx, fy * x_dist[1] + cy])
  return z


def pinhole_equi4_backproject(
  proj_params,
  dist_params,
  z,
):
  """Pinhole + Equi-distant back-project"""
  assert len(proj_params) == 4
  assert len(dist_params) == 4
  assert len(z) == 2

  # Convert image pixel coordinates to normalized retinal coordintes
  fx, fy, cx, cy = proj_params
  x = np.array([(z[0] - cx) / fx, (z[1] - cy) / fy])

  # Undistort
  x = equi4_undistort(dist_params, x)

  # 3D ray
  p = np.array([x[0], x[1], 1.0])
  return p


def pinhole_equi4_undistort(
  proj_params,
  dist_params,
  z,
):
  """Pinhole + Equi-distant undistort"""
  assert len(proj_params) == 4
  assert len(dist_params) == 4
  assert len(z) == 2

  # Back project and undistort
  fx, fy, cx, cy = proj_params
  p = np.array([(z[0] - cx) / fx, (z[1] - cy) / fy])
  p_undist = equi4_undistort(dist_params, p)

  # Project undistorted point to image plane
  return np.array([p_undist[0] * fx + cx, p_undist[1] * fy + cy])


def pinhole_equi4_project_jacobian(
  proj_params,
  dist_params,
  p_C,
):
  """Pinhole + Equi-distant project jacobian"""
  assert len(proj_params) == 4
  assert len(dist_params) == 4
  assert len(p_C) == 3

  # Project 3D point
  x = np.array([p_C[0] / p_C[2], p_C[1] / p_C[2]])

  # Jacobian
  J_proj = zeros((2, 3))
  J_proj[0, :] = [1 / p_C[2], 0, -p_C[0] / p_C[2]**2]
  J_proj[1, :] = [0, 1 / p_C[2], -p_C[1] / p_C[2]**2]
  J_dist_point = equi4_point_jacobian(dist_params, x)
  J_proj_point = pinhole_point_jacobian(proj_params)
  return J_proj_point @ J_dist_point @ J_proj


def pinhole_equi4_params_jacobian(
  proj_params,
  dist_params,
  p_C,
):
  """Pinhole + Equi-distant params jacobian"""
  assert len(proj_params) == 4
  assert len(dist_params) == 4
  assert len(p_C) == 3

  x = np.array([p_C[0] / p_C[2], p_C[1] / p_C[2]])  # Project 3D point
  x_dist = equi4_distort(dist_params, x)  # Distort point

  J_proj_point = pinhole_point_jacobian(proj_params)
  J_dist_params = equi4_params_jacobian(dist_params, x)

  J = zeros((2, 8))
  J[0:2, 0:4] = pinhole_params_jacobian(x_dist)
  J[0:2, 4:8] = J_proj_point @ J_dist_params
  return J


# CAMERA GEOMETRY #############################################################


@dataclass
class CameraGeometry:
  """Camera Geometry"""

  cam_idx: int
  resolution: tuple
  proj_model: str
  dist_model: str
  proj_params_size: int
  dist_params_size: int

  project_fn: Callable[[Vec4, Vec4, Vec3], Vec2]
  backproject_fn: Callable[[Vec4, Vec4, Vec2], Vec3]
  undistort_fn: Callable[[Vec4, Vec4, Vec2], Vec2]
  J_proj_fn: Callable[[Vec4, Vec4, Vec3], Mat2x3]
  J_params_fn: Callable[[Vec4, Vec4, Vec3], Mat2xN]

  def get_proj_params_size(self):
    """Return projection parameter size"""
    return self.proj_params_size

  def get_dist_params_size(self):
    """Return distortion parameter size"""
    return self.dist_params_size

  def get_params_size(self):
    """Return parameter size"""
    return self.get_proj_params_size() + self.get_dist_params_size()

  def proj_params(self, params):
    """Extract projection parameters"""
    return params[:self.proj_params_size]

  def dist_params(self, params):
    """Extract distortion parameters"""
    return params[-self.dist_params_size:]

  def project(self, params, p_C):
    """Project point `p_C` with camera parameters `params`"""
    # Project
    proj_params = params[:self.proj_params_size]
    dist_params = params[-self.dist_params_size:]
    z = self.project_fn(proj_params, dist_params, p_C)

    # Make sure point is infront of camera
    if p_C[2] < 0.0:
      return False, z

    # Make sure image point is within image bounds
    x_ok = z[0] >= 0.0 and z[0] <= self.resolution[0]
    y_ok = z[1] >= 0.0 and z[1] <= self.resolution[1]
    if x_ok and y_ok:
      return True, z

    return False, z

  def backproject(self, params, z):
    """Back-project image point `z` with camera parameters `params`"""
    proj_params = params[:self.proj_params_size]
    dist_params = params[-self.dist_params_size:]
    return self.backproject_fn(proj_params, dist_params, z)

  def undistort(self, params, z):
    """Undistort image point `z` with camera parameters `params`"""
    proj_params = params[:self.proj_params_size]
    dist_params = params[-self.dist_params_size:]
    return self.undistort_fn(proj_params, dist_params, z)

  def J_proj(self, params, p_C):
    """Form Jacobian w.r.t. p_C"""
    proj_params = params[:self.proj_params_size]
    dist_params = params[-self.dist_params_size:]
    return self.J_proj_fn(proj_params, dist_params, p_C)

  def J_params(self, params, p_C):
    """Form Jacobian w.r.t. camera parameters"""
    proj_params = params[:self.proj_params_size]
    dist_params = params[-self.dist_params_size:]
    return self.J_params_fn(proj_params, dist_params, p_C)


def pinhole_radtan4_setup(cam_idx, cam_res):
  """Setup Pinhole + Radtan4 camera geometry"""
  return CameraGeometry(
    cam_idx,
    cam_res,
    "pinhole",
    "radtan4",
    4,
    4,
    pinhole_radtan4_project,
    pinhole_radtan4_backproject,
    pinhole_radtan4_undistort,
    pinhole_radtan4_project_jacobian,
    pinhole_radtan4_params_jacobian,
  )


def pinhole_equi4_setup(cam_idx, cam_res):
  """Setup Pinhole + Equi camera geometry"""
  return CameraGeometry(
    cam_idx,
    cam_res,
    "pinhole",
    "equi4",
    4,
    4,
    pinhole_equi4_project,
    pinhole_equi4_backproject,
    pinhole_equi4_undistort,
    pinhole_equi4_project_jacobian,
    pinhole_equi4_params_jacobian,
  )


def camera_geometry_setup(cam_idx, cam_res, proj_model, dist_model):
  """Setup camera geometry"""
  if proj_model == "pinhole" and dist_model == "radtan4":
    return pinhole_radtan4_setup(cam_idx, cam_res)
  elif proj_model == "pinhole" and dist_model == "equi4":
    return pinhole_equi4_setup(cam_idx, cam_res)

  raise RuntimeError(f"Unrecognized [{proj_model}]-[{dist_model}] combo!")


# ChessboardDetector


# class ChessboardDetector:
#   def __init__(self):
#     pass
#
#   def correlation_patch(self, angle_1: float, angle_2: float, radius: float):
#     """
#     Form correlation patch
#     """
#     # Width and height
#     width = int(radius * 2 + 1)
#     height = int(radius * 2 + 1)
#     if width == 0 or height == 0:
#       return None
#
#     # Initialize template
#     template = []
#     for i in range(4):
#       x = np.zeros((height, width))
#       template.append(x)
#
#     # Midpoint
#     mu = radius
#     mv = radius
#
#     # Compute normals from angles
#     n1 = [-np.sin(angle_1), np.cos(angle_1)]
#     n2 = [-np.sin(angle_2), np.cos(angle_2)]
#
#     # For all points in template do
#     for u in range(width):
#       for v in range(height):
#         # Vector
#         vec = [u - mu, v - mv]
#         dist = np.linalg.norm(vec)
#
#         # Check on which side of the normals we are
#         s1 = np.dot(vec, n1)
#         s2 = np.dot(vec, n2)
#
#         if dist <= radius:
#           if s1 <= -0.1 and s2 <= -0.1:
#             template[0][v, u] = 1
#           elif s1 >= 0.1 and s2 >= 0.1:
#             template[1][v, u] = 1
#           elif s1 <= -0.1 and s2 >= 0.1:
#             template[2][v, u] = 1
#           elif s1 >= 0.1 and s2 <= -0.1:
#             template[3][v, u] = 1
#
#     # Normalize
#     for i in range(4):
#       template[i] /= np.sum(template[i])
#
#     return template
#
#   def non_maxima_suppression(self,
#                              image: Image,
#                              n: int = 3,
#                              tau: float = 0.1,
#                              margin: int = 2):
#     """
#     Non Maximum Suppression
#
#     Args:
#
#       image: Input image
#       n: Kernel size
#       tau: Corner response threshold
#       margin: Offset away from image boundaries
#
#     Returns:
#
#       List of corners with maximum response
#
#     """
#     height, width = image.shape
#     maxima = []
#
#     for i in range(n + margin, width - n - margin, n + 1):
#       for j in range(n + margin, height - n - margin, n + 1):
#         # Initialize max value
#         maxi = i
#         maxj = j
#         maxval = image[j, i]
#
#         # Get max value in kernel
#         for i2 in range(i, i + n):
#           for j2 in range(j, j + n):
#             currval = image[j2, i2]
#             if currval > maxval:
#               maxi = i2
#               maxj = j2
#               maxval = currval
#
#         # Make sure maxval is larger than neighbours
#         failed = 0
#         for i2 in range(maxi - n, min(maxi + n, width - margin)):
#           for j2 in range(maxj - n, min(maxj + n, height - margin)):
#             currval = image[j2, i2]
#             if currval > maxval and (i2 < i or i2 > i + n or j2 < j or
#                                      j2 > j + n):
#               failed = 1
#               break
#           if failed:
#             break
#
#         # Store maxval
#         if maxval >= tau and failed == 0:
#           maxima.append([maxi, maxj])
#
#     return maxima
#
#   def edge_orientations(self, img_angle: Image,
#                         img_weight: Image) -> tuple[Vec2, Vec2]:
#     """
#     Calculate Edge Orientations
#
#     Args:
#
#       img_angle: Image angles
#       img_weight: Image weight
#
#     Returns:
#
#       Refined edge orientation vectors v1, v2
#
#     """
#     # Initialize v1 and v2
#     v1 = np.array([0, 0])
#     v2 = np.array([0, 0])
#
#     # Number of bins (histogram parameter)
#     bin_num = 32
#
#     # Convert images to vectors
#     vec_angle = img_angle.flatten()
#     vec_weight = img_weight.flatten()
#
#     # Convert angles from normals to directions
#     vec_angle = vec_angle + np.pi / 2
#     vec_angle[vec_angle > np.pi] -= np.pi
#
#     # Create histogram
#     angle_hist = np.zeros(bin_num)
#     for i in range(len(vec_angle)):
#       bin_idx = min(max(int(np.floor(vec_angle[i] / (np.pi / bin_num))), 0),
#                     bin_num - 1)
#       angle_hist[bin_idx] += vec_weight[i]
#
#     # Find modes of smoothed histogram
#     modes, _ = find_modes_mean_shift(angle_hist, 1)
#
#     # If only one or no mode => return invalid corner
#     if modes.shape[0] <= 1:
#       return v1, v2
#
#     # Compute orientation at modes
#     modes = np.hstack(
#       (modes, ((modes[:, 0] - 1) * np.pi / bin_num).reshape(-1, 1)))
#
#     # Extract 2 strongest modes and sort by angle
#     modes = modes[:2]
#     modes = modes[np.argsort(modes[:, 2])]
#
#     # Compute angle between modes
#     delta_angle = min(modes[1, 2] - modes[0, 2],
#                       modes[0, 2] + np.pi - modes[1, 2])
#
#     # If angle too small => return invalid corner
#     if delta_angle <= 0.3:
#       return v1, v2
#
#     # Set statistics: orientations
#     v1 = np.array([np.cos(modes[0, 2]), np.sin(modes[0, 2])])
#     v2 = np.array([np.cos(modes[1, 2]), np.sin(modes[1, 2])])
#
#     return v1, v2
#
#   def refine_corners(
#     self,
#     img_shape: tuple[int, ...],
#     img_angle: MatN,
#     img_weight: MatN,
#     corners,
#     r=10,
#   ):
#     """
#     Refine detected corners
#
#     Args:
#
#       img_shape: Image shape (rows, cols)
#       img_angle: Image angles [degrees]
#       img_weight: Image weight
#       corners: List of corners to refine
#       r: Patch radius size [pixels]
#
#     Returns
#
#       corners, v1, v2
#
#     """
#     # Image dimensions
#     assert len(img_shape) == 2
#     height, width = img_shape
#
#     # Init orientations to invalid (corner is invalid iff orientation=0)
#     corners_inliers = []
#     v1 = []
#     v2 = []
#
#     # for all corners do
#     for i, (cu, cv, _) in enumerate(corners):
#       # Estimate edge orientations
#       cu, cv = int(cu), int(cv)
#       rs = max(cv - r, 1)
#       re = min(cv + r, height)
#       cs = max(cu - r, 1)
#       ce = min(cu + r, width)
#       img_angle_sub = img_angle[rs:re, cs:ce]
#       img_weight_sub = img_weight[rs:re, cs:ce]
#       v1_edge, v2_edge = self.edge_orientations(img_angle_sub, img_weight_sub)
#
#       # Check invalid edge
#       if np.array_equal(v1_edge, [0.0, 0.0]):
#         continue
#       if np.array_equal(v2_edge, [0.0, 0.0]):
#         continue
#
#       corners_inliers.append(corners[i])
#       v1.append(v1_edge)
#       v2.append(v2_edge)
#
#     return corners, v1, v2
#
#   def subpixel_refine(self, image: Image):
#     """
#     Sub-pixel Refinement.
#     """
#     dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
#     dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
#
#     matsum = np.zeros((2, 2))
#     pointsum = np.zeros(2)
#     for i in range(dx.shape[0]):
#       for j in range(dx.shape[1]):
#         vec = np.array([dy[i, j], dx[i, j]])
#         pos = (i, j)
#         mat = np.outer(vec, vec)
#         pointsum += mat @ pos
#         matsum += mat
#
#     try:
#       minv = np.linalg.inv(matsum)
#     except np.linalg.LinAlgError:
#       return None
#
#     newp = minv.dot(pointsum)
#
#     return newp
#
#   def detect_corners(self, image: Image, radiuses: list[int] = [6, 8, 10]):
#     """
#     Detect corners
#     """
#     # Convert gray image to double
#     assert len(image.shape) == 2
#     image = image / 255
#
#     # Find corners
#     template_props = [[0.0, pi / 2.0], [pi / 4.0, -pi / 4.0]]
#     corr = np.zeros(image.shape)
#     for angle_1, angle_2 in template_props:
#       for radius in radiuses:
#         template = self.correlation_patch(angle_1, angle_2, radius)
#         if template is None:
#           continue
#
#         img_corners = [
#           scipy.signal.convolve2d(image, template[0], mode="same"),
#           scipy.signal.convolve2d(image, template[1], mode="same"),
#           scipy.signal.convolve2d(image, template[2], mode="same"),
#           scipy.signal.convolve2d(image, template[3], mode="same"),
#         ]
#         img_corners_mu = np.mean(img_corners, axis=0)
#         arr = np.array([
#           img_corners[0] - img_corners_mu,
#           img_corners[1] - img_corners_mu,
#           img_corners_mu - img_corners[2],
#           img_corners_mu - img_corners[3],
#         ])
#         img_corners_1 = np.min(arr, axis=0)  # Case 1: a = white, b = black
#         img_corners_2 = np.min(-arr, axis=0)  # Case 2: b = white, a = black
#
#         # Combine both
#         img_corners = np.max([img_corners_1, img_corners_2], axis=0)
#
#         # Max
#         corr = np.max([img_corners, corr], axis=0)
#
#     # Max pooling
#     # step = 40
#     # threshold = float(np.max(corr) * 0.2)
#     # corners = self.max_pooling(corr, step, threshold)
#
#     # print(np.max(corr))
#     # print(np.min(corr))
#
#     # import matplotlib.pylab as plt
#     # plt.imshow(corr, cmap="gray")
#     # plt.colorbar()
#     # plt.show()
#
#   def checkerboard_score(self, corners, size=(9, 6)):
#     corners_reshaped = corners[:, :2].reshape(*size, 2)
#     maxm = 0
#     for rownum in range(size[0]):
#       for colnum in range(1, size[1] - 1):
#         pts = corners_reshaped[rownum, [colnum - 1, colnum, colnum + 1]]
#         top = np.linalg.norm(pts[2] + pts[0] - 2 * pts[1])
#         bot = np.linalg.norm(pts[2] - pts[0])
#         if np.abs(bot) < 1e-9:
#           return 1
#         maxm = max(top / bot, maxm)
#     for colnum in range(0, size[1]):
#       for rownum in range(1, size[0] - 1):
#         pts = corners_reshaped[[rownum - 1, rownum, rownum + 1], colnum]
#         top = np.linalg.norm(pts[2] + pts[0] - 2 * pts[1])
#         bot = np.linalg.norm(pts[2] - pts[0])
#         if np.abs(bot) < 1e-9:
#           return 1
#         maxm = max(top / bot, maxm)
#     return maxm


# UNITESTS #####################################################################


class TestCV(unittest.TestCase):
  """Test computer vision functions"""
  def setUp(self):
    # Camera
    img_w = 640
    img_h = 480
    fx = focal_length(img_w, 90.0)
    fy = focal_length(img_w, 90.0)
    cx = img_w / 2.0
    cy = img_h / 2.0
    self.proj_params = np.array([fx, fy, cx, cy])

    # Camera pose in world frame
    C_WC = euler321(-pi / 2, 0.0, -pi / 2)
    r_WC = np.array([0.0, 0.0, 0.0])
    self.T_WC = tf(C_WC, r_WC)

    # 3D World point
    self.p_W = np.array([10.0, 0.0, 0.0])

    # Point w.r.t camera
    self.p_C = tf_point(inv(self.T_WC), self.p_W)
    self.x = np.array([self.p_C[0] / self.p_C[2], self.p_C[1] / self.p_C[2]])

  def test_linear_triangulation(self):
    """Test linear_triangulation()"""
    # Camera i - Camera j extrinsics
    C_CiCj = eye(3)
    r_CiCj = np.array([0.05, 0.0, 0.0])
    T_CiCj = tf(C_CiCj, r_CiCj)

    # Camera 0 pose in world frame
    C_WCi = euler321(-pi / 2, 0.0, -pi / 2)
    r_WCi = np.array([0.0, 0.0, 0.0])
    T_WCi = tf(C_WCi, r_WCi)

    # Camera 1 pose in world frame
    T_WCj = T_WCi @ T_CiCj

    # Projection matrices P_i and P_j
    P_i = pinhole_P(self.proj_params, eye(4))
    P_j = pinhole_P(self.proj_params, T_CiCj)

    # Test multiple times
    nb_tests = 100
    for _ in range(nb_tests):
      # Project feature point p_W to image plane
      x = np.random.uniform(-0.05, 0.05)
      y = np.random.uniform(-0.05, 0.05)
      p_W = np.array([10.0, x, y])
      p_Ci_gnd = tf_point(inv(T_WCi), p_W)
      p_Cj_gnd = tf_point(inv(T_WCj), p_W)
      z_i = pinhole_project(self.proj_params, p_Ci_gnd)
      z_j = pinhole_project(self.proj_params, p_Cj_gnd)

      # Triangulate
      p_Ci_est = linear_triangulation(P_i, P_j, z_i, z_j)
      self.assertTrue(np.allclose(p_Ci_est, p_Ci_gnd))

  def test_parallax(self):
    """Test parallax"""
    # Camera 0 pose in world frame
    C_WCi = euler321(-pi / 2, 0.0, -pi / 2)
    r_WCi = np.array([0.0, 0.1, 0.0])
    T_WCi = tf(C_WCi, r_WCi)

    # Camera 1 pose in world frame
    C_WCj = euler321(-pi / 2, 0.0, -pi / 2)
    r_WCj = np.array([0.0, -0.1, 0.0])
    T_WCj = tf(C_WCj, r_WCj)

    # Calculate parallax
    p_W = np.array([0.2, 0, 0])
    p_Ci_gnd = tf_point(inv(T_WCi), p_W)
    p_Cj_gnd = tf_point(inv(T_WCj), p_W)

    angle = parallax(p_Ci_gnd, p_Cj_gnd)
    self.assertTrue(angle > 0)

    # Visualize
    debug = False
    if debug:
      plt.figure()
      ax = plt.axes(projection="3d")
      plot_tf(ax, T_WCi, size=0.1)
      plot_tf(ax, T_WCj, size=0.1)
      ax.plot(*p_W, "r.")
      plot_set_axes_equal(ax)
      ax.set_xlabel("x [m]")
      ax.set_ylabel("y [m]")
      ax.set_zlabel("z [m]")  # pyright: ignore
      plt.show()

  def test_homography_find(self):
    """Test homography_find()"""
    # Camera
    img_w = 640
    img_h = 480
    fx = focal_length(img_w, 90.0)
    fy = focal_length(img_w, 90.0)
    cx = img_w / 2.0
    cy = img_h / 2.0
    proj_params = np.array([fx, fy, cx, cy])

    # Camera pose i
    C_WC_i = euler321(-pi / 2 - deg2rad(45), 0.0, -pi / 2)
    r_WC_i = np.array([0.0, 1.0, 0.0])
    T_WC_i = tf(C_WC_i, r_WC_i)

    # Camera pose j
    C_WC_j = euler321(-pi / 2 + deg2rad(45), 0.0, -pi / 2)
    r_WC_j = np.array([0.0, -1.0, 0.0])
    T_WC_j = tf(C_WC_j, r_WC_j)

    # Generate image points
    num_points = 10
    points = []
    pts_i = []
    pts_j = []
    for _ in range(num_points):
      # Project feature point p_W to image plane
      x = np.random.uniform(-0.5, 0.5)
      y = np.random.uniform(-0.5, 0.5)
      p_W = np.array([1.0, x, y])
      p_Ci_gnd = tf_point(inv(T_WC_i), p_W)
      p_Cj_gnd = tf_point(inv(T_WC_j), p_W)
      z_i = pinhole_project(proj_params, p_Ci_gnd)
      z_j = pinhole_project(proj_params, p_Cj_gnd)
      pt_i = pinhole_back_project(proj_params, z_i)
      pt_j = pinhole_back_project(proj_params, z_j)

      points.append(p_W)
      pts_i.append(pt_i)
      pts_j.append(pt_j)
    points = np.array(points)
    pts_i = np.array(pts_i)
    pts_j = np.array(pts_j)

    H = homography_find(pts_i, pts_j)
    for i in range(num_points):
      pt_j_gnd = np.array([pts_j[i, 0], pts_j[i, 1], 1.0])
      pt_j_est = H @ np.array([pts_i[i, 0], pts_i[i, 1], 1.0])

      pt_j_est[0] /= pt_j_est[2]
      pt_j_est[1] /= pt_j_est[2]
      pt_j_est[2] /= pt_j_est[2]

      diff = norm(pt_j_gnd - pt_j_est)
      self.assertTrue(diff < 1e-5)

    # Plot 3D
    plt.figure()
    ax = plt.axes(projection='3d')
    plot_tf(ax, T_WC_i, size=0.1, name="pose_i")
    plot_tf(ax, T_WC_j, size=0.1, name="pose_j")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")  # pyright: ignore
    plot_set_axes_equal(ax)
    plt.show()

  def test_homography_pose(self):
    """Test homography_pose()"""
    # Camera
    img_w = 640
    img_h = 480
    fx = focal_length(img_w, 90.0)
    fy = focal_length(img_w, 90.0)
    cx = img_w / 2.0
    cy = img_h / 2.0
    proj_params = np.array([fx, fy, cx, cy])

    # Camera pose T_WC
    C_WC = euler321(-pi / 2, 0.0, -pi / 2)
    r_WC = np.array([0.0, 0.0, 0.0])
    T_WC = tf(C_WC, r_WC)

    # Calibration target pose T_WF
    num_rows = 4
    num_cols = 4
    tag_size = 0.1

    target_x = ((num_cols - 1) * tag_size) / 2.0
    target_y = -((num_rows - 1) * tag_size) / 2.0
    C_WF = euler321(-pi / 2, 0.0, pi / 2)
    r_WF = np.array([0.5, target_x, target_y])
    T_WF = tf(C_WF, r_WF)

    # Generate data
    world_points = []
    object_points = []
    image_points = []

    for i in range(num_rows):
      for j in range(num_cols):
        p_F = np.array([i * tag_size, j * tag_size, 0.0])
        p_W = tf_point(T_WF, p_F)
        p_C = tf_point(inv(T_WC), p_W)
        z = pinhole_project(proj_params, p_C)

        object_points.append(p_F)
        world_points.append(p_W)
        image_points.append(z)

    object_points = np.array(object_points)
    world_points = np.array(world_points)
    image_points = np.array(image_points)

    T_CF = homography_pose(object_points, image_points, fx, fy, cx, cy)
    T_WC_est = T_WF @ inv(T_CF)

    # Compare estimated and ground-truth
    (dr, dtheta) = tf_diff(T_WC, T_WC_est)
    self.assertTrue(norm(dr) < 1e-2)
    self.assertTrue(abs(dtheta) < 1e-4)

    # Plot 3D
    # debug = True
    debug = False
    if debug:
      plt.figure()
      ax = plt.axes(projection="3d")
      plot_tf(ax, T_WC, size=0.1, name="camera")
      plot_tf(ax, T_WC_est, size=0.1, name="camera estimate")
      plot_tf(ax, T_WF, size=0.1, name="fiducial")
      ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2])
      ax.set_xlabel("x [m]")
      ax.set_ylabel("y [m]")
      ax.set_zlabel("z [m]")  # pyright: ignore
      plot_set_axes_equal(ax)
      plt.show()

  def test_dlt_pose(self):
    """Test dlt_pose()"""
    # Camera
    img_w = 640
    img_h = 480
    fx = focal_length(img_w, 90.0)
    fy = focal_length(img_w, 90.0)
    cx = img_w / 2.0
    cy = img_h / 2.0
    proj_params = np.array([fx, fy, cx, cy])

    # Camera pose T_WC
    C_WC = euler321(deg2rad(50.0), 0.0, 0.0)
    r_WC = np.array([0.0, 0.0, 1.0])
    T_WC = tf(C_WC, r_WC)

    # Calibration target pose T_WF
    num_rows = 4
    num_cols = 4
    tag_size = 0.1

    C_WF = euler321(0.0, 0.0, 0.0)
    r_WF = np.array([0, 0, 0])
    T_WF = tf(C_WF, r_WF)

    # Generate data
    world_points = []
    object_points = []
    image_points = []

    for i in range(num_rows):
      for j in range(num_cols):
        p_F = np.array([i * tag_size, j * tag_size, random.uniform(0.0, 1.0)])
        p_W = tf_point(T_WF, p_F)
        p_C = tf_point(inv(T_WC), p_W)
        z = pinhole_project(proj_params, p_C)

        object_points.append(p_F)
        world_points.append(p_W)
        image_points.append(z)

    object_points = np.array(object_points)
    world_points = np.array(world_points)
    image_points = np.array(image_points)

    T_CF = dlt_pose(object_points, image_points, fx, fy, cx, cy)
    T_WC_est = T_WF @ inv(T_CF)

    # Compare estimated and ground-truth
    (dr, dtheta) = tf_diff(T_WC, T_WC_est)
    self.assertTrue(norm(dr) < 1e-4)
    self.assertTrue(abs(dtheta) < 1e-4)

    # Plot 3D
    debug = False
    if debug:
      plt.figure()
      ax = plt.axes(projection="3d")
      plot_tf(ax, T_WC, size=0.1, name="camera")
      plot_tf(ax, T_WC_est, size=0.1, name="camera estimate")
      plot_tf(ax, T_WF, size=0.1, name="fiducial")
      ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2])
      ax.set_xlabel("x [m]")
      ax.set_ylabel("y [m]")
      ax.set_zlabel("z [m]")  # pyright: ignore
      plot_set_axes_equal(ax)
      plt.show()

  def test_solvepnp(self):
    """Test solvepnp()"""
    # Camera
    img_w = 640
    img_h = 480
    fx = focal_length(img_w, 90.0)
    fy = focal_length(img_w, 90.0)
    cx = img_w / 2.0
    cy = img_h / 2.0
    proj_params = np.array([fx, fy, cx, cy])

    num_points = 10
    for _ in range(num_points):
      # Camera pose T_WC
      dr = np.random.uniform(-0.01, 0.01, size=(3,))
      drot = np.random.uniform(-0.01, 0.01, size=(3,))
      C_WC = euler321(-pi / 2 + drot[0], 0.0 + drot[1], -pi / 2 + drot[2])
      r_WC = np.array([0.0 + dr[0], 0.0 + dr[1], 0.0 + dr[2]])
      T_WC = tf(C_WC, r_WC)

      # Calibration target pose T_WF
      num_rows = 4
      num_cols = 4
      tag_size = 0.1

      C_WF = euler321(-pi / 2, 0.0, pi / 2)
      r_WF = np.array([0.1, 0, 0])
      T_WF = tf(C_WF, r_WF)

      # Generate data
      world_points = []
      object_points = []
      image_points = []

      for i in range(num_rows):
        for j in range(num_cols):
          p_F = np.array([i * tag_size, j * tag_size, 0.0])
          p_W = tf_point(T_WF, p_F)
          p_C = tf_point(inv(T_WC), p_W)
          z = pinhole_project(proj_params, p_C)

          object_points.append(p_F)
          world_points.append(p_W)
          image_points.append(z)

      object_points = np.array(object_points)
      world_points = np.array(world_points)
      image_points = np.array(image_points)

      # Get initial T_CF using DLT and perturb it
      T_CF = homography_pose(object_points, image_points, fx, fy, cx, cy)
      trans_rand = np.random.rand(3) * 0.01
      rvec_rand = np.random.rand(3) * 0.01
      T_CF = tf_update(T_CF, np.block([*trans_rand, *rvec_rand]))

      # Test solvepnp
      t_start = datetime.now()
      T_CF = solvepnp(
        object_points,
        image_points,
        fx,
        fy,
        cx,
        cy,
        T_CF_init=T_CF,
        verbose=False,
      )
      t_end = datetime.now()
      solvepnp_time = (t_end - t_start).total_seconds()
      T_WC_est = T_WF @ inv(T_CF)

      # Compare estimated and ground-truth
      (dr, dtheta) = tf_diff(T_WC, T_WC_est)
      self.assertTrue(norm(dr) < 1e-1)
      self.assertTrue(abs(dtheta) < 1e-1)
      self.assertTrue(solvepnp_time < 1.0)

      # Solve pnp with OpenCV
      K = pinhole_K(np.array([fx, fy, cx, cy]))
      D = np.array([0.0, 0.0, 0.0, 0.0])
      flags = cv2.SOLVEPNP_ITERATIVE
      t_start = datetime.now()
      _, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        D,
        False,
        flags=flags,
      )
      C, _ = cv2.Rodrigues(rvec)
      r = tvec.flatten()
      T_CF_opencv = tf(C, r)
      t_end = datetime.now()
      opencv_time = (t_end - t_start).total_seconds()

      # Compare against opencv
      (dr, dtheta) = tf_diff(T_CF, T_CF_opencv)
      self.assertTrue(norm(dr) < 1e-1)
      self.assertTrue(abs(dtheta) < 1e-1)
      self.assertTrue(opencv_time < 1.0)

      # Plot 3D
      # debug = True
      debug = False
      if debug:
        plt.figure()
        ax = plt.axes(projection="3d")
        plot_tf(ax, T_WC, size=0.1, name="camera")
        plot_tf(ax, T_WC_est, size=0.1, name="camera estimate")
        plot_tf(ax, T_WF, size=0.1, name="fiducial")
        ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")  # pyright: ignore
        plot_set_axes_equal(ax)
        plt.show()

  @unittest.skip("Fix Me!")
  def test_illumination_invariant_transform(self):
    """Test illumination_invariant_transform()"""
    img_path = os.path.join(SCRIPT_DIR, "./test_data/images/flower.jpg")
    img = cv2.imread(img_path)
    img = illumination_invariant_transform(img)

    debug = False
    if debug:
      cv2.imshow("Image", img)
      cv2.waitKey(0)

    self.assertTrue(True)

  def test_pinhole_K(self):
    """Test pinhole_K()"""
    fx = 1.0
    fy = 2.0
    cx = 3.0
    cy = 4.0
    proj_params = np.array([fx, fy, cx, cy])
    K = pinhole_K(proj_params)
    expected = np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]])

    self.assertTrue(np.array_equal(K, expected))

  def test_pinhole_project(self):
    """Test pinhole_project()"""
    z = pinhole_project(self.proj_params, self.p_C)
    self.assertTrue(isclose(z[0], 320.0))
    self.assertTrue(isclose(z[1], 240.0))

  def test_pinhole_params_jacobian(self):
    """Test pinhole_params_jacobian()"""
    # Pinhole params jacobian
    fx, fy, cx, cy = self.proj_params
    z = np.array([fx * self.x[0] + cx, fy * self.x[1] + cy])
    J = pinhole_params_jacobian(self.x)

    # Perform numerical diff to obtain finite difference
    step_size = 1e-6
    tol = 1e-4
    finite_diff = zeros((2, 4))

    for i in range(4):
      params_diff = list(self.proj_params)
      params_diff[i] += step_size
      fx, fy, cx, cy = params_diff

      z_diff = np.array([fx * self.x[0] + cx, fy * self.x[1] + cy])
      finite_diff[0:2, i] = (z_diff - z) / step_size

    self.assertTrue(matrix_equal(finite_diff, J, tol, True))

  def test_pinhole_point_jacobian(self):
    """Test pinhole_point_jacobian()"""
    # Pinhole params jacobian
    fx, fy, cx, cy = self.proj_params
    z = np.array([fx * self.x[0] + cx, fy * self.x[1] + cy])
    J = pinhole_point_jacobian(self.proj_params)

    # Perform numerical diff to obtain finite difference
    step_size = 1e-6
    tol = 1e-4
    finite_diff = zeros((2, 2))

    for i in range(2):
      x_diff = list(self.x)
      x_diff[i] += step_size

      z_diff = np.array([fx * x_diff[0] + cx, fy * x_diff[1] + cy])
      finite_diff[0:2, i] = (z_diff - z) / step_size

    self.assertTrue(matrix_equal(finite_diff, J, tol, True))

  def test_chessboard_detector(self):
    # Load the image
    # euroc_data = Path("/data/euroc")
    # calib_dir = euroc_data / "cam_checkerboard" / "mav0" / "cam0" / "data"
    # calib_image = calib_dir / "1403709080437837056.png"
    # image = cv2.imread(str(calib_image), cv2.COLOR_BGR2GRAY)
    # image = image.astype(np.float32)
    # cb_size = (7, 6)
    # winsize = 9

    # detect_corners(image)
    # compute_edge_orientation(image)
    pass


################################################################################
# DATASET
################################################################################

# TIMELINE #####################################################################


@dataclass
class CameraEvent:
  """Camera Event"""

  ts: int
  cam_idx: int
  image: Image | list[tuple[int, Vec2]]


@dataclass
class ImuEvent:
  """IMU Event"""

  ts: int
  imu_idx: int
  acc: Vec3
  gyr: Vec3


@dataclass
class Timeline:
  """Timeline"""
  def __init__(self):
    self.data = {}

  def num_timestamps(self):
    """Return number of timestamps"""
    return len(self.data)

  def num_events(self):
    """Return number of events"""
    nb_events = 0
    for _, events in self.data:
      nb_events += len(events)
    return nb_events

  def get_timestamps(self):
    """Get timestamps"""
    return sorted(list(self.data.keys()))

  def add_event(self, ts, event):
    """Add event"""
    if ts not in self.data:
      self.data[ts] = [event]
    else:
      self.data[ts].append(event)

  def get_events(self, ts):
    """Get events"""
    return self.data[ts]


# EUROC ########################################################################


class EurocSensor:
  """Euroc Sensor"""
  def __init__(self, yaml_path):
    # Load yaml file
    config = load_yaml(yaml_path)

    # General sensor definitions.
    self.sensor_type = config.sensor_type
    self.comment = config.comment

    # Sensor extrinsics wrt. the body-frame.
    self.T_BS = np.array(config.T_BS.data).reshape((4, 4))

    # Camera specific definitions.
    if config.sensor_type == "camera":
      self.rate_hz = config.rate_hz
      self.resolution = config.resolution
      self.camera_model = config.camera_model
      self.intrinsics = config.intrinsics
      self.distortion_model = config.distortion_model
      self.distortion_coefficients = config.distortion_coefficients

    elif config.sensor_type == "imu":
      self.rate_hz = config.rate_hz
      self.gyro_noise_density = config.gyroscope_noise_density
      self.gyro_random_walk = config.gyroscope_random_walk
      self.accel_noise_density = config.accelerometer_noise_density
      self.accel_random_walk = config.accelerometer_random_walk


class EurocImuData:
  """Euroc Imu data"""
  def __init__(self, data_dir):
    import pandas

    self.imu_dir = Path(data_dir, "mav0", "imu0")
    self.config = EurocSensor(Path(self.imu_dir, "sensor.yaml"))
    self.timestamps = []
    self.acc = {}
    self.gyr = {}

    # Load data
    imu_path = Path(self.imu_dir, "data.csv")
    df = pandas.read_csv(imu_path)
    df = df.rename(columns=lambda x: x.strip())

    # -- Timestamp
    timestamps = df["#timestamp [ns]"].to_numpy()
    # -- Accelerometer measurement
    acc_x = df["a_RS_S_x [m s^-2]"].to_numpy()
    acc_y = df["a_RS_S_y [m s^-2]"].to_numpy()
    acc_z = df["a_RS_S_z [m s^-2]"].to_numpy()
    # -- Gyroscope measurement
    gyr_x = df["w_RS_S_x [rad s^-1]"].to_numpy()
    gyr_y = df["w_RS_S_y [rad s^-1]"].to_numpy()
    gyr_z = df["w_RS_S_z [rad s^-1]"].to_numpy()
    # -- Load
    for i, ts in enumerate(timestamps):
      self.timestamps.append(ts)
      self.acc[ts] = np.array([acc_x[i], acc_y[i], acc_z[i]])
      self.gyr[ts] = np.array([gyr_x[i], gyr_y[i], gyr_z[i]])


class EurocCameraData:
  """Euroc Camera data"""
  def __init__(self, data_dir, cam_idx):
    self.cam_idx = cam_idx
    self.cam_dir = Path(data_dir, "mav0", "cam" + str(cam_idx))
    self.config = EurocSensor(Path(self.cam_dir, "sensor.yaml"))
    self.timestamps = []
    self.image_paths = {}

    # Load image paths
    cam_data_dir = str(Path(self.cam_dir, "data", "*.png"))
    for img_file in sorted(glob.glob(cam_data_dir)):
      ts_str, _ = os.path.basename(img_file).split(".")
      ts = int(ts_str)
      self.timestamps.append(ts)
      self.image_paths[ts] = img_file

  def get_image_path_list(self):
    """Return list of image paths"""
    return [img_path for _, img_path in self.image_paths]


class EurocGroundTruth:
  """Euroc ground truth"""
  def __init__(self, data_dir):
    import pandas

    self.timestamps = []
    self.T_WB = {}
    self.v_WB = {}
    self.w_WB = {}
    self.a_WB = {}

    # Load data
    dir_name = "state_groundtruth_estimate0"
    data_csv = Path(data_dir, "mav0", dir_name, "data.csv")
    df = pandas.read_csv(data_csv)
    df = df.rename(columns=lambda x: x.strip())
    # -- Timestamp
    timestamps = df["#timestamp"].to_numpy()
    # -- Body pose in world frame
    rx_list = df["p_RS_R_x [m]"].to_numpy()
    ry_list = df["p_RS_R_y [m]"].to_numpy()
    rz_list = df["p_RS_R_z [m]"].to_numpy()
    qw_list = df["q_RS_w []"].to_numpy()
    qx_list = df["q_RS_x []"].to_numpy()
    qy_list = df["q_RS_y []"].to_numpy()
    qz_list = df["q_RS_z []"].to_numpy()
    # -- Body velocity in world frame
    vx_list = df["v_RS_R_x [m s^-1]"].to_numpy()
    vy_list = df["v_RS_R_y [m s^-1]"].to_numpy()
    vz_list = df["v_RS_R_z [m s^-1]"].to_numpy()
    # -- Add to class
    for i, ts in enumerate(timestamps):
      r_WB = np.array([rx_list[i], ry_list[i], rz_list[i]])
      q_WB = np.array([qw_list[i], qx_list[i], qy_list[i], qz_list[i]])
      v_WB = np.array([vx_list[i], vy_list[i], vz_list[i]])

      self.timestamps.append(ts)
      self.T_WB[ts] = tf(q_WB, r_WB)
      self.v_WB[ts] = v_WB


class EurocDataset:
  """Euroc Dataset"""
  def __init__(self, data_path):
    # Data path
    self.data_path = data_path
    if os.path.isdir(data_path) is False:
      raise RuntimeError(f"Path {data_path} does not exist!")

    # Data
    self.imu0_data = EurocImuData(self.data_path)
    self.cam0_data = EurocCameraData(self.data_path, 0)
    self.cam1_data = EurocCameraData(self.data_path, 1)
    self.ground_truth = EurocGroundTruth(self.data_path)
    self.timeline = self._form_timeline()

  def _form_timeline(self):
    timeline = Timeline()

    # Form timeline
    # -- Add imu0 events
    for ts in self.imu0_data.timestamps:
      acc = self.imu0_data.acc[ts]
      gyr = self.imu0_data.gyr[ts]
      timeline.add_event(ts, ImuEvent(ts, 0, acc, gyr))

    # -- Add cam0 events
    for ts, img_path in self.cam0_data.image_paths.items():
      timeline.add_event(ts, CameraEvent(ts, 0, img_path))

    # -- Add cam1 events
    for ts, img_path in self.cam1_data.image_paths.items():
      timeline.add_event(ts, CameraEvent(ts, 1, img_path))

    return timeline

  def get_camera_image(self, cam_idx, ts):
    """Get camera image"""
    img_path = None
    if cam_idx == 0:
      img_path = self.cam0_data.image_paths[ts]
    elif cam_idx == 1:
      img_path = self.cam1_data.image_paths[ts]
    else:
      raise RuntimeError("cam_idx has to be 0 or 1")
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

  def get_ground_truth_pose(self, ts):
    """Get ground truth pose T_WB at timestamp `ts`"""
    # Pre-check
    if ts <= self.ground_truth.timestamps[0]:
      return None
    elif ts >= self.ground_truth.timestamps[-1]:
      return None

    # Loop throught timestamps
    for k, ground_truth_ts in enumerate(self.ground_truth.timestamps):
      if ts == ground_truth_ts:
        return self.ground_truth.T_WB[ts]
      elif self.ground_truth.timestamps[k] > ts:
        ts_i = self.ground_truth.timestamps[k - 1]
        ts_j = self.ground_truth.timestamps[k]
        alpha = float(ts_j - ts) / float(ts_j - ts_i)
        pose_i = self.ground_truth.T_WB[ts_i]
        pose_j = self.ground_truth.T_WB[ts_j]
        return tf_lerp(pose_i, pose_j, alpha)

    return None


class TestEuroc(unittest.TestCase):
  """Test Euroc dataset loader"""
  def test_load(self):
    """Test load"""
    dataset = EurocDataset(EUROC_DATA_PATH)
    self.assertTrue(dataset is not None)


# KITTI #######################################################################


class KittiCameraData:
  """Kitti Camera Data"""
  def __init__(self, cam_idx, seq_dir):
    self.cam_idx = cam_idx
    self.seq_dir = seq_dir
    self.cam_path = self.seq_dir / ("image_" + str(self.cam_idx).zfill(2))
    self.img_dir = self.cam_path / "data"
    self.img_paths = sorted(glob.glob(str(Path(self.img_dir, "*.png"))))


class KittiVelodyneData:
  """Kitti Velodyne Data"""
  def __init__(self, seq_dir):
    self.seq_dir = seq_dir
    self.velodyne_path = Path(self.seq_dir, "velodyne_points")
    self.bins_dir = self.velodyne_path / "data"

    ts_file = self.velodyne_path / "timestamps.txt"
    ts_start_file = self.velodyne_path / "timestamps_start.txt"
    ts_end_file = self.velodyne_path / "timestamps_end.txt"

    self.timestamps = self._load_timestamps(ts_file)
    self.timestamps_start = self._load_timestamps(ts_start_file)
    self.timestamps_end = self._load_timestamps(ts_end_file)
    self.bin_paths = sorted(self.bins_dir.glob("*.bin"))

  def _load_timestamps(self, data_file):
    """Load timestamps from file"""
    f = open(data_file, "r")

    timestamps = []
    for line in f:
      line = line.strip()
      dt = line.split(".")[0]
      dt_obj = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
      seconds = dt_obj.timestamp()
      nanoseconds = int(line.split(".")[1])
      timestamps.append(int(seconds * 1e9) + nanoseconds)

    f.close()
    return timestamps

  def load_scan(self, ts):
    """Load scan based on timestamp"""
    index = self.timestamps.index(ts)
    bin_path = self.bin_paths[index]
    pcd = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pcd


class KittiRawDataset:
  """KittiRawDataset"""
  def __init__(self, data_dir, date, seq, is_sync):
    # Paths
    self.data_dir = data_dir
    self.date = date
    self.date_dir = data_dir / self.date
    self.seq = seq.zfill(4)
    self.sync = "sync" if is_sync else "extract"
    self.seq_name = "_".join([self.date, "drive", self.seq, self.sync])
    self.seq_dir = Path(self.data_dir, self.date, self.seq_name)

    # Camera data
    self.cam0_data = KittiCameraData(0, self.seq_dir)
    self.cam1_data = KittiCameraData(1, self.seq_dir)
    self.cam2_data = KittiCameraData(2, self.seq_dir)
    self.cam3_data = KittiCameraData(3, self.seq_dir)

    # Velodyne data
    self.velodyne_data = KittiVelodyneData(self.seq_dir)

    # Calibration
    calib_cam_to_cam_filepath = Path(self.date_dir, "calib_cam_to_cam.txt")
    calib_imu_to_velo_filepath = Path(self.date_dir, "calib_imu_to_velo.txt")
    calib_velo_to_cam_filepath = Path(self.date_dir, "calib_velo_to_cam.txt")
    self.calib_cam_to_cam = self._read_calib_file(calib_cam_to_cam_filepath)
    self.calib_imu_to_velo = self._read_calib_file(calib_imu_to_velo_filepath)
    self.calib_velo_to_cam = self._read_calib_file(calib_velo_to_cam_filepath)

  @classmethod
  def _read_calib_file(cls, fp):
    data = {}
    with open(fp, "r") as f:
      for line in f.readlines():
        key, value = line.split(":", 1)
        # The only non-float values in these files are dates, which
        # we don't care about anyway
        try:
          data[key] = np.array([float(x) for x in value.split()])
        except ValueError:
          pass
      return data

  def num_camera_images(self, cam_idx=0):
    """Return number of camera images"""
    assert cam_idx >= 0
    assert cam_idx <= 3
    if cam_idx == 0:
      return len(self.cam0_data.img_paths)
    elif cam_idx == 1:
      return len(self.cam1_data.img_paths)
    elif cam_idx == 2:
      return len(self.cam2_data.img_paths)
    elif cam_idx == 3:
      return len(self.cam3_data.img_paths)

    raise RuntimeError(f"Invalid cam_idx: {cam_idx}")

  def get_velodyne_extrinsics(self):
    """Get velodyne extrinsics"""
    # Form imu-velo extrinsics T_BV
    C_VB = self.calib_imu_to_velo["R"].reshape((3, 3))
    r_VB = self.calib_imu_to_velo["T"]
    T_VB = tf(C_VB, r_VB)
    T_BV = inv(T_VB)
    return T_BV

  def get_camera_extrinsics(self, cam_idx):
    """Get camera extrinsics T_BCi"""
    # Form imu-velo extrinsics T_VB
    C_VB = self.calib_imu_to_velo["R"].reshape((3, 3))
    r_VB = self.calib_imu_to_velo["T"]
    T_VB = tf(C_VB, r_VB)

    # Form velo-cam extrinsics T_C0V
    C_C0V = self.calib_velo_to_cam["R"].reshape((3, 3))
    r_C0V = self.calib_velo_to_cam["T"]
    T_C0V = tf(C_C0V, r_C0V)

    # Form cam-cam extrinsics T_CiC0
    cam_str = str(cam_idx)
    C_CiC0 = self.calib_cam_to_cam["R_" + cam_str.zfill(2)].reshape((3, 3))
    r_CiC0 = self.calib_cam_to_cam["T_" + cam_str.zfill(2)]
    T_CiC0 = tf(C_CiC0, r_CiC0)

    # Form camera extrinsics T_BC0
    T_CiB = T_CiC0 @ T_C0V @ T_VB
    T_BCi = inv(T_CiB)

    return T_BCi

  def get_camera_image(self, cam_idx, **kwargs):
    """Get camera image"""
    assert cam_idx >= 0 and cam_idx <= 3
    imread_flag = kwargs.get("imread_flag", cv2.IMREAD_GRAYSCALE)
    img_idx = kwargs["index"]

    if cam_idx == 0:
      return cv2.imread(self.cam0_data.img_paths[img_idx], imread_flag)
    elif cam_idx == 1:
      return cv2.imread(self.cam1_data.img_paths[img_idx], imread_flag)
    elif cam_idx == 2:
      return cv2.imread(self.cam2_data.img_paths[img_idx], imread_flag)
    elif cam_idx == 3:
      return cv2.imread(self.cam3_data.img_paths[img_idx], imread_flag)

    raise RuntimeError(f"Invalid cam_idx: {cam_idx}")

  def plot_frames(self):
    """Plot Frames"""
    T_BV = self.get_velodyne_extrinsics()
    T_BC0 = self.get_camera_extrinsics(0)
    T_BC1 = self.get_camera_extrinsics(1)
    T_BC2 = self.get_camera_extrinsics(2)
    T_BC3 = self.get_camera_extrinsics(3)

    plt.figure()
    ax = plt.axes(projection="3d")
    plot_tf(ax, eye(4), size=0.1, name="imu")
    plot_tf(ax, T_BV, size=0.1, name="velo")
    plot_tf(ax, T_BC0, size=0.1, name="cam0")
    plot_tf(ax, T_BC1, size=0.1, name="cam1")
    plot_tf(ax, T_BC2, size=0.1, name="cam2")
    plot_tf(ax, T_BC3, size=0.1, name="cam3")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")  # pyright: ignore
    plot_set_axes_equal(ax)
    plt.show()


# class TestKitti(unittest.TestCase):
#   """Test KITTI dataset loader"""
#
#   # @unittest.skip("")
#   def test_load(self):
#     """Test load"""
#     data_dir = Path("/data/kitti_raw")
#     date = "2011_09_26"
#     seq = "93"
#     dataset = KittiRawDataset(data_dir, date, seq, True)
#
#     lidar_timestamps = dataset.velodyne_data.timestamps
#     xyzi = dataset.velodyne_data.load_scan(lidar_timestamps[0])
#
#     # fig = plt.figure(figsize=(12, 10))
#     # ax = plt.axes(projection='3d')
#     # ax.scatter(xyzi[::100, 0], xyzi[::100, 1], xyzi[::100, 2])
#     # ax.set_xlabel("x [m]")
#     # ax.set_ylabel("y [m]")
#     # ax.set_zlabel("z [m]")
#     # plot_set_axes_equal(ax)
#     # plt.show()
#
#     # for ts in lidar_timestamps[:10]:
#     #   xyzi = dataset.velodyne_data.load_scan(ts)
#
#     # for i in range(dataset.num_camera_images()):
#     #   cam0_img = dataset.get_camera_image(0, index=i)
#     #   cam1_img = dataset.get_camera_image(1, index=i)
#     #   cam2_img = dataset.get_camera_image(2, index=i)
#     #   cam3_img = dataset.get_camera_image(3, index=i)
#     #
#     #   img_size = cam0_img.shape
#     #   img_new_size = (int(img_size[1] / 2.0), int(img_size[0] / 2.0))
#     #
#     #   cam0_img = cv2.resize(cam0_img, img_new_size)
#     #   cam1_img = cv2.resize(cam1_img, img_new_size)
#     #   cam2_img = cv2.resize(cam2_img, img_new_size)
#     #   cam3_img = cv2.resize(cam3_img, img_new_size)
#     #
#     #   cv2.imshow("viz", cv2.vconcat([cam0_img, cam1_img, cam2_img, cam3_img]))
#     #   cv2.waitKey(0)
#     #
#     # self.assertTrue(dataset is not None)




###############################################################################
# STATE ESTIMATION
###############################################################################

# STATE VARIABLES #############################################################


class StateVariableType(Enum):
  """State Variable Type"""

  POSE = 1
  EXTRINSICS = 2
  FEATURE = 3
  CAMERA = 4
  SPEED_AND_BIASES = 5


@dataclass
class StateVariable:
  """State variable"""

  ts: int
  var_type: str
  param: VecN
  parameterization: str | None
  min_dims: int
  fix: bool
  data: Any | None = None
  param_id: int | None = None
  marginalize: bool = False

  def set_param_id(self, pid):
    """Set parameter id"""
    self.param_id = pid

  def __hash__(self):
    """Hash function"""
    return hash(repr(self))


class FeatureMeasurements:
  """Feature measurements"""
  def __init__(self):
    self._init = False
    self._data = {}

  def initialized(self):
    """Check if feature is initialized"""
    return self._init

  def has_overlap(self, ts):
    """Check if feature has overlap at timestamp `ts`"""
    return len(self._data[ts]) > 1

  def set_initialized(self):
    """Set feature as initialized"""
    self._init = True

  def update(self, ts, cam_idx, z):
    """Add feature measurement"""
    assert len(z) == 2
    if ts not in self._data:
      self._data[ts] = {}
    self._data[ts][cam_idx] = z

  def get(self, ts, cam_idx):
    """Get feature measurement"""
    return self._data[ts][cam_idx]

  def get_overlaps(self, ts):
    """Get feature overlaps"""
    overlaps = []
    for cam_idx, z in self._data[ts].items():
      overlaps.append((cam_idx, z))
    return overlaps


def pose_setup(ts, param, **kwargs):
  """Form pose state-variable"""
  fix = kwargs.get("fix", False)
  param = tf2pose(param) if param.shape == (4, 4) else param
  return StateVariable(ts, "pose", param, None, 6, fix)


def extrinsics_setup(param, **kwargs):
  """Form extrinsics state-variable"""
  fix = kwargs.get("fix", False)
  param = tf2pose(param) if param.shape == (4, 4) else param
  return StateVariable(-1, "extrinsics", param, None, 6, fix)


def screw_axis_setup(param, **kwargs):
  """Form screw axis state-variable"""
  fix = kwargs.get("fix", False)
  return StateVariable(-1, "screw_axis", param, None, 6, fix)


def camera_params_setup(cam_idx, res, proj_model, dist_model, param, **kwargs):
  """Form camera parameters state-variable"""
  fix = kwargs.get("fix", False)
  data = camera_geometry_setup(cam_idx, res, proj_model, dist_model)
  return StateVariable(-1, "camera", param, None, len(param), fix, data)


def feature_setup(param, **kwargs):
  """Form feature state-variable"""
  fix = kwargs.get("fix", False)
  data = FeatureMeasurements()
  return StateVariable(-1, "feature", param, None, len(param), fix, data)


def speed_biases_setup(ts, vel, ba, bg, **kwargs):
  """Form speed and biases state-variable"""
  fix = kwargs.get("fix", False)
  param = np.block([vel, ba, bg])
  return StateVariable(ts, "speed_and_biases", param, None, len(param), fix)


def inverse_depth_setup(param, **kwargs):
  """Form inverse depth state-variable"""
  fix = kwargs.get("fix", False)
  return StateVariable(-1, "inverse_depth", np.array([param]), None, 1, fix)


def time_delay_setup(param, **kwargs):
  """Form time delay state-variable"""
  fix = kwargs.get("fix", False)
  return StateVariable(-1, "time_delay", np.array([param]), None, 1, fix)


def joint_angle_setup(param, **kwargs):
  """Form time delay state-variable"""
  fix = kwargs.get("fix", False)
  return StateVariable(-1, "joint_angle", np.array([param]), None, 1, fix)


def perturb_state_variable(sv, i, step_size):
  """Perturb state variable"""
  if sv.var_type == "pose" or sv.var_type == "extrinsics":
    T = pose2tf(sv.param)
    T_dash = tf_perturb(T, i, step_size)
    sv.param = tf2pose(T_dash)
  else:
    sv.param[i] += step_size

  return sv


def perturb_pose(pose, dr, drot):
  """Perturb pose"""
  T = pose2tf(pose)

  T = tf_perturb(T, 0, dr[0])
  T = tf_perturb(T, 1, dr[1])
  T = tf_perturb(T, 2, dr[2])

  T = tf_perturb(T, 3, drot[0])
  T = tf_perturb(T, 4, drot[1])
  T = tf_perturb(T, 5, drot[2])

  return tf2pose(T)


def perturb_pose_random(pose, pos_range, rot_range):
  """Perturb pose randomly"""
  dr = np.random.uniform(pos_range[0], pos_range[1], size=(3,))
  drot = np.random.uniform(rot_range[0], rot_range[1], size=(3,))
  return perturb_pose(pose, dr, drot)


def perturb_tf_random(T, pos_range, rot_range):
  """Perturb TF randomly"""
  pose = tf2pose(T)
  pose = perturb_pose_random(pose, pos_range, rot_range)
  return pose2tf(pose)


def update_state_variable(sv, dx):
  """Update state variable"""
  if sv.var_type == "pose" or sv.var_type == "extrinsics":
    T = pose2tf(sv.param)
    T_prime = tf_update(T, dx)
    sv.param = tf2pose(T_prime)
  else:
    sv.param += dx


def idp_param(cam_params, T_WC, z):
  """Create inverse depth parameter"""
  # Back project image pixel measurmeent to 3D ray
  cam_geom = cam_params.data
  x = cam_geom.backproject(cam_params.params, z)

  # Convert 3D ray from camera frame to world frame
  r_WC = tf_trans(T_WC)
  C_WC = tf_rot(T_WC)
  h_W = C_WC @ x

  # Obtain bearing (theta, phi) and inverse depth (rho)
  theta = atan2(h_W[0], h_W[2])
  phi = atan2(-h_W[1], sqrt(h_W[0] * h_W[0] + h_W[2] * h_W[2]))
  rho = 0.1
  # sigma_rho = 0.5  # variance of inverse depth

  # Form inverse depth parameter
  param = np.array([r_WC, theta, phi, rho])
  return param


def idp_param_jacobian(param):
  """Inverse depth parameter jacobian"""
  _, _, _, theta, phi, rho = param
  p_W = np.array([cos(phi) * sin(theta), -sin(phi), cos(phi) * cos(theta)])
  d = 1.0 / rho
  cphi = cos(phi)
  sphi = sin(phi)
  ctheta = cos(theta)
  stheta = sin(theta)

  J_x = np.array([1.0, 0.0, 0.0])
  J_y = np.array([0.0, 1.0, 0.0])
  J_z = np.array([0.0, 0.0, 1.0])
  J_theta = d * np.array([cphi * ctheta, 0.0, cphi * -stheta])
  J_phi = d * np.array([-sphi * stheta, -cphi, -sphi * ctheta])
  J_rho = -1.0 / rho**2 @ p_W
  J_param = np.block([J_x, J_y, J_z, J_theta, J_phi, J_rho])

  return J_param


def idp_point(param):
  """Inverse depth parmaeter to point"""
  # Extract parameter values
  x, y, z, theta, phi, rho = param

  # Camera position in world frame
  r_WC = np.array([x, y, z])

  # Convert bearing to 3D ray from camera frame
  m = np.array([cos(phi) * sin(theta), -sin(phi), cos(phi) * cos(theta)])

  # Form 3D point in world frame
  p_W = r_WC + (1.0 / rho) @ m

  return p_W


# FACTORS ######################################################################


class Factor:
  """Factor"""
  def __init__(self, ftype, pids, z, covar, r_size):
    self.factor_id = None
    self.factor_type = ftype
    self.param_ids = pids
    self.r_size = r_size
    self.measurement = None
    self.covar = None
    self.sqrt_info = None

    if isinstance(z, (np.ndarray, tuple)):
      self.measurement = z
    else:
      self.measurement = np.array([z])

    if covar is not None and isinstance(covar, np.ndarray):
      self.covar = covar
      self.sqrt_info = chol(inv(self.covar)).T
    elif covar is not None:
      self.covar = np.array([covar])
      self.sqrt_info = np.sqrt(1.0 / covar)

  def set_factor_id(self, fid):
    """Set factor id"""
    self.factor_id = fid

  def eval(self, params, **kwargs):
    """Evalulate Factor"""
    assert params
    assert kwargs
    raise NotImplementedError()

  def calculate_jacobian(self, fvars, var_idx):
    """Calculate Jacobian"""
    params = [sv.param for sv in fvars]
    _, jacs = self.eval(params)
    return jacs[var_idx]

  def get_numerical_jacobian(self, fvars, var_idx, **kwargs):
    """Get numerical jacobian"""
    # Step size and threshold
    h = kwargs.get("step_size", 1e-8)

    # Calculate baseline
    params = [sv.param for sv in fvars]
    r, _ = self.eval(params)

    # Numerical diff
    J_fdiff = zeros((len(r), fvars[var_idx].min_dims))
    for i in range(fvars[var_idx].min_dims):
      # Forward difference and evaluate
      vars_fwd = copy.deepcopy(fvars)
      vars_fwd[var_idx] = perturb_state_variable(vars_fwd[var_idx], i, 0.5 * h)
      r_fwd, _ = self.eval([sv.param for sv in vars_fwd])

      # Backward difference and evaluate
      vars_bwd = copy.deepcopy(fvars)
      vars_bwd[var_idx] = perturb_state_variable(vars_bwd[var_idx], i, -0.5 * h)
      r_bwd, _ = self.eval([sv.param for sv in vars_bwd])

      # Central finite difference
      J_fdiff[:, i] = (r_fwd - r_bwd) / h

    return J_fdiff

  def check_jacobian(self, fvars, var_idx, jac_name, **kwargs):
    """Check factor Jacobian"""
    threshold = kwargs.get("threshold", 1e-4)
    verbose = kwargs.get("verbose", False)

    J_fdiff = self.get_numerical_jacobian(fvars, var_idx, **kwargs)
    params = [sv.param for sv in fvars]
    _, jacs = self.eval(params)
    J = jacs[var_idx]

    return check_jacobian(jac_name, J_fdiff, J, threshold, verbose)

  def __hash__(self):
    """Hash function"""
    return hash(repr(self))


class MeasurementFactor(Factor):
  """Measurement Factor"""
  def __init__(self, pids, z, covar):
    assert len(pids) == 1
    r_size = 1 if type(z) in [np.float64, float] else len(z)
    Factor.__init__(self, "MeasurementFactor", pids, z, covar, r_size)

  def eval(self, params, **kwargs):
    """Evaluate"""
    assert self.sqrt_info

    # Form residuals
    z = self.measurement
    z_hat = params[0]

    if isinstance(self.sqrt_info, np.ndarray):
      r = self.sqrt_info @ (z - z_hat)
    else:
      r = self.sqrt_info * (z - z_hat)

    if kwargs.get("only_residuals", False):
      return r

    # Form Jacobians
    J_rows = r.shape[0]
    J = zeros((J_rows, J_rows))
    if isinstance(self.sqrt_info, np.ndarray):
      J = self.sqrt_info @ -eye(J_rows)
    else:
      J = self.sqrt_info * -eye(J_rows)

    return (r, [J])


class PoseFactor(Factor):
  """Pose Factor"""
  def __init__(self, pids, z, covar):
    assert len(pids) == 1
    assert z.shape == (4, 4)
    assert covar.shape == (6, 6)
    Factor.__init__(self, "PoseFactor", pids, z, covar, 6)

  def eval(self, params, **kwargs):
    """Evaluate"""
    assert len(params) == 1
    assert len(params[0]) == 7
    assert self.sqrt_info is not None
    assert self.measurement is not None

    # Measured pose
    assert self.measurement is not None
    T_meas = typing.cast(Mat4, self.measurement)
    q_meas = tf_quat(T_meas)
    r_meas = tf_trans(T_meas)

    # Estimated pose
    T_est = pose2tf(params[0])
    q_est = tf_quat(T_est)
    r_est = tf_trans(T_est)

    # Form residuals (pose - pose_est)
    dr = r_meas - r_est
    dq = quat_mul(quat_inv(q_meas), q_est)
    dtheta = 2 * dq[1:4]
    r = self.sqrt_info @ np.block([dr, dtheta])
    if kwargs.get("only_residuals", False):
      return r

    # Form jacobians
    J = zeros((6, 6))
    J[0:3, 0:3] = -eye(3)
    J[3:6, 3:6] = quat_left(dq)[1:4, 1:4]
    J = self.sqrt_info @ J

    return (r, [J])


class MultiCameraBuffer:
  """Multi-camera buffer"""
  def __init__(self, nb_cams=0):
    self.nb_cams = nb_cams
    self._ts = []
    self._data = {}

  def reset(self):
    """Reset buffer"""
    self._ts = []
    self._data = {}

  def add(self, ts, cam_idx, data):
    """Add camera event"""
    if self.nb_cams == 0:
      raise RuntimeError("MulitCameraBuffer not initialized yet!")

    self._ts.append(ts)
    self._data[cam_idx] = data

  def ready(self):
    """Check whether buffer has all the camera frames ready"""
    if self.nb_cams == 0:
      raise RuntimeError("MulitCameraBuffer not initialized yet!")

    check_ts_same = len(set(self._ts)) == 1
    check_ts_len = len(self._ts) == self.nb_cams
    check_data = len(self._data) == self.nb_cams
    check_cam_indices = len(set(self._data.keys())) == self.nb_cams

    return check_ts_same and check_ts_len and check_data and check_cam_indices

  def get_camera_indices(self):
    """Get camera indices"""
    return self._data.keys()

  def get_data(self):
    """Get camera data"""
    if self.nb_cams is None:
      raise RuntimeError("MulitCameraBuffer not initialized yet!")

    return self._data


class BAFactor(Factor):
  """BA Factor"""
  def __init__(self, cam_geom, pids, z, covar=eye(2)):
    assert len(pids) == 3
    assert len(z) == 2
    assert covar.shape == (2, 2)
    Factor.__init__(self, "BAFactor", pids, z, covar, 2)
    self.cam_geom = cam_geom

  def get_residual(self, cam_pose, feature, cam_params):
    """Get residual"""
    T_WC = pose2tf(cam_pose)
    p_W = feature
    p_C = tf_point(inv(T_WC), p_W)
    status, z_hat = self.cam_geom.project(cam_params, p_C)

    z = self.measurement
    r = z - z_hat

    return status, r

  def get_reproj_error(self, cam_pose, feature, cam_params):
    """Get reprojection error"""
    status, r = self.get_residual(cam_pose, feature, cam_params)
    reproj_error = norm(r)
    return status, reproj_error

  def eval(self, params, **kwargs):
    """Evaluate"""
    assert self.sqrt_info is not None
    assert len(params) == 3
    assert len(params[0]) == 7  # Camera pose T_WC
    assert len(params[1]) == 3  # Feature position (x, y, z)
    assert len(params[2]) == self.cam_geom.get_params_size()  # Camera params

    # Setup
    r = np.array([0.0, 0.0])
    J0 = zeros((2, 6))
    J1 = zeros((2, 3))
    J2 = zeros((2, self.cam_geom.get_params_size()))

    # Map params
    cam_pose, feature, cam_params = params

    # Project point in world frame to image plane
    T_WC = pose2tf(cam_pose)
    z_hat = zeros((2, 1))
    p_W = zeros((3, 1))
    p_W = feature
    p_C = tf_point(inv(T_WC), p_W)
    status, z_hat = self.cam_geom.project(cam_params, p_C)

    # Calculate residual
    sqrt_info = self.sqrt_info
    z = self.measurement
    r = sqrt_info @ (z - z_hat)
    if kwargs.get("only_residuals", False):
      return r

    # Calculate Jacobians
    if status is False:
      return (r, [J0, J1, J2])
    # -- Measurement model jacobian
    neg_sqrt_info = -1.0 * sqrt_info
    Jh = self.cam_geom.J_proj(cam_params, p_C)
    Jh_weighted = neg_sqrt_info @ Jh
    # -- Jacobian w.r.t. camera pose T_WC
    C_WC = tf_rot(T_WC)
    C_CW = C_WC.T
    r_WC = tf_trans(T_WC)
    J0 = zeros((2, 6))  # w.r.t Camera pose T_WC
    J0[0:2, 0:3] = Jh_weighted @ -C_CW
    J0[0:2, 3:6] = Jh_weighted @ -C_CW @ hat(p_W - r_WC) @ -C_WC
    # -- Jacobian w.r.t. feature
    J1 = zeros((2, 3))
    J1 = Jh_weighted @ C_CW
    # -- Jacobian w.r.t. camera parameters
    J_cam_params = self.cam_geom.J_params(cam_params, p_C)
    J2 = zeros((2, self.cam_geom.get_params_size()))
    J2 = neg_sqrt_info @ J_cam_params

    return (r, [J0, J1, J2])


class VisionFactor(Factor):
  """Vision Factor"""
  def __init__(self, cam_geom, pids, z, covar=eye(2)):
    assert len(pids) == 4
    assert len(z) == 2
    assert covar.shape == (2, 2)
    Factor.__init__(self, "VisionFactor", pids, z, covar, 2)
    self.cam_geom = cam_geom

  def get_residual(self, pose, cam_exts, feature, cam_params):
    """Get residual"""
    T_WB = pose2tf(pose)
    T_BCi = pose2tf(cam_exts)
    p_W = feature
    p_C = tf_point(inv(T_WB @ T_BCi), p_W)
    status, z_hat = self.cam_geom.project(cam_params, p_C)

    z = self.measurement
    r = z - z_hat

    return status, r

  def get_reproj_error(self, pose, cam_exts, feature, cam_params):
    """Get reprojection error"""
    status, r = self.get_residual(pose, cam_exts, feature, cam_params)
    reproj_error = norm(r)
    return status, reproj_error

  def eval(self, params, **kwargs):
    """Evaluate"""
    assert self.sqrt_info is not None
    assert len(params) == 4
    assert len(params[0]) == 7
    assert len(params[1]) == 7
    assert len(params[2]) == 3
    assert len(params[3]) == self.cam_geom.get_params_size()

    # Setup
    r = np.array([0.0, 0.0])
    J0 = zeros((2, 6))
    J1 = zeros((2, 6))
    J2 = zeros((2, 3))
    J3 = zeros((2, self.cam_geom.get_params_size()))

    # Project point in world frame to image plane
    pose, cam_exts, feature, cam_params = params
    T_WB = pose2tf(pose)
    T_BCi = pose2tf(cam_exts)
    p_W = feature
    p_C = tf_point(inv(T_WB @ T_BCi), p_W)
    status, z_hat = self.cam_geom.project(cam_params, p_C)

    # Calculate residual
    sqrt_info = self.sqrt_info
    z = self.measurement
    r = sqrt_info @ (z - z_hat)
    if kwargs.get("only_residuals", False):
      return r

    # Calculate Jacobians
    if status is False:
      return (r, [J0, J1, J2, J3])

    C_BCi = tf_rot(T_BCi)
    C_WB = tf_rot(T_WB)
    C_CB = C_BCi.T
    C_BW = C_WB.T
    C_CW = C_CB @ C_WB.T
    r_WB = tf_trans(T_WB)
    neg_sqrt_info = -1.0 * sqrt_info
    # -- Measurement model jacobian
    Jh = self.cam_geom.J_proj(cam_params, p_C)
    Jh_weighted = neg_sqrt_info @ Jh
    # -- Jacobian w.r.t. pose T_WB
    J0 = zeros((2, 6))
    J0[0:2, 0:3] = Jh_weighted @ C_CB @ -C_BW
    J0[0:2, 3:6] = Jh_weighted @ C_CB @ -C_BW @ hat(p_W - r_WB) @ -C_WB
    # -- Jacobian w.r.t. camera extrinsics T_BCi
    J1 = zeros((2, 6))
    J1[0:2, 0:3] = Jh_weighted @ -C_CB
    J1[0:2, 3:6] = Jh_weighted @ -C_CB @ hat(C_BCi @ p_C) @ -C_BCi
    # -- Jacobian w.r.t. feature
    J2 = zeros((2, 3))
    J2 = Jh_weighted @ C_CW
    # -- Jacobian w.r.t. camera parameters
    J_cam_params = self.cam_geom.J_params(cam_params, p_C)
    J3 = zeros((2, 8))
    J3 = neg_sqrt_info @ J_cam_params

    return (r, [J0, J1, J2, J3])


class CalibVisionFactor(Factor):
  """Calibration Vision Factor"""
  def __init__(self, cam_geom, pids, calib_target, covar=eye(2)):
    assert len(pids) == 3
    assert len(calib_target) == 4
    assert covar.shape == (2, 2)
    tag_id, corner_idx, r_FFi, z = calib_target
    Factor.__init__(self, "CalibVisionFactor", pids, z, covar, 2)
    self.cam_geom = cam_geom
    self.tag_id = tag_id
    self.corner_idx = corner_idx
    self.r_FFi = r_FFi

  def get_residual(self, pose, cam_exts, cam_params):
    """Get residual"""
    T_BF = pose2tf(pose)
    T_BCi = pose2tf(cam_exts)
    T_CiB = inv(T_BCi)
    r_CiFi = tf_point(T_CiB @ T_BF, self.r_FFi)
    status, z_hat = self.cam_geom.project(cam_params, r_CiFi)
    r = self.measurement - z_hat
    return status, r

  def get_reproj_error(self, pose, cam_exts, cam_params):
    """Get reprojection error"""
    status, r = self.get_residual(pose, cam_exts, cam_params)
    return status, norm(r)

  def eval(self, params, **kwargs):
    """Evaluate"""
    assert len(params) == 3
    assert len(params[0]) == 7
    assert len(params[1]) == 7
    assert len(params[2]) == self.cam_geom.get_params_size()
    assert self.sqrt_info is not None

    # Setup
    r = np.array([0.0, 0.0])
    J0 = zeros((2, 6))
    J1 = zeros((2, 6))
    J2 = zeros((2, self.cam_geom.get_params_size()))

    # Map parameters out
    pose, cam_exts, cam_params = params
    T_BF = pose2tf(pose)
    T_BCi = pose2tf(cam_exts)

    # Transform and project point to image plane
    T_CiB = inv(T_BCi)
    r_CiFi = tf_point(T_CiB @ T_BF, self.r_FFi)
    status, z_hat = self.cam_geom.project(cam_params, r_CiFi)

    # Calculate residual
    sqrt_info = self.sqrt_info
    z = self.measurement
    r = sqrt_info @ (z - z_hat)
    if kwargs.get("only_residuals", False):
      return r

    # Calculate Jacobians
    if status is False:
      return (r, [J0, J1, J2])

    neg_sqrt_info = -1.0 * sqrt_info
    Jh = self.cam_geom.J_proj(cam_params, r_CiFi)
    Jh_weighted = neg_sqrt_info @ Jh
    # -- Jacobians w.r.t relative camera pose T_BF
    C_CiB = tf_rot(T_CiB)
    C_BF = tf_rot(T_BF)
    J0 = zeros((2, 6))
    J0[0:2, 0:3] = Jh_weighted @ C_CiB
    J0[0:2, 3:6] = Jh_weighted @ C_CiB @ -C_BF @ hat(self.r_FFi)
    # -- Jacobians w.r.t T_BCi
    r_BFi = tf_point(T_BF, self.r_FFi)
    r_BCi = tf_trans(T_BCi)
    C_BCi = tf_rot(T_BCi)
    J1 = zeros((2, 6))
    J1[0:2, 0:3] = Jh_weighted @ -C_CiB
    J1[0:2, 3:6] = Jh_weighted @ -C_CiB @ hat(r_BFi - r_BCi) @ -C_BCi
    # -- Jacobians w.r.t cam params
    J_cam_params = self.cam_geom.J_params(cam_params, r_CiFi)
    J2 = neg_sqrt_info @ J_cam_params

    return (r, [J0, J1, J2])


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

  def add_event(self, imu_event):
    """Add imu event"""
    self.ts.append(imu_event.ts)
    self.acc.append(imu_event.acc)
    self.gyr.append(imu_event.gyr)

  def length(self):
    """Return length of imu buffer"""
    return len(self.ts)

  def extract(self, ts_start, ts_end):
    """Form ImuBuffer"""
    assert ts_start >= self.ts[0]
    assert ts_end <= self.ts[-1]
    imu_ts = []
    imu_acc = []
    imu_gyr = []

    # Extract data between ts_start and ts_end
    remove_idx = 0
    ts_km1 = None
    acc_km1 = None
    gyr_km1 = None

    for k, ts_k in enumerate(self.ts):
      # Check if within the extraction zone
      if ts_k < ts_start:
        continue

      # Setup
      acc_k = self.acc[k]
      gyr_k = self.gyr[k]

      # Interpolate start or end?
      if len(imu_ts) == 0 and ts_k > ts_start:
        # Interpolate start
        ts_km1 = self.ts[k - 1]
        acc_km1 = self.acc[k - 1]
        gyr_km1 = self.gyr[k - 1]

        alpha = (ts_start - ts_km1) / (ts_k - ts_km1)
        acc_km1 = (1.0 - alpha) * acc_km1 + alpha * acc_k
        gyr_km1 = (1.0 - alpha) * gyr_km1 + alpha * gyr_k
        ts_km1 = ts_start

        imu_ts.append(ts_km1)
        imu_acc.append(acc_km1)
        imu_gyr.append(gyr_km1)

      elif ts_k > ts_end:
        # Interpolate end
        ts_km1 = self.ts[k - 1]
        acc_km1 = self.acc[k - 1]
        gyr_km1 = self.gyr[k - 1]

        alpha = (ts_end - ts_km1) / (ts_k - ts_km1)
        acc_k = (1.0 - alpha) * acc_km1 + alpha * acc_k
        gyr_k = (1.0 - alpha) * gyr_km1 + alpha * gyr_k
        ts_k = ts_end

      # Add to subset
      imu_ts.append(ts_k)
      imu_acc.append(acc_k)
      imu_gyr.append(gyr_k)

      # End?
      if ts_k == ts_end:
        break

      # Update
      remove_idx = k

    # Remove data before ts_end
    self.ts = self.ts[remove_idx:]
    self.acc = self.acc[remove_idx:]
    self.gyr = self.gyr[remove_idx:]

    return ImuBuffer(imu_ts, imu_acc, imu_gyr)

  def print(self, extra_newline=False):
    """Print"""
    for ts, acc, gyr in zip(self.ts, self.acc, self.gyr):
      print(f"ts: [{ts}], acc: {acc}, gyr: {gyr}")

    if extra_newline:
      print()


@dataclass
class ImuParams:
  """IMU parameters"""

  noise_acc: float
  noise_gyr: float
  noise_ba: float
  noise_bg: float
  g: VecN = field(default_factory=lambda : np.array([0.0, 0.0, 9.81]))


@dataclass
class ImuFactorData:
  """IMU Factor data"""

  state_F: VecN
  state_P: VecN
  dr: VecN
  dv: VecN
  dC: VecN
  ba: VecN
  bg: VecN
  g: VecN = field(default_factory=lambda : np.array([0.0, 0.0, 9.81]))
  Dt: float | np.float64 = 0.0


@dataclass
class ImuFactorData2:
  """IMU Factor2 data"""

  state_F: VecN
  state_P: VecN
  dr: VecN
  dv: VecN
  dq: VecN
  ba: VecN
  bg: VecN
  g: VecN = field(default_factory=lambda : np.array([0.0, 0.0, 9.81]))
  Dt: float | np.float64 = 0.0


class ImuFactor(Factor):
  """Imu Factor"""
  def __init__(self, pids, imu_params, imu_buf, sb_i):
    assert len(pids) == 4
    self.imu_params = imu_params
    self.imu_buf = imu_buf

    data = self.propagate(imu_buf, imu_params, sb_i)
    Factor.__init__(self, "ImuFactor", pids, None, data.state_P, 15)

    self.state_F = data.state_F
    self.state_P = data.state_P
    self.dr = data.dr
    self.dv = data.dv
    self.dC = data.dC
    self.ba = data.ba
    self.bg = data.bg
    self.g = data.g
    self.Dt = data.Dt

  @staticmethod
  def propagate(imu_buf, imu_params, sb_i):
    """Propagate imu measurements"""
    # Setup
    Dt = 0.0
    g = imu_params.g
    state_F = eye(15)  # State jacobian
    state_P = zeros((15, 15))  # State covariance

    # Noise matrix Q
    Q = zeros((12, 12))
    Q[0:3, 0:3] = imu_params.noise_acc**2 * eye(3)
    Q[3:6, 3:6] = imu_params.noise_gyr**2 * eye(3)
    Q[6:9, 6:9] = imu_params.noise_ba**2 * eye(3)
    Q[9:12, 9:12] = imu_params.noise_bg**2 * eye(3)

    # Pre-integrate relative position, velocity, rotation and biases
    dr = np.array([0.0, 0.0, 0.0])  # Relative position
    dv = np.array([0.0, 0.0, 0.0])  # Relative velocity
    dC = eye(3)  # Relative rotation
    ba_i = sb_i.param[3:6]  # Accel biase at i
    bg_i = sb_i.param[6:9]  # Gyro biase at i
    ba = sb_i.param[3:6]  # Accel biase
    bg = sb_i.param[6:9]  # Gyro biase

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
      dr = dr + (dv * dt) + (0.5 * dC @ (acc_i - ba_i) * dt_sq)
      dv = dv + dC @ (acc_i - ba_i) * dt
      dC = dC @ Exp((gyr_i - bg_i) * dt)
      ba = ba_i
      bg = bg_i

      # Make sure determinant of rotation is 1 by normalizing the quaternion
      dq = quat_normalize(rot2quat(dC))
      dC = quat2rot(dq)

      # Continuous time transition matrix F
      F = zeros((15, 15))
      F[0:3, 3:6] = eye(3)
      F[3:6, 6:9] = -1.0 * dC @ hat(acc_i - ba_i)
      F[3:6, 9:12] = -1.0 * dC
      F[6:9, 6:9] = -1.0 * hat(gyr_i - bg_i)
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
      state_F = I_F_dt @ state_F
      state_P = I_F_dt @ state_P @ I_F_dt.T + G_dt @ Q @ G_dt.T
      Dt += dt

    state_P = (state_P + state_P.T) / 2.0
    return ImuFactorData(state_F, state_P, dr, dv, dC, ba, bg, g, Dt)

  def eval(self, params, **kwargs):
    """Evaluate IMU factor"""
    assert len(params) == 4
    assert len(params[0]) == 7
    assert len(params[1]) == 9
    assert len(params[2]) == 7
    assert len(params[3]) == 9

    # Map params
    pose_i, sb_i, pose_j, sb_j = params

    # Timestep i
    T_i = pose2tf(pose_i)
    r_i = tf_trans(T_i)
    C_i = tf_rot(T_i)
    q_i = tf_quat(T_i)
    v_i = sb_i[0:3]
    ba_i = sb_i[3:6]
    bg_i = sb_i[6:9]

    # Timestep j
    T_j = pose2tf(pose_j)
    r_j = tf_trans(T_j)
    C_j = tf_rot(T_j)
    q_j = tf_quat(T_j)
    v_j = sb_j[0:3]

    # Correct the relative position, velocity and orientation
    # -- Extract jacobians from error-state jacobian
    dr_dba = self.state_F[0:3, 9:12]
    dr_dbg = self.state_F[0:3, 12:15]
    dv_dba = self.state_F[3:6, 9:12]
    dv_dbg = self.state_F[3:6, 12:15]
    dq_dbg = self.state_F[6:9, 12:15]
    dba = ba_i - self.ba
    dbg = bg_i - self.bg

    # -- Correct the relative position, velocity and rotation
    dr = self.dr + dr_dba @ dba + dr_dbg @ dbg
    dv = self.dv + dv_dba @ dba + dv_dbg @ dbg
    dC = self.dC @ Exp(dq_dbg @ dbg)
    dq = quat_normalize(rot2quat(dC))

    # Form residuals
    assert self.sqrt_info is not None
    sqrt_info = self.sqrt_info
    g = self.g
    Dt = self.Dt
    Dt_sq = Dt * Dt

    dr_meas = C_i.T @ ((r_j - r_i) - (v_i * Dt) + (0.5 * g * Dt_sq))
    dv_meas = C_i.T @ ((v_j - v_i) + (g * Dt))

    err_pos = dr_meas - dr
    err_vel = dv_meas - dv
    err_rot = (2.0 * quat_mul(quat_inv(dq), quat_mul(quat_inv(q_i), q_j)))[1:4]
    err_ba = np.array([0.0, 0.0, 0.0])
    err_bg = np.array([0.0, 0.0, 0.0])
    r = sqrt_info @ np.block([err_pos, err_vel, err_rot, err_ba, err_bg])

    if kwargs.get("only_residuals", False):
      return r

    # Form jacobians
    J0 = zeros((15, 6))  # residuals w.r.t pose i
    J1 = zeros((15, 9))  # residuals w.r.t speed and biase i
    J2 = zeros((15, 6))  # residuals w.r.t pose j
    J3 = zeros((15, 9))  # residuals w.r.t speed and biase j

    # -- Jacobian w.r.t. pose i
    # yapf: disable
    J0[0:3, 0:3] = -C_i.T  # dr w.r.t r_i
    J0[0:3, 3:6] = hat(dr_meas)  # dr w.r.t C_i
    J0[3:6, 3:6] = hat(dv_meas)  # dv w.r.t C_i
    J0[6:9, 3:6] = -1.0 * (quat_left(rot2quat(C_j.T @ C_i)) @ quat_right(dq))[1:4, 1:4]  # dtheta w.r.t C_i
    J0 = sqrt_info @ J0
    # yapf: enable

    # -- Jacobian w.r.t. speed and biases i
    # yapf: disable
    J1[0:3, 0:3] = -C_i.T * Dt  # dr w.r.t v_i
    J1[0:3, 3:6] = -dr_dba  # dr w.r.t ba
    J1[0:3, 6:9] = -dr_dbg  # dr w.r.t bg
    J1[3:6, 0:3] = -C_i.T  # dv w.r.t v_i
    J1[3:6, 3:6] = -dv_dba  # dv w.r.t ba
    J1[3:6, 6:9] = -dv_dbg  # dv w.r.t bg
    J1[6:9, 6:9] = -quat_left(rot2quat(C_j.T @ C_i @ self.dC))[1:4, 1:4] @ dq_dbg  # dtheta w.r.t C_i
    J1 = sqrt_info @ J1
    # yapf: enable

    # -- Jacobian w.r.t. pose j
    # yapf: disable
    J2[0:3, 0:3] = C_i.T  # dr w.r.t r_j
    J2[6:9, 3:6] = quat_left(rot2quat(dC.T @ C_i.T @ C_j))[1:4, 1:4]  # dtheta w.r.t C_j
    J2 = sqrt_info @ J2
    # yapf: enable

    # -- Jacobian w.r.t. sb j
    J3[3:6, 0:3] = C_i.T  # dv w.r.t v_j
    J3 = sqrt_info @ J3

    return (r, [J0, J1, J2, J3])


class ImuFactor2(Factor):
  """Imu Factor2"""
  def __init__(self, pids, imu_params, imu_buf, sb_i):
    assert len(pids) == 4
    self.imu_params = imu_params
    self.imu_buf = imu_buf

    data = self.propagate(imu_buf, imu_params, sb_i)
    Factor.__init__(self, "ImuFactor", pids, None, data.state_P, 15)

    self.state_F = data.state_F
    self.state_P = data.state_P
    self.dr = data.dr
    self.dv = data.dv
    self.dq = data.dq
    self.ba = data.ba
    self.bg = data.bg
    self.g = data.g
    self.Dt = data.Dt

  @staticmethod
  def propagate(imu_buf, imu_params, sb_i):
    """Propagate imu measurements"""
    # Setup
    Dt = 0.0
    g = imu_params.g
    state_F = eye(15)  # State jacobian
    state_P = zeros((15, 15))  # State covariance

    # Noise matrix Q
    Q = zeros((18, 18))
    Q[0:3, 0:3] = imu_params.noise_acc**2 * eye(3)
    Q[3:6, 3:6] = imu_params.noise_gyr**2 * eye(3)
    Q[6:9, 6:9] = imu_params.noise_acc**2 * eye(3)
    Q[9:12, 9:12] = imu_params.noise_gyr**2 * eye(3)
    Q[12:15, 12:15] = imu_params.noise_ba**2 * eye(3)
    Q[15:18, 15:18] = imu_params.noise_bg**2 * eye(3)

    # Pre-integrate relative position, velocity, rotation and biases
    dr = np.array([0.0, 0.0, 0.0])  # Relative position
    dv = np.array([0.0, 0.0, 0.0])  # Relative velocity
    dq = np.array([1.0, 0.0, 0.0, 0.0])  # Relative rotation
    ba = sb_i.param[3:6]  # Accel biase at i
    bg = sb_i.param[6:9]  # Gyro biase at i

    # Pre-integrate imu measuremenets
    for k in range(len(imu_buf.ts) - 1):
      # Timestep
      ts_i = imu_buf.ts[k]
      ts_j = imu_buf.ts[k + 1]
      dt = ts2sec(ts_j - ts_i)
      dt_sq = dt * dt

      # Setup
      dq_i = dq
      dr_i = dr
      dv_i = dv
      ba_i = ba
      bg_i = bg

      # Gyroscope measurement
      w = 0.5 * (imu_buf.gyr[k] + imu_buf.gyr[k + 1]) - bg_i
      dq_perturb = np.array([1.0, w[0] * dt / 2, w[1] * dt / 2, w[2] * dt / 2])

      # Accelerometer measurement
      acc_i = quat_rot(dq_i, imu_buf.acc[k] - ba_i)
      acc_j = quat_rot(quat_mul(dq_i, dq_perturb), (imu_buf.acc[k + 1] - ba_i))
      a = 0.5 * (acc_i + acc_j)

      # Propagate IMU state using mid-point method
      dq_j = quat_mul(dq_i, dq_perturb)
      dr_j = dr_i + dv_i * dt + 0.5 * a * dt_sq
      dv_j = dv_i + a * dt
      ba_j = ba_i
      bg_j = bg_i

      # Continuous time transition matrix F
      gyr_x = hat(0.5 * (imu_buf.gyr[k] + imu_buf.gyr[k + 1]) - bg_i)
      acc_i_x = hat(imu_buf.acc[k] - ba_i)
      acc_j_x = hat(imu_buf.acc[k + 1] - ba_i)
      dC_i = quat2rot(dq_i)
      dC_j = quat2rot(dq_j)

      # -- F row block 1
      F11 = eye(3)
      F12 = -0.25 * dC_i @ acc_i_x * dt_sq
      F12 += -0.25 * dC_j @ acc_j_x @ (eye(3) - gyr_x * dt) * dt_sq
      F13 = eye(3) * dt
      F14 = -0.25 * (dC_i + dC_j) * dt_sq
      F15 = 0.25 * -dC_j @ acc_j_x * dt_sq * -dt
      # -- F row block 2
      F22 = eye(3) - gyr_x * dt
      F25 = -eye(3) * dt
      # -- F row block 3
      F32 = -0.5 * dC_i @ acc_i_x * dt
      F32 += -0.5 * dC_j @ acc_j_x @ (eye(3) - gyr_x * dt) * dt
      F33 = eye(3)
      F34 = -0.5 * (dC_i + dC_j) * dt
      F35 = 0.5 * -dC_j @ acc_j_x * dt * -dt
      # -- F row block 4
      F44 = eye(3)
      # -- F row block 5
      F55 = eye(3)

      F = zeros((15, 15))

      F[0:3, 0:3] = F11
      F[0:3, 3:6] = F12
      F[0:3, 6:9] = F13
      F[0:3, 9:12] = F14
      F[0:3, 12:15] = F15
      F[3:6, 3:6] = F22
      F[3:6, 12:15] = F25
      F[6:9, 3:6] = F32
      F[6:9, 6:9] = F33
      F[6:9, 9:12] = F34
      F[6:9, 12:15] = F35
      F[9:12, 9:12] = F44
      F[12:15, 12:15] = F55

      # Continuous time input jacobian G
      G11 = 0.25 * dC_i * dt_sq
      G12 = 0.25 * -dC_j @ acc_i_x * dt_sq * 0.5 * dt
      G13 = 0.25 * dC_j @ acc_i_x * dt_sq
      G14 = 0.25 * -dC_j @ acc_i_x * dt_sq * 0.5 * dt
      G22 = eye(3) * dt
      G24 = eye(3) * dt
      G31 = 0.5 * dC_i * dt
      G32 = 0.5 * -dC_j @ acc_i_x * dt * 0.5 * dt
      G33 = 0.5 * dC_j * dt
      G34 = 0.5 * -dC_j @ acc_i_x * dt * 0.5 * dt
      G45 = eye(3) * dt
      G56 = eye(3) * dt

      G = zeros((15, 18))
      G[0:3, 0:3] = G11
      G[0:3, 3:6] = G12
      G[0:3, 6:9] = G13
      G[0:3, 9:12] = G14
      G[3:6, 3:6] = G22
      G[3:6, 9:12] = G24
      G[6:9, 0:3] = G31
      G[6:9, 3:6] = G32
      G[6:9, 6:9] = G33
      G[6:9, 9:12] = G34
      G[9:12, 12:15] = G45
      G[12:15, 15:18] = G56

      # Map results
      dq = dq_j
      dr = dr_j
      dv = dv_j
      ba = ba_j
      bg = bg_j

      # Update
      state_F = F @ state_F
      state_P = F @ state_P @ F.T + G @ Q @ G.T
      Dt += dt

    # Enforce semi-positive-definite
    state_P = (state_P + state_P.T) / 2.0

    return ImuFactorData2(state_F, state_P, dr, dv, dq, ba, bg, g, Dt)

  def eval(self, params, **kwargs):
    """Evaluate IMU factor"""
    assert len(params) == 4
    assert len(params[0]) == 7
    assert len(params[1]) == 9
    assert len(params[2]) == 7
    assert len(params[3]) == 9

    # Map params
    pose_i, sb_i, pose_j, sb_j = params

    # Timestep i
    T_i = pose2tf(pose_i)
    r_i = tf_trans(T_i)
    C_i = tf_rot(T_i)
    q_i = tf_quat(T_i)
    v_i = sb_i[0:3]
    ba_i = sb_i[3:6]
    bg_i = sb_i[6:9]

    # Timestep j
    T_j = pose2tf(pose_j)
    r_j = tf_trans(T_j)
    C_j = tf_rot(T_j)
    q_j = tf_quat(T_j)
    v_j = sb_j[0:3]
    ba_j = sb_j[3:6]
    bg_j = sb_j[6:9]

    # Correct the relative position, velocity and orientation
    # -- Extract jacobians from error-state jacobian
    dr_dba = self.state_F[0:3, 9:12]
    dr_dbg = self.state_F[0:3, 12:15]
    dq_dbg = self.state_F[3:6, 12:15]
    dv_dba = self.state_F[6:9, 9:12]
    dv_dbg = self.state_F[6:9, 12:15]
    dba = ba_i - self.ba
    dbg = bg_i - self.bg

    # -- Correct the relative position, velocity and rotation
    dr = self.dr + dr_dba @ dba + dr_dbg @ dbg
    dv = self.dv + dv_dba @ dba + dv_dbg @ dbg
    dq = quat_mul(self.dq, quat_delta(dq_dbg @ dbg))

    # Form residuals
    assert self.sqrt_info is not None
    sqrt_info = self.sqrt_info
    g = self.g
    Dt = self.Dt
    Dt_sq = Dt * Dt

    dr_meas = C_i.T @ ((r_j - r_i) - (v_i * Dt) + (0.5 * g * Dt_sq))
    dv_meas = C_i.T @ ((v_j - v_i) + (g * Dt))

    err_pos = dr_meas - dr
    err_vel = dv_meas - dv
    err_rot = (2.0 * quat_mul(quat_inv(dq), quat_mul(quat_inv(q_i), q_j)))[1:4]
    err_ba = ba_j - ba_i
    err_bg = bg_j - bg_i
    r = sqrt_info @ np.block([err_pos, err_vel, err_rot, err_ba, err_bg])

    if kwargs.get("only_residuals", False):
      return r

    # Form jacobians
    J0 = zeros((15, 6))  # residuals w.r.t pose i
    J1 = zeros((15, 9))  # residuals w.r.t speed and biase i
    J2 = zeros((15, 6))  # residuals w.r.t pose j
    J3 = zeros((15, 9))  # residuals w.r.t speed and biase j

    # -- Jacobian w.r.t. pose i
    # yapf: disable
    J0[0:3, 0:3] = -C_i.T  # dr w.r.t r_i
    J0[0:3, 3:6] = hat(dr_meas)  # dr w.r.t C_i
    J0[3:6, 3:6] = hat(dv_meas)  # dv w.r.t C_i
    J0[6:9, 3:6] = -1.0 * (quat_left(rot2quat(C_j.T @ C_i)) @ quat_right(dq))[1:4, 1:4]  # dtheta w.r.t C_i
    J0 = sqrt_info @ J0
    # yapf: enable

    # -- Jacobian w.r.t. speed and biases i
    # yapf: disable
    J1[0:3, 0:3] = -C_i.T * Dt  # dr w.r.t v_i
    J1[0:3, 3:6] = -dr_dba  # dr w.r.t ba
    J1[0:3, 6:9] = -dr_dbg  # dr w.r.t bg
    J1[3:6, 0:3] = -C_i.T  # dv w.r.t v_i
    J1[3:6, 3:6] = -dv_dba  # dv w.r.t ba
    J1[3:6, 6:9] = -dv_dbg  # dv w.r.t bg
    J1[6:9, 6:9] = -quat_left(rot2quat(C_j.T @ C_i @ quat2rot(self.dq)))[1:4, 1:4] @ dq_dbg  # dtheta w.r.t C_i
    J1[9:12, 3:6] = -eye(3)
    J1[12:15, 6:9] = -eye(3)
    J1 = sqrt_info @ J1
    # yapf: enable

    # -- Jacobian w.r.t. pose j
    # yapf: disable
    J2[0:3, 0:3] = C_i.T  # dr w.r.t r_j
    J2[6:9, 3:6] = quat_left(rot2quat(quat2rot(dq).T @ C_i.T @ C_j))[1:4, 1:4]  # dtheta w.r.t C_j
    J2 = sqrt_info @ J2
    # yapf: enable

    # -- Jacobian w.r.t. sb j
    J3[3:6, 0:3] = C_i.T  # dv w.r.t v_j
    J3[9:12, 3:6] = eye(3)
    J3[12:15, 6:9] = eye(3)
    J3 = sqrt_info @ J3

    return (r, [J0, J1, J2, J3])


# class LidarFactor(Factor):
#   """Lidar Factor"""
#   def __init__(self, pids, map, lidar_scan):
#     assert len(pids) == 3
#     assert map is not None
#     self.map = map
#     self.lidar_scan = lidar_scan
#     Factor.__init__(self, "LidarFactor", pids, None, None, None)
#
#   def eval(self, params, **kwargs):
#     """Evaluate"""
#     assert self.sqrt_info is not None
#     assert len(params) == 3
#
#     # Map params
#     pose, extrinsic = params
#     T_world_body = pose2tf(pose)
#     T_body_lidar = pose2tf(extrinsic)
#     T_world_lidar = T_world_body @ T_body_lidar
#     C_world_lidar = tf_rot(T_world_lidar)
#     r_world_lidar = tf_trans(T_world_lidar)
#
#     # Find closest points and transform to world frame
#     # pts_world = (C_world_lidar @ pts_lidar) + r_world_lidar[:, np.newaxis]
#
#     # Setup
#     num_points = 0
#     r = []
#     J0 = zeros((3, 6))
#     J1 = zeros((3, 6))
#
#     # for i in range(N):
#     # r.append(p_gnd[:, i] - p_est[:, i])
#     # J0[0:3, 0:3] += -1.0 * eye(3)
#     # J0[0:3, 3:6] += C_world_lidar @ hat(p_est[:, i])
#     # J1[0:3, 0:3] += -1.0 * eye(3)
#     # J1[0:3, 3:6] += C_world_lidar @ hat(p_est[:, i])
#
#     if kwargs.get("only_residuals", False):
#       return r
#
#     return (r, [J0, J1])


class MargFactor(Factor):
  """Marginalization Factor"""
  def __init__(self):
    Factor.__init__(self, "MargFactor", [], None, None, None)

    self.marg_param_ids = set()  # Parameters to be marginalized
    self.remain_param_ids = set()  # Parameters to remain
    self.factors = []  # Factors to be marginalized
    self.param_idxs = {}  # Param column indicies

    self.r0 = None  # Linearized residuals
    self.J0 = None  # Linearized jacobians
    self.x0 = {}  # Linearized estimates

  def add_factor(self, factor, marg_param_idxs):
    """Add factors to be marginalized"""
    # Track factor
    self.factors.append(factor)

    # Track parameters to remain / marginalized
    for idx, param_id in enumerate(factor.param_ids):
      if idx not in marg_param_idxs:
        self.remain_param_ids.add(param_id)
      else:
        self.marg_param_ids.add(param_id)

  def _linearize(self, param_blocks):
    """Linearize Nonlinear System"""
    #  Determine parameter block column indicies for Hessian matrix H
    H_idx = 0
    marg_size = 0
    remain_size = 0
    marg_params = []
    remain_params = []

    # -- Column indices for parameter blocks to be marginalized
    for param_id in self.marg_param_ids:
      param_block = param_blocks[param_id]
      self.param_idxs[param_id] = H_idx
      H_idx += param_block.min_dims
      marg_size += param_block.min_dims
      marg_params.append(param_block)

    # -- Column indices for parameter blocks to remain
    for param_id in self.remain_param_ids:
      param_block = param_blocks[param_id]
      self.param_idxs[param_id] = H_idx
      H_idx += param_block.min_dims
      remain_size += param_block.min_dims
      remain_params.append(param_block)

    # Form the H and g. Left and RHS of Gauss-Newton
    # H = J.T * J
    # g = -J.T * e
    param_size = marg_size + remain_size
    H = zeros((param_size, param_size))
    g = zeros(param_size)
    for factor in self.factors:
      factor_params = [param_blocks[pid].param for pid in factor.param_ids]
      r, jacobians = factor.eval(factor_params)

      # Form Hessian
      nb_params = len(factor_params)
      for i in range(nb_params):
        param_i = param_blocks[factor.param_ids[i]]
        if param_i.fix:
          continue
        idx_i = self.param_idxs[factor.param_ids[i]]
        size_i = param_i.min_dims
        J_i = jacobians[i]

        for j in range(i, nb_params):
          param_j = param_blocks[factor.param_ids[j]]
          if param_j.fix:
            continue
          idx_j = self.param_idxs[factor.param_ids[j]]
          size_j = param_j.min_dims
          J_j = jacobians[j]

          rs = idx_i
          re = idx_i + size_i
          cs = idx_j
          ce = idx_j + size_j

          if i == j:  # Diagonal
            H[rs:re, cs:ce] += J_i.T @ J_j
          else:  # Off-Diagonal
            H[rs:re, cs:ce] += J_i.T @ J_j
            H[cs:ce, rs:re] += (J_i.T @ J_j).T

        # Form R.H.S. Gauss Newton g
        rs = idx_i
        re = idx_i + size_i
        g[rs:re] += -J_i.T @ r

    return (H, g, marg_size, remain_size)

  @staticmethod
  def _shur_complement(H, b, m, eps=1e-8):
    """Schur complement"""
    # H = [H_mm, H_mr,
    #      H_rm, H_rr]
    H_mm = H[:m, :m]
    H_mr = H[:m, m:]
    H_rm = H[m:, :m]
    H_rr = H[m:, m:]

    # b = [b_mm,
    #      b_rr]
    b_mm = b[:m]
    b_rr = b[m:]

    # Invert H_mm matrix sub-block via Eigen-decomposition
    H_mm = 0.5 * (H_mm + H_mm.T)  # Enforce symmetry
    w, V = eig(H_mm)
    w_inv = np.zeros(w.shape)

    for idx, w_i in enumerate(w):
      if w_i > eps:
        w_inv[idx] = 1.0 / w_i
      else:
        w[idx] = 0.0
        w_inv[idx] = 0.0

    Lambda_inv = np.diag(w_inv)
    H_mm_inv = V @ Lambda_inv @ V.T

    # Check inverse
    check_inverse = True
    if check_inverse:
      inv_norm = np.linalg.norm((H_mm @ H_mm_inv) - eye(H_mm.shape[0]))
      if inv_norm > 1e-8:
        print("Hmmm... inverse check failed!")

    # Apply Shur-Complement
    H_marg = H_rr - H_rm @ H_mm_inv @ H_mr
    b_marg = b_rr - H_rm @ H_mm_inv @ b_mm

    return (H_marg, b_marg)

  @staticmethod
  def _decomp_hessian(H, eps=1e-12):
    """Decompose Hessian into J.T * J"""
    # Decompose Hessian via Eigen-Decomposition
    w, V = np.linalg.eigh(H)

    # Check eigen-values
    w_inv = np.zeros(w.shape)
    for idx, w_i in enumerate(w):
      if w_i > eps:
        w_inv[idx] = 1.0 / w_i
      else:
        w[idx] = 0.0
        w_inv[idx] = 0.0

    # Form J.T and J
    S_sqrt = np.diag(np.sqrt(w))
    S_inv_sqrt = np.diag(np.sqrt(w_inv))
    J = S_sqrt @ V.T
    J_inv = S_inv_sqrt @ V.T

    # Check decomposition
    # decomp_norm = np.linalg.norm((J.T @ J) - H)

    return J, J_inv

  def _calc_delta_chi(self, params):
    """Calculate Delta Chi"""
    dchi = np.array([])

    for param in params:
      x0 = self.x0[param.param_id]
      x = param.param

      if param.var_type in ["pose", "extrinsics"]:
        # Map out pose vector [rx, ry, rz, qx, qy, qz, qw]
        # into translation and rotation components
        # -- First linearization point
        r0 = x0[0:3]
        q0 = np.array([x0[6], x0[3], x0[4], x0[5]])
        # -- Current linearization point
        r = x[0:3]
        q = np.array([x[6], x[3], x[4], x[5]])

        # Calculate delta chi
        dr = r - r0
        dq = quat_mul(q, quat_inv(q0))
        dchi = np.append(dchi, [dr, dq[3], dq[0], dq[1], dq[2]])

      else:
        dchi = np.append(dchi, x - x0)

    return dchi

  def marginalize(self, param_blocks):
    """Marginalize"""
    # Marginalize
    H, g, marg_size, _ = self._linearize(param_blocks)
    H_marg, b_marg = self._shur_complement(H, g, marg_size)
    J, J_inv = self._decomp_hessian(H_marg)

    # First-Estimate Jacobians (FEJ)
    # Keep track of:
    # - First linearized Jacobians
    # - First linearized residuals
    # - First linearized state variable estimates
    self.J0 = J  # Linearized jacobians
    self.r0 = -1.0 * J_inv @ b_marg  # Linearized residuals
    for param_id in self.remain_param_ids:
      param_block = param_blocks[param_id]
      self.x0[param_id] = param_block.param

  def eval(self, params, **kwargs):
    """Evaluate Marginalization Factor"""
    assert self.r0
    assert self.J0

    # Calculate residuals
    dchi = self._calc_delta_chi(params)
    r = self.r0 + self.J0 @ dchi
    if kwargs.get("only_residuals", False):
      return r

    # Get First-Estimate Jacobians
    jacs = []
    r_size = r.shape[0]
    for param in params:
      J_idx = self.param_idxs[param.param_id]
      J_size = param.min_dims
      J_param = self.J0[J_idx:J_idx + r_size, J_idx:J_idx + J_size]
      jacs.append(J_param)

    return (r, jacs)


class TestPoseFactor(unittest.TestCase):
  """Test Pose factor"""
  def test_pose_factor(self):
    """Test pose factor"""
    # Setup camera pose T_WC
    rot = euler2quat(-pi / 2.0, 0.0, -pi / 2.0)
    trans = np.array([0.1, 0.2, 0.3])
    T_WC = tf(rot, trans)

    rot = euler2quat(-pi / 2.0 + 0.01, 0.0 + 0.01, -pi / 2.0 + 0.01)
    trans = np.array([0.1 + 0.01, 0.2 + 0.01, 0.3 + 0.01])
    T_WC_diff = tf(rot, trans)
    pose_est = pose_setup(0, T_WC_diff)

    # Create factor
    param_ids = [0]
    covar = eye(6)
    factor = PoseFactor(param_ids, T_WC, covar)

    # Test jacobians
    fvars = [pose_est]
    self.assertTrue(factor.check_jacobian(fvars, 0, "J_pose"))


class TestBAFactor(unittest.TestCase):
  """Test BA factor"""
  def test_ba_factor(self):
    """Test ba factor"""
    # Setup camera pose T_WC
    rot = euler2quat(-pi / 2.0, 0.0, -pi / 2.0)
    trans = np.array([0.1, 0.2, 0.3])
    T_WC = tf(rot, trans)
    cam_pose = pose_setup(0, T_WC)

    # Setup cam0
    cam_idx = 0
    img_w = 640
    img_h = 480
    res = [img_w, img_h]
    fov = 60.0
    fx = focal_length(img_w, fov)
    fy = focal_length(img_h, fov)
    cx = img_w / 2.0
    cy = img_h / 2.0
    params = [fx, fy, cx, cy, -0.01, 0.01, 1e-4, 1e-4]
    cam_params = camera_params_setup(cam_idx, res, "pinhole", "radtan4", params)
    cam_geom = camera_geometry_setup(cam_idx, res, "pinhole", "radtan4")

    # Setup feature
    p_W = np.array([10, random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)])
    # -- Feature XYZ parameterization
    feature = feature_setup(p_W)
    # # -- Feature inverse depth parameterization
    # param = idp_param(camera, T_WC, z)
    # feature = feature_init(0, param)
    # -- Calculate image point
    p_C = tf_point(inv(T_WC), p_W)
    status, z = cam_geom.project(cam_params.param, p_C)
    self.assertTrue(status)

    # Setup factor
    param_ids = [0, 1, 2]
    factor = BAFactor(cam_geom, param_ids, z)

    # Test jacobians
    fvars = [cam_pose, feature, cam_params]
    self.assertTrue(factor.check_jacobian(fvars, 0, "J_cam_pose"))
    self.assertTrue(factor.check_jacobian(fvars, 1, "J_feature"))
    self.assertTrue(factor.check_jacobian(fvars, 2, "J_cam_params"))

    # params = [sv.param for sv in fvars]
    # r, jacs = factor.eval(params)
    # for J in jacs:
    #   print(rank(J), J.shape)


class TestVisionFactor(unittest.TestCase):
  """Test Vision factor"""
  def test_vision_factor(self):
    """Test vision factor"""
    # Setup camera pose T_WB
    rot = euler2quat(0.01, 0.01, 0.03)
    trans = np.array([0.001, 0.002, 0.003])
    T_WB = tf(rot, trans)
    pose = pose_setup(0, T_WB)

    # Setup camera extrinsics T_BCi
    rot = euler2quat(-pi / 2.0, 0.0, -pi / 2.0)
    trans = np.array([0.1, 0.2, 0.3])
    T_BCi = tf(rot, trans)
    cam_exts = extrinsics_setup(T_BCi)

    # Setup cam0
    cam_idx = 0
    img_w = 640
    img_h = 480
    res = [img_w, img_h]
    fov = 60.0
    fx = focal_length(img_w, fov)
    fy = focal_length(img_h, fov)
    cx = img_w / 2.0
    cy = img_h / 2.0
    params = [fx, fy, cx, cy, -0.01, 0.01, 1e-4, 1e-4]
    cam_params = camera_params_setup(cam_idx, res, "pinhole", "radtan4", params)
    cam_geom = camera_geometry_setup(cam_idx, res, "pinhole", "radtan4")

    # Setup feature
    p_W = np.array([10, random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)])
    # -- Feature XYZ parameterization
    feature = feature_setup(p_W)
    # # -- Feature inverse depth parameterization
    # param = idp_param(camera, T_WC, z)
    # feature = feature_init(0, param)
    # -- Calculate image point
    T_WCi = T_WB @ T_BCi
    p_C = tf_point(inv(T_WCi), p_W)
    status, z = cam_geom.project(cam_params.param, p_C)
    self.assertTrue(status)

    # Setup factor
    param_ids = [0, 1, 2, 3]
    factor = VisionFactor(cam_geom, param_ids, z)

    # Test jacobians
    fvars = [pose, cam_exts, feature, cam_params]
    self.assertTrue(factor.check_jacobian(fvars, 0, "J_pose"))
    self.assertTrue(factor.check_jacobian(fvars, 1, "J_cam_exts"))
    self.assertTrue(factor.check_jacobian(fvars, 2, "J_feature"))
    self.assertTrue(factor.check_jacobian(fvars, 3, "J_cam_params"))


class TestCalibVisionFactor(unittest.TestCase):
  """Test CalibVision factor"""
  def test_calib_vision_factor(self):
    """Test CalibVisionFactor"""
    # Calibration target pose T_WF
    C_WF = euler321(-pi / 2.0, 0.0, deg2rad(80.0))
    r_WF = np.array([0.001, 0.001, 0.001])
    T_WF = tf(C_WF, r_WF)

    # Body pose T_WB
    rot = euler2quat(-pi / 2.0, 0.0, -pi / 2.0)
    trans = np.array([-10.0, 0.0, 0.0])
    T_WB = tf(rot, trans)

    # Relative pose T_BF
    T_BF = inv(T_WB) @ T_WF

    # Camera extrinsics T_BCi
    rot = eye(3)
    trans = np.array([0.001, 0.002, 0.003])
    T_BCi = tf(rot, trans)

    # Camera 0
    cam_idx = 0
    img_w = 640
    img_h = 480
    res = [img_w, img_h]
    fov = 90.0
    fx = focal_length(img_w, fov)
    fy = focal_length(img_h, fov)
    cx = img_w / 2.0
    cy = img_h / 2.0
    params = [fx, fy, cx, cy, -0.01, 0.01, 1e-4, 1e-4]
    cam_params = camera_params_setup(cam_idx, res, "pinhole", "radtan4", params)
    cam_geom = camera_geometry_setup(cam_idx, res, "pinhole", "radtan4")

    # Test factor
    target = CalibTarget()
    tag_id = 1
    corner_idx = 2
    r_FFi = target.get_object_point(tag_id, corner_idx)
    T_CiF = inv(T_BCi) @ T_BF
    r_CiFi = tf_point(T_CiF, r_FFi)
    status, z = cam_geom.project(cam_params.param, r_CiFi)
    self.assertTrue(status)

    pids = [0, 1, 2]
    target_data = (tag_id, corner_idx, r_FFi, z)
    factor = CalibVisionFactor(cam_geom, pids, target_data)

    # Test jacobianstf(rot, trans)
    rel_pose = pose_setup(0, T_BF)
    cam_exts = extrinsics_setup(T_BCi)
    fvars = [rel_pose, cam_exts, cam_params]
    self.assertTrue(factor.check_jacobian(fvars, 0, "J_rel_pose"))
    self.assertTrue(factor.check_jacobian(fvars, 1, "J_cam_exts"))
    self.assertTrue(factor.check_jacobian(fvars, 2, "J_cam_params"))


class TestImuFactor(unittest.TestCase):
  """Test Imu factor"""
  def test_imu_buffer(self):
    """Test IMU Buffer"""
    # Extract measurements from ts: 4 - 7
    imu_buf = ImuBuffer()
    for k in range(10):
      ts = k
      acc = np.array([0.0 + k, 0.0 + k, 0.0 + k])
      gyr = np.array([0.0 + k, 0.0 + k, 0.0 + k])
      imu_buf.add(ts, acc, gyr)

    # print("Original imu_buf:")
    # imu_buf.print(True)
    imu_buf2 = imu_buf.extract(4, 7)
    # print("Extracted imu_buf2 (ts: 4 - 7):")
    # imu_buf2.print(True)
    # print("Modified imu_buf:")
    # imu_buf.print(True)

    self.assertTrue(imu_buf.length() == 4)
    self.assertTrue(imu_buf.ts[0] == 6)
    self.assertTrue(imu_buf.ts[-1] == 9)

    self.assertTrue(imu_buf2.length() == 4)
    self.assertTrue(imu_buf2.ts[0] == 4)
    self.assertTrue(imu_buf2.ts[-1] == 7)

  def test_imu_buffer_with_interpolation(self):
    """Test IMU Buffer with interpolation"""
    # Interpolation test
    imu_buf = ImuBuffer()
    for k in range(10):
      ts = k
      acc = np.array([0.0 + k, 0.0 + k, 0.0 + k])
      gyr = np.array([0.0 + k, 0.0 + k, 0.0 + k])
      imu_buf.add(ts, acc, gyr)

    # print("Original imu_buf:")
    # imu_buf.print(True)
    imu_buf2 = imu_buf.extract(4.25, 8.9)
    # print("Extracted imu_buf2 (ts: 4.25 - 8.9):")
    # imu_buf2.print(True)
    # print("Modified imu_buf:")
    # imu_buf.print(True)

    self.assertTrue(imu_buf.length() == 2)
    self.assertTrue(imu_buf.ts[0] == 8)
    self.assertTrue(imu_buf.ts[-1] == 9)

    self.assertTrue(imu_buf2.length() == 6)
    self.assertTrue(imu_buf2.ts[0] == 4.25)
    self.assertTrue(imu_buf2.ts[-1] == 8.9)

  def test_imu_factor_propagate(self):
    """Test IMU factor propagate"""
    # Sim imu data
    circle_r = 1.0
    circle_v = 0.1
    sim_data = SimData(circle_r, circle_v, sim_cams=False)
    imu_data = sim_data.imu0_data
    assert imu_data

    # Setup imu parameters
    noise_acc = 0.08  # accelerometer measurement noise stddev.
    noise_gyr = 0.004  # gyroscope measurement noise stddev.
    noise_ba = 0.00004  # accelerometer bias random work noise stddev.
    noise_bg = 2.0e-6  # gyroscope bias random work noise stddev.
    g = np.array([0.0, 0.0, 9.81])
    imu_params = ImuParams(noise_acc, noise_gyr, noise_ba, noise_bg, g)

    # Setup imu buffer
    start_idx = 0
    end_idx = 20
    # end_idx = len(imu_data.timestamps) - 1
    imu_buf = imu_data.form_imu_buffer(start_idx, end_idx)

    # Pose i
    ts_i = imu_buf.ts[start_idx]
    T_WS_i = imu_data.poses[ts_i]

    # Speed and bias i
    ts_i = imu_buf.ts[start_idx]
    vel_i = imu_data.vel[ts_i]
    ba_i = np.array([0.0, 0.0, 0.0])
    bg_i = np.array([0.0, 0.0, 0.0])
    sb_i = speed_biases_setup(ts_i, vel_i, bg_i, ba_i)

    # Propagate imu measurements
    data = ImuFactor.propagate(imu_buf, imu_params, sb_i)

    # Check propagation
    ts_j = imu_data.timestamps[end_idx - 1]
    T_WS_j_est = T_WS_i @ tf(data.dC, data.dr)
    C_WS_j_est = tf_rot(T_WS_j_est)
    T_WS_j_gnd = imu_data.poses[ts_j]
    C_WS_j_gnd = tf_rot(T_WS_j_gnd)
    # -- Position
    trans_diff = norm(tf_trans(T_WS_j_gnd) - tf_trans(T_WS_j_est))
    self.assertTrue(trans_diff < 0.05)
    # -- Rotation
    dC = C_WS_j_gnd.T * C_WS_j_est
    dq = quat_normalize(rot2quat(dC))
    dC = quat2rot(dq)
    rpy_diff = rad2deg(acos((trace(dC) - 1.0) / 2.0))
    self.assertTrue(rpy_diff < 1.0)

  def test_imu_factor(self):
    """Test IMU factor"""
    # Simulate imu data
    circle_r = 1.0
    circle_v = 0.1
    sim_data = SimData(circle_r, circle_v, sim_cams=False)
    imu_data = sim_data.imu0_data
    assert imu_data

    # Setup imu parameters
    noise_acc = 0.08  # accelerometer measurement noise stddev.
    noise_gyr = 0.004  # gyroscope measurement noise stddev.
    noise_ba = 0.00004  # accelerometer bias random work noise stddev.
    noise_bg = 2.0e-6  # gyroscope bias random work noise stddev.
    g = np.array([0.0, 0.0, 9.81])
    imu_params = ImuParams(noise_acc, noise_gyr, noise_ba, noise_bg, g)

    # Setup imu buffer
    start_idx = 0
    end_idx = 20
    imu_buf = imu_data.form_imu_buffer(start_idx, end_idx)

    # Pose i
    ts_i = imu_buf.ts[start_idx]
    T_WS_i = imu_data.poses[ts_i]
    pose_i = pose_setup(ts_i, T_WS_i)

    # Pose j
    ts_j = imu_buf.ts[end_idx - 1]
    T_WS_j = imu_data.poses[ts_j]
    pose_j = pose_setup(ts_j, T_WS_j)

    # Speed and bias i
    vel_i = imu_data.vel[ts_i]
    ba_i = np.array([0.0, 0.0, 0.0])
    bg_i = np.array([0.0, 0.0, 0.0])
    sb_i = speed_biases_setup(ts_i, vel_i, ba_i, bg_i)

    # Speed and bias j
    vel_j = imu_data.vel[ts_j]
    ba_j = np.array([0.0, 0.0, 0.0])
    bg_j = np.array([0.0, 0.0, 0.0])
    sb_j = speed_biases_setup(ts_j, vel_j, ba_j, bg_j)

    # Setup IMU factor
    param_ids = [0, 1, 2, 3]
    fvars = [pose_i, sb_i, pose_j, sb_j]
    factor = ImuFactor(param_ids, imu_params, imu_buf, sb_i)

    # Evaluate and obtain residuals, jacobians
    # params = [sv.param for sv in fvars]
    # (_, [J0, J1, J2, J3]) = factor.eval(params)

    # # Form Hessian
    # J = np.block([J0, J1, J2, J3])
    # H = J.T @ J

    # # Perform Schur Complement
    # m = 6 + 9
    # Hmm = H[0:m, 0:m]
    # Hmr = H[0:m, m:]
    # Hrm = H[m:, 0:m]
    # Hrr = H[m:, m:]
    # Hmm_inv = inv(Hmm)
    # H_marg = Hrr - Hrm @ Hmm_inv @ Hmr
    # print(f"rank(Hmm): {rank(Hmm)}")
    # print(f"rank(H_marg): {rank(H_marg)}")
    #
    # # Check inverse Hmm_inv
    # check_inverse = True
    # if check_inverse:
    #   inv_norm = np.linalg.norm((Hmm @ Hmm_inv) - eye(Hmm.shape[0]))
    #   if inv_norm > 1e-8:
    #     print("Hmmm... inverse check failed!")

    # Test jacobians
    factor.sqrt_info = eye(15)
    self.assertTrue(factor.check_jacobian(fvars, 0, "J_pose_i"))
    self.assertTrue(factor.check_jacobian(fvars, 1, "J_sb_i"))
    self.assertTrue(factor.check_jacobian(fvars, 2, "J_pose_j"))
    self.assertTrue(factor.check_jacobian(fvars, 3, "J_sb_j"))

  # def test_imu_propagation_jacobians(self):
  #   """Test IMU Propagation Jacobians"""
  #   # -- Setup
  #   dt = 0.001
  #   I3 = eye(3)
  #
  #   # -- State
  #   p = np.array([0.0, 0.0, 0.0])
  #   v = np.array([0.1, 0.2, 0.3])
  #   q = euler2quat(0.1, 0.2, 0.3)
  #   a_b = np.array([0.1, 0.2, 0.3])
  #   w_b = np.array([0.1, 0.2, 0.3])
  #   C = eye(3)
  #   g = np.array([0.0, 0.0, -10.0])
  #
  #   # -- Input
  #   a_m = np.array([0.1, 0.2, 10.0])
  #   w_m = np.array([0.1, 0.2, 0.3])
  #
  #   # -- Nominal state kinematics
  #   # p = p + v * dt + 0.5 * C * (a_m - a_b) + g) * dt**2
  #   # v = v + C * (a_m - a_b) + g) * dt
  #   # q = q * (w_m - w_b) * dt
  #   # a_b = 0
  #   # w_b = 0
  #   p_kp1 = p + v * dt + 0.5 * C @ ((a_m - a_b) + g) * dt**2
  #   v_kp1 = v + C @ ((a_m - a_b) + g) * dt
  #   q_kp1 = quat_mul(q, quat_delta((w_m - w_b) * dt))
  #
  #   # -- Error state kinematics
  #   # dp = dp + dv * dt
  #   # dv = dv + (-C * hat(a_m - a_b) * dtheta - C * da_b + dg) * dt
  #   # dtheta = C.T{w_m - w_b) * dt} * dtheta - dw_b * dt
  #   # da_b = a_w
  #   # dw_b = w_w
  #
  #   # -- Transition matrix F
  #   F = np.zeros((15, 15))
  #   # -- Row block 1
  #   F[0:3, 0:3] = I3
  #   F[0:3, 3:6] = I3 * dt
  #   # -- Row block 2
  #   F[3:6, 3:6] = I3
  #   F[3:6, 6:9] = -C @ (a_m - a_b) * dt
  #   F[3:6, 9:12] = -C * dt
  #   # -- Row block 3
  #   F[6:9, 6:9] = I3 - hat(w_m - w_b) * dt
  #   F[6:9, 12:15] = -I3 * dt
  #   # -- Row block 4
  #   F[9:12, 9:12] = I3
  #   # -- Row block 5
  #   F[12:15, 12:15] = I3

  def test_imu_factor2_propagate(self):
    """Test IMU factor propagate"""
    # Sim imu data
    circle_r = 1.0
    circle_v = 0.1
    sim_data = SimData(circle_r, circle_v, sim_cams=False)
    imu_data = sim_data.imu0_data
    assert imu_data

    # Setup imu parameters
    noise_acc = 0.08  # accelerometer measurement noise stddev.
    noise_gyr = 0.004  # gyroscope measurement noise stddev.
    noise_ba = 0.00004  # accelerometer bias random work noise stddev.
    noise_bg = 2.0e-6  # gyroscope bias random work noise stddev.
    g = np.array([0.0, 0.0, 9.81])
    imu_params = ImuParams(noise_acc, noise_gyr, noise_ba, noise_bg, g)

    # Setup imu buffer
    start_idx = 0
    end_idx = 10
    # end_idx = len(imu_data.timestamps) - 1
    imu_buf = imu_data.form_imu_buffer(start_idx, end_idx)

    # Pose i
    ts_i = imu_buf.ts[start_idx]
    T_WS_i = imu_data.poses[ts_i]

    # Speed and bias i
    ts_i = imu_buf.ts[start_idx]
    vel_i = imu_data.vel[ts_i]
    ba_i = np.array([0.0, 0.0, 0.0])
    bg_i = np.array([0.0, 0.0, 0.0])
    sb_i = speed_biases_setup(ts_i, vel_i, bg_i, ba_i)

    # Propagate imu measurements
    data = ImuFactor2.propagate(imu_buf, imu_params, sb_i)

    # Check propagation
    ts_j = imu_data.timestamps[end_idx]
    T_WS_j_est = T_WS_i @ tf(data.dq, data.dr)
    T_WS_j_gnd = imu_data.poses[ts_j]
    C_WS_j_est = tf_rot(T_WS_j_est)
    C_WS_j_gnd = tf_rot(T_WS_j_gnd)
    # -- Position
    trans_diff = norm(tf_trans(T_WS_j_gnd) - tf_trans(T_WS_j_est))
    self.assertTrue(trans_diff < 0.05)
    # -- Rotation
    dC = C_WS_j_gnd.T * C_WS_j_est
    dq = quat_normalize(rot2quat(dC))
    dC = quat2rot(dq)
    rpy_diff = rad2deg(acos((trace(dC) - 1.0) / 2.0))
    self.assertTrue(rpy_diff < 1.0)

  def test_imu_factor2(self):
    """Test IMU factor 2"""
    # Simulate imu data
    circle_r = 5.0
    circle_v = 1.0
    sim_data = SimData(circle_r, circle_v, sim_cams=False)
    imu_data = sim_data.imu0_data
    assert imu_data

    # Setup imu parameters
    noise_acc = 0.08  # accelerometer measurement noise stddev.
    noise_gyr = 0.004  # gyroscope measurement noise stddev.
    noise_ba = 0.00004  # accelerometer bias random work noise stddev.
    noise_bg = 2.0e-6  # gyroscope bias random work noise stddev.
    g = np.array([0.0, 0.0, 9.81])
    imu_params = ImuParams(noise_acc, noise_gyr, noise_ba, noise_bg, g)

    # Setup imu buffer
    start_idx = 0
    end_idx = 20
    imu_buf = imu_data.form_imu_buffer(start_idx, end_idx)

    # Pose i
    ts_i = imu_buf.ts[start_idx]
    T_WS_i = imu_data.poses[ts_i]
    pose_i = pose_setup(ts_i, T_WS_i)

    # Pose j
    ts_j = imu_buf.ts[end_idx - 1]
    T_WS_j = imu_data.poses[ts_j]
    pose_j = pose_setup(ts_j, T_WS_j)

    # Speed and bias i
    vel_i = imu_data.vel[ts_i]
    ba_i = np.array([0.0, 0.0, 0.0])
    bg_i = np.array([0.0, 0.0, 0.0])
    sb_i = speed_biases_setup(ts_i, vel_i, ba_i, bg_i)

    # Speed and bias j
    vel_j = imu_data.vel[ts_j]
    ba_j = np.array([0.0, 0.0, 0.0])
    bg_j = np.array([0.0, 0.0, 0.0])
    sb_j = speed_biases_setup(ts_j, vel_j, ba_j, bg_j)

    # Setup IMU factor
    param_ids = [0, 1, 2, 3]
    fvars = [pose_i, sb_i, pose_j, sb_j]
    factor = ImuFactor2(param_ids, imu_params, imu_buf, sb_i)

    # Test jacobians
    factor.sqrt_info = eye(15)
    self.assertTrue(factor.check_jacobian(fvars, 0, "J_pose_i"))
    self.assertTrue(factor.check_jacobian(fvars, 1, "J_sb_i"))
    self.assertTrue(factor.check_jacobian(fvars, 2, "J_pose_j"))
    self.assertTrue(factor.check_jacobian(fvars, 3, "J_sb_j"))


# class TestLidarFactor(unittest.TestCase):
#   """ Test Lidar factor """
#   def test_lidar_factor(self):


class TestMargFactor(unittest.TestCase):
  """Test Marg factor"""
  def test_marg_factor(self):
    """Test MargFactor"""
    # Setup cam0 parameters and geometry
    cam_idx = 0
    img_w = 640
    img_h = 480
    res = [img_w, img_h]
    fov = 60.0
    fx = focal_length(img_w, fov)
    fy = focal_length(img_h, fov)
    cx = img_w / 2.0
    cy = img_h / 2.0
    params = [fx, fy, cx, cy, -0.01, 0.01, 1e-4, 1e-4]
    cam_params = camera_params_setup(cam_idx, res, "pinhole", "radtan4", params)
    cam_geom = camera_geometry_setup(cam_idx, res, "pinhole", "radtan4")

    # Setup camera poses T_WC
    rot0 = euler2quat(-pi / 2.0, 0.0, -pi / 2.0)
    rot1 = euler2quat(-pi / 2.0 + 0.01, 0.0 + 0.01, -pi / 2.0 + 0.01)
    rot2 = euler2quat(-pi / 2.0 + 0.02, 0.0 + 0.02, -pi / 2.0 + 0.02)
    rot3 = euler2quat(-pi / 2.0 + 0.03, 0.0 + 0.03, -pi / 2.0 + 0.03)
    pos0 = np.array([0.1, 0.2, 0.3])
    pos1 = np.array([0.11, 0.21, 0.31])
    pos2 = np.array([0.12, 0.22, 0.32])
    pos3 = np.array([0.13, 0.23, 0.33])
    T_WC_0 = tf(rot0, pos0)
    T_WC_1 = tf(rot1, pos1)
    T_WC_2 = tf(rot2, pos2)
    T_WC_3 = tf(rot3, pos3)
    cam_tfs = [T_WC_0, T_WC_1, T_WC_2, T_WC_3]
    cam_pose_0 = pose_setup(0, T_WC_0)
    cam_pose_1 = pose_setup(1, T_WC_1)
    cam_pose_2 = pose_setup(2, T_WC_2)
    cam_pose_3 = pose_setup(3, T_WC_3)

    cam_pose_0.param[0] += random.uniform(-0.1, 0.1)
    cam_pose_0.param[1] += random.uniform(-0.1, 0.1)
    cam_pose_0.param[2] += random.uniform(-0.1, 0.1)

    cam_pose_1.param[0] += random.uniform(-0.1, 0.1)
    cam_pose_1.param[1] += random.uniform(-0.1, 0.1)
    cam_pose_1.param[2] += random.uniform(-0.1, 0.1)

    cam_pose_2.param[0] += random.uniform(-0.1, 0.1)
    cam_pose_2.param[1] += random.uniform(-0.1, 0.1)
    cam_pose_2.param[2] += random.uniform(-0.1, 0.1)

    cam_pose_3.param[0] += random.uniform(-0.1, 0.1)
    cam_pose_3.param[1] += random.uniform(-0.1, 0.1)
    cam_pose_3.param[2] += random.uniform(-0.1, 0.1)

    cam_poses = [cam_pose_0]

    # Setup feature
    nb_features = 10
    features = []
    feature_positions = []
    for i in range(nb_features):
      p_W = np.array([10, random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)])
      features.append(feature_setup(p_W))
      feature_positions.append(p_W)

    # Setup parameter blocks
    param_blocks = {}
    param_idx = 0
    param_ids = []
    # -- Camera poses
    for cam_pose in cam_poses:
      param_blocks[param_idx] = cam_pose
      param_ids.append(param_idx)
      param_idx += 1
    # -- Features
    for feature in features:
      param_blocks[param_idx] = feature
      param_ids.append(param_idx)
      param_idx += 1
    # -- Camera params
    param_blocks[param_idx] = cam_params
    param_ids.append(param_idx)

    # Setup BAFactors
    ba_factors = []
    for k, cam_pose in enumerate(cam_poses):  # Iterate through camera poses
      for i, feature in enumerate(features):  # Iterate through features
        # Calculate image point
        T_WC = cam_tfs[k]
        p_W = feature_positions[i]
        p_C = tf_point(inv(T_WC), p_W)
        status, z = cam_geom.project(cam_params.param, p_C)
        self.assertTrue(status)

        # Create BA Factor
        pids = [k, len(cam_poses) + i, param_ids[-1]]
        fvars = [cam_pose, feature, cam_params]
        factor = BAFactor(cam_geom, pids, z)
        ba_factors.append(factor)

        # Check BA factor jacobians
        self.assertTrue(factor)
        self.assertTrue(factor.check_jacobian(fvars, 0, "J_cam_pose"))
        self.assertTrue(factor.check_jacobian(fvars, 1, "J_feature"))
        self.assertTrue(factor.check_jacobian(fvars, 2, "J_cam_params"))

    # Setup MargFactor
    marg_factor = MargFactor()
    for ba_factor in ba_factors:
      # Form marginalization parameter index
      marg_param_idxs = []
      if ba_factor.param_ids[0] == 0:
        marg_param_idxs = [0]

      # Add BA factor to marginalization factor
      marg_factor.add_factor(ba_factor, marg_param_idxs)

    # Test Linearize
    H, g, m, r = marg_factor._linearize(param_blocks)
    # plt.matshow(H)
    # plt.colorbar()
    # plt.show()

    H_size = len(cam_poses) * 6 + len(features) * 3 + len(cam_params.param)
    g_size = H_size
    self.assertTrue(H.shape == (H_size, H_size))
    self.assertTrue(g.shape == (g_size,))
    self.assertTrue(H_size == (m + r))

    # # Check if Gauss-Newton is solve-able: H dx = g (solve for dx)
    # check_hessian = False
    # if check_hessian:
    #   # -- Get cost before Gauss-Newton step
    #   residuals = np.array([])
    #   for factor in marg_factor.factors:
    #     factor_params = [param_blocks[pid].param for pid in factor.param_ids]
    #     r = factor.eval(factor_params, only_residuals=True)
    #     residuals = np.append(residuals, r)
    #   cost = 0.5 * (r.T @ r)
    #   print(f"cost: {cost}")
    #   # -- Solve for dx
    #   lambda_k = 1e-4
    #   H = H + lambda_k * eye(H.shape[0])
    #   c, low = scipy.linalg.cho_factor(H)
    #   dx = scipy.linalg.cho_solve((c, low), g)
    #   # -- Update state-variables
    #   for param_id, param in param_blocks.items():
    #     start = param_idxs[param_id]
    #     end = start + param.min_dims
    #     param_dx = dx[start:end]
    #     update_state_variable(param, param_dx)
    #   # -- Get cost after Gauss-Newton step
    #   residuals = np.array([])
    #   for factor in marg_factor.factors:
    #     factor_params = [param_blocks[pid].param for pid in factor.param_ids]
    #     r = factor.eval(factor_params, only_residuals=True)
    #     residuals = np.append(residuals, r)
    #   cost = 0.5 * (r.T @ r)
    #   print(f"cost: {cost}")

    # Test Shurs-Complement
    H_marg, b_marg = marg_factor._shur_complement(H, g, m)
    self.assertTrue(H_marg.shape == (H_size - m, H_size - m))
    self.assertTrue(b_marg.shape == (H_size - m,))

    # Test marginalization
    marg_factor.marginalize(param_blocks)
    # self.assertEqual(len(marg_factor.remain_param_ids), 2)
    # self.assertEqual(len(marg_factor.marg_param_ids), 5)


# Factor Graph #################################################################


class FactorGraph:
  """FactorGraph"""
  def __init__(self):
    self._next_param_id = 0
    self._next_factor_id = 0
    self.params = {}
    self.factors = {}
    self.solver_max_iter = 5
    self.solver_lambda = 1e4

  def add_param(self, param):
    """Add param"""
    param_id = self._next_param_id
    self.params[param_id] = param
    self.params[param_id].set_param_id(param_id)
    self._next_param_id += 1
    return param_id

  def add_factor(self, factor):
    """Add factor"""
    # Double check if params exists
    for param_id in factor.param_ids:
      if param_id not in self.params:
        raise RuntimeError(f"Parameter [{param_id}] does not exist!")

    # Add factor
    factor_id = self._next_factor_id
    self.factors[factor_id] = factor
    self.factors[factor_id].set_factor_id(factor_id)
    self._next_factor_id += 1
    return factor_id

  def remove_param(self, param):
    """Remove param"""
    assert param.param_id in self.params
    del self.params[param.param_id]

  def remove_factor(self, factor):
    """Remove factor"""
    assert factor.factor_id in self.factors
    del self.factors[factor.factor_id]

  @staticmethod
  def _print_to_console(iter_k, lambda_k, cost_kp1, cost_k):
    """Print to console"""

    print(f"iter[{iter_k}]:", end=" ")
    print(f"lambda: {lambda_k:.2e}", end=", ")
    print(f"cost: {cost_kp1:.2e}", end=", ")
    print(f"dcost: {cost_kp1 - cost_k:.2e}", end=" ")
    print()

    # status, rmse_vision = rmse(self._get_reproj_errors())
    # print(f"rms_reproj_error: {rmse_vision:.2f} px")

    sys.stdout.flush()

  def _form_param_indices(self):
    """Form parameter indices"""
    # Parameter ids
    param_ids = {
      "pose": set(),
      "speed_and_biases": set(),
      "feature": set(),
      "camera": set(),
      "extrinsics": set(),
      "joint_angle": set(),
    }

    # Track parameters
    nb_params = 0
    for _, factor in self.factors.items():
      for _, param_id in enumerate(factor.param_ids):
        param = self.params[param_id]
        if param.fix:
          continue
        else:
          param_ids[param.var_type].add(param_id)
        nb_params += 1

    # Assign global parameter order
    param_order = []
    param_order.append("joint_angle")
    param_order.append("pose")
    param_order.append("speed_and_biases")
    param_order.append("feature")
    param_order.append("camera")
    param_order.append("extrinsics")

    param_idxs = {}
    param_size = 0
    for param_type in param_order:
      for param_id in param_ids[param_type]:
        param_idxs[param_id] = param_size
        param_size += self.params[param_id].min_dims

    return (param_idxs, param_size)

  def _linearize(self, params):
    """Linearize non-linear problem"""
    # Setup
    (param_idxs, param_size) = self._form_param_indices()
    H = zeros((param_size, param_size))
    g = zeros(param_size)

    # Form Hessian and R.H.S of Gauss newton
    for _, factor in self.factors.items():
      factor_params = [params[pid].param for pid in factor.param_ids]
      r, jacobians = factor.eval(factor_params)

      # Form Hessian
      nb_params = len(factor_params)
      for i in range(nb_params):
        param_i = params[factor.param_ids[i]]
        if param_i.fix:
          continue
        idx_i = param_idxs[factor.param_ids[i]]
        size_i = param_i.min_dims
        J_i = jacobians[i]

        for j in range(i, nb_params):
          param_j = params[factor.param_ids[j]]
          if param_j.fix:
            continue
          idx_j = param_idxs[factor.param_ids[j]]
          size_j = param_j.min_dims
          J_j = jacobians[j]

          rs = idx_i
          re = idx_i + size_i
          cs = idx_j
          ce = idx_j + size_j

          if i == j:  # Diagonal
            H[rs:re, cs:ce] += J_i.T @ J_j
          else:  # Off-Diagonal
            H[rs:re, cs:ce] += J_i.T @ J_j
            H[cs:ce, rs:re] += (J_i.T @ J_j).T

        # Form R.H.S. Gauss Newton g
        rs = idx_i
        re = idx_i + size_i
        g[rs:re] += -J_i.T @ r

    return (H, g, param_idxs)

  def _linearize2(self, params):
    """Linearize non-linear problem

    This function forms the Jacboian J instead of the Hessian H.

    """
    # Setup
    (param_idxs, param_size) = self._form_param_indices()
    r_size = np.sum([f.r_size for _, f in self.factors.items()])
    J = zeros((r_size, param_size))
    g = zeros(param_size)

    # Form Jacobian and R.H.S of Gauss newton
    J_idx = 0
    for _, factor in self.factors.items():
      factor_params = [params[pid].param for pid in factor.param_ids]
      r, jacobians = factor.eval(factor_params)
      J_rs = J_idx
      J_re = J_rs + len(r)

      # Form Jacobian
      nb_params = len(factor_params)
      for i in range(nb_params):
        param_i = params[factor.param_ids[i]]
        if param_i.fix:
          continue
        idx_i = param_idxs[factor.param_ids[i]]
        size_i = param_i.min_dims
        J_i = jacobians[i]

        J_cs = idx_i
        J_ce = J_cs + size_i
        J[J_rs:J_re, J_cs:J_ce] = J_i

        # Form R.H.S. Gauss Newton g
        rs = idx_i
        re = idx_i + size_i
        g[rs:re] += -J_i.T @ r

      # Update row index
      J_idx = J_rs + len(r)

    # print(f"J.shape: {J.shape}")
    return (J, g, param_idxs)

  def _calculate_residuals(self, params):
    """Calculate Residuals"""
    residuals = np.array([])

    for _, factor in self.factors.items():
      factor_params = [params[pid].param for pid in factor.param_ids]
      r = factor.eval(factor_params, only_residuals=True)
      residuals = np.append(residuals, r)

    return residuals

  def _calculate_cost(self, params):
    """Calculate Cost"""
    r = self._calculate_residuals(params)
    return 0.5 * (r.T @ r)

  @staticmethod
  def _update(params_k, param_idxs, dx):
    """Update"""
    params_kp1 = copy.deepcopy(params_k)

    for param_id, param in params_kp1.items():
      # Check if param even exists
      if param_id not in param_idxs:
        continue

      # Update parameter
      start = param_idxs[param_id]
      end = start + param.min_dims
      param_dx = dx[start:end]
      update_state_variable(param, param_dx)

    return params_kp1

  @staticmethod
  def _solve_for_dx(lambda_k, H, g):
    """Solve for dx"""
    # Damp Hessian
    H_damped = H + lambda_k * eye(H.shape[0])
    # H_damped = H + lambda_k * np.diag(H.diagonal())

    # # Pseudo inverse
    # dx = pinv(H) @ g

    # Cholesky decomposition
    dx = None
    try:
      c, low = scipy.linalg.cho_factor(H_damped)
      dx = scipy.linalg.cho_solve((c, low), g)
    except:
      dx = np.zeros((H_damped.shape[0],))

    # SVD
    # dx = solve_svd(H_damped, g)

    # # QR
    # q, r = np.linalg.qr(H_damped)
    # p = np.dot(q.T, g)
    # dx = np.dot(np.linalg.inv(r), p)

    # Sparse cholesky decomposition
    # sH = scipy.sparse.csc_matrix(H_damped)
    # dx = scipy.sparse.linalg.spsolve(sH, g)

    return dx

  def solve(self, verbose=False):
    """Solve"""
    lambda_k = self.solver_lambda
    params_k = copy.deepcopy(self.params)
    cost_k = self._calculate_cost(params_k)

    # First evaluation
    if verbose:
      print(f"num_factors: {len(self.factors)}")
      print(f"num_params: {len(self.params)}")
      self._print_to_console(0, lambda_k, cost_k, cost_k)

    # Iterate
    success = False
    H = None
    g = None
    param_idxs = None

    for i in range(1, self.solver_max_iter):
      # Update and calculate cost
      if i == 1 or success:
        (H, g, param_idxs) = self._linearize(params_k)

      dx = self._solve_for_dx(lambda_k, H, g)
      params_kp1 = self._update(params_k, param_idxs, dx)
      cost_kp1 = self._calculate_cost(params_kp1)

      # Verbose
      if verbose:
        self._print_to_console(i, lambda_k, cost_kp1, cost_k)

      # Accept or reject update
      dcost = cost_kp1 - cost_k
      success = dcost < 0

      if success:
        # Accept update
        cost_k = cost_kp1
        params_k = params_kp1
        lambda_k /= 10.0

      else:
        # Reject update
        # params_k = params_k
        lambda_k *= 10.0

      # # Termination criteria
      # if dcost > -1e-5:
      #   break

    # Finish - set the original params the optimized values
    # Note: The reason we don't just do `self.params = params_k` is because
    # that would destroy the references to outside `FactorGraph()`.
    for param_id, param in params_k.items():
      self.params[param_id].param = param.param


class TestFactorGraph(unittest.TestCase):
  """Test Factor Graph"""
  def setUp(self):
    circle_r = 5.0
    circle_v = 1.0
    pickle_path = "/tmp/sim_data.pickle"
    self.sim_data = SimData.create_or_load(circle_r, circle_v, pickle_path)

  def test_add_param(self):
    """Test graph.add_param()"""
    # Setup camera pose T_WC
    rot = euler2quat(-pi / 2.0, 0.0, -pi / 2.0)
    trans = np.array([0.1, 0.2, 0.3])
    T_WC = tf(rot, trans)
    pose0 = pose_setup(0, T_WC)
    pose1 = pose_setup(1, T_WC)

    # Add params
    graph = FactorGraph()
    pose0_id = graph.add_param(pose0)
    pose1_id = graph.add_param(pose1)

    # Assert
    self.assertEqual(pose0_id, 0)
    self.assertEqual(pose1_id, 1)
    self.assertNotEqual(pose0, pose1)
    self.assertEqual(graph.params[pose0_id], pose0)
    self.assertEqual(graph.params[pose1_id], pose1)

  def test_add_factor(self):
    """Test graph.add_factor()"""
    # Setup factor graph
    graph = FactorGraph()

    # Setup camera pose T_WC
    rot = euler2quat(-pi / 2.0, 0.0, -pi / 2.0)
    trans = np.array([0.1, 0.2, 0.3])
    T_WC = tf(rot, trans)
    pose = pose_setup(0, T_WC)
    pose_id = graph.add_param(pose)

    # Create factor
    param_ids = [pose_id]
    covar = eye(6)
    pose_factor = PoseFactor(param_ids, T_WC, covar)
    pose_factor_id = graph.add_factor(pose_factor)

    # Assert
    self.assertEqual(len(graph.params), 1)
    self.assertEqual(len(graph.factors), 1)
    self.assertEqual(graph.factors[pose_factor_id], pose_factor)

  @unittest.skip("")
  def test_solve_vo(self):
    """Test solving a visual odometry problem"""
    # Sim data
    cam0_data = self.sim_data.get_camera_data(0)
    cam0_params = self.sim_data.get_camera_params(0)
    cam0_geom = self.sim_data.get_camera_geometry(0)

    # Setup factor graph
    poses_gnd = []
    poses_init = []
    poses_est = []
    graph = FactorGraph()

    # -- Add features
    features = self.sim_data.features
    feature_ids = []
    for i in range(features.shape[0]):
      p_W = features[i, :]
      # p_W += np.random.rand(3) * 0.1  # perturb feature
      feature = feature_setup(p_W, fix=True)
      feature_ids.append(graph.add_param(feature))

    # -- Add cam0
    cam0_id = graph.add_param(cam0_params)

    # -- Build bundle adjustment problem
    nb_poses = 0
    for ts in cam0_data.timestamps:
      # Camera frame at ts
      cam_frame = cam0_data.frames[ts]

      # Add camera pose T_WC0
      T_WC0_gnd = cam0_data.poses[ts]
      # -- Perturb camera pose
      trans_rand = np.random.rand(3)
      rvec_rand = np.random.rand(3) * 0.05
      T_WC0_init = tf_update(T_WC0_gnd, np.block([*trans_rand, *rvec_rand]))
      # -- Add to graph
      pose = pose_setup(ts, T_WC0_init)
      pose_id = graph.add_param(pose)
      poses_gnd.append(T_WC0_gnd)
      poses_init.append(T_WC0_init)
      poses_est.append(pose_id)
      nb_poses += 1

      # Add ba factors
      for i, idx in enumerate(cam_frame.feature_ids):
        z = cam_frame.measurements[i]
        param_ids = [pose_id, feature_ids[idx], cam0_id]
        graph.add_factor(BAFactor(cam0_geom, param_ids, z))

    # Solve
    # debug = True
    debug = False
    # prof = profile_start()
    graph.solve(debug)
    # profile_stop(prof)

    # Visualize
    if debug:
      pos_gnd = np.array([tf_trans(T) for T in poses_gnd])
      pos_init = np.array([tf_trans(T) for T in poses_init])
      pos_est = []
      for pose_pid in poses_est:
        pose = graph.params[pose_pid]
        pos_est.append(tf_trans(pose2tf(pose.param)))
      pos_est = np.array(pos_est)

      plt.figure()
      plt.plot(pos_gnd[:, 0], pos_gnd[:, 1], "g-", label="Ground Truth")
      plt.plot(pos_init[:, 0], pos_init[:, 1], "r-", label="Initial")
      plt.plot(pos_est[:, 0], pos_est[:, 1], "b-", label="Estimated")
      plt.xlabel("Displacement [m]")
      plt.ylabel("Displacement [m]")
      plt.legend(loc=0)
      plt.show()

    # Asserts
    # errors = graph.get_reproj_errors()
    # self.assertTrue(rmse(errors) < 0.1)

  @unittest.skip("")
  def test_solve_io(self):
    """Test solving a pure inertial odometry problem"""
    # Imu params
    noise_acc = 0.08  # accelerometer measurement noise stddev.
    noise_gyr = 0.004  # gyroscope measurement noise stddev.
    noise_ba = 0.00004  # accelerometer bias random work noise stddev.
    noise_bg = 2.0e-6  # gyroscope bias random work noise stddev.
    g = np.array([0.0, 0.0, 9.81])
    imu_params = ImuParams(noise_acc, noise_gyr, noise_ba, noise_bg, g)

    # Setup factor graph
    imu0_data = self.sim_data.imu0_data
    assert imu0_data
    assert imu0_data.timestamps
    window_size = 20
    start_idx = 0
    # end_idx = 200
    # end_idx = 2000
    # end_idx = int((len(imu0_data.timestamps) - 1) / 2.0)
    end_idx = len(imu0_data.timestamps)

    poses_init = []
    poses_est = []
    poses_gnd = []
    sb_est = []
    sb_gnd = []
    graph = FactorGraph()

    # -- Pose i
    ts_i = imu0_data.timestamps[start_idx]
    T_WS_i = imu0_data.poses[ts_i]
    pose_i = pose_setup(ts_i, T_WS_i)
    pose_i_id = graph.add_param(pose_i)
    poses_init.append(T_WS_i)
    poses_est.append(pose_i_id)
    poses_gnd.append(T_WS_i)

    # -- Speed and biases i
    vel_i = imu0_data.vel[ts_i]
    ba_i = np.array([0.0, 0.0, 0.0])
    bg_i = np.array([0.0, 0.0, 0.0])
    sb_i = speed_biases_setup(ts_i, vel_i, ba_i, bg_i)
    sb_i_id = graph.add_param(sb_i)
    sb_est.append(sb_i_id)
    sb_gnd = [[*vel_i, *ba_i, *bg_i]]

    for ts_idx in range(start_idx + window_size, end_idx, window_size):
      # -- Pose j
      ts_j = imu0_data.timestamps[ts_idx]
      T_WS_j_gnd = imu0_data.poses[ts_j]
      # ---- Pertrub pose j
      trans_rand = np.random.rand(3) * 0.5
      rvec_rand = np.random.rand(3) * 0.01
      T_WS_j = tf_update(T_WS_j_gnd, np.block([*trans_rand, *rvec_rand]))
      # ---- Add to factor graph
      pose_j = pose_setup(ts_j, T_WS_j)
      pose_j_id = graph.add_param(pose_j)

      # -- Speed and biases j
      vel_j = imu0_data.vel[ts_j]
      ba_j = np.array([0.0, 0.0, 0.0])
      bg_j = np.array([0.0, 0.0, 0.0])
      sb_j = speed_biases_setup(ts_j, vel_j, ba_j, bg_j)
      sb_j_id = graph.add_param(sb_j)

      # ---- Keep track of initial and estimate pose
      poses_init.append(T_WS_j)
      poses_est.append(pose_j_id)
      poses_gnd.append(T_WS_j_gnd)
      sb_est.append(sb_j_id)
      sb_gnd.append([*vel_j, *ba_j, *bg_j])

      # -- Imu Factor
      param_ids = [pose_i_id, sb_i_id, pose_j_id, sb_j_id]
      imu_buf = imu0_data.form_imu_buffer(ts_idx - window_size, ts_idx)
      # factor = ImuFactor(param_ids, imu_params, imu_buf, sb_i) # Euler method
      factor = ImuFactor2(param_ids, imu_params, imu_buf,
                          sb_i)  # Midpoint method
      graph.add_factor(factor)

      # -- Update
      pose_i_id = pose_j_id
      pose_i = pose_j
      sb_i_id = sb_j_id
      sb_i = sb_j

    # Solve
    debug = False
    prof = profile_start()
    graph.solver_max_iter = 10
    graph.solve(debug)
    profile_stop(prof)

    if debug:
      pos_init = np.array([tf_trans(T) for T in poses_init])
      pos_gnd = np.array([tf_trans(T) for T in poses_gnd])

      pos_est = []
      for pose_pid in poses_est:
        pose = graph.params[pose_pid]
        pos_est.append(tf_trans(pose2tf(pose.param)))
      pos_est = np.array(pos_est)

      sb_gnd = np.array(sb_gnd)
      sb_est = [graph.params[pid] for pid in sb_est]
      sb_ts0 = sb_est[0].ts
      sb_time = np.array([ts2sec(sb.ts - sb_ts0) for sb in sb_est])
      vel_est = np.array([sb.param[0:3] for sb in sb_est])
      ba_est = np.array([sb.param[3:6] for sb in sb_est])
      bg_est = np.array([sb.param[6:9] for sb in sb_est])
      vel_gnd = np.array([sb[0:3] for sb in sb_gnd])
      ba_gnd = np.array([sb[3:6] for sb in sb_gnd])
      bg_gnd = np.array([sb[6:9] for sb in sb_gnd])

      # Plot X-Y position
      plt.figure()
      plt.plot(pos_init[:, 0], pos_init[:, 1], "r-", alpha=0.2, label="Initial")
      plt.plot(pos_gnd[:, 0], pos_gnd[:, 1], "k--", label="Ground-Truth")
      plt.plot(pos_est[:, 0], pos_est[:, 1], "b-", label="Estimate")
      plt.legend(loc=0)
      plt.xlabel("Displacement [m]")
      plt.ylabel("Displacement [m]")

      # Plot velocity
      plt.figure()
      plt.subplot(311)
      plt.plot(sb_time, vel_gnd[:, 0], "k-", label="Ground-Truth")
      plt.plot(sb_time, vel_est[:, 0], "r-", label="Estimate")
      plt.xlabel("Time [s]")
      plt.ylabel("Velocity [ms^-1]")
      plt.subplot(312)
      plt.plot(sb_time, vel_gnd[:, 1], "k-", label="Ground-Truth")
      plt.plot(sb_time, vel_est[:, 1], "g-", label="Estimate")
      plt.xlabel("Time [s]")
      plt.ylabel("Velocity [ms^-1]")
      plt.subplot(313)
      plt.plot(sb_time, vel_gnd[:, 2], "k-", label="Ground-Truth")
      plt.plot(sb_time, vel_est[:, 2], "b-", label="Estimate")
      plt.xlabel("Time [s]")
      plt.ylabel("Velocity [ms^-1]")

      # Plot accelerometer bias
      plt.figure()
      plt.subplot(311)
      plt.plot(sb_time, ba_gnd[:, 0], "k-", label="Ground-Truth")
      plt.plot(sb_time, ba_est[:, 0], "r-", label="Estimate")
      plt.xlabel("Time [s]")
      plt.ylabel("Accelerometer Bias [m s^-2]")
      plt.subplot(312)
      plt.plot(sb_time, ba_gnd[:, 1], "k-", label="Ground-Truth")
      plt.plot(sb_time, ba_est[:, 1], "g-", label="Estimate")
      plt.xlabel("Time [s]")
      plt.ylabel("Accelerometer Bias [m s^-2]")
      plt.subplot(313)
      plt.plot(sb_time, ba_gnd[:, 2], "k-", label="Ground-Truth")
      plt.plot(sb_time, ba_est[:, 2], "b-", label="Estimate")
      plt.xlabel("Time [s]")
      plt.ylabel("Accelerometer Bias [m s^-2]")

      # Plot gyroscope bias
      plt.figure()
      plt.subplot(311)
      plt.plot(sb_time, bg_gnd[:, 0], "k-", label="Ground-Truth")
      plt.plot(sb_time, bg_est[:, 0], "r-", label="Estimate")
      plt.xlabel("Time [s]")
      plt.ylabel("Gyroscope Bias [rad s^-1]")
      plt.subplot(312)
      plt.plot(sb_time, bg_gnd[:, 1], "k-", label="Ground-Truth")
      plt.plot(sb_time, bg_est[:, 1], "g-", label="Estimate")
      plt.xlabel("Time [s]")
      plt.ylabel("Gyroscope Bias [rad s^-1]")
      plt.subplot(313)
      plt.plot(sb_time, bg_gnd[:, 2], "k-", label="Ground-Truth")
      plt.plot(sb_time, bg_est[:, 2], "b-", label="Estimate")
      plt.xlabel("Time [s]")
      plt.ylabel("Gyroscope Bias [rad s^-1]")

      plt.show()

  @unittest.skip("")
  def test_solve_vio(self):
    """Test solving a visual inertial odometry problem"""
    # Imu params
    noise_acc = 0.08  # accelerometer measurement noise stddev.
    noise_gyr = 0.004  # gyroscope measurement noise stddev.
    noise_ba = 0.00004  # accelerometer bias random work noise stddev.
    noise_bg = 2.0e-6  # gyroscope bias random work noise stddev.
    g = np.array([0.0, 0.0, 9.81])
    imu_params = ImuParams(noise_acc, noise_gyr, noise_ba, noise_bg, g)

    # Sim data
    cam_idx = 0
    cam_data = self.sim_data.get_camera_data(cam_idx)
    cam_params = self.sim_data.get_camera_params(cam_idx)
    cam_geom = self.sim_data.get_camera_geometry(cam_idx)
    cam_exts = self.sim_data.get_camera_extrinsics(cam_idx)
    cam_params.fix = True
    cam_exts.fix = True

    # Setup factor graph
    poses_gnd = []
    poses_init = []
    poses_est = []
    graph = FactorGraph()
    graph.solver_lambda = 1e4
    graph.solver_max_iter = 10

    # -- Add features
    features = self.sim_data.features
    feature_ids = []
    for i in range(features.shape[0]):
      p_W = features[i, :]
      # p_W += np.random.rand(3) * 0.1  # perturb feature
      feature = feature_setup(p_W, fix=False)
      feature_ids.append(graph.add_param(feature))

    # -- Add cam
    cam_id = graph.add_param(cam_params)
    exts_id = graph.add_param(cam_exts)
    T_BC_gnd = pose2tf(cam_exts.param)
    T_CB_gnd = inv(T_BC_gnd)

    # -- Build bundle adjustment problem
    assert self.sim_data.imu0_data
    imu_data = ImuBuffer()
    poses = []
    sbs = []

    for ts_k in self.sim_data.timeline.get_timestamps():
      for event in self.sim_data.timeline.get_events(ts_k):
        if isinstance(event, ImuEvent):
          imu_data.add_event(event)

        elif isinstance(event, CameraEvent) and event.cam_idx == cam_idx:
          if imu_data.length() == 0:
            continue

          # Vision factors
          # -- Add camera pose T_WC
          T_WC_gnd = cam_data.poses[ts_k]
          T_WB_gnd = T_WC_gnd @ T_CB_gnd
          # ---- Perturb camera pose
          trans_rand = np.random.rand(3) * 0.5
          rvec_rand = np.random.rand(3) * 0.5
          T_perturb = np.block([*trans_rand, *rvec_rand])
          T_WB_init = tf_update(T_WB_gnd, T_perturb)
          # T_WB_init = T_WB_gnd
          # ---- Add to graph
          pose = pose_setup(ts_k, T_WB_init)
          poses.append(pose)
          pose_id = graph.add_param(pose)
          poses_gnd.append(T_WB_gnd)
          poses_init.append(T_WB_init)
          poses_est.append(pose_id)
          # -- Speed and biases
          vel_j = self.sim_data.imu0_data.vel[ts_k]
          ba_j = np.array([0.0, 0.0, 0.0])
          bg_j = np.array([0.0, 0.0, 0.0])
          sb = speed_biases_setup(ts_k, vel_j, bg_j, ba_j)
          graph.add_param(sb)
          sbs.append(sb)
          # -- Add vision factors
          for i, idx in enumerate(cam_data.frames[ts_k].feature_ids):
            z = cam_data.frames[ts_k].measurements[i]
            param_ids = [pose_id, exts_id, feature_ids[idx], cam_id]
            graph.add_factor(VisionFactor(cam_geom, param_ids, z))

          # Imu factor
          if len(poses) >= 2:
            ts_km1 = poses[-2].ts
            pose_i_id = poses[-2].param_id
            pose_j_id = poses[-1].param_id
            sb_i_id = sbs[-2].param_id
            sb_j_id = sbs[-1].param_id
            param_ids = [pose_i_id, sb_i_id, pose_j_id, sb_j_id]

            if ts_k <= imu_data.ts[-1]:
              imu_buf = imu_data.extract(ts_km1, ts_k)
              graph.add_factor(
                ImuFactor2(param_ids, imu_params, imu_buf, sbs[-2]))

      if len(poses) > 20:
        break

    # Solve
    # debug = True
    debug = False
    # prof = profile_start()
    graph.solve(debug)
    # profile_stop(prof)

    # Visualize
    if debug:
      pos_gnd = np.array([tf_trans(T) for T in poses_gnd])
      pos_init = np.array([tf_trans(T) for T in poses_init])
      pos_est = []
      for pose_pid in poses_est:
        pose = graph.params[pose_pid]
        pos_est.append(tf_trans(pose2tf(pose.param)))
      pos_est = np.array(pos_est)

      plt.figure()
      plt.plot(pos_gnd[:, 0], pos_gnd[:, 1], "g-", label="Ground Truth")
      plt.plot(pos_init[:, 0], pos_init[:, 1], "r-", label="Initial")
      plt.plot(pos_est[:, 0], pos_est[:, 1], "b-", label="Estimated")
      plt.xlabel("Displacement [m]")
      plt.ylabel("Displacement [m]")
      plt.legend(loc=0)
      plt.show()

    # Asserts
    # errors = graph.get_reproj_errors()
    # self.assertTrue(rmse(errors) < 0.1)


###############################################################################
# FEATURE TRACKING
###############################################################################


def draw_matches(img_i, img_j, kps_i, kps_j):
  """
  Draw keypoint matches between images `img_i` and `img_j` with keypoints
  `kps_i` and `kps_j`
  """
  assert len(kps_i) == len(kps_j)

  nb_kps = len(kps_i)
  viz = cv2.hconcat([img_i, img_j])  # pyright: ignore
  if len(viz.shape) != 3:
    viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2RGB)

  color = (0, 255, 0)
  radius = 1
  thickness = cv2.FILLED
  linetype = cv2.LINE_AA

  for n in range(nb_kps):
    pt_i = None
    pt_j = None
    if hasattr(kps_i[n], "pt"):
      pt_i = (int(kps_i[n].pt[0]), int(kps_i[n].pt[1]))
      pt_j = (int(kps_j[n].pt[0] + img_i.shape[1]), int(kps_j[n].pt[1]))
    else:
      pt_i = (int(kps_i[n][0]), int(kps_i[n][1]))
      pt_j = (int(kps_j[n][0] + img_i.shape[1]), int(kps_j[n][1]))

    cv2.circle(viz, pt_i, radius, color, thickness, lineType=linetype)
    cv2.circle(viz, pt_j, radius, color, thickness, lineType=linetype)
    cv2.line(viz, pt_i, pt_j, color, 1, linetype)

  return viz


def draw_keypoints(
    img,
    kps,
    inliers=None,
    radius = 1,
    color = (0, 255, 0),
):
  """
  Draw points `kps` on image `img`. The `inliers` boolean list is optional
  and is expected to be the same size as `kps` denoting whether the point
  should be drawn or not.
  """
  thickness = cv2.FILLED
  linetype = cv2.LINE_AA
  inliers = [1 for _ in range(len(kps))] if inliers is None else inliers

  viz = img
  if len(img.shape) == 2:
    viz = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

  for n, kp in enumerate(kps):
    if inliers[n]:
      p = None
      if hasattr(kp, "pt"):
        p = (int(kp.pt[0]), int(kp.pt[1]))
      else:
        p = (int(kp[0]), int(kp[1]))

      cv2.circle(viz, p, radius, color, thickness, lineType=linetype)

  return viz


def sort_keypoints(kps, des=None):
  """Sort a list of cv2.KeyPoint based on their response"""
  responses = [kp.response for kp in kps]
  indices = range(len(responses))
  indices = sorted(indices, key=lambda i: responses[i], reverse=True)
  if des is None:
    return [kps[i] for i in indices]

  kps_sorted = []
  des_sorted = np.zeros(des.shape)
  for i in range(des.shape[0]):
    des_sorted[i, :] = des[indices[i], :]
    kps_sorted.append(kps[indices[i]])

  return kps_sorted, des_sorted


def spread_corners(img, corners, min_dist, **kwargs):
  """
  Given a set of corners `corners` make sure they are atleast `min_dist` pixels
  away from each other, if they are not remove them.
  """
  # Pre-check
  if not corners:
    return corners

  # Setup
  debug = kwargs.get("debug", False)
  prev_corners = kwargs.get("prev_corners", [])
  min_dist = int(min_dist)
  img_h, img_w = img.shape
  A = np.zeros(img.shape)  # Allowable areas are marked 0 else not allowed

  # Loop through previous keypoints
  for c in prev_corners:
    # Convert from keypoint to tuple
    p = (int(c[0]), int(c[1]))

    # Fill the area of the matrix where the next keypoint cannot be around
    rs = int(max(p[1] - min_dist, 0.0))
    re = int(min(p[1] + min_dist + 1, img_h))
    cs = int(max(p[0] - min_dist, 0.0))
    ce = int(min(p[0] + min_dist + 1, img_w))
    A[rs:re, cs:ce] = np.ones((re - rs, ce - cs))

  # Loop through keypoints
  corners_results = []
  for c in corners:
    # Convert from keypoint to tuple
    p = (int(c[0]), int(c[1]))

    # Check if point is ok to be added to results
    if A[p[1], p[0]] > 0.0:
      continue

    # Fill the area of the matrix where the next keypoint cannot be around
    rs = int(max(p[1] - min_dist, 0.0))
    re = int(min(p[1] + min_dist + 1, img_h))
    cs = int(max(p[0] - min_dist, 0.0))
    ce = int(min(p[0] + min_dist + 1, img_w))
    A[rs:re, cs:ce] = np.ones((re - rs, ce - cs))
    A[p[1], p[0]] = 2

    # Add to results
    corners_results.append(c)

  # Debug
  if debug:
    img = draw_keypoints(img, corners_results, radius=3)

    plt.figure()

    ax = plt.subplot(121)
    ax.imshow(A)
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax = plt.subplot(122)
    ax.imshow(img)
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    plt.show()

  return corners_results


def spread_keypoints(img, kps, min_dist, prev_kps=[], debug = False):
  """
  Given a set of keypoints `kps` make sure they are atleast `min_dist` pixels
  away from each other, if they are not remove them.
  """
  # Pre-check
  if not kps:
    return kps

  # Setup
  min_dist = int(min_dist)
  img_h, img_w = img.shape
  A = np.zeros(img.shape)  # Allowable areas are marked 0 else not allowed

  # Loop through previous keypoints
  for kp in prev_kps:
    # Convert from keypoint to tuple
    p = None
    if hasattr(kp, "pt"):
      p = (int(kp.pt[0]), int(kp.pt[1]))
    else:
      p = (int(kp[0]), int(kp[1]))

    # Fill the area of the matrix where the next keypoint cannot be around
    rs = int(max(p[1] - min_dist, 0.0))
    re = int(min(p[1] + min_dist + 1, img_h))
    cs = int(max(p[0] - min_dist, 0.0))
    ce = int(min(p[0] + min_dist + 1, img_w))
    A[rs:re, cs:ce] = np.ones((re - rs, ce - cs))

  # Loop through keypoints
  kps_results = []
  for kp in kps:
    # Convert from keypoint to tuple
    p = None
    if hasattr(kp, "pt"):
      p = (int(kp.pt[0]), int(kp.pt[1]))
    else:
      p = (int(kp[0]), int(kp[1]))

    # Check if point is ok to be added to results
    if A[p[1], p[0]] > 0.0:
      continue

    # Fill the area of the matrix where the next keypoint cannot be around
    rs = int(max(p[1] - min_dist, 0.0))
    re = int(min(p[1] + min_dist + 1, img_h))
    cs = int(max(p[0] - min_dist, 0.0))
    ce = int(min(p[0] + min_dist + 1, img_w))
    A[rs:re, cs:ce] = np.ones((re - rs, ce - cs))
    A[p[1], p[0]] = 2

    # Add to results
    kps_results.append(kp)

  # Debug
  if debug:
    img = draw_keypoints(img, kps_results, radius=3)

    plt.figure()

    ax = plt.subplot(121)
    ax.imshow(A)
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax = plt.subplot(122)
    ax.imshow(img)
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    plt.show()

  return kps_results


def spread_features(img, kps, des, min_dist, **kwargs):
  """
  Given a set of keypoints `kps` and descriptors are atleast `min_dist` pixels
  away from each other, if they are not remove them.
  """
  # Pre-check
  if not kps:
    return kps

  # Setup
  debug = kwargs.get("debug", False)
  prev_kps = kwargs.get("prev_kps", [])
  min_dist = int(min_dist)
  img_h, img_w = img.shape
  A = np.zeros(img.shape)  # Allowable areas are marked 0 else not allowed

  # Loop through previous keypoints
  for kp in prev_kps:
    # Convert from keypoint to tuple
    p = (int(kp.pt[0]), int(kp.pt[1]))

    # Fill the area of the matrix where the next keypoint cannot be around
    rs = int(max(p[1] - min_dist, 0.0))
    re = int(min(p[1] + min_dist + 1, img_h))
    cs = int(max(p[0] - min_dist, 0.0))
    ce = int(min(p[0] + min_dist + 1, img_w))
    A[rs:re, cs:ce] = np.ones((re - rs, ce - cs))

  # Loop through keypoints and descriptors
  kps_results = []
  des_results = []
  kps, des = sort_keypoints(kps, des)
  for kp, des in zip(kps, des):
    # Convert from keypoint to tuple
    p = (int(kp.pt[0]), int(kp.pt[1]))

    # Check if point is ok to be added to results
    if A[p[1], p[0]] > 0.0:
      continue

    # Fill the area of the matrix where the next keypoint cannot be around
    rs = int(max(p[1] - min_dist, 0.0))
    re = int(min(p[1] + min_dist + 1, img_h))
    cs = int(max(p[0] - min_dist, 0.0))
    ce = int(min(p[0] + min_dist + 1, img_w))
    A[rs:re, cs:ce] = np.ones((re - rs, ce - cs))
    A[p[1], p[0]] = 2

    # Add to results
    kps_results.append(kp)
    des_results.append(des)

  # Debug
  if debug:
    img = draw_keypoints(img, kps_results, radius=3)

    plt.figure()

    ax = plt.subplot(121)
    ax.imshow(A)
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax = plt.subplot(122)
    ax.imshow(img)
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    plt.show()

  return kps_results, np.array(des_results)


class FeatureGrid:
  """
  FeatureGrid

  The idea is to take all the feature positions and put them into grid cells
  across the full image space. This is so that one could keep track of how many
  feautures are being tracked in each individual grid cell and act accordingly.

  o-----> x
  | ---------------------
  | |  0 |  1 |  2 |  3 |
  V ---------------------
  y |  4 |  5 |  6 |  7 |
    ---------------------
    |  8 |  9 | 10 | 11 |
    ---------------------
    | 12 | 13 | 14 | 15 |
    ---------------------

    grid_x = ceil((max(1, pixel_x) / img_w) * grid_cols) - 1.0
    grid_y = ceil((max(1, pixel_y) / img_h) * grid_rows) - 1.0
    cell_id = int(grid_x + (grid_y * grid_cols))

  """
  def __init__(self, grid_rows, grid_cols, image_shape, keypoints):
    assert len(image_shape) == 2
    self.grid_rows = grid_rows
    self.grid_cols = grid_cols
    self.image_shape = image_shape
    self.keypoints = keypoints

    self.cell = [0 for _ in range(self.grid_rows * self.grid_cols)]
    for kp in keypoints:
      if hasattr(kp, "pt"):
        # cv2.KeyPoint
        assert kp.pt[0] >= 0 and kp.pt[0] <= image_shape[1]
        assert kp.pt[1] >= 0 and kp.pt[1] <= image_shape[0]
        self.cell[self.cell_index(kp.pt)] += 1
      else:
        # Tuple
        assert kp[0] >= 0 and kp[0] <= image_shape[1]
        assert kp[1] >= 0 and kp[1] <= image_shape[0]
        self.cell[self.cell_index(kp)] += 1

  def cell_index(self, pt):
    """Return cell index based on point `pt`"""
    pixel_x, pixel_y = pt
    img_h, img_w = self.image_shape
    grid_x = ceil((max(1, pixel_x) / img_w) * self.grid_cols) - 1.0
    grid_y = ceil((max(1, pixel_y) / img_h) * self.grid_rows) - 1.0
    cell_id = int(grid_x + (grid_y * self.grid_cols))
    return cell_id

  def count(self, cell_idx):
    """Return cell count"""
    return self.cell[cell_idx]


def grid_detect(
  image,
  max_keypoints = 2000,
  grid_rows = 3,
  grid_cols = 4,
  prev_kps=[],
  debug = False,
):
  """
  Detect features uniformly using a grid system.
  """
  assert image is not None
  if prev_kps is None:
    prev_kps = []

  # Calculate number of grid cells and max corners per cell
  detector = cv2.FastFeatureDetector_create(threshold=50)
  image_height, image_width = image.shape
  dx = int(ceil(float(image_width) / float(grid_cols)))
  dy = int(ceil(float(image_height) / float(grid_rows)))
  nb_cells = grid_rows * grid_cols
  max_per_cell = floor(max_keypoints / nb_cells)

  # Detect corners in each grid cell
  feature_grid = FeatureGrid(grid_rows, grid_cols, image.shape, prev_kps)
  kps_all = []

  cell_idx = 0
  for y in range(0, image_height, dy):
    for x in range(0, image_width, dx):
      # Make sure roi width and height are not out of bounds
      w = image_width - x if (x + dx > image_width) else dx
      h = image_height - y if (y + dy > image_height) else dy

      # Detect corners in grid cell
      cs, ce, rs, re = (x, x + w, y, y + h)
      roi_image = image[rs:re, cs:ce]
      kps = detector.detect(roi_image)
      # kps = sort_keypoints(kps)

      # Offset keypoints
      cell_vacancy = max_per_cell - feature_grid.count(cell_idx)
      if cell_vacancy <= 0:
        continue

      limit = min(len(kps), cell_vacancy)
      for i in range(limit):
        kp = kps[i]
        kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
        kps_all.append(kp)

      # Update cell_idx
      cell_idx += 1

  # Space out the keypoints
  kps_all = spread_keypoints(image, kps_all, 20, prev_kps=prev_kps)

  # Debug
  if debug:
    # Setup
    viz = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    kps_grid = FeatureGrid(grid_rows, grid_cols, image.shape, kps_all)

    # Visualization properties
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    linetype = cv2.LINE_AA
    font = cv2.FONT_HERSHEY_SIMPLEX

    # -- Draw horizontal lines
    for x in range(0, image_width, dx):
      cv2.line(viz, (x, 0), (x, image_height), red, 1, linetype)

    # -- Draw vertical lines
    for y in range(0, image_height, dy):
      cv2.line(viz, (0, y), (image_width, y), red, 1, linetype)

    # -- Draw bin numbers
    cell_idx = 0
    for y in range(0, image_height, dy):
      for x in range(0, image_width, dx):
        text = str(kps_grid.count(cell_idx))
        origin = (x + 10, y + 20)
        viz = cv2.putText(viz, text, origin, font, 0.5, red, 1, linetype)

        # text = str(feature_grid.count(cell_idx))
        # origin = (x + 10, y + 20)
        # viz = cv2.putText(viz, text, origin, font, 0.5, yellow, 1, linetype)

        cell_idx += 1

    # -- Draw keypoints
    viz = draw_keypoints(viz, kps_all, color=red)
    viz = draw_keypoints(viz, prev_kps, color=yellow)
    cv2.imshow("viz", viz)
    cv2.waitKey(0)

  # Return
  return kps_all


def good_grid(
  image,
  max_keypoints = 2000,
  quality_level = 0.001,
  use_harris = True,
  min_dist = 20,
  grid_rows = 2,
  grid_cols = 3,
  prev_kps=[],
  debug = False,
):
  """
  Detect features uniformly using a grid system.
  """
  if prev_kps is None:
    prev_kps = []

  # Calculate number of grid cells and max corners per cell
  image_height, image_width = image.shape
  dx = int(ceil(float(image_width) / float(grid_cols)))
  dy = int(ceil(float(image_height) / float(grid_rows)))
  nb_cells = grid_rows * grid_cols
  max_per_cell = floor(max_keypoints / nb_cells)

  # Detect corners in each grid cell
  feature_grid = FeatureGrid(grid_rows, grid_cols, image.shape, prev_kps)
  kps_new = []

  cell_idx = 0
  for y in range(0, image_height, dy):
    for x in range(0, image_width, dx):
      # Make sure roi width and height are not out of bounds
      w = image_width - x if (x + dx > image_width) else dx
      h = image_height - y if (y + dy > image_height) else dy

      # Calculate cell vacancy
      cell_vacancy = max_per_cell - feature_grid.count(cell_idx)
      if cell_vacancy <= 0:
        continue

      # Detect corners in grid cell
      cs, ce, rs, re = (x, x + w, y, y + h)
      roi_image = image[rs:re, cs:ce]
      corners = cv2.goodFeaturesToTrack(
        roi_image,
        cell_vacancy,
        quality_level,
        min_dist,
        blockSize=3,
        useHarrisDetector=use_harris,
      )
      if corners is None:
        cell_idx += 1
        continue

      # Add to results
      for corner in corners:
        if corner is None:
          break
        cx, cy = corner[0]
        pt = (cx + x, cy + y)
        kps_new.append(pt)

      # Update cell_idx
      cell_idx += 1

  # Spread
  kps_new = spread_corners(image, kps_new, min_dist, prev_corners=prev_kps)
  kps_new = np.array(kps_new, dtype=np.float32)

  # Subpixel refinement
  if len(kps_new):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    kps_new = cv2.cornerSubPix(image, kps_new, (5, 5), (-1, -1), criteria)

  # Debug
  if debug:
    # Setup
    viz = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    kps_grid = FeatureGrid(grid_rows, grid_cols, image.shape, kps_new)

    # Visualization properties
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    linetype = cv2.LINE_AA
    font = cv2.FONT_HERSHEY_SIMPLEX

    # -- Draw horizontal lines
    for x in range(0, image_width, dx):
      cv2.line(viz, (x, 0), (x, image_height), red, 1, linetype)

    # -- Draw vertical lines
    for y in range(0, image_height, dy):
      cv2.line(viz, (0, y), (image_width, y), red, 1, linetype)

    # -- Draw bin numbers
    cell_idx = 0
    for y in range(0, image_height, dy):
      for x in range(0, image_width, dx):
        text = str(kps_grid.count(cell_idx))
        origin = (x + 10, y + 20)
        viz = cv2.putText(viz, text, origin, font, 0.5, red, 1, linetype)

        text = str(feature_grid.count(cell_idx))
        origin = (x + 10, y + 40)
        viz = cv2.putText(viz, text, origin, font, 0.5, yellow, 1, linetype)

        cell_idx += 1

    # -- Draw keypoints
    viz = draw_keypoints(viz, kps_new, color=red)
    viz = draw_keypoints(viz, prev_kps, color=yellow)
    cv2.imshow("good_grid", viz)
    cv2.waitKey(1)

  # Return
  return np.array(kps_new)


def optflow_track(
  img_i,
  img_j,
  pts_i,
  pts_j = None,
  patch_size = 30,
  max_iter = 100,
  epsilon = 0.001,
  debug = False,
):
  """
  Track keypoints `pts_i` from image `img_i` to image `img_j` using optical
  flow. Returns a tuple of `(pts_i, pts_j, inliers)` points in image i, j and a
  vector of inliers.
  """
  # Setup
  pts_j = np.array(pts_i) if pts_j is None else pts_j
  crit = (cv2.TermCriteria_COUNT | cv2.TermCriteria_EPS, max_iter, epsilon)

  # Optical flow settings
  config = {}
  config["winSize"] = (patch_size, patch_size)
  config["maxLevel"] = 3
  config["criteria"] = crit
  config["flags"] = cv2.OPTFLOW_USE_INITIAL_FLOW

  # Track using optical flow
  track_results = cv2.calcOpticalFlowPyrLK(img_i, img_j, pts_i, pts_j, **config)
  (pts_j, optflow_inliers, _) = track_results

  # Make sure keypoints are within image dimensions
  bound_inliers = []
  img_h, img_w = img_j.shape
  assert pts_j is not None
  for p in pts_j:
    x_ok = p[0] >= 0 and p[0] <= img_w
    y_ok = p[1] >= 0 and p[1] <= img_h
    if x_ok and y_ok:
      bound_inliers.append(True)
    else:
      bound_inliers.append(False)

  # Update or mark feature as lost
  assert len(pts_i) == optflow_inliers.shape[0]
  assert len(pts_i) == len(bound_inliers)
  inliers = []
  for i in range(len(pts_i)):
    if optflow_inliers[i, 0] and bound_inliers[i]:
      inliers.append(True)
    else:
      inliers.append(False)

  if debug:
    viz_i = draw_keypoints(img_i, pts_i, inliers)
    viz_j = draw_keypoints(img_j, pts_j, inliers)
    viz = np.hstack([viz_i, viz_j])
    cv2.imshow("viz", viz)
    cv2.waitKey(0)

  return (pts_i, pts_j, inliers)


def filter_outliers(pts_i, pts_j, inliers):
  """Filter outliers"""
  pts_out_i = []
  pts_out_j = []

  for n, status in enumerate(inliers):
    if status:
      pts_out_i.append(pts_i[n])
      pts_out_j.append(pts_j[n])
  pts_out_i = np.array(pts_out_i)
  pts_out_j = np.array(pts_out_j)

  return (pts_out_i, pts_out_j)


def check_parallax(
  cam0_params,
  cam1_params,
  cam0_exts,
  cam1_exts,
  kps0,
  kps1,
  parallax_threshold,
):
  """Check Parallax"""
  cam0_geom = cam0_params.data
  cam1_geom = cam1_params.data
  cam0_intrinsic = cam0_params.param
  cam1_intrinsic = cam1_params.param

  # Form projection matrices P_i and P_j
  T_BC0 = pose2tf(cam0_exts.param)
  T_BC1 = pose2tf(cam1_exts.param)
  T_C0C1 = inv(T_BC0) @ T_BC1
  P0 = pinhole_P(cam0_geom.proj_params(cam0_intrinsic), eye(4))
  P1 = pinhole_P(cam1_geom.proj_params(cam1_intrinsic), T_C0C1)

  # Check parallax
  inliers = []
  for pt0, pt1 in zip(kps0, kps1):
    # Undistort
    z0 = cam0_geom.undistort(cam0_intrinsic, pt0)
    z1 = cam1_geom.undistort(cam1_intrinsic, pt1)

    # Triangulate and check parallax
    p_C0 = linear_triangulation(P0, P1, z0, z1)
    p_C1 = tf_point(inv(T_C0C1), p_C0)
    angle = parallax(p_C0, p_C1)

    if angle < parallax_threshold:
      inliers.append(False)
    else:
      inliers.append(True)

  return inliers


def ransac(pts_i, pts_j, cam_i, cam_j):
  """RANSAC"""
  # Setup
  cam_geom_i = cam_i.data
  cam_geom_j = cam_j.data

  # Undistort points
  pts_i_ud = np.array([cam_geom_i.undistort(cam_i.param, p) for p in pts_i])
  pts_j_ud = np.array([cam_geom_j.undistort(cam_j.param, p) for p in pts_j])

  # Ransac via OpenCV's find fundamental matrix
  method = cv2.FM_RANSAC
  reproj_thresh = 0.5
  confidence = 0.999
  args = [pts_i_ud, pts_j_ud, method, reproj_thresh, confidence]
  _, inliers = cv2.findFundamentalMat(*args)

  return inliers.flatten()


def estimate_pose(
  param_i,
  param_j,
  ext_i,
  ext_j,
  kps_i,
  kps_j,
  features,
  pose_i,
  **kwargs,
):
  """Estimate pose"""
  # Settings
  verbose = kwargs.get("verbose", False)
  max_iter = kwargs.get("max_iter", 5)

  # Setup
  cam_geom_i = param_i.data
  cam_geom_j = param_j.data
  graph = FactorGraph()
  graph.solver_max_iter = max_iter

  # Add params
  param_i_id = graph.add_param(param_i)
  param_j_id = graph.add_param(param_j)
  ext_i_id = graph.add_param(ext_i)
  ext_j_id = graph.add_param(ext_j)
  pose_i_id = graph.add_param(pose_i)
  pose_j_id = graph.add_param(pose_setup(1, pose_i.param))

  # Add factors
  for z_i, z_j, p_W in zip(kps_i, kps_j, features):
    feature = feature_setup(p_W, fix=True)
    feature_id = graph.add_param(feature)

    param_ids = [pose_i_id, ext_i_id, feature_id, param_i_id]
    factor_i = VisionFactor(cam_geom_i, param_ids, z_i)
    graph.add_factor(factor_i)

    param_ids = [pose_j_id, ext_j_id, feature_id, param_j_id]
    factor_j = VisionFactor(cam_geom_j, param_ids, z_j)
    graph.add_factor(factor_j)

  # Solve
  graph.solve(verbose)
  # reproj_error = graph.get_reproj_errors()

  # if verbose:
  #   print(f"reproj_error: {np.linalg.norm(reproj_error):.4f}")
  #   print(f"max:    {np.max(reproj_error):.4f}")
  #   print(f"min:    {np.min(reproj_error):.4f}")
  #   print(f"mean:   {np.mean(reproj_error):.4f}")
  #   print(f"median: {np.median(reproj_error):.4f}")
  #   print(f"std:    {np.std(reproj_error):.4f}")

  return graph.params[pose_j_id]


class FeatureTrack:
  """
  Feature Track
  """
  def __init__(self, feature_id, cam_params, cam_exts, **kwargs):
    self.feature_id = feature_id
    self.cam_params = cam_params
    self.cam_exts = cam_exts
    self.max_length = kwargs.get("max_length", 30)
    self.min_length = kwargs.get("min_length", 5)

    self.timestamps = []
    self.data = {}

    self.init = False
    self.init_ts = None
    self.param = None

  def get_timestamps(self):
    """Get timestamps"""
    return self.timestamps

  def get_lifetime(self):
    """Get lifetime"""
    return len(self.timestamps)

  def get_keypoints(self, ts=None):
    """Get Keypoints"""
    ts = self.timestamps[-1] if ts is None else ts
    return self.data[ts]

  def add(self, ts, cam_idx, kp, des=None):
    """Add observation"""
    if ts not in self.data:
      self.data[ts] = {}

    if cam_idx not in self.data[ts]:
      self.data[ts][cam_idx] = {"kp": None, "desc": None}

    self.data[ts][cam_idx]["kp"] = kp
    self.data[ts][cam_idx]["des"] = des
    self.timestamps.append(ts)

  def _triangulate(self, ts, T_WB, measurements):
    # Triangulate
    cam0 = self.cam_params[0]
    cam1 = self.cam_params[1]
    cam0_ext = self.cam_exts[0]
    cam1_ext = self.cam_exts[1]

    # -- Form projection matrices P0 and P1
    T_BC0 = pose2tf(cam0_ext.param)
    T_BC1 = pose2tf(cam1_ext.param)
    T_C0C1 = inv(T_BC0) @ T_BC1
    cam0_geom = cam0.data
    cam1_geom = cam1.data
    P0 = pinhole_P(cam0_geom.proj_params(cam0.param), eye(4))
    P1 = pinhole_P(cam1_geom.proj_params(cam1.param), T_C0C1)

    # -- Undistort image points z0 and z1
    z0 = measurements[0]["kp"]
    z1 = measurements[1]["kp"]
    z0 = cam0_geom.undistort(cam0.param, z0)
    z1 = cam1_geom.undistort(cam1.param, z1)

    # -- Triangulate
    p_C0 = linear_triangulation(P0, P1, z0, z1)
    p_W = tf_point(T_WB @ T_BC0, p_C0)

    # Update
    self.init = True
    self.init_ts = ts
    self.param = np.array([p_W[0], p_W[1], p_W[2]])

  def initialize(self, ts, T_WB):
    """Initialize"""
    # Initialized?
    if self.init:
      return True

    # Do we have data?
    if ts not in self.data:
      return False

    # Is the feature tracked long enough?
    if self.get_lifetime() < self.min_length:
      return False

    # Two or more measurements?
    measurements = self.data[ts]
    if len(measurements) < 2:
      return False

    # Triangulate
    self._triangulate(ts, T_WB, measurements)
    return True


class TestFeatureTracking(unittest.TestCase):
  """Test feature tracking functions"""
  @classmethod
  def setUpClass(cls):
    super(TestFeatureTracking, cls).setUpClass()
    cls.dataset = EurocDataset(EUROC_DATA_PATH)

  def setUp(self):
    # Setup test images
    self.dataset = TestFeatureTracking.dataset
    ts = self.dataset.cam0_data.timestamps[800]
    img0_path = self.dataset.cam0_data.image_paths[ts]
    img1_path = self.dataset.cam1_data.image_paths[ts]
    self.img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    self.img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    # -- cam0 intrinsic
    res = self.dataset.cam0_data.config.resolution
    proj_params = self.dataset.cam0_data.config.intrinsics
    dist_params = self.dataset.cam0_data.config.distortion_coefficients
    proj_model = "pinhole"
    dist_model = "radtan4"
    cam0 = np.block([*proj_params, *dist_params])
    self.cam0_params = camera_params_setup(0, res, proj_model, dist_model, cam0)

    # -- cam1 intrinsic
    res = self.dataset.cam1_data.config.resolution
    proj_params = self.dataset.cam1_data.config.intrinsics
    dist_params = self.dataset.cam1_data.config.distortion_coefficients
    proj_model = "pinhole"
    dist_model = "radtan4"
    cam1 = np.block([*proj_params, *dist_params])
    self.cam1_params = camera_params_setup(1, res, proj_model, dist_model, cam1)

    # -- cam0 extrinsic
    T_BC0 = self.dataset.cam0_data.config.T_BS
    self.cam0_ext = extrinsics_setup(T_BC0)

    # -- cam1 extrinsic
    T_BC1 = self.dataset.cam1_data.config.T_BS
    self.cam1_ext = extrinsics_setup(T_BC1)

  def test_spread_keypoints(self):
    """Test spread_keypoints()"""
    debug = False
    kps = grid_detect(self.img0, debug=debug)
    kps = spread_keypoints(self.img0, kps, 100, debug=debug)

    for i in range(len(kps)):
      for j in range(len(kps)):
        if i == j:
          continue
        z_i = np.array(kps[i].pt)
        z_j = np.array(kps[j].pt)
        self.assertTrue(np.linalg.norm(z_i - z_j) > 100)

  def test_feature_grid_cell_index(self):
    """Test FeatureGrid.grid_cell_index()"""
    grid_rows = 4
    grid_cols = 4
    image_shape = (280, 320)
    keypoints = [[0, 0], [320, 0], [0, 280], [320, 280]]
    grid = FeatureGrid(grid_rows, grid_cols, image_shape, keypoints)

    self.assertEqual(grid.cell[0], 1)
    self.assertEqual(grid.cell[3], 1)
    self.assertEqual(grid.cell[12], 1)
    self.assertEqual(grid.cell[15], 1)

  def test_feature_grid_count(self):
    """Test FeatureGrid.count()"""
    grid_rows = 4
    grid_cols = 4
    image_shape = (280, 320)
    pts = [[0, 0], [320, 0], [0, 280], [320, 280]]
    grid = FeatureGrid(grid_rows, grid_cols, image_shape, pts)

    self.assertEqual(grid.count(0), 1)
    self.assertEqual(grid.count(3), 1)
    self.assertEqual(grid.count(12), 1)
    self.assertEqual(grid.count(15), 1)

  def test_grid_detect(self):
    """Test grid_detect()"""
    debug = False
    kps = grid_detect(self.img0, debug=debug)
    self.assertTrue(len(kps) > 0)

  def test_good_grid(self):
    """Test grid_detect()"""
    debug = False
    kwargs = {"debug": debug}
    kps = good_grid(self.img0, **kwargs)
    self.assertTrue(len(kps) > 0)

  def test_optflow_track(self):
    """Test optflow_track()"""
    debug = False

    # Detect
    kps = grid_detect(self.img0)
    self.assertTrue(len(kps))

    # Track
    pts_i = np.array([kp.pt for kp in kps], dtype=np.float32)
    track_results = optflow_track(self.img0, self.img1, pts_i, debug=debug)
    (pts_i, pts_j, inliers) = track_results

    self.assertTrue(len(pts_i) == len(pts_j))
    self.assertTrue(len(pts_i) == len(inliers))

  def test_feature_track(self):
    """Test FeatureTrack"""
    feature_id = 123
    intrinsics = {0: self.cam0_params, 1: self.cam1_params}
    extrinsics = {0: self.cam0_ext, 1: self.cam1_ext}
    track = FeatureTrack(feature_id, intrinsics, extrinsics)

    track.add(0, 0, [0, 0])
    track.add(0, 1, [0, 0])

    track.add(1, 0, [0, 0])
    track.add(1, 1, [0, 0])

    track.add(2, 0, [0, 0])
    track.add(2, 1, [0, 0])

  def test_estimate_pose(self):
    """Test estimate_pose()"""
    # Detect
    kps0 = good_grid(self.img0, max_keypoints=200)

    # Track
    kps0, kps1, inliers = optflow_track(self.img0, self.img1, kps0)
    kps0, kps1 = filter_outliers(kps0, kps1, inliers)

    # RANSAC
    inliers = ransac(kps0, kps1, self.cam0_params, self.cam1_params)
    kps0, kps1 = filter_outliers(kps0, kps1, inliers)

    # Visualize
    # viz_i = draw_keypoints(self.img0, kps0)
    # viz_j = draw_keypoints(self.img1, kps1)
    # viz = cv2.hconcat([viz_i, viz_j])
    # cv2.imshow("Viz", viz)
    # cv2.waitKey(0)

    # Triangulate
    assert self.cam0_params.data
    assert self.cam1_params.data
    features = []
    T_WB = eye(4)
    T_BC0 = pose2tf(self.cam0_ext.param)
    T_BC1 = pose2tf(self.cam1_ext.param)
    T_C0C1 = inv(T_BC0) @ T_BC1

    for z0, z1 in zip(kps0, kps1):
      # -- Form projection matrices P0 and P1
      param0 = self.cam0_params.param
      param1 = self.cam1_params.param
      P0 = pinhole_P(self.cam0_params.data.proj_params(param0), eye(4))
      P1 = pinhole_P(self.cam1_params.data.proj_params(param1), T_C0C1)

      # -- Undistort image points z0 and z1
      z0 = self.cam0_params.data.undistort(param0, z0)
      z1 = self.cam1_params.data.undistort(param1, z1)

      # -- Triangulate
      p_C0 = linear_triangulation(P0, P1, z0, z1)
      features.append(p_C0)

    # Estimate relative pose
    # time_start = time.time()
    pose_i = pose_setup(0, T_WB, fix=True)
    pose_j = estimate_pose(
      self.cam0_params,
      self.cam1_params,
      extrinsics_setup(eye(4), fix=True),
      extrinsics_setup(eye(4), fix=True),
      kps0,
      kps1,
      features,
      pose_i,
    )
    # elapsed = time.time() - time_start
    # T_C0C1_est = pose2tf(pose_j.param)

    self.assertTrue(pose_i is not None)
    self.assertTrue(pose_j is not None)

    # print(f"elapsed: {elapsed:.4f} [s]")
    # print(f"est:\n{np.round(T_C0C1_est, 3)}\n")
    # print(f"gnd:\n{np.round(T_C0C1, 3)}\n")


if __name__ == "__main__":
  unittest.main(failfast=True)
