#!/usr/bin/env python3
import os

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ["QT_LOGGING_RULES"] = "qt.qpa.wayland.warning=false"

import scipy
import numpy as np
import matplotlib.pyplot as plt


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

  cpsi = np.cos(psi)
  spsi = np.sin(psi)
  ctheta = np.cos(theta)
  stheta = np.sin(theta)
  cphi = np.cos(phi)
  sphi = np.sin(phi)

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


def load_bunny_scan(path, step=10):
  """Load bunny scan"""
  data = np.loadtxt(path)
  points = data[::step, :3]
  normals = data[::step, 3:6]
  return points, normals


def umeyama(X, Y, rigid=False):
  """
  Estimates scale `c`, rotation matrix `R` and translation vector `t` between
  two sets of points `X` and `Y` such that:

    Y ~= c * R @ X + t

  Args:

    X: src 3D points (3, N)
    Y: dest 3D points (3, N)
    rigid: If True, forces c = 1 (rigid transformation)

  Returns:

    c: Scale factor
    R: Rotation matrix
    t: Translation vector

  """
  # Compute centroid
  mu_x = X.mean(axis=1).reshape(-1, 1)
  mu_y = Y.mean(axis=1).reshape(-1, 1)

  # Form covariance matrix and decompose with SVD
  var_x = np.square(X - mu_x).sum(axis=0).mean()
  cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
  U, D, VH = np.linalg.svd(cov_xy)

  # Check to see if rotation matrix det(R) is 1
  S = np.eye(X.shape[0])
  if np.linalg.det(U) * np.linalg.det(VH) < 0:
    S[-1, -1] = -1

  # Calculate rotation matrix and translation vector
  R = U @ S @ VH
  if rigid:
    c = 1.0
  else:
    c = np.trace(np.diag(D) @ S) / var_x
  t = mu_y - c * R @ mu_x

  return c, R, t


def icp(X, Y, **kwargs):
  # Parameters
  prev_error = float("inf")
  max_iter = kwargs.get("max_iter", 30)
  tol = kwargs.get("tol", 1e-8)
  plot = kwargs.get("plot", True)

  # Setup
  R = None
  t = None
  ax = None

  # -- Setup plotting
  if plot:
    plt.ion()
    _ = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color="r", label="src", alpha=0.2)
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color="g", label="dest", alpha=0.2)
    plt.legend(loc=0)
    plt.draw()
    plt.pause(0.5)

  # Optimize
  R_total = np.eye(3)
  t_total = np.zeros((3, 1))
  est_ax = None
  for _ in range(max_iter):
    # Step 1: Find closest points in Y for each point in X
    tree = scipy.spatial.KDTree(Y)
    distances, indices = tree.query(X)
    closest_Y = Y[indices]

    # Step 2: Compute rigid transformation using Least Squares
    _, R, t = umeyama(X.T, closest_Y.T, rigid=True)

    # Step 3: Accumulate transformation
    R_total = R @ R_total
    t_total = R @ t_total + t

    # Step 4: Apply transformation
    X = (X @ R.T) + t.T

    # Plot
    if plot:
      assert ax is not None
      if est_ax:
        est_ax.remove()
      est_ax = ax.scatter(X[:, 0],
                          X[:, 1],
                          X[:, 2],
                          color="k",
                          label="est",
                          alpha=0.1)
      plot_set_axes_equal(ax)
      plt.draw()
      plt.pause(0.5)

    # Step 4: Check for convergence
    mean_error = np.mean(distances)
    if abs(prev_error - mean_error) < tol:
      break
    prev_error = mean_error

  return X, R_total, t_total


if __name__ == "__main__":
  points, _ = load_bunny_scan("data/bunny/pcd-00.xyz")
  R_gnd = euler321(*np.random.rand(3) * 0.5)
  t_gnd = np.random.rand(3) * 0.1

  X = points
  Y = points @ R_gnd.T + t_gnd.T
  _, R, t = icp(X, Y)

  assert np.allclose(R, R_gnd, atol=1e-4)
  assert np.allclose(t, t_gnd, atol=1e-4)
