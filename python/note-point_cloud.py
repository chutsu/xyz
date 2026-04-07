#!/usr/bin/env python3
import unittest
from pathlib import Path

import scipy
import numpy as np
from numpy import eye
from numpy import zeros
import matplotlib.pyplot as plt

from xyz import hat
from xyz import euler321
from xyz import Exp
from xyz import rot_diff
from xyz import solve_svd
from xyz import plot_set_axes_equal
from xyz import KittiRawDataset


def umeyama(X, Y):
  """
  Estimates scale `c`, rotation matrix `R` and translation vector `t` between
  two sets of points `X` and `Y` such that:

    Y ~= c * R @ X + t

  Args:

    X: src 3D points
    Y: dest 3D points

  Returns:

    c: Scale factor
    R: Rotation matrix
    t: translation vector

  """
  # Compute centroid
  mu_x = X.mean(axis=1).reshape(-1, 1)
  mu_y = Y.mean(axis=1).reshape(-1, 1)

  # Form covariance matrix and decompose with SVD
  var_x = np.square(X - mu_x).sum(axis=0).mean()
  cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
  U, D, VH = np.linalg.svd(cov_xy)

  # Check to see if rotation matrix det(R) is 1
  S = eye(X.shape[0])
  if np.linalg.det(U) * np.linalg.det(VH) < 0:
    S[-1, -1] = -1

  # Calculate scale, rotation matrix and translation vector
  c = np.trace(np.diag(D) @ S) / var_x
  R = U @ S @ VH
  t = mu_y - c * R @ mu_x

  return c, R, t


def icp(X, Y, **kwargs):
  # Parameters
  prev_error = float("inf")
  max_iter = kwargs.get("max_iter", 2)
  tol = kwargs.get("tol", 1e-8)

  # Setup
  R = None
  t = None

  # -- Setup plotting
  plt.figure(figsize=(12, 10))
  ax = plt.axes(projection="3d")
  ax.scatter(X[:, 0], X[:, 1], X[:, 2], color="r", label="src", alpha=0.2)
  ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color="g", label="dest", alpha=0.2)
  plt.legend(loc=0)
  # plt.show()

  # Optimize
  est_ax = None
  for _ in range(max_iter):
    # Step 1: Find closest points in Y for each point in X
    tree = scipy.spatial.KDTree(Y)
    distances, indices = tree.query(X)
    closest_Y = Y[indices]

    # # Step 2: Compute transformation using Least Squares
    # X_flat = np.hstack((X, np.ones((X.shape[0], 1))))
    # Y_flat = np.hstack((Y, np.ones((Y.shape[0], 1))))
    #
    # # Solve Ax = b in Least Squares sense (x = transformation matrix)
    # est, _, _, _ = np.linalg.lstsq(X_flat, Y_flat, rcond=None)  # Solving A * X ≈ B
    # R = est[:3, :3]  # Rotation matrix
    # t = est[3, :]    # Translation vector
    _, R, t = umeyama(X.T, closest_Y.T)

    # Step 3: Apply transformation
    X = (X @ R.T) + t.T

    # Plot
    if est_ax:
      est_ax.remove()
    est_ax = ax.scatter(X[:, 0],
                        X[:, 1],
                        X[:, 2],
                        color="k",
                        label="est",
                        alpha=0.2)
    plot_set_axes_equal(ax)
    plt.draw()
    plt.pause(0.5)
    # plt.show()

    # Step 4: Check for convergence
    mean_error = np.mean(distances)
    print(f"mean_error: {mean_error}")
    if abs(prev_error - mean_error) < tol:
      break
    prev_error = mean_error

  return X, R, t


class TestPointCloud(unittest.TestCase):
  """Test point cloud functions"""
  def test_umeyama(self):
    debug = False
    R_gnd = euler321(*np.random.rand(3))
    t_gnd = np.random.rand(3, 1) * 0.1

    points = np.random.rand(int(1e7), 3)
    src = points
    dst = points @ R_gnd.T + t_gnd.T
    # time_start = time.time()
    c, R, t = umeyama(src.T, dst.T)
    # elapsed = time.time() - time_start
    # print(f"python umeyama elapsed: {elapsed:.2f} [s]")
    est = c * src @ R.T + t.T

    self.assertTrue(np.allclose(R, R_gnd, atol=1e-4))
    self.assertTrue(np.allclose(t, t_gnd, atol=1e-4))
    self.assertTrue(np.allclose(est, dst, atol=1e-4))

    # Visualize
    if debug:
      plt.figure(figsize=(12, 10))
      ax = plt.axes(projection="3d")
      ax.scatter(src[:, 0], src[:, 1], src[:, 2], "r", label="src", alpha=0.2)
      ax.scatter(dst[:, 0], dst[:, 1], dst[:, 2], "g", label="dest", alpha=0.2)
      ax.scatter(est[:, 0],
                 est[:, 1],
                 est[:, 2],
                 "k",
                 label="aligned",
                 alpha=0.2)
      ax.legend(loc=0)
      plot_set_axes_equal(ax)
      plt.show()

  def test_icp(self):
    # Ground truth
    R_gnd = euler321(*np.random.rand(3))
    t_gnd = np.random.rand(3) * 2

    # Estimate
    R_est = R_gnd @ euler321(*np.random.rand(3) * 0.2)
    t_est = t_gnd + np.random.rand(3)

    # Create ground truth points
    N = 10000
    p_src = np.random.rand(3, N)
    p_dst_gnd = (R_gnd @ p_src) + t_gnd[:, np.newaxis]

    # ICP
    max_iter = 10
    for _ in range(max_iter):
      p_dst_est = (R_est @ p_src) + t_est[:, np.newaxis]

      jacobians = []
      residuals = []
      for i in range(N):
        residuals.append(p_dst_gnd[:, i] - p_dst_est[:, i])
        J = zeros((3, 6))
        J[0:3, 0:3] = -1.0 * eye(3)
        J[0:3, 3:6] = R_est @ hat(p_dst_est[:, i])
        jacobians.append(J)
      J = np.vstack(jacobians)
      r = np.hstack(residuals)

      cost = 0.5 * (r.T @ r)
      H = J.T @ J
      H += 1e-4 * eye(6)
      b = -1.0 * J.T @ r
      dx = solve_svd(H, b)

      print(f"{cost=:.2e}")
      t_est += dx[0:3]
      R_est = R_est @ Exp(dx[3:6])

    # Assert
    self.assertTrue(np.linalg.norm(t_est - t_gnd) < 1e-2)
    self.assertTrue(rot_diff(R_est, R_gnd) < 1e-2)

  @unittest.skip("Fix Me!")
  def test_icp_kitti(self):
    # Setup
    data_dir = Path("/data/kitti_raw")
    date = "2011_09_26"
    seq = "93"
    dataset = KittiRawDataset(data_dir, date, seq, True)

    # Load scans
    lidar_timestamps = dataset.velodyne_data.timestamps
    pcd0 = dataset.velodyne_data.load_scan(lidar_timestamps[0])[:, :3]
    pcd1 = dataset.velodyne_data.load_scan(lidar_timestamps[20])[:, :3]

    import open3d as o3d
    pcd_src = o3d.geometry.PointCloud()
    pcd_dst = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(pcd0)
    pcd_dst.points = o3d.utility.Vector3dVector(pcd1)
    threshold = 0.001
    trans_init = eye(4)
    result = o3d.pipelines.registration.registration_icp(
      pcd_src,
      pcd_dst,
      threshold,
      trans_init,
      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
      o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000),
    )
    # print("Estimated transformation:")
    print(result.transformation)
    pcd_src_icp = pcd_src.transform(result.transformation)

    # max_iter = 100
    # tree = scipy.spatial.KDTree(pcd1)
    # R_est = eye(3)
    # t_est = np.array([0.0, 0.0, 0.0])
    #
    # for _ in range(max_iter):
    #   pcd1_est = (R_est @ pcd0.T).T + t_est
    #
    #   distances, indicies = tree.query(pcd1_est)
    #   pcd1_gnd = pcd1[indicies]
    #   N = len(indicies)
    #
    #   # print(pcd1_est.shape)
    #   # print(pcd1_gnd.shape)
    #   # c, R_est, t_est = umeyama(pcd1_est, pcd1_gnd)
    #
    #   jacobians = []
    #   residuals = []
    #   for i in range(N):
    #     residuals.append(pcd1_gnd[i] - pcd1_est[i])
    #     J = zeros((3, 6))
    #     J[0:3, 0:3] = -1.0 * eye(3)
    #     J[0:3, 3:6] = R_est @ hat(pcd1_est[i])
    #     jacobians.append(J)
    #   J = np.vstack(jacobians)
    #   r = np.hstack(residuals)
    #   cost = 0.5 * (r.T @ r)
    #   print(f"{cost=:.2e}")
    #
    #   H = J.T @ J
    #   # H += 1e-20 * eye(6)
    #   b = -1.0 * J.T @ r
    #
    #   c, low = scipy.linalg.cho_factor(H)
    #   dx = scipy.linalg.cho_solve((c, low), b)
    #   # dx = solve_svd(H, b)
    #
    #   t_est += dx[0:3]
    #   R_est = R_est @ Exp(dx[3:6])

    voxel_size = 0.5
    pcd_src = pcd_src.voxel_down_sample(voxel_size)
    pcd_src_icp = pcd_src_icp.voxel_down_sample(voxel_size)
    pcd_dst = pcd_dst.voxel_down_sample(voxel_size)

    pcd = np.asarray(pcd_src.points)
    pcd0 = np.asarray(pcd_src_icp.points)
    pcd1 = np.asarray(pcd_dst.points)

    # Visualize
    _ = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')
    ax.scatter(pcd0[:, 0], pcd0[:, 1], pcd0[:, 2], 'r', alpha=0.1)
    ax.scatter(pcd1[:, 0], pcd1[:, 1], pcd1[:, 2], 'g', alpha=0.1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")  # type: ignore
    plot_set_axes_equal(ax)
    plt.show()
