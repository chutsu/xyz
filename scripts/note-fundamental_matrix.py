#!/usr/bin/env python3
import numpy as np
from numpy import pi
import matplotlib.pylab as plt

from xyz import inv
from xyz import tf
from xyz import tf_point
from xyz import euler321
from xyz import focal_length
from xyz import pinhole_project
from xyz import plot_tf
from xyz import plot_set_axes_equal


def random_point_cloud(n=30, center=(0.5, 0, 0), spread=0.3):
  """Random 3D point cloud centered at `center` with given spread."""
  return np.random.uniform(-spread, spread, (n, 3)) + center


def skew(t):
  """Skew-symmetric matrix [t]_×."""
  return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def essential_matrix(R, t):
  """Essential matrix E = [t]_× @ R."""
  return skew(t) @ R


def fundamental_from_pose(T_WC0, T_WC1, K):
  """Ground truth fundamental matrix from two camera poses.

  F satisfies cam1^T @ F @ cam0 = 0.
  """
  T_C1C0 = inv(T_WC1) @ T_WC0
  R = T_C1C0[:3, :3]
  t = T_C1C0[:3, 3]
  E = essential_matrix(R, t)
  F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
  F /= F[2, 2]
  return F


def _normalize(pts):
  """Hartley normalization. Returns (T, pts_norm)."""
  centroid = np.mean(pts, axis=0)
  pts_shifted = pts - centroid
  mean_dist = np.mean(np.sqrt(np.sum(pts_shifted**2, axis=1)))
  scale = np.sqrt(2) / mean_dist
  T = np.array([[scale, 0, -scale * centroid[0]],
                [0, scale, -scale * centroid[1]], [0, 0, 1]])
  pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
  pts_norm_h = (T @ pts_h.T).T
  return T, pts_norm_h[:, :2]


def estimate_fundamental_matrix(pts1, pts2):
  """Normalized 8-point algorithm. Returns 3x3 fundamental matrix F."""
  assert pts1.shape == pts2.shape
  assert pts1.shape[0] >= 8

  # Normalize points
  T1, pts1_norm = _normalize(pts1)
  T2, pts2_norm = _normalize(pts2)

  # Build A matrix
  A = []
  for (x1, y1), (x2, y2) in zip(pts1_norm, pts2_norm):
    A.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0])
  A = np.asarray(A)

  # Solve Af=0
  _, _, Vt = np.linalg.svd(A)
  f = Vt[-1]  # last column = smallest eigenvalue
  F = f.reshape(3, 3)

  # Enforce rank 2 constraint
  U, S, Vt = np.linalg.svd(F)
  S[2] = 0  # set smallest eigenvalue to 0
  F = U @ np.diag(S) @ Vt

  # Denormalize
  F = T2.T @ F @ T1

  # Scale
  F /= F[2, 2]

  return F


def triangulate_dlt(P1, P2, pts1, pts2):
  """Linear triangulation via DLT. Returns Nx3 points."""
  assert len(pts1) == len(pts2)

  points = []
  for (x1, y1), (x2, y2) in zip(pts1, pts2):
    A = np.array([
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1],
        x2 * P2[2] - P2[0],
        y2 * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    points.append(X[:3] / X[3])

  return np.array(points)


def decompose_essential(E, pts1, pts2, K):
  """Decompose E into (R, t) with cheirality disambiguation using all points."""
  U, _, Vt = np.linalg.svd(E)
  W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

  candidates = [
      (U @ W @ Vt, U[:, 2]),
      (U @ W @ Vt, -U[:, 2]),
      (U @ W.T @ Vt, U[:, 2]),
      (U @ W.T @ Vt, -U[:, 2]),
  ]

  P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
  best_count = -1
  best_Rt = None

  for R, t in candidates:
    if np.linalg.det(R) < 0:
      R = -R
      t = -t
    P2 = K @ np.hstack([R, t.reshape(-1, 1)])
    X = triangulate_dlt(P1, P2, pts1, pts2)
    in_front = sum(
        np.dot(P1, [*Xw, 1])[2] > 1e-6 and np.dot(P2, [*Xw, 1])[2] > 1e-6
        for Xw in X)
    if in_front > best_count:
      best_count = in_front
      best_Rt = (R, t)

  if best_Rt:
    return best_Rt[0], best_Rt[1]
  return None


if __name__ == "__main__":
  np.random.seed(0)

  # Camera
  img_w = 640
  img_h = 480
  fx = focal_length(img_w, 90.0)
  fy = focal_length(img_w, 90.0)
  cx = img_w / 2.0
  cy = img_h / 2.0
  proj_params = np.array([fx, fy, cx, cy])

  # Stereo camera pair pointing inward at the point cloud
  baseline = 0.15
  converge_angle = np.deg2rad(10.0)
  # -- cam0
  C_WC0 = euler321(-pi / 2 + converge_angle, 0.0, -pi / 2)
  r_WC0 = np.array([0.0, -baseline / 2, 0.0])
  T_WC0 = tf(C_WC0, r_WC0)
  # -- cam1
  C_WC1 = euler321(-pi / 2 - converge_angle, 0.0, -pi / 2)
  r_WC1 = np.array([0.0, baseline / 2, 0.0])
  T_WC1 = tf(C_WC1, r_WC1)

  # Generate random point cloud in world frame
  world_points = random_point_cloud(n=30, center=(0.5, 0, 0), spread=0.3)
  cam0_kps = []
  cam1_kps = []
  for p_W in world_points:
    cam0_kps.append(pinhole_project(proj_params, tf_point(inv(T_WC0), p_W)))
    cam1_kps.append(pinhole_project(proj_params, tf_point(inv(T_WC1), p_W)))
  cam0_kps = np.array(cam0_kps)
  cam1_kps = np.array(cam1_kps)

  # 3D scene
  fig = plt.figure(figsize=(12, 5))
  ax = fig.add_subplot(1, 2, 1, projection="3d")
  plot_tf(ax, T_WC0, size=0.1, name="cam0")
  plot_tf(ax, T_WC1, size=0.1, name="cam1")
  ax.scatter(world_points[:, 0],
             world_points[:, 1],
             world_points[:, 2],
             c="tab:blue",
             s=10)
  ax.set_xlabel("x [m]")
  ax.set_ylabel("y [m]")
  ax.set_zlabel("z [m]")  # pyright: ignore
  ax.view_init(elev=35, azim=-145, roll=0)
  plot_set_axes_equal(ax)
  ax.set_title("3D scene")

  # 2D image projections from both cameras
  ax2 = fig.add_subplot(1, 2, 2)
  ax2.scatter(cam0_kps[:, 0], cam0_kps[:, 1], c="tab:blue", s=10, label="cam0")
  ax2.scatter(cam1_kps[:, 0],
              cam1_kps[:, 1],
              c="tab:orange",
              s=10,
              label="cam1")
  for p0, p1 in zip(cam0_kps, cam1_kps):
    ax2.plot([p0[0], p1[0]], [p0[1], p1[1]], "gray", lw=0.5, alpha=0.5)
  ax2.set_aspect("equal")
  ax2.invert_yaxis()
  ax2.set_xlabel("u [px]")
  ax2.set_ylabel("v [px]")
  ax2.legend()
  ax2.set_title("Correspondences")
  plt.tight_layout()
  plt.show()

  # Ground truth
  K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
  F_gt = fundamental_from_pose(T_WC0, T_WC1, K)
  T_C1C0 = inv(T_WC1) @ T_WC0
  R_gt = T_C1C0[:3, :3]
  t_gt = T_C1C0[:3, 3]
  print("Ground truth F:\n", F_gt)

  # Estimated
  F_est = estimate_fundamental_matrix(cam0_kps, cam1_kps)
  print("\nEstimated F:\n", F_est)

  # Epipolar errors
  print("\nEpipolar errors (cam1^T @ F @ cam0):")
  max_err = 0.0
  for kp0, kp1 in zip(cam0_kps, cam1_kps):
    x0 = np.array([kp0[0], kp0[1], 1])
    x1 = np.array([kp1[0], kp1[1], 1])
    err = x1.T @ F_est @ x0
    max_err = max(max_err, abs(err))
  print(f"Max epipolar error: {max_err:.3e}")

  # Decompose estimated F into R, t
  E_est = K.T @ F_est @ K
  result = decompose_essential(E_est, cam0_kps, cam1_kps, K)
  if result:
    R_est, t_est = result
  else:
    raise RuntimeError("Failed to decompose Essential matrix")

  # Compare
  t_gt_n = t_gt / np.linalg.norm(t_gt)
  t_est_n = t_est / np.linalg.norm(t_est)
  rot_err = np.rad2deg(
      np.arccos(np.clip((np.trace(R_gt.T @ R_est) - 1) / 2, -1, 1)))
  t_err = np.rad2deg(np.arccos(np.clip(np.dot(t_gt_n, t_est_n), -1, 1)))
  print(f"Rotation error: {rot_err:.4f} deg")
  print(f"Translation direction error: {t_err:.4f} deg")
