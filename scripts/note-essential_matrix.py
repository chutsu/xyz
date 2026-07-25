import cv2
import numpy as np
import sympy as sp

###############################################################################
# UTILS
###############################################################################


def skew(v):
  """Returns 3x3 skew-symmetric matrix for vector v."""
  return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rodrigues(w):
  """Exponential map from lie algebra so(3) vector to SO(3) rotation matrix."""
  theta = np.linalg.norm(w)
  if theta < 1e-8:
    return np.eye(3)
  k = w / theta
  K = skew(k)
  return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def decompose_essential_matrix(E):
  """Decompose an essential matrix into 4 possible (R, t) pose hypotheses."""
  U, _, Vt = np.linalg.svd(E)
  if np.linalg.det(U) < 0:
    U *= -1
  if np.linalg.det(Vt) < 0:
    Vt *= -1

  W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

  R1 = U @ W @ Vt
  R2 = U @ W.T @ Vt
  t1 = U[:, 2]
  t2 = -U[:, 2]

  return [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]


def cheirality_check(R, t, pts1, pts2):
  """Triangulate points and count how many have positive depth in both views."""
  P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
  P2 = np.hstack((R, t.reshape(3, 1)))

  front_count = 0
  for i in range(len(pts1)):
    x1, y1 = pts1[i, 0], pts1[i, 1]
    x2, y2 = pts2[i, 0], pts2[i, 1]

    A = np.array([
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1],
        x2 * P2[2] - P2[0],
        y2 * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[3]

    depth1 = X[2]
    depth2 = (R[2] @ X[:3]) + t[2]

    if depth1 > 0 and depth2 > 0:
      front_count += 1

  return front_count


def sampson_distance(E, pts1, pts2):
  """Symmetric epipolar distance (Sampson) for all point pairs, summed."""
  n = len(pts1)
  pts1_h = np.hstack([pts1, np.ones((n, 1))])
  pts2_h = np.hstack([pts2, np.ones((n, 1))])

  Ex1 = (E @ pts1_h.T).T
  Etx2 = (E.T @ pts2_h.T).T

  numerator = np.sum(pts2_h * Ex1, axis=1)**2
  denominator = (Ex1[:, 0]**2 + Ex1[:, 1]**2 + Etx2[:, 0]**2 + Etx2[:, 1]**2)
  with np.errstate(divide='ignore', invalid='ignore'):
    sd = np.where(denominator > 1e-12, numerator / denominator, 0.0)
  return np.sum(sd)


###############################################################################
# NISTER 5-POINT ALGORITHM
###############################################################################


def _compute_nullspace_basis(pts1, pts2):
  """
  Builds the 5x9 linear design matrix from 5 normalized point pairs and
  computes the 4 basis matrices (Ex, Ey, Ez, Ew) spanning the null space.
  """
  A = np.zeros((5, 9))
  for i in range(5):
    x1, y1 = pts1[i, 0], pts1[i, 1]
    x2, y2 = pts2[i, 0], pts2[i, 1]
    A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0]

  _, _, Vt = np.linalg.svd(A)
  nullspace = Vt[5:].reshape(4, 3, 3)

  return nullspace[0], nullspace[1], nullspace[2], nullspace[3]


def _build_constraint_matrix(Ex, Ey, Ez, Ew):
  """
  Substitutes E = x*Ex + y*Ey + z*Ez + Ew into the 9 matrix constraints:

    2 * E * E^T * E - trace(E * E^T) * E = 0

  and det(E) = 0. Returns a 10x20 coefficient matrix.

  Uses sympy to perform the symbolic expansion.
  """
  # Null-space basis matrices
  x, y, z = sp.symbols('x y z')
  _M = lambda a: sp.Matrix(a.tolist())
  E = x * _M(Ex) + y * _M(Ey) + z * _M(Ez) + _M(Ew)

  # Contraints
  EET = E * E.T
  constraint = 2 * E * E.T * E - sp.trace(EET) * E
  det_constraint = E.det()

  # Graded lexicographic ordering of monomials in (x, y, z),
  # total degree descending, then x-exponent descending.
  # Columns 0-9  (degree 3) : x^3, x^2y, x^2z, xy^2, xyz, xz^2, y^3, y^2z, yz^2, z^3
  # Columns 10-15 (degree 2) : x^2, xy, xz, y^2, yz, z^2
  # Columns 16-18 (degree 1) : x, y, z
  # Column 19    (degree 0) : 1
  #
  # The split at col 10 separates cubic coefficients (L = M[:, :10]) from
  # lower-degree coefficients (R = M[:, 10:]), enabling the elimination step.
  monomial_map = {
      (3, 0, 0): 0,
      (2, 1, 0): 1,
      (2, 0, 1): 2,
      (1, 2, 0): 3,
      (1, 1, 1): 4,
      (1, 0, 2): 5,
      (0, 3, 0): 6,
      (0, 2, 1): 7,
      (0, 1, 2): 8,
      (0, 0, 3): 9,
      (2, 0, 0): 10,
      (1, 1, 0): 11,
      (1, 0, 1): 12,
      (0, 2, 0): 13,
      (0, 1, 1): 14,
      (0, 0, 2): 15,
      (1, 0, 0): 16,
      (0, 1, 0): 17,
      (0, 0, 1): 18,
      (0, 0, 0): 19
  }
  M = np.zeros((10, 20))

  row = 0
  for r in range(3):
    for c in range(3):
      poly = sp.Poly(sp.expand(constraint[r, c]), x, y, z)
      for monom, coeff in poly.terms():
        if monom in monomial_map:
          M[row, monomial_map[monom]] = float(coeff)
      row += 1

  poly_det = sp.Poly(sp.expand(det_constraint), x, y, z)
  for monom, coeff in poly_det.terms():
    if monom in monomial_map:
      M[9, monomial_map[monom]] = float(coeff)

  return M


def _solve_system_nister(M):
  """
  Reduces the 10x20 matrix using Gauss-Jordan elimination and solves for z
  using an 10x10 action matrix, then recovers (x, y).
  """
  # Split M = [L | R] where L holds cubic coeffs and R holds lower-degree coeffs
  L = M[:, :10]
  R = M[:, 10:]
  if np.linalg.matrix_rank(L) < 10:
    return []

  # L @ cubic + R @ lower = 0  =>  cubic = -L^{-1} @ R @ lower = -B @ lower
  B = np.linalg.solve(L, R)

  # B is 10x10: L * cubic = -R * lower  =>  cubic = -L^{-1} * R * lower = -B * lower
  # So cubic_monomial_i = -sum_j B[i,j] * lower_monomial_j
  #
  # Action matrix implements multiplication by z in the quotient ring.
  # Basis vector v = [x^2, xy, xz, y^2, yz, z^2, x, y, z, 1]^T.
  # Action satisfies  Action @ v  =  z * v,  so its eigenvalues are z-solutions.
  #
  # For each basis element multiplied by z, if the result is a cubic monomial,
  # substitute using cubic = -B * lower.  If it is already a lower monomial,
  # write a one-hot row.
  Action = np.zeros((10, 10))
  Action[0] = -B[2]  # z * x^2  = x^2*z  = -B[2] @ lower
  Action[1] = -B[4]  # z * xy   = x*y*z   = -B[4] @ lower
  Action[2] = -B[5]  # z * xz   = x*z^2   = -B[5] @ lower
  Action[3] = -B[7]  # z * y^2  = y^2*z   = -B[7] @ lower
  Action[4] = -B[8]  # z * yz   = y*z^2   = -B[8] @ lower
  Action[5] = -B[9]  # z * z^2  = z^3     = -B[9] @ lower
  Action[6] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # z * x   = xz  -> basis index 2
  Action[7] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # z * y   = yz  -> basis index 4
  Action[8] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # z * z   = z^2 -> basis index 5
  Action[9] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # z * 1   = z   -> basis index 8
  eigvals, eigvecs = np.linalg.eig(Action)

  solutions = []
  for i in range(10):
    if np.abs(np.imag(eigvals[i])) < 1e-6:
      z_val = np.real(eigvals[i])
      vec = np.real(eigvecs[:, i])

      if np.abs(vec[-1]) > 1e-8:
        vec /= vec[-1]
        x_val = vec[6]
        y_val = vec[7]
        solutions.append((x_val, y_val, z_val))

  return solutions


def nister_5point_algorithm(pts1_norm, pts2_norm):
  """Estimate relative pose using the Nistér 5-point algorithm."""
  Ex, Ey, Ez, Ew = _compute_nullspace_basis(pts1_norm, pts2_norm)
  M = _build_constraint_matrix(Ex, Ey, Ez, Ew)
  xyz_sols = _solve_system_nister(M)

  best_R, best_t = None, None
  max_front_pts = -1
  best_sd = float('inf')

  for x, y, z in xyz_sols:
    # Form candidate
    E_candidate = x * Ex + y * Ey + z * Ez + Ew

    # Ensure singular values are positive and rank-2
    U, S, Vt = np.linalg.svd(E_candidate)
    E_clean = U @ np.diag([(S[0] + S[1]) / 2.0, (S[0] + S[1]) / 2.0, 0.0]) @ Vt

    # Decompose Essential matrix into R, t
    poses = decompose_essential_matrix(E_clean)
    for R, t in poses:
      valid_pts = cheirality_check(R, t, pts1_norm, pts2_norm)
      sd = sampson_distance(E_clean, pts1_norm, pts2_norm)
      if valid_pts > max_front_pts or (valid_pts == max_front_pts and
                                       sd < best_sd):
        max_front_pts = valid_pts
        best_sd = sd
        best_R, best_t = R, t

  return best_R, best_t


###############################################################################
# LUI 5-POINT ALGORITHM
###############################################################################


def _compute_angular_residuals(R1, R2, v1, v2):
  """
  Projects unit vectors v1, v2 using candidate rotations R1, R2.
  Measures 2D angular difference on the x-y tangent plane.
  """
  # 1. Rotate 3D unit vectors
  p1_3d = (R1 @ v1.T).T  # Shape: (5, 3)
  p2_3d = (R2 @ v2.T).T  # Shape: (5, 3)

  # 2. Extract 2D polar angles on x-y plane (looking along baseline e_z)
  theta1 = np.arctan2(p1_3d[:, 1], p1_3d[:, 0])
  theta2 = np.arctan2(p2_3d[:, 1], p2_3d[:, 0])

  # 3. Signed angular difference wrapped to [-pi, pi]
  residuals = theta1 - theta2
  residuals = (residuals + np.pi) % (2 * np.pi) - np.pi
  return residuals, p1_3d, p2_3d


def lui_5point_algorithm(v1, v2, max_iters=100, tol=1e-20):
  """
  Iterative 5-point solver by Vincent Lui & Tom Drummond.

  v1, v2: (N, 3) arrays of unit bearing vectors in camera 1 and camera 2 frames,
           where N >= 5.
  Returns: R (3x3) relative rotation from camera 1 to camera 2,
           t (3,) translation direction in camera 2's frame.
  """
  v1, v2 = v1.copy(), v2.copy()
  assert v1.shape[1] == 3 and v2.shape[1] == 3
  assert v1.shape[0] >= 5 and v2.shape[0] >= 5

  R1 = np.eye(3)
  R2 = np.eye(3)
  n_pts = len(v1)

  lam = 1e-3
  r, p1, p2 = _compute_angular_residuals(R1, R2, v1, v2)
  prev_err = np.linalg.norm(r)
  stagnation_count = 0

  for _ in range(max_iters):
    if prev_err < tol:
      break

    x1, y1, z1 = p1[:, 0], p1[:, 1], p1[:, 2]
    x2, y2, z2 = p2[:, 0], p2[:, 1], p2[:, 2]
    sq1 = x1*x1 + y1*y1
    sq2 = x2*x2 + y2*y2
    mask1 = sq1 > 1e-12
    mask2 = sq2 > 1e-12

    J = np.zeros((n_pts, 5))
    J[:, 0] = np.where(mask1, -x1 * z1 / sq1, 0.0)
    J[:, 1] = np.where(mask1, -y1 * z1 / sq1, 0.0)
    J[:, 2] = 1.0
    J[:, 3] = np.where(mask2, x2 * z2 / sq2, 0.0)
    J[:, 4] = np.where(mask2, y2 * z2 / sq2, 0.0)

    JtJ = J.T @ J
    Jt_r = J.T @ -r

    accepted = False
    for _ in range(30):
      try:
        delta = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ)), Jt_r)
      except np.linalg.LinAlgError:
        lam *= 2
        continue

      R1_test = rodrigues(delta[0:3]) @ R1
      w2_test = np.array([delta[3], delta[4], 0.0])
      r_test, p1_test, p2_test = _compute_angular_residuals(
          R1_test, rodrigues(w2_test) @ R2, v1, v2)
      new_err = np.linalg.norm(r_test)

      if new_err < prev_err:
        R1 = R1_test
        R2 = rodrigues(w2_test) @ R2
        r, p1, p2 = r_test, p1_test, p2_test

        rel_change = abs(prev_err - new_err) / max(prev_err, 1e-12)
        if rel_change < 1e-6:
          stagnation_count += 1
          if stagnation_count >= 3:
            ez = np.array([0.0, 0.0, 1.0])
            return R2.T @ R1, R2.T @ ez
        else:
          stagnation_count = 0

        prev_err = new_err
        lam /= 2
        accepted = True
        break
      else:
        lam *= 2

    if not accepted:
      break

  ez = np.array([0.0, 0.0, 1.0])
  t = R2.T @ ez
  R = R2.T @ R1

  return R, t


def _sampson_per_point(E, pts1, pts2):
  """Per-point Sampson distance for RANSAC inlier counting."""
  n = len(pts1)
  pts1_h = np.hstack([pts1, np.ones((n, 1))])
  pts2_h = np.hstack([pts2, np.ones((n, 1))])
  Ex1 = (E @ pts1_h.T).T
  Etx2 = (E.T @ pts2_h.T).T
  numerator = np.sum(pts2_h * Ex1, axis=1)**2
  denominator = (Ex1[:, 0]**2 + Ex1[:, 1]**2 + Etx2[:, 0]**2 + Etx2[:, 1]**2)
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.where(denominator > 1e-12, numerator / denominator, 0.0)


def ransac_5pt_lui(pts1, pts2, max_iters=500, threshold=1e-4):
  """RANSAC wrapper around the Lui 5-point solver.

  Randomly samples 5-point subsets, fits the model, and returns the
  pose with the most inliers (Sampson distance < threshold).
  The winning hypothesis is refined on all inlier points.
  """
  N = len(pts1)
  best_R, best_t = np.eye(3), np.zeros(3)
  best_score = -1

  # Precompute bearing vectors for all points
  v1_all = np.column_stack([pts1, np.ones(N)])
  v1_all /= np.linalg.norm(v1_all, axis=1, keepdims=True)
  v2_all = np.column_stack([pts2, np.ones(N)])
  v2_all /= np.linalg.norm(v2_all, axis=1, keepdims=True)

  # RANSAC
  for _ in range(max_iters):
    idx = np.random.choice(N, 5, replace=False)
    try:
      R, t = lui_5point_algorithm(v1_all[idx], v2_all[idx])
    except Exception:
      continue

    E = skew(t) @ R
    scores = _sampson_per_point(E, pts1, pts2)
    inliers = int(np.sum(scores < threshold))

    if inliers > best_score:
      best_score = inliers
      best_R, best_t = R.copy(), t.copy()

  # Refinement: re-fit on all inliers
  E_best = skew(best_t) @ best_R
  scores = _sampson_per_point(E_best, pts1, pts2)
  inlier_mask = scores < threshold
  if np.sum(inlier_mask) >= 5:
    try:
      R_ref, t_ref = lui_5point_algorithm(v1_all[inlier_mask], v2_all[inlier_mask],
                                   max_iters=100, tol=1e-12)
      # Accept refinement if it doesn't regress
      E_ref = skew(t_ref) @ R_ref
      scores_ref = _sampson_per_point(E_ref, pts1, pts2)
      if np.sum(scores_ref < threshold) >= best_score:
        best_R, best_t = R_ref, t_ref
    except Exception:
      pass

  return best_R, best_t


###############################################################################
# OPENCV 5-POINT ALGORITHM
###############################################################################


def _try_decompose_essential(E_cv, pts1_norm, pts2_norm):
  """Decompose E and recover pose using OpenCV's recoverPose."""
  K = np.eye(3)
  _, R_cv, t_cv, _ = cv2.recoverPose(E_cv, pts1_norm, pts2_norm, K)
  return R_cv, t_cv.ravel()


def opencv_5point_algorithm(pts1_norm, pts2_norm):
  """
  Estimates relative pose using OpenCV's built-in 5-point solver safely.
  """
  # 1. Estimate Essential Matrix
  E_cv, _ = cv2.findEssentialMat(pts1_norm,
                                 pts2_norm,
                                 cameraMatrix=np.eye(3),
                                 method=cv2.RANSAC,
                                 prob=0.99,
                                 threshold=1e-3)

  # 2. Check if E_cv is None or empty
  if E_cv is None or E_cv.size == 0:
    raise RuntimeError("OpenCV findEssentialMat failed to find a matrix!")

  # 3. Handle cases where multiple essential matrices are returned.
  # OpenCV stacks them horizontally (3x(3*N)) or vertically (N*3 x 3).
  candidates = []
  if E_cv.shape == (3, 3):
    candidates = [E_cv]
  elif E_cv.shape[0] == 3 and E_cv.shape[1] > 3:
    for i in range(E_cv.shape[1] // 3):
      candidates.append(E_cv[:, i * 3:(i + 1) * 3])
  elif E_cv.shape[1] == 3 and E_cv.shape[0] > 3:
    for i in range(E_cv.shape[0] // 3):
      candidates.append(E_cv[i * 3:(i + 1) * 3, :])
  else:
    raise RuntimeError(f"Unexpected Essential Matrix shape: {E_cv.shape}")

  # Try each candidate and return the one with most cheirally consistent points
  best_R, best_t = np.eye(3), np.zeros(3)
  best_count = -1
  for E_candidate in candidates:
    R_cv, t_cv = _try_decompose_essential(E_candidate, pts1_norm, pts2_norm)
    count = cheirality_check(R_cv, t_cv, pts1_norm, pts2_norm)
    if count > best_count:
      best_count = count
      best_R, best_t = R_cv, t_cv

  return best_R, best_t


###############################################################################
# VERIFICATION & COMPARISON BENCHMARK
###############################################################################

def run_benchmark(pts1, pts2, R_gt, t_gt, label, noise_sigma=0.0):
  """Run all solvers on the given point correspondences and print results."""
  N = len(pts1)

  # Build unit bearing vectors for raw Lui solver (first 5 points)
  v1 = np.column_stack([pts1[:5], np.ones(5)])
  v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
  v2 = np.column_stack([pts2[:5], np.ones(5)])
  v2 /= np.linalg.norm(v2, axis=1, keepdims=True)

  import time

  t0 = time.time()
  R_custom, t_custom = nister_5point_algorithm(pts1, pts2)
  t_custom_elapsed = time.time() - t0

  t0 = time.time()
  R_cv, t_cv = opencv_5point_algorithm(pts1, pts2)
  t_cv_elapsed = time.time() - t0

  t0 = time.time()
  R_lui_raw, t_lui_raw = lui_5point_algorithm(v1, v2)
  t_lui_raw_elapsed = time.time() - t0

  t0 = time.time()
  R_lui_ransac, t_lui_ransac = ransac_5pt_lui(pts1, pts2, max_iters=500, threshold=1e-8)
  t_lui_ransac_elapsed = time.time() - t0

  # Align sign of translation vectors
  for t in [t_cv, t_custom, t_lui_raw, t_lui_ransac]:
    if np.dot(t, t_gt) < 0:
      t *= -1

  results = []
  for name, R, t, elapsed in [
      ("OpenCV 5-pt", R_cv, t_cv, t_cv_elapsed),
      ("Nistér 5-pt", R_custom, t_custom, t_custom_elapsed),
      ("Lui raw 5-pt", R_lui_raw, t_lui_raw, t_lui_raw_elapsed),
      ("Lui RANSAC", R_lui_ransac, t_lui_ransac, t_lui_ransac_elapsed),
  ]:
    R_err = np.linalg.norm(R_gt - R)
    t_ang = np.degrees(np.arccos(np.clip(np.dot(t_gt, t), -1.0, 1.0)))
    results.append((name, R_err, t_ang, elapsed))

  header = f"--- {label} (noise σ={noise_sigma:.0e}) ---"
  print(f"\n{'='*len(header)}")
  print(header)
  print(f"{'='*len(header)}")
  print(f"{'Method':<20} {'R Err (Frob)':<16} {'t Err (deg)':<16} {'Time':<12}")
  print("-" * 64)
  for name, R_err, t_ang, elapsed in results:
    print(f"{name:<20} {R_err:<16.2e} {t_ang:<16.4f} {elapsed*1000:<12.1f} ms")
  print()

  return results


if __name__ == "__main__":
  np.random.seed(42)

  theta = np.radians(15.0)
  R_gt = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
  t_gt = np.array([0.5, -0.2, 0.84])
  t_gt /= np.linalg.norm(t_gt)

  N = 20
  X_3D = np.random.uniform(-1, 1, (N, 3))
  X_3D[:, 2] += 3.0

  pts1_clean = X_3D[:, :2] / X_3D[:, 2:]
  X_cam2 = (R_gt @ X_3D.T).T + t_gt
  pts2_clean = X_cam2[:, :2] / X_cam2[:, 2:]

  # Add noise to image correspondences
  NOISE_SIGMAS = [0.0, 1e-4, 5e-4, 1e-3, 5e-3]

  for noise_sigma in NOISE_SIGMAS:
    pts1_noisy = pts1_clean.copy()
    pts2_noisy = pts2_clean.copy()
    if noise_sigma > 0:
      pts1_noisy += np.random.randn(N, 2) * noise_sigma
      pts2_noisy += np.random.randn(N, 2) * noise_sigma

    label = f"Noise σ={noise_sigma:.0e}"
    if noise_sigma == 0.0:
      label = "Noiseless"

    run_benchmark(pts1_noisy, pts2_noisy, R_gt, t_gt, label, noise_sigma)
