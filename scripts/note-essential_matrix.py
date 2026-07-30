import cv2
import numpy as np

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


# Monomial column lookup for _build_constraint_matrix.
# Variable encoding: 0→1 (constant), 1→x, 2→y, 3→z.
# Column layout (same as original sympy version):
#   0-9:   degree 3 (x^3 … z^3)
#   10-15: degree 2 (x^2 … z^2)
#   16-18: degree 1 (x, y, z)
#   19:    degree 0 (1)
_MONOMIAL_COL = np.zeros((4, 4, 4), dtype=np.int32)
_C3 = {
    (3, 0, 0): 0,
    (2, 1, 0): 1,
    (2, 0, 1): 2,
    (1, 2, 0): 3,
    (1, 1, 1): 4,
    (1, 0, 2): 5,
    (0, 3, 0): 6,
    (0, 2, 1): 7,
    (0, 1, 2): 8,
    (0, 0, 3): 9
}
_C2 = {
    (2, 0, 0): 10,
    (1, 1, 0): 11,
    (1, 0, 1): 12,
    (0, 2, 0): 13,
    (0, 1, 1): 14,
    (0, 0, 2): 15
}
_C1 = {(1, 0, 0): 16, (0, 1, 0): 17, (0, 0, 1): 18}
for v1 in range(4):
  for v2 in range(4):
    for v3 in range(4):
      cnt = [0, 0, 0]
      for v in (v1, v2, v3):
        if v > 0:
          cnt[v - 1] += 1
      a, b, c = cnt
      d = a + b + c
      if d == 3:
        _MONOMIAL_COL[v1, v2, v3] = _C3[(a, b, c)]
      elif d == 2:
        _MONOMIAL_COL[v1, v2, v3] = _C2[(a, b, c)]
      elif d == 1:
        _MONOMIAL_COL[v1, v2, v3] = _C1[(a, b, c)]
      else:
        _MONOMIAL_COL[v1, v2, v3] = 19


def _build_constraint_matrix(Ex, Ey, Ez, Ew):
  """
  Substitutes E = x*Ex + y*Ey + z*Ez + Ew into the 9 matrix constraints:

    2 * E * E^T * E - trace(E * E^T) * E = 0

  and det(E) = 0. Returns a 10x20 coefficient matrix.

  Computes the polynomial coefficients numerically (no symbolic algebra).
  """
  # Stack basis coefficients: C[p,q] = [Ew[p,q], Ex[p,q], Ey[p,q], Ez[p,q]]
  C = np.stack([Ew, Ex, Ey, Ez], axis=-1)

  M = np.zeros((10, 20))

  # Accumulate coefficients from a triple product of three E entries.
  # L1 * L2 * L3 where each Li = Σ_k C[pi,qi,k] * var_k
  def _acc(row, p1, q1, p2, q2, p3, q3, scale):
    c1 = C[p1, q1]
    c2 = C[p2, q2]
    c3 = C[p3, q3]
    for i in range(4):
      ci = c1[i]
      if ci == 0:
        continue
      for j in range(4):
        cij = ci * c2[j]
        if cij == 0:
          continue
        for k in range(4):
          ck = c3[k]
          if ck == 0:
            continue
          M[row, _MONOMIAL_COL[i, j, k]] += scale * cij * ck

  # 9 matrix constraints: C[i,j] = 2*E*E^T*E - trace(E*E^T)*E = 0
  #
  # C[i,j] = 2 * Σ_k Σ_l E[i,l] * E[k,l] * E[k,j]
  #        - Σ_m Σ_n E[m,n] * E[m,n] * E[i,j]
  row = 0
  for i in range(3):
    for j in range(3):
      for k in range(3):
        for l in range(3):
          _acc(row, i, l, k, l, k, j, 2.0)
      for m in range(3):
        for n in range(3):
          _acc(row, m, n, m, n, i, j, -1.0)
      row += 1

  # det(E) = 0  (Leibniz formula)
  for p1, q1, p2, q2, p3, q3, s in [
      (0, 0, 1, 1, 2, 2, 1.0),
      (0, 1, 1, 2, 2, 0, 1.0),
      (0, 2, 1, 0, 2, 1, 1.0),
      (0, 0, 1, 2, 2, 1, -1.0),
      (0, 1, 1, 0, 2, 2, -1.0),
      (0, 2, 1, 1, 2, 0, -1.0),
  ]:
    _acc(9, p1, q1, p2, q2, p3, q3, s)

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
  """Estimate relative pose using the Nister 5-point algorithm."""
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


def rotation_from_vectors(a, b):
  """Find R such that R @ a = b for unit vectors a, b."""
  v = np.cross(a, b)
  s = np.linalg.norm(v)
  c = np.dot(a, b)
  if s < 1e-12:
    return np.eye(3) if c > 0 else -np.eye(3)
  V = skew(v)
  return np.eye(3) + V + V @ V * (1.0 - c) / (s * s)


def lui_5point_algorithm(v1,
                         v2,
                         max_iters=100,
                         tol=1e-20,
                         R_init=None,
                         t_init=None):
  """
  Iterative 5-point solver by Vincent Lui & Tom Drummond.

  v1, v2: (N, 3) arrays of unit bearing vectors in camera 1 and camera 2 frames,
           where N >= 5.
  R_init, t_init: optional initial pose guess. If provided, the solver
                  initializes from this pose instead of identity.
  Returns: R (3x3) relative rotation from camera 1 to camera 2,
           t (3,) translation direction in camera 2's frame.
  """
  v1, v2 = v1.copy(), v2.copy()
  assert v1.shape[1] == 3 and v2.shape[1] == 3
  assert v1.shape[0] >= 5 and v2.shape[0] >= 5

  if R_init is not None and t_init is not None:
    R2 = rotation_from_vectors(t_init / np.linalg.norm(t_init),
                               np.array([0.0, 0.0, 1.0]))
    R1 = R2 @ R_init
  else:
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
    sq1 = x1 * x1 + y1 * y1
    sq2 = x2 * x2 + y2 * y2
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
          R1_test,
          rodrigues(w2_test) @ R2, v1, v2)
      new_err = np.linalg.norm(r_test)

      if new_err < prev_err:
        R1 = R1_test
        R2 = rodrigues(w2_test) @ R2
        r, p1, p2 = r_test, p1_test, p2_test

        rel_change = abs(prev_err - new_err) / max(float(prev_err), 1e-12)
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


def _sampson_from_pose(R, t, pts1, pts2):
  """Per-point Sampson distance from pose (R, t) for RANSAC scoring."""
  E = skew(t) @ R
  return _sampson_per_point(E, pts1, pts2)


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

  # Refinement: re-fit on all inliers (initialize from best RANSAC hypothesis)
  E_best = skew(best_t) @ best_R
  scores = _sampson_per_point(E_best, pts1, pts2)
  inlier_mask = scores < threshold
  if np.sum(inlier_mask) >= 5:
    try:
      R_ref, t_ref = lui_5point_algorithm(v1_all[inlier_mask],
                                          v2_all[inlier_mask],
                                          max_iters=100,
                                          tol=1e-12,
                                          R_init=best_R,
                                          t_init=best_t)
      # Accept refinement if it doesn't regress
      E_ref = skew(t_ref) @ R_ref
      scores_ref = _sampson_per_point(E_ref, pts1, pts2)
      if np.sum(scores_ref < threshold) >= best_score:
        best_R, best_t = R_ref, t_ref
    except Exception:
      pass

  return best_R, best_t


###############################################################################
# HEDBORG 5-POINT ALGORITHM (Levenberg-Marquardt)
###############################################################################


def _essential_from_params(w):
  """Extract essential matrix E from parameter vector w = [ax, ay, az, theta, phi].

  R = rodrigues(w[:3]),  t = [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)].
  """
  R = rodrigues(w[:3])
  theta, phi = w[3], w[4]
  st = np.sin(theta)
  t = np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])
  return skew(t) @ R, R, t


def _epipolar_dist_residuals(w, pts1_h, pts2_h):
  """Epipolar distance r_i = (x'^T E x) / sqrt((x'^T E)_1^2 + (x'^T E)_2^2)."""
  E, _, _ = _essential_from_params(w)
  Ex1 = (E @ pts1_h.T).T
  numerator = np.sum(pts2_h * Ex1, axis=1)
  denominator = np.sqrt(Ex1[:, 0]**2 + Ex1[:, 1]**2)
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.where(denominator > 1e-12, numerator / denominator, 0.0)


def _so3_right_jacobian(r):
  """Right Jacobian of SO(3) exponential map at r (axis-angle vector)."""
  theta = np.linalg.norm(r)
  if theta < 1e-8:
    return np.eye(3)
  rx = skew(r / theta)
  a = (1.0 - np.cos(theta)) / theta
  b = (theta - np.sin(theta)) / theta
  return np.eye(3) - a * rx + b * rx @ rx


def _dE_dparams(w):
  """Analytical derivatives of E w.r.t. each parameter in w.

  Returns list [dE_da, dE_db, dE_dg, dE_dtheta, dE_dphi] of 3x3 arrays.
  """
  R = rodrigues(w[:3])
  theta, phi = w[3], w[4]
  st, ct = np.sin(theta), np.cos(theta)
  sp, cp = np.sin(phi), np.cos(phi)
  t = np.array([st * cp, st * sp, ct])

  # Translation derivatives
  dt_dtheta = np.array([ct * cp, ct * sp, -st])
  dt_dphi = np.array([-st * sp, st * cp, 0.0])

  # Rotation derivatives: dR/dr_k = R @ skew(J_r[:, k])
  Jr = _so3_right_jacobian(w[:3])
  dE = []
  for k in range(3):
    dR = R @ skew(Jr[:, k])
    dE.append(skew(t) @ dR)
  dE.append(skew(dt_dtheta) @ R)
  dE.append(skew(dt_dphi) @ R)
  return dE


def _jacobian_epipolar(w, pts1_h, pts2_h):
  """Analytical Nx5 Jacobian of epipolar distance residuals."""
  E, _, _ = _essential_from_params(w)
  dE = _dE_dparams(w)
  N = len(pts1_h)
  J = np.zeros((N, 5))

  Ex1 = (E @ pts1_h.T).T
  a = Ex1[:, 0]  # (x'^T E)_1
  b = Ex1[:, 1]  # (x'^T E)_2
  d_sq = a * a + b * b
  d = np.sqrt(d_sq)
  n = np.sum(pts2_h * Ex1, axis=1)

  for j in range(5):
    dEx1 = (dE[j] @ pts1_h.T).T
    dn = np.sum(pts2_h * dEx1, axis=1)
    da = dEx1[:, 0]
    db = dEx1[:, 1]
    dd = np.where(d > 1e-12, (a * da + b * db) / d, 0.0)

    denom = d_sq + 1e-12
    J[:, j] = (dn * d - n * dd) / denom

  return J


def _lm_solve(w, pts1_h, pts2_h, max_iters, tol):
  """Run Levenberg-Marquardt optimization from a single initial w.

  Uses analytical Jacobians of the epipolar distance residuals.

  Returns (w_opt, cost) where cost = 0.5 * ||r||^2.
  """
  lam = 1e-3
  nu = 2.0
  eps_jac = 1e-8

  r = _epipolar_dist_residuals(w, pts1_h, pts2_h)
  cost = 0.5 * np.dot(r, r)

  for _ in range(max_iters):
    J = _jacobian_epipolar(w, pts1_h, pts2_h)
    g = J.T @ r
    H = J.T @ J

    if np.linalg.norm(g, np.inf) < tol:
      break

    diag_H = np.diag(H)
    accepted = False
    for _ in range(30):
      try:
        h = np.linalg.solve(H + lam * np.diag(diag_H), -g)
      except np.linalg.LinAlgError:
        lam *= nu
        nu *= 2
        continue

      w_new = w + h
      r_new = _epipolar_dist_residuals(w_new, pts1_h, pts2_h)
      new_cost = 0.5 * np.dot(r_new, r_new)

      l_pred = r + J @ h
      pred_cost = 0.5 * np.dot(l_pred, l_pred)
      actual_reduction = cost - new_cost
      pred_reduction = cost - pred_cost

      rho = 0.0 if abs(
          pred_reduction) < 1e-16 else actual_reduction / pred_reduction

      if rho > 0:
        w = w_new
        r = r_new
        cost = new_cost
        lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0)**3)
        nu = 2.0
        accepted = True
        if np.linalg.norm(h) < tol:
          return w, cost
        break
      else:
        lam *= nu
        nu *= 2.0

    if not accepted:
      break

  return w, cost


def _generate_seeds(n_seeds, rng):
  """Generate candidate initial parameter vectors for multi-start."""
  seeds = [np.zeros(5)]
  for _ in range(n_seeds - 1):
    w = np.zeros(5)
    w[:3] = rng.uniform(-0.5, 0.5, 3)
    theta = rng.uniform(0.1, np.pi - 0.1)
    phi = rng.uniform(-np.pi, np.pi)
    w[3] = theta
    w[4] = phi
    seeds.append(w)
  return seeds


def hedborg_5point_algorithm(pts1_norm,
                             pts2_norm,
                             max_iters=50,
                             tol=1e-20,
                             R_init=None,
                             t_init=None):
  """Fast iterative 5-point relative pose via Levenberg-Marquardt (Hedborg & Felsberg 2013).

  Uses multi-start initialisation: tries several candidate parameter vectors
  and picks the one with the lowest final epipolar cost.

  The solver parameterises the essential matrix with 5 parameters
  w = [ax, ay, az, theta, phi] (axis-angle rotation + spherical translation)
  and minimises the epipolar distance using LM with analytical Jacobians.

  Args:
    pts1_norm, pts2_norm: (N, 2) normalized image coordinates (N >= 5).
    max_iters: maximum LM iterations per seed.
    tol: convergence threshold (gradient infinity-norm).
    R_init: optional 3x3 initial rotation.
    t_init: optional 3D initial translation direction.

  Returns:
    R (3x3), t (3,) describing the relative pose.
  """
  n_pts = len(pts1_norm)
  pts1_h = np.column_stack([pts1_norm, np.ones(n_pts)])
  pts2_h = np.column_stack([pts2_norm, np.ones(n_pts)])

  rng = np.random.default_rng(0)
  n_seeds = 4 if R_init is None else 1

  if R_init is not None and t_init is not None:
    w_rot = cv2.Rodrigues(R_init)[0].ravel()
    t_dir = t_init / np.linalg.norm(t_init)
    theta = np.arccos(np.clip(t_dir[2], -1.0, 1.0))
    phi = np.arctan2(t_dir[1], t_dir[0])
    single_seed = np.array([w_rot[0], w_rot[1], w_rot[2], theta, phi])
    seeds = [single_seed]
  else:
    seeds = _generate_seeds(n_seeds, rng)

  best_w = seeds[0]
  best_cost = float('inf')
  for seed in seeds:
    w_opt, cost = _lm_solve(seed.copy(), pts1_h, pts2_h, max_iters, tol)
    if cost < best_cost:
      best_cost = cost
      best_w = w_opt

  _, R, t = _essential_from_params(best_w)
  t = t / np.linalg.norm(t)
  return R, t


def ransac_5pt_hedborg(pts1, pts2, max_iters=500, threshold=1e-4):
  """RANSAC wrapper around the Hedborg 5-point solver.

  Uses inlier count as primary score and total Sampson distance as
  tiebreaker, then refines the winner on all inliers.

  Args:
    pts1, pts2: (N, 2) image point correspondences.
    max_iters: number of RANSAC iterations.
    threshold: Sampson distance inlier threshold.

  Returns:
    R (3x3), t (3,).
  """
  N = len(pts1)
  best_R, best_t = np.eye(3), np.zeros(3)
  best_score = -1
  best_sampson = float('inf')

  for _ in range(max_iters):
    idx = np.random.choice(N, 5, replace=False)
    try:
      R, t = hedborg_5point_algorithm(pts1[idx], pts2[idx])
    except Exception:
      continue

    E = skew(t) @ R
    scores = _sampson_per_point(E, pts1, pts2)
    inliers = int(np.sum(scores < threshold))
    total_sampson = np.sum(scores)

    if inliers > best_score or (inliers == best_score and
                                total_sampson < best_sampson):
      best_score = inliers
      best_sampson = total_sampson
      best_R, best_t = R.copy(), t.copy()

  # Refinement: re-fit on all inliers (initialize from best RANSAC hypothesis)
  E_best = skew(best_t) @ best_R
  scores = _sampson_per_point(E_best, pts1, pts2)
  inlier_mask = scores < threshold
  if np.sum(inlier_mask) >= 5:
    try:
      R_ref, t_ref = hedborg_5point_algorithm(pts1[inlier_mask],
                                              pts2[inlier_mask],
                                              max_iters=20,
                                              tol=1e-12,
                                              R_init=best_R,
                                              t_init=best_t)
      E_ref = skew(t_ref) @ R_ref
      scores_ref = _sampson_per_point(E_ref, pts1, pts2)
      if np.sum(scores_ref < threshold) >= best_score:
        best_R, best_t = R_ref, t_ref
    except Exception:
      pass

  return best_R, best_t


###############################################################################
# PRE-EMPTIVE RANSAC (shared scoring scheme for any solver)
###############################################################################


def ransac_preemptive(pts1,
                      pts2,
                      solver_fn,
                      sample_size=5,
                      num_hypotheses=100,
                      threshold=1e-4,
                      stage_sizes=None,
                      keep_frac=0.5,
                      min_inliers=None,
                      refine=True,
                      refine_fn=None,
                      rng=None):
  """Pre-emptive RANSAC with progressive hypothesis pruning.

  Generates ``num_hypotheses`` model hypotheses from random minimal
  samples, then evaluates them in stages on increasingly large subsets
  of the data. At each stage the worst-performing hypotheses are
  discarded, focusing computation on the most promising candidates.

  This implements the pre-emptive scoring scheme popularised by
  Nister's pre-emptive RANSAC.

  All three 5-point solvers (Nister, Lui, Hedborg) use this same
  function via thin adapter wrappers below.

  Args:
    pts1, pts2: (N, 2) normalized image coordinates.
    solver_fn: callable(pts1_sample, pts2_sample) -> (R, t).
    sample_size: minimal sample size (default 5).
    num_hypotheses: number of hypotheses to generate (default 500).
    threshold: Sampson distance inlier threshold (default 1e-4).
    stage_sizes: explicit list of subset sizes for each pre-emptive
        stage. If None, computed as a geometric series from
        10*sample_size to N with log2(num_hypotheses) steps.
    keep_frac: fraction of hypotheses to keep at each stage (default 0.5).
    min_inliers: early termination threshold.
    refine: if True, refit on all inliers after pre-emptive scoring.
    refine_fn: optional callable(pts1, pts2, R_init, t_init) -> (R, t)
        for refinement. If provided, used instead of ``solver_fn`` so
        the solver can initialise from the best hypothesis.
    rng: numpy random generator.

  Returns:
    (R, t) relative pose with the most inliers.
  """
  N = len(pts1)
  if rng is None:
    rng = np.random.default_rng()

  # Stage 0: generate candidate hypotheses
  hypotheses = []
  for _ in range(num_hypotheses):
    idx = rng.choice(N, sample_size, replace=False)
    try:
      R, t = solver_fn(pts1[idx], pts2[idx])
      if R is not None:
        hypotheses.append((R, t))
    except Exception:
      continue

  if not hypotheses:
    return np.eye(3), np.zeros(3)

  M = len(hypotheses)

  # Determine pre-emptive stage sizes (geometric progression)
  if stage_sizes is None:
    n_stages = max(2, int(np.log2(M)))
    start = min(N, max(sample_size * 10, 50))
    stage_sizes = np.geomspace(start, N, n_stages).astype(int)
    stage_sizes = np.unique(stage_sizes)
    if len(stage_sizes) == 0 or stage_sizes[-1] < N:
      stage_sizes = np.append(stage_sizes, N)
  else:
    stage_sizes = [s for s in stage_sizes if s >= sample_size]
    if len(stage_sizes) == 0:
      stage_sizes = [N]
    elif stage_sizes[-1] < N:
      stage_sizes.append(N)

  # Fixed permutation so all hypotheses see the same subsets
  perm = rng.permutation(N)
  active = list(range(M))

  # Pre-emptive stages
  for sz in stage_sizes:
    if len(active) <= 1:
      break
    n_test = min(sz, N)
    subset = perm[:n_test]

    inliers_arr = np.zeros(len(active))
    sampson_arr = np.full(len(active), float('inf'))

    for i, hyp_idx in enumerate(active):
      R, t = hypotheses[hyp_idx]
      E = skew(t) @ R
      scores = _sampson_per_point(E, pts1[subset], pts2[subset])
      inliers_arr[i] = np.sum(scores < threshold)
      sampson_arr[i] = np.sum(scores)

    n_keep = max(1, int(len(active) * keep_frac))
    order = np.lexsort((sampson_arr, -inliers_arr))
    active = [active[i] for i in order[:n_keep]]

    if min_inliers is not None and inliers_arr[order[0]] >= min_inliers:
      break

  # Final scoring on all points
  best_R, best_t = np.eye(3), np.zeros(3)
  best_score = -1
  best_sampson = float('inf')

  for hyp_idx in active:
    R, t = hypotheses[hyp_idx]
    E = skew(t) @ R
    scores = _sampson_per_point(E, pts1, pts2)
    inliers = int(np.sum(scores < threshold))
    total = np.sum(scores)

    if inliers > best_score or (inliers == best_score and total < best_sampson):
      best_score = inliers
      best_sampson = total
      best_R, best_t = R.copy(), t.copy()

  # Refinement on all inliers
  if refine:
    E_best = skew(best_t) @ best_R
    scores = _sampson_per_point(E_best, pts1, pts2)
    mask = scores < threshold
    if np.sum(mask) >= sample_size:
      try:
        if refine_fn is not None:
          R_ref, t_ref = refine_fn(pts1[mask], pts2[mask], best_R, best_t)
        else:
          R_ref, t_ref = solver_fn(pts1[mask], pts2[mask])
        if R_ref is not None:
          t_ref = t_ref / (np.linalg.norm(t_ref) + 1e-12)
          E_ref = skew(t_ref) @ R_ref
          scores_ref = _sampson_per_point(E_ref, pts1, pts2)
          if np.sum(scores_ref < threshold) >= best_score:
            best_R, best_t = R_ref, t_ref
      except Exception:
        pass

  return best_R, best_t


def _lui_adapter(pts1_sample, pts2_sample):
  """Adapter: (N,2) image points -> bearing vectors for Lui solver."""
  v1 = np.column_stack([pts1_sample, np.ones(len(pts1_sample))])
  v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
  v2 = np.column_stack([pts2_sample, np.ones(len(pts2_sample))])
  v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
  return lui_5point_algorithm(v1, v2)


def ransac_5pt_nister_preemptive(pts1, pts2, **kwargs):
  """Pre-emptive RANSAC with the Nister 5-point solver."""
  return ransac_preemptive(pts1, pts2, nister_5point_algorithm, **kwargs)


def ransac_5pt_lui_preemptive(pts1, pts2, **kwargs):
  """Pre-emptive RANSAC with the Lui 5-point solver."""
  return ransac_preemptive(pts1, pts2, _lui_adapter, **kwargs)


def _hedborg_refine(pts1, pts2, R_init, t_init):
  """Refine Hedborg from an initial guess (avoids multi-start)."""
  return hedborg_5point_algorithm(pts1,
                                  pts2,
                                  max_iters=20,
                                  tol=1e-12,
                                  R_init=R_init,
                                  t_init=t_init)


def ransac_5pt_hedborg_preemptive(pts1, pts2, **kwargs):
  """Pre-emptive RANSAC with the Hedborg 5-point solver.

  The refinement step initialises the LM solver from the best RANSAC
  hypothesis, avoiding random multi-start on the inlier set.
  """
  kwargs.setdefault('refine_fn', _hedborg_refine)
  return ransac_preemptive(pts1, pts2, hedborg_5point_algorithm, **kwargs)


###############################################################################
# VERIFICATION & COMPARISON BENCHMARK
###############################################################################


def _run_solvers(pts1, pts2, R_gt, t_gt, ransac_iters=200):
  """Run all solvers once and return dict of {name: (R_err, t_ang, elapsed_ms)}."""
  N = len(pts1)
  import time

  v1_all = np.column_stack([pts1, np.ones(N)])
  v1_all /= np.linalg.norm(v1_all, axis=1, keepdims=True)
  v2_all = np.column_stack([pts2, np.ones(N)])
  v2_all /= np.linalg.norm(v2_all, axis=1, keepdims=True)

  def _t_err(t):
    if np.dot(t, t_gt) < 0:
      t *= -1
    return np.degrees(np.arccos(np.clip(np.dot(t_gt, t), -1.0, 1.0)))

  def _r_err(R):
    return np.linalg.norm(R_gt - R)

  out = {}

  t0 = time.time()
  R, t = nister_5point_algorithm(pts1, pts2)
  if R is None:
    R, t = np.eye(3), np.zeros(3)
  out['Nister'] = (_r_err(R), _t_err(t), (time.time() - t0) * 1000)

  t0 = time.time()
  R, t = opencv_5point_algorithm(pts1, pts2)
  out['OpenCV'] = (_r_err(R), _t_err(t), (time.time() - t0) * 1000)

  t0 = time.time()
  R, t = lui_5point_algorithm(v1_all[:5], v2_all[:5])
  out['Lui'] = (_r_err(R), _t_err(t), (time.time() - t0) * 1000)

  t0 = time.time()
  R, t = hedborg_5point_algorithm(pts1, pts2)
  out['Hedborg'] = (_r_err(R), _t_err(t), (time.time() - t0) * 1000)

  t0 = time.time()
  R, t = ransac_5pt_nister_preemptive(pts1,
                                      pts2,
                                      num_hypotheses=ransac_iters,
                                      threshold=1e-4)
  out['Nister RANSAC'] = (_r_err(R), _t_err(t), (time.time() - t0) * 1000)

  t0 = time.time()
  R, t = ransac_5pt_lui_preemptive(pts1,
                                   pts2,
                                   num_hypotheses=ransac_iters,
                                   threshold=1e-4)
  out['Lui RANSAC'] = (_r_err(R), _t_err(t), (time.time() - t0) * 1000)

  t0 = time.time()
  R, t = ransac_5pt_hedborg_preemptive(pts1,
                                       pts2,
                                       num_hypotheses=ransac_iters,
                                       threshold=1e-4)
  out['Hedborg RANSAC'] = (_r_err(R), _t_err(t), (time.time() - t0) * 1000)

  return out


def _add_pixel_noise(pts, sigma_px, f=500.0, rng=None):
  """Add zero-mean Gaussian noise in pixel space, then convert back."""
  if rng is None:
    rng = np.random
  if sigma_px <= 0:
    return pts.copy()
  return pts + rng.randn(*pts.shape) * sigma_px / f


def benchmark_summary(R_gt,
                      t_gt,
                      pts1_clean,
                      pts2_clean,
                      noise_sigmas_px=(0.0, 0.5, 1.0, 2.0, 4.0),
                      focal_length=500.0,
                      n_trials=10,
                      ransac_iters=100):
  """Run multiple trials at each pixel-noise level and print statistical summary."""
  method_names = [
      'OpenCV',
      'Nister',
      'Lui',
      'Hedborg',
      'Nister RANSAC',
      'Lui RANSAC',
      'Hedborg RANSAC',
  ]

  for sigma_px in noise_sigmas_px:
    r_errs = {m: [] for m in method_names}
    t_errs = {m: [] for m in method_names}

    for _ in range(n_trials):
      pts1 = _add_pixel_noise(pts1_clean, sigma_px, focal_length)
      pts2 = _add_pixel_noise(pts2_clean, sigma_px, focal_length)

      try:
        out = _run_solvers(pts1, pts2, R_gt, t_gt, ransac_iters=ransac_iters)
        for m in method_names:
          if m in out:
            r_errs[m].append(out[m][0])
            t_errs[m].append(out[m][1])
      except Exception:
        pass

    def _fmt(a):
      if len(a) == 0:
        return f"{'—':>28}"
      return f"{a.mean():>12.4e} ± {a.std():>12.4e}"

    hdr = f"Noise {sigma_px:.1f} px"
    print(f"\n{'='*len(hdr)}")
    print(hdr)
    print(f"{'='*len(hdr)}")
    print(f"{'Method':<20} {'R_err':>28} {'t_err (deg)':>28}  {'n':>4}")
    print("-" * 84)
    for m in method_names:
      r = np.array(r_errs[m])
      t = np.array(t_errs[m])
      print(f"{m:<20} {_fmt(r)}  {_fmt(t)}  {len(r):>4}")


def outlier_benchmark(R_gt,
                      t_gt,
                      pts1_clean,
                      pts2_clean,
                      outlier_fracs=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                      noise_sigma_px=1.0,
                      focal_length=500.0,
                      n_trials=10,
                      ransac_iters=100):
  """Benchmark all solvers with varying fractions of outlier correspondences.

  At each outlier fraction, a subset of point pairs are replaced with
  random uniform correspondences (outliers), and mild Gaussian noise
  (in pixels) is added to all points.
  """
  method_names = [
      'OpenCV',
      'Nister',
      'Lui',
      'Hedborg',
      'Nister RANSAC',
      'Lui RANSAC',
      'Hedborg RANSAC',
  ]

  N = len(pts1_clean)
  for outlier_frac in outlier_fracs:
    r_errs = {m: [] for m in method_names}
    t_errs = {m: [] for m in method_names}

    for _ in range(n_trials):
      pts1 = _add_pixel_noise(pts1_clean, noise_sigma_px, focal_length)
      pts2 = _add_pixel_noise(pts2_clean, noise_sigma_px, focal_length)

      if outlier_frac > 0:
        n_out = max(1, int(N * outlier_frac))
        idx = np.random.choice(N, n_out, replace=False)
        bound = max(np.abs(pts1_clean).max(), np.abs(pts2_clean).max()) * 2
        pts1[idx] = np.random.uniform(-bound, bound, (n_out, 2))
        pts2[idx] = np.random.uniform(-bound, bound, (n_out, 2))

      try:
        out = _run_solvers(pts1, pts2, R_gt, t_gt, ransac_iters=ransac_iters)
        for m in method_names:
          if m in out:
            r_errs[m].append(out[m][0])
            t_errs[m].append(out[m][1])
      except Exception:
        pass

    def _fmt(a):
      if len(a) == 0:
        return f"{'—':>28}"
      return f"{a.mean():>12.4e} ± {a.std():>12.4e}"

    hdr = f"Outliers {outlier_frac*100:.0f}%  (noise {noise_sigma_px:.1f} px)"
    print(f"\n{'='*len(hdr)}")
    print(hdr)
    print(f"{'='*len(hdr)}")
    print(f"{'Method':<20} {'R_err':>28} {'t_err (deg)':>28}  {'n':>4}")
    print("-" * 84)
    for m in method_names:
      r = np.array(r_errs[m])
      t = np.array(t_errs[m])
      print(f"{m:<20} {_fmt(r)}  {_fmt(t)}  {len(r):>4}")


if __name__ == "__main__":
  np.random.seed(42)

  theta = np.radians(15.0)
  R_gt = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
  t_gt = np.array([0.5, -0.2, 0.84])
  t_gt /= np.linalg.norm(t_gt)

  N = 60
  X_3D = np.random.uniform(-1, 1, (N, 3))
  X_3D[:, 2] += 3.0

  pts1_clean = X_3D[:, :2] / X_3D[:, 2:]
  X_cam2 = (R_gt @ X_3D.T).T + t_gt
  pts2_clean = X_cam2[:, :2] / X_cam2[:, 2:]

  benchmark_summary(
      R_gt=R_gt,
      t_gt=t_gt,
      pts1_clean=pts1_clean,
      pts2_clean=pts2_clean,
      n_trials=10,
      ransac_iters=50,
  )

  outlier_benchmark(
      R_gt=R_gt,
      t_gt=t_gt,
      pts1_clean=pts1_clean,
      pts2_clean=pts2_clean,
      n_trials=10,
      ransac_iters=50,
  )
