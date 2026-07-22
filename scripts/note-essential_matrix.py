import cv2
import numpy as np
import sympy as sp

# ==============================================================================
# 1. NULLSPACE & POLYNOMIAL CONSTRAINT GENERATION
# ==============================================================================


def compute_nullspace_basis(pts1, pts2):
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


def build_constraint_matrix(Ex, Ey, Ez, Ew):
  """
  Substitutes E = x*Ex + y*Ey + z*Ez + Ew into the 9 matrix constraints:

    2 * E * E^T * E - trace(E * E^T) * E = 0

  and det(E) = 0. Returns a 10x20 coefficient matrix.

  Uses sympy to perform the symbolic expansion.
  """
  x, y, z = sp.symbols('x y z')
  _M = lambda a: sp.Matrix(a.tolist())
  E = x * _M(Ex) + y * _M(Ey) + z * _M(Ez) + _M(Ew)

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


# ==============================================================================
# 2. ELIMINATION & ROOT SOLVING FOR (x, y, z)
# ==============================================================================


def solve_system_nister(M):
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


# ==============================================================================
# 3. POSE RECOVERY & CHEIRALITY TEST
# ==============================================================================


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


def tri_cheirality_check(R, t, pts1, pts2):
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


def _sampson_distance(E, pts1, pts2):
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


def nister_5point_algorithm(pts1_norm, pts2_norm):
  """Estimate relative pose using the Nistér 5-point algorithm."""
  Ex, Ey, Ez, Ew = compute_nullspace_basis(pts1_norm, pts2_norm)
  M = build_constraint_matrix(Ex, Ey, Ez, Ew)
  xyz_sols = solve_system_nister(M)

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
      valid_pts = tri_cheirality_check(R, t, pts1_norm, pts2_norm)
      sd = _sampson_distance(E_clean, pts1_norm, pts2_norm)
      if valid_pts > max_front_pts or (valid_pts == max_front_pts and
                                       sd < best_sd):
        max_front_pts = valid_pts
        best_sd = sd
        best_R, best_t = R, t

  return best_R, best_t


# ==============================================================================
# 4. OPENCV COMPARISON FUNCTION
# ==============================================================================


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
    count = tri_cheirality_check(R_cv, t_cv, pts1_norm, pts2_norm)
    if count > best_count:
      best_count = count
      best_R, best_t = R_cv, t_cv

  return best_R, best_t


# ==============================================================================
# VERIFICATION & COMPARISON BENCHMARK
# ==============================================================================
if __name__ == "__main__":
  np.random.seed(42)

  # 1. Ground truth relative pose setup
  theta = np.radians(15.0)  # 15 degree rotation around Y-axis
  R_gt = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
  t_gt = np.array([0.5, -0.2, 0.84])
  t_gt /= np.linalg.norm(t_gt)  # Scale to unit vector

  # 2. Generate many 3D points. The solver uses the first 5 to find E,
  #    but we need extra points to disambiguate among the up-to-10
  #    real solutions via cheirality + Sampson distance.
  N = 20
  X_3D = np.random.uniform(-1, 1, (N, 3))
  X_3D[:, 2] += 3.0  # Depth Z in [2.0, 4.0]

  # Project to normalized coordinates
  pts1 = X_3D[:, :2] / X_3D[:, 2:]
  X_cam2 = (R_gt @ X_3D.T).T + t_gt
  pts2 = X_cam2[:, :2] / X_cam2[:, 2:]

  # 3. Execute Solvers
  R_custom, t_custom = nister_5point_algorithm(pts1, pts2)
  if R_custom is None or t_custom is None:
    raise RuntimeError("Custom 5-point algorithm failed to find a solution")

  R_cv, t_cv = opencv_5point_algorithm(pts1, pts2)
  if R_cv is None or t_cv is None:
    raise RuntimeError("OpenCV 5-point algorithm failed to find a solution")

  # Align sign of translation vectors if inverted (t vs -t ambiguity)
  if np.dot(t_cv, t_gt) < 0:
    t_cv *= -1
  if np.dot(t_custom, t_gt) < 0:
    t_custom *= -1

  # 4. Display Results
  print("==========================================================")
  print(" GROUND TRUTH POSE")
  print("==========================================================")
  print("R:\n", np.round(R_gt, 5))
  print("t:", np.round(t_gt, 5))

  print("\n==========================================================")
  print(" CUSTOM NISTÉR 5-POINT IMPLEMENTATION")
  print("==========================================================")
  print("R:\n", np.round(R_custom, 5))
  print("t:", np.round(t_custom, 5))
  print(f"R Error (Frobenius Norm): {np.linalg.norm(R_gt - R_custom):.2e}")
  print(f"t Error (Euclidean):      {np.linalg.norm(t_gt - t_custom):.2e}")

  print("\n==========================================================")
  print(" OPENCV `cv2.findEssentialMat` IMPLEMENTATION")
  print("==========================================================")
  print("R:\n", np.round(R_cv, 5))
  print("t:", np.round(t_cv, 5))
  print(f"R Error (Frobenius Norm): {np.linalg.norm(R_gt - R_cv):.2e}")
  print(f"t Error (Euclidean):      {np.linalg.norm(t_gt - t_cv):.2e}")
