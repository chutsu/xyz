import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable, Optional

Arr = NDArray[Any]

###############################################################################
# UTILS
###############################################################################


def skew(v: Arr) -> Arr:
  """Returns 3x3 skew-symmetric matrix for vector v."""
  return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rodrigues(w: Arr) -> Arr:
  """Exponential map from lie algebra so(3) vector to SO(3) rotation matrix."""
  theta: float = float(np.linalg.norm(w))
  if theta < 1e-8:
    return np.eye(3)
  k: Arr = w / theta
  K: Arr = skew(k)
  return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def decompose_essential_matrix(E: Arr) -> list[tuple[Arr, Arr]]:
  """Decompose an essential matrix into 4 possible (R, t) pose hypotheses."""
  u: Arr
  vt: Arr
  u, _, vt = np.linalg.svd(E)
  if np.linalg.det(u) < 0:
    u *= -1
  if np.linalg.det(vt) < 0:
    vt *= -1

  W: Arr = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

  R1: Arr = u @ W @ vt
  R2: Arr = u @ W.T @ vt
  t1: Arr = u[:, 2]
  t2: Arr = -u[:, 2]

  return [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]


def cheirality_check(R: Arr, t: Arr, pts1: Arr, pts2: Arr) -> int:
  """Triangulate points and count how many have positive depth in both views."""
  P1: Arr = np.hstack((np.eye(3), np.zeros((3, 1))))
  P2: Arr = np.hstack((R, t.reshape(3, 1)))

  front_count: int = 0
  for i in range(len(pts1)):
    x1: float = pts1[i, 0]
    y1: float = pts1[i, 1]
    x2: float = pts2[i, 0]
    y2: float = pts2[i, 1]

    A: Arr = np.array([
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1],
        x2 * P2[2] - P2[0],
        y2 * P2[2] - P2[1],
    ])
    vt: Arr
    _, _, vt = np.linalg.svd(A)
    x_vec: Arr = vt[-1]
    x_vec /= x_vec[3]

    depth1: float = x_vec[2]
    depth2: float = (R[2] @ x_vec[:3]) + t[2]

    if depth1 > 0 and depth2 > 0:
      front_count += 1

  return front_count


def sampson_distance(E: Arr, pts1: Arr, pts2: Arr) -> float:
  """Symmetric epipolar distance (Sampson) for all point pairs, summed."""
  n: int = len(pts1)
  pts1_h: Arr = np.hstack([pts1, np.ones((n, 1))])
  pts2_h: Arr = np.hstack([pts2, np.ones((n, 1))])

  Ex1: Arr = (E @ pts1_h.T).T
  Etx2: Arr = (E.T @ pts2_h.T).T

  numerator: Arr = np.sum(pts2_h * Ex1, axis=1)**2
  denominator: Arr = (Ex1[:, 0]**2 + Ex1[:, 1]**2 + Etx2[:, 0]**2 +
                      Etx2[:, 1]**2)
  with np.errstate(divide='ignore', invalid='ignore'):
    sd: Arr = np.where(denominator > 1e-12, numerator / denominator, 0.0)
  return np.sum(sd)


###############################################################################
# OPENCV 5-POINT ALGORITHM
###############################################################################


def _try_decompose_essential(E_cv: Arr, pts1_norm: Arr,
                             pts2_norm: Arr) -> tuple[Arr, Arr]:
  """Decompose E and recover pose using OpenCV's recoverPose."""
  K: Arr = np.eye(3)
  R_cv: Arr
  t_cv: Arr
  _, R_cv, t_cv, _ = cv2.recoverPose(E_cv, pts1_norm, pts2_norm,
                                     K)  # type: ignore[reportUnknownMemberType]
  return R_cv, t_cv.ravel()


def opencv_5point_algorithm(pts1_norm: Arr, pts2_norm: Arr) -> tuple[Arr, Arr]:
  """
  Estimates relative pose using OpenCV's built-in 5-point solver safely.
  """
  # 1. Estimate Essential Matrix
  E_cv, _ = cv2.findEssentialMat(
      pts1_norm,  # type: ignore[reportUnknownMemberType]
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
  candidates: list[Arr] = []
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
  best_R: Arr = np.eye(3)
  best_t: Arr = np.zeros(3)
  best_count: int = -1
  for E_candidate in candidates:
    R_cv, t_cv = _try_decompose_essential(E_candidate, pts1_norm, pts2_norm)
    count: int = cheirality_check(R_cv, t_cv, pts1_norm, pts2_norm)
    if count > best_count:
      best_count = count
      best_R, best_t = R_cv, t_cv

  return best_R, best_t


###############################################################################
# NISTER 5-POINT ALGORITHM
###############################################################################


def _compute_nullspace_basis(pts1: Arr, pts2: Arr) -> tuple[Arr, Arr, Arr, Arr]:
  """
  Builds the 5x9 linear design matrix from 5 normalized point pairs and
  computes the 4 basis matrices (Ex, Ey, Ez, Ew) spanning the null space.
  """
  A: Arr = np.zeros((5, 9))
  for i in range(5):
    x1: float = pts1[i, 0]
    y1: float = pts1[i, 1]
    x2: float = pts2[i, 0]
    y2: float = pts2[i, 1]
    A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0]

  vt: Arr
  _, _, vt = np.linalg.svd(A)
  nullspace: Arr = vt[5:].reshape(4, 3, 3)

  return nullspace[0], nullspace[1], nullspace[2], nullspace[3]


# Monomial column lookup for _build_constraint_matrix.
# Variable encoding: 0-1 (constant), 1-x, 2-y, 3-z.
# Column layout (same as original sympy version):
#   0-9:   degree 3 (x^3 … z^3)
#   10-15: degree 2 (x^2 … z^2)
#   16-18: degree 1 (x, y, z)
#   19:    degree 0 (1)
_MONOMIAL_COL: Arr = np.zeros((4, 4, 4), dtype=np.int32)
_C3: dict[tuple[int, int, int], int] = {
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
_C2: dict[tuple[int, int, int], int] = {
    (2, 0, 0): 10,
    (1, 1, 0): 11,
    (1, 0, 1): 12,
    (0, 2, 0): 13,
    (0, 1, 1): 14,
    (0, 0, 2): 15
}
_C1: dict[tuple[int, int, int], int] = {
    (1, 0, 0): 16,
    (0, 1, 0): 17,
    (0, 0, 1): 18
}
for v1 in range(4):
  for v2 in range(4):
    for v3 in range(4):
      cnt: list[int] = [0, 0, 0]
      for v in (v1, v2, v3):
        if v > 0:
          cnt[v - 1] += 1
      a: int
      b: int
      c: int
      a, b, c = cnt
      d: int = a + b + c
      if d == 3:
        _MONOMIAL_COL[v1, v2, v3] = _C3[(a, b, c)]
      elif d == 2:
        _MONOMIAL_COL[v1, v2, v3] = _C2[(a, b, c)]
      elif d == 1:
        _MONOMIAL_COL[v1, v2, v3] = _C1[(a, b, c)]
      else:
        _MONOMIAL_COL[v1, v2, v3] = 19


def _build_constraint_matrix(Ex: Arr, Ey: Arr, Ez: Arr, Ew: Arr) -> Arr:
  """
  Substitutes E = x*Ex + y*Ey + z*Ez + Ew into the 9 matrix constraints:

    2 * E * E^T * E - trace(E * E^T) * E = 0

  and det(E) = 0. Returns a 10x20 coefficient matrix.

  Computes the polynomial coefficients numerically (no symbolic algebra).
  """
  # Stack basis coefficients: C[p,q] = [Ew[p,q], Ex[p,q], Ey[p,q], Ez[p,q]]
  C: Arr = np.stack([Ew, Ex, Ey, Ez], axis=-1)

  M: Arr = np.zeros((10, 20))

  # Accumulate coefficients from a triple product of three E entries.
  # L1 * L2 * L3 where each Li = .k C[pi,qi,k] * var_k
  def _acc(row: int, p1: int, q1: int, p2: int, q2: int, p3: int, q3: int,
           scale: float) -> None:
    c1: Arr = C[p1, q1]
    c2: Arr = C[p2, q2]
    c3: Arr = C[p3, q3]
    for i in range(4):
      ci: float = c1[i]
      if ci == 0:
        continue
      for j in range(4):
        cij: float = ci * c2[j]
        if cij == 0:
          continue
        for k in range(4):
          ck: float = c3[k]
          if ck == 0:
            continue
          M[row, _MONOMIAL_COL[i, j, k]] += scale * cij * ck

  # 9 matrix constraints: C[i,j] = 2*E*E^T*E - trace(E*E^T)*E = 0
  #
  # C[i,j] = 2 * .k .l E[i,l] * E[k,l] * E[k,j]
  #        - .m .n E[m,n] * E[m,n] * E[i,j]
  row: int = 0
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


def _solve_system_nister(M: Arr) -> list[tuple[float, float, float]]:
  """
  Reduces the 10x20 matrix using Gauss-Jordan elimination and solves for z
  using an 10x10 action matrix, then recovers (x, y).
  """
  # Split M = [L | R] where L holds cubic coeffs and R holds lower-degree coeffs
  L: Arr = M[:, :10]
  R: Arr = M[:, 10:]
  if np.linalg.matrix_rank(L) < 10:
    return []

  # L @ cubic + R @ lower = 0  =>  cubic = -L^{-1} @ R @ lower = -B @ lower
  B: Arr = np.linalg.solve(L, R)

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
  Action: Arr = np.zeros((10, 10))
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
  eigvals: Arr
  eigvecs: Arr
  eigvals, eigvecs = np.linalg.eig(Action)

  solutions: list[tuple[float, float, float]] = []
  for i in range(10):
    if np.abs(np.imag(
        eigvals[i])) < 1e-6:  # type: ignore[reportUnknownArgumentType]
      z_val: float = float(np.real(eigvals[i]))
      vec: Arr = np.real(eigvecs[:, i])

      if np.abs(vec[-1]) > 1e-8:
        vec /= vec[-1]
        x_val: float = vec[6]
        y_val: float = vec[7]
        solutions.append((x_val, y_val, z_val))

  return solutions


def nister_5point_algorithm(
    pts1_norm: Arr, pts2_norm: Arr) -> tuple[Optional[Arr], Optional[Arr]]:
  """Estimate relative pose using the Nister 5-point algorithm."""
  Ex: Arr
  Ey: Arr
  Ez: Arr
  Ew: Arr
  Ex, Ey, Ez, Ew = _compute_nullspace_basis(pts1_norm, pts2_norm)
  M: Arr = _build_constraint_matrix(Ex, Ey, Ez, Ew)
  xyz_sols: list[tuple[float, float, float]] = _solve_system_nister(M)

  best_R: Optional[Arr] = None
  best_t: Optional[Arr] = None
  max_front_pts: int = -1
  best_sd: float = float('inf')

  for x, y, z in xyz_sols:
    # Form candidate
    e_candidate: Arr = x * Ex + y * Ey + z * Ez + Ew

    # Ensure singular values are positive and rank-2
    u: Arr
    s: Arr
    vt: Arr
    u, s, vt = np.linalg.svd(e_candidate)
    e_clean: Arr = u @ np.diag([(s[0] + s[1]) / 2.0,
                                (s[0] + s[1]) / 2.0, 0.0]) @ vt

    # Decompose Essential matrix into R, t
    poses: list[tuple[Arr, Arr]] = decompose_essential_matrix(e_clean)
    for r, t in poses:
      valid_pts: int = cheirality_check(r, t, pts1_norm, pts2_norm)
      sd: float = sampson_distance(e_clean, pts1_norm, pts2_norm)
      if valid_pts > max_front_pts or (valid_pts == max_front_pts and
                                       sd < best_sd):
        max_front_pts = valid_pts
        best_sd = sd
        best_R, best_t = r, t

  return best_R, best_t


###############################################################################
# LUI 5-POINT ALGORITHM
###############################################################################


def _compute_angular_residuals(r1: Arr, r2: Arr, v1: Arr,
                               v2: Arr) -> tuple[Arr, Arr, Arr]:
  """
  Projects unit vectors v1, v2 using candidate rotations R1, R2.
  Measures 2D angular difference on the x-y tangent plane.
  """
  # 1. Rotate 3D unit vectors
  p1_3d: Arr = (r1 @ v1.T).T  # Shape: (5, 3)
  p2_3d: Arr = (r2 @ v2.T).T  # Shape: (5, 3)

  # 2. Extract 2D polar angles on x-y plane (looking along baseline e_z)
  theta1: Arr = np.arctan2(p1_3d[:, 1], p1_3d[:, 0])
  theta2: Arr = np.arctan2(p2_3d[:, 1], p2_3d[:, 0])

  # 3. Signed angular difference wrapped to [-pi, pi]
  residuals: Arr = theta1 - theta2
  residuals = (residuals + np.pi) % (2 * np.pi) - np.pi
  return residuals, p1_3d, p2_3d


def rotation_from_vectors(a: Arr, b: Arr) -> Arr:
  """Find R such that R @ a = b for unit vectors a, b."""
  v: Arr = np.cross(a, b)
  s: float = float(np.linalg.norm(v))
  c: float = float(np.dot(a, b))
  if s < 1e-12:
    return np.eye(3) if c > 0 else -np.eye(3)
  V: Arr = skew(v)
  return np.eye(3) + V + V @ V * (1.0 - c) / (s * s)


def lui_5point_algorithm(v1: Arr,
                         v2: Arr,
                         max_iters: int = 100,
                         tol: float = 1e-20,
                         R_init: Optional[Arr] = None,
                         t_init: Optional[Arr] = None) -> tuple[Arr, Arr]:
  """
  Iterative 5-point solver by Vincent Lui & Tom Drummond.

  v1, v2: (N, 3) arrays of unit bearing vectors in camera 1 and camera 2 frames,
           where N >= 5.
  R_init, t_init: optional initial pose guess. If provided, the solver
                  initializes from this pose instead of identity.
  Returns: R (3x3) relative rotation from camera 1 to camera 2,
           t (3,) translation direction in camera 2's frame.
  """
  v1 = v1.copy()
  v2 = v2.copy()
  assert v1.shape[1] == 3 and v2.shape[1] == 3
  assert v1.shape[0] >= 5 and v2.shape[0] >= 5

  if R_init is not None and t_init is not None:
    r2: Arr = rotation_from_vectors(t_init / np.linalg.norm(t_init),
                                    np.array([0.0, 0.0, 1.0]))
    r1: Arr = r2 @ R_init
  else:
    r1 = np.eye(3)
    r2 = np.eye(3)
  n_pts: int = len(v1)

  lam: float = 1e-3
  r: Arr
  p1: Arr
  p2: Arr
  r, p1, p2 = _compute_angular_residuals(r1, r2, v1, v2)
  prev_err: float = float(np.linalg.norm(r))
  stagnation_count: int = 0

  for _ in range(max_iters):
    if prev_err < tol:
      break

    x1: Arr = p1[:, 0]
    y1: Arr = p1[:, 1]
    z1: Arr = p1[:, 2]
    x2: Arr = p2[:, 0]
    y2: Arr = p2[:, 1]
    z2: Arr = p2[:, 2]
    sq1: Arr = x1 * x1 + y1 * y1
    sq2: Arr = x2 * x2 + y2 * y2
    mask1: Arr = sq1 > 1e-12
    mask2: Arr = sq2 > 1e-12

    J: Arr = np.zeros((n_pts, 5))
    J[:, 0] = np.where(mask1, -x1 * z1 / sq1, 0.0)
    J[:, 1] = np.where(mask1, -y1 * z1 / sq1, 0.0)
    J[:, 2] = 1.0
    J[:, 3] = np.where(mask2, x2 * z2 / sq2, 0.0)
    J[:, 4] = np.where(mask2, y2 * z2 / sq2, 0.0)

    JtJ: Arr = J.T @ J
    Jt_r: Arr = J.T @ -r

    accepted: bool = False
    for _ in range(30):
      try:
        delta: Arr = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ)), Jt_r)
      except np.linalg.LinAlgError:
        lam *= 2
        continue

      r1_test: Arr = rodrigues(delta[0:3]) @ r1
      w2: Arr = np.array([delta[3], delta[4], 0.0])
      r_test: Arr
      p1_test: Arr
      p2_test: Arr
      r_test, p1_test, p2_test = _compute_angular_residuals(
          r1_test,
          rodrigues(w2) @ r2, v1, v2)
      new_err: float = float(np.linalg.norm(r_test))

      if new_err < prev_err:
        r1 = r1_test
        r2 = rodrigues(w2) @ r2
        r, p1, p2 = r_test, p1_test, p2_test

        rel_change: float = abs(prev_err - new_err) / max(
            float(prev_err), 1e-12)
        if rel_change < 1e-6:
          stagnation_count += 1
          if stagnation_count >= 3:
            ez: Arr = np.array([0.0, 0.0, 1.0])
            return r2.T @ r1, r2.T @ ez
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
  t_vec: Arr = r2.T @ ez
  r: Arr = r2.T @ r1

  return r, t_vec


def _sampson_per_point(E: Arr, pts1: Arr, pts2: Arr) -> Arr:
  """Per-point Sampson distance for RANSAC inlier counting."""
  n: int = len(pts1)
  pts1_h: Arr = np.hstack([pts1, np.ones((n, 1))])
  pts2_h: Arr = np.hstack([pts2, np.ones((n, 1))])
  Ex1: Arr = (E @ pts1_h.T).T
  Etx2: Arr = (E.T @ pts2_h.T).T
  numerator: Arr = np.sum(pts2_h * Ex1, axis=1)**2
  denominator: Arr = (Ex1[:, 0]**2 + Ex1[:, 1]**2 + Etx2[:, 0]**2 +
                      Etx2[:, 1]**2)
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.where(denominator > 1e-12, numerator / denominator, 0.0)


def ransac_5pt_lui(pts1: Arr,
                   pts2: Arr,
                   max_iters: int = 500,
                   threshold: float = 1e-4) -> tuple[Arr, Arr]:
  """RANSAC wrapper around the Lui 5-point solver.

  Randomly samples 5-point subsets, fits the model, and returns the
  pose with the most inliers (Sampson distance < threshold).
  The winning hypothesis is refined on all inlier points.
  """
  N: int = len(pts1)
  best_R: Arr = np.eye(3)
  best_t: Arr = np.zeros(3)
  best_score: int = -1

  # Precompute bearing vectors for all points
  v1_all: Arr = np.column_stack([pts1, np.ones(N)])
  v1_all /= np.linalg.norm(v1_all, axis=1, keepdims=True)
  v2_all: Arr = np.column_stack([pts2, np.ones(N)])
  v2_all /= np.linalg.norm(v2_all, axis=1, keepdims=True)

  # RANSAC
  for _ in range(max_iters):
    idx: Arr = np.random.choice(N, 5, replace=False)
    try:
      r, t = lui_5point_algorithm(v1_all[idx], v2_all[idx])
    except Exception:
      continue

    e: Arr = skew(t) @ r
    scores: Arr = _sampson_per_point(e, pts1, pts2)
    inliers: int = int(np.sum(scores < threshold))

    if inliers > best_score:
      best_score = inliers
      best_R, best_t = r.copy(), t.copy()

  # Refinement: re-fit on all inliers (initialize from best RANSAC hypothesis)
  E_best: Arr = skew(best_t) @ best_R
  scores = _sampson_per_point(E_best, pts1, pts2)
  inlier_mask: Arr = scores < threshold
  if np.sum(inlier_mask) >= 5:
    try:
      R_ref, t_ref = lui_5point_algorithm(v1_all[inlier_mask],
                                          v2_all[inlier_mask],
                                          max_iters=100,
                                          tol=1e-12,
                                          R_init=best_R,
                                          t_init=best_t)
      # Accept refinement if it doesn't regress
      E_ref: Arr = skew(t_ref) @ R_ref
      scores_ref: Arr = _sampson_per_point(E_ref, pts1, pts2)
      if np.sum(scores_ref < threshold) >= best_score:
        best_R, best_t = R_ref, t_ref
    except Exception:
      pass

  return best_R, best_t


###############################################################################
# HEDBORG 5-POINT ALGORITHM (Levenberg-Marquardt)
###############################################################################


def _tangent_basis(t: Arr) -> tuple[Arr, Arr]:
  """Build orthonormal basis [e1, e2] for the tangent space of S^2 at t."""
  e1: Arr = np.cross(t, np.array([1.0, 0.0, 0.0]))
  if np.linalg.norm(e1) < 1e-8:
    e1 = np.cross(t, np.array([0.0, 1.0, 0.0]))
  e1 = e1 / np.linalg.norm(e1)
  e2: Arr = np.cross(t, e1)
  e2 = e2 / np.linalg.norm(e2)
  return e1, e2


def _sphere_exp_map(t: Arr, v: Arr) -> Arr:
  """Exponential map on S^2 at t with tangent vector v (orthogonal to t)."""
  theta: float = float(np.linalg.norm(v))
  if theta < 1e-12:
    return t
  return np.cos(theta) * t + np.sin(theta) * (v / theta)


def _essential_from_params(w: Arr, t_cur: Arr) -> tuple[Arr, Arr, Arr]:
  """Extract essential matrix E from parameter vector w = [ax, ay, az, du, dv].

  R = rodrigues(w[:3]), t = exp_map(t_cur, w[3:5] projected to tangent space).
  """
  r: Arr = rodrigues(w[:3])
  du: float = w[3]
  dv: float = w[4]
  e1: Arr
  e2: Arr
  e1, e2 = _tangent_basis(t_cur)
  v: Arr = du * e1 + dv * e2
  t_vec: Arr = _sphere_exp_map(t_cur, v)
  return skew(t_vec) @ r, r, t_vec


def _epipolar_dist_residuals(
    w: Arr,
    t_cur: Arr,
    pts1_h: Arr,
    pts2_h: Arr,
) -> Arr:
  """Epipolar distance r_i = (x'^T E x) / sqrt((x'^T E)_1^2 + (x'^T E)_2^2)."""
  e: Arr
  e, _, _ = _essential_from_params(w, t_cur)
  Ex1: Arr = (e @ pts1_h.T).T
  numerator: Arr = np.sum(pts2_h * Ex1, axis=1)
  denominator: Arr = np.sqrt(Ex1[:, 0]**2 + Ex1[:, 1]**2)
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.where(denominator > 1e-12, numerator / denominator, 0.0)


def _so3_right_jacobian(r: Arr) -> Arr:
  """Right Jacobian of SO(3) exponential map at r (axis-angle vector)."""
  theta: float = float(np.linalg.norm(r))
  if theta < 1e-8:
    return np.eye(3)
  rx: Arr = skew(r / theta)
  a: float = (1.0 - np.cos(theta)) / theta
  b: float = (theta - np.sin(theta)) / theta
  return np.eye(3) - a * rx + b * rx @ rx


def _dE_dparams(w: Arr, t_cur: Arr) -> list[Arr]:
  """Analytical derivatives of E w.r.t. each parameter in w.

  Returns list [dE_da, dE_db, dE_dg, dE_du, dE_dv] of 3x3 arrays.
  w = [ax, ay, az, du, dv] where (du, dv) are tangent space coords at t_cur.
  """
  r: Arr = rodrigues(w[:3])
  e1: Arr
  e2: Arr
  e1, e2 = _tangent_basis(t_cur)
  du: float = w[3]
  dv: float = w[4]
  v: Arr = du * e1 + dv * e2
  t_vec: Arr = _sphere_exp_map(t_cur, v)

  # Jacobian of exp_map at v=0 is identity: dt/du = e1, dt/dv = e2
  # (re-linearisation happens after each accepted step, keeping w[3:5] near 0)

  # Rotation derivatives: dR/dr_k = R @ skew(J_r[:, k])
  Jr: Arr = _so3_right_jacobian(w[:3])
  dE: list[Arr] = []
  for k in range(3):
    dR: Arr = r @ skew(Jr[:, k])
    dE.append(skew(t_vec) @ dR)
  dE.append(skew(e1) @ r)  # dE/du
  dE.append(skew(e2) @ r)  # dE/dv
  return dE


def _jacobian_epipolar(w: Arr, t_cur: Arr, pts1_h: Arr, pts2_h: Arr) -> Arr:
  """Analytical Nx5 Jacobian of epipolar distance residuals."""
  e: Arr
  e, _, _ = _essential_from_params(w, t_cur)
  dE: list[Arr] = _dE_dparams(w, t_cur)
  N: int = len(pts1_h)
  J: Arr = np.zeros((N, 5))

  Ex1: Arr = (e @ pts1_h.T).T
  a: Arr = Ex1[:, 0]  # (x'^T E)_1
  b: Arr = Ex1[:, 1]  # (x'^T E)_2
  d_sq: Arr = a * a + b * b
  d: Arr = np.sqrt(d_sq)
  n: Arr = np.sum(pts2_h * Ex1, axis=1)

  for j in range(5):
    dEx1: Arr = (dE[j] @ pts1_h.T).T
    dn: Arr = np.sum(pts2_h * dEx1, axis=1)
    da: Arr = dEx1[:, 0]
    db: Arr = dEx1[:, 1]
    dd: Arr = np.where(d > 1e-12, (a * da + b * db) / d, 0.0)

    denom: Arr = d_sq + 1e-12
    J[:, j] = (dn * d - n * dd) / denom

  return J


def _lm_solve(
    w: Arr,
    t_cur: Arr,
    pts1_h: Arr,
    pts2_h: Arr,
    max_iters: int,
    tol: float,
) -> tuple[Arr, Arr, float]:
  """Run Levenberg-Marquardt optimization from a single initial w.

  Uses analytical Jacobians of the epipolar distance residuals.
  w = [ax, ay, az, du, dv] with translation in tangent space at t_cur.
  After each accepted step, t_cur is updated and w[3:5] reset to 0
  (re-linearisation on the S^2 manifold).

  Returns (w_opt, t_cur_opt, cost) where cost = 0.5 * ||r||^2.
  """
  lam: float = 1e-3
  nu: float = 2.0

  r: Arr = _epipolar_dist_residuals(w, t_cur, pts1_h, pts2_h)
  cost: float = 0.5 * float(np.dot(r, r))

  for _ in range(max_iters):
    j: Arr = _jacobian_epipolar(w, t_cur, pts1_h, pts2_h)
    g: Arr = j.T @ r
    hess: Arr = j.T @ j

    if float(np.linalg.norm(g, np.inf)) < tol:
      break

    diag_hess: Arr = np.diag(hess)
    accepted: bool = False
    for _ in range(30):
      try:
        h: Arr = np.linalg.solve(hess + lam * np.diag(diag_hess), -g)
      except np.linalg.LinAlgError:
        lam *= nu
        nu *= 2
        continue

      # Update rotation (axis-angle accumulates)
      w_new: Arr = w.copy()
      w_new[:3] = w[:3] + h[:3]
      # Update translation via exponential map on S^2
      e1: Arr
      e2: Arr
      e1, e2 = _tangent_basis(t_cur)
      v: Arr = h[3] * e1 + h[4] * e2
      t_new: Arr = _sphere_exp_map(t_cur, v)
      # Reset tangent-space parameters after the step
      w_new[3] = 0.0
      w_new[4] = 0.0

      r_new: Arr = _epipolar_dist_residuals(w_new, t_new, pts1_h, pts2_h)
      new_cost: float = 0.5 * float(np.dot(r_new, r_new))

      jh: Arr = j @ h  # type: ignore[reportUnknownVariableType]
      l_pred: Arr = r + jh  # type: ignore[reportUnknownVariableType]
      pred_cost: float = 0.5 * float(np.dot(
          l_pred, l_pred))  # type: ignore[reportUnknownArgumentType]
      actual_reduction: float = cost - new_cost
      pred_reduction: float = cost - pred_cost

      rho: float = 0.0 if abs(
          pred_reduction) < 1e-16 else actual_reduction / pred_reduction

      if rho > 0:
        w = w_new
        t_cur = t_new
        r = r_new
        cost = new_cost
        lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0)**3)
        nu = 2.0
        accepted = True
        if float(np.linalg.norm(h)) < tol:
          return w, t_cur, cost
        break
      else:
        lam *= nu
        nu *= 2.0

    if not accepted:
      break

  return w, t_cur, cost


def _generate_seeds(
    n_seeds: int,
    rng: np.random.Generator,
) -> list[tuple[Arr, Arr]]:
  """Generate candidate initial (w, t_cur) pairs for multi-start.

  w = [ax, ay, az, 0, 0]  (tangent-space params start at zero).
  t_cur is a random unit vector on S^2.
  """
  t0: Arr = np.array([0.0, 0.0, 1.0])
  seeds: list[tuple[Arr, Arr]] = [(np.zeros(5), t0.copy())]
  for _ in range(n_seeds - 1):
    w: Arr = np.zeros(5)
    w[:3] = rng.uniform(-0.5, 0.5, 3)
    theta: float = rng.uniform(0.1, np.pi - 0.1)
    phi: float = rng.uniform(-np.pi, np.pi)
    t_cur: Arr = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    seeds.append((w, t_cur))
  return seeds


def hedborg_5point_algorithm(
    pts1_norm: Arr,
    pts2_norm: Arr,
    max_iters: int = 50,
    tol: float = 1e-20,
    R_init: Optional[Arr] = None,
    t_init: Optional[Arr] = None,
) -> tuple[Arr, Arr]:
  """Fast iterative 5-point relative pose via Levenberg-Marquardt (Hedborg & Felsberg 2013).

  Uses multi-start initialisation: tries several candidate parameter vectors
  and picks the one with the lowest final epipolar cost.

  The solver parameterises the essential matrix with 5 parameters
  w = [ax, ay, az, du, dv] where (du, dv) are coordinates in the tangent
  space of S^2 at the current translation estimate t_cur.  After each
  accepted LM step the translation is updated via the exponential map on
  S^2 and the tangent-space parameters are reset to zero (re-linearisation).

  Args:
    pts1_norm, pts2_norm: (N, 2) normalized image coordinates (N >= 5).
    max_iters: maximum LM iterations per seed.
    tol: convergence threshold (gradient infinity-norm).
    R_init: optional 3x3 initial rotation.
    t_init: optional 3D initial translation direction.

  Returns:
    R (3x3), t (3,) describing the relative pose.
  """
  n_pts: int = len(pts1_norm)
  pts1_h: Arr = np.column_stack([pts1_norm, np.ones(n_pts)])
  pts2_h: Arr = np.column_stack([pts2_norm, np.ones(n_pts)])

  rng: np.random.Generator = np.random.default_rng(0)
  n_seeds: int = 4 if R_init is None else 1

  if R_init is not None and t_init is not None:
    w_rot: Arr = cv2.Rodrigues(R_init)[0].ravel()
    t_dir: Arr = t_init / np.linalg.norm(t_init)
    single_seed_w: Arr = np.array([w_rot[0], w_rot[1], w_rot[2], 0.0, 0.0])
    seeds: list[tuple[Arr, Arr]] = [(single_seed_w, t_dir)]
  else:
    seeds = _generate_seeds(n_seeds, rng)

  best_w: Arr = seeds[0][0]
  best_t_cur: Arr = seeds[0][1]
  best_cost: float = float('inf')
  for seed_w, seed_t_cur in seeds:
    w_opt, t_cur_opt, cost = _lm_solve(seed_w.copy(), seed_t_cur, pts1_h,
                                       pts2_h, max_iters, tol)
    if cost < best_cost:
      best_cost = cost
      best_w = w_opt
      best_t_cur = t_cur_opt

  _, r, t = _essential_from_params(best_w, best_t_cur)
  t = t / np.linalg.norm(t)
  return r, t


def ransac_5pt_hedborg(
    pts1: Arr,
    pts2: Arr,
    max_iters: int = 500,
    threshold: float = 1e-4,
) -> tuple[Arr, Arr]:
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
  N: int = len(pts1)
  best_R: Arr = np.eye(3)
  best_t: Arr = np.zeros(3)
  best_score: int = -1
  best_sampson: float = float('inf')

  for _ in range(max_iters):
    idx: Arr = np.random.choice(N, 5, replace=False)
    try:
      r, t = hedborg_5point_algorithm(pts1[idx], pts2[idx])
    except Exception:
      continue

    e: Arr = skew(t) @ r
    scores: Arr = _sampson_per_point(e, pts1, pts2)
    inliers: int = int(np.sum(scores < threshold))
    total_sampson: float = np.sum(scores)

    if inliers > best_score or (inliers == best_score and
                                total_sampson < best_sampson):
      best_score = inliers
      best_sampson = total_sampson
      best_R, best_t = r.copy(), t.copy()

  # Refinement: re-fit on all inliers (initialize from best RANSAC hypothesis)
  E_best: Arr = skew(best_t) @ best_R
  scores = _sampson_per_point(E_best, pts1, pts2)
  inlier_mask: Arr = scores < threshold
  if np.sum(inlier_mask) >= 5:
    try:
      R_ref, t_ref = hedborg_5point_algorithm(pts1[inlier_mask],
                                              pts2[inlier_mask],
                                              max_iters=50,
                                              tol=1e-12,
                                              R_init=best_R,
                                              t_init=best_t)
      E_ref: Arr = skew(t_ref) @ R_ref
      scores_ref: Arr = _sampson_per_point(E_ref, pts1, pts2)
      if np.sum(scores_ref < threshold) >= best_score:
        best_R, best_t = R_ref, t_ref
    except Exception:
      pass

  return best_R, best_t


###############################################################################
# PRE-EMPTIVE RANSAC (shared scoring scheme for any solver)
###############################################################################


def ransac_preemptive(
    pts1: Arr,
    pts2: Arr,
    solver_fn: Callable[[Arr, Arr], tuple[Optional[Arr], Optional[Arr]]],
    sample_size: int = 5,
    num_hypotheses: int = 100,
    threshold: float = 1e-4,
    stage_sizes: Optional[list[int]] = None,
    keep_frac: float = 0.5,
    min_inliers: Optional[int] = None,
    refine: bool = True,
    refine_fn: Optional[Callable[[Arr, Arr, Arr, Arr], tuple[Arr, Arr]]] = None,
    rng: Optional[np.random.Generator] = None) -> tuple[Arr, Arr]:
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
  N: int = len(pts1)
  if rng is None:
    rng = np.random.default_rng()

  # Stage 0: generate candidate hypotheses
  hypotheses: list[tuple[Arr, Arr]] = []
  for _ in range(num_hypotheses):
    idx: Arr = rng.choice(N, sample_size, replace=False)
    try:
      r_raw, t_raw = solver_fn(pts1[idx], pts2[idx])
      if r_raw is not None and t_raw is not None:
        hypotheses.append((r_raw, t_raw))
    except Exception:
      continue

  if not hypotheses:
    return np.eye(3), np.zeros(3)

  M: int = len(hypotheses)

  # Determine pre-emptive stage sizes (geometric progression)
  if stage_sizes is None:
    n_stages: int = max(2, int(np.log2(M)))
    start: int = min(N, max(sample_size * 10, 50))
    stage_sizes_arr: Arr = np.geomspace(start, N, n_stages).astype(int)
    stage_sizes = list(np.unique(stage_sizes_arr))
    if len(stage_sizes) == 0 or stage_sizes[-1] < N:
      stage_sizes.append(N)
  else:
    stage_sizes = [s for s in stage_sizes if s >= sample_size]
    if len(stage_sizes) == 0:
      stage_sizes = [N]
    elif stage_sizes[-1] < N:
      stage_sizes.append(N)

  # Fixed permutation so all hypotheses see the same subsets
  perm: Arr = rng.permutation(N)
  active: list[int] = list(range(M))

  # Pre-emptive stages
  for sz in stage_sizes:
    if len(active) <= 1:
      break
    n_test: int = min(sz, N)
    subset: Arr = perm[:n_test]

    inliers_arr: Arr = np.zeros(len(active))
    sampson_arr: Arr = np.full(len(active), float('inf'))

    for i, hyp_idx in enumerate(active):
      r, t = hypotheses[hyp_idx]
      e: Arr = skew(t) @ r
      scores: Arr = _sampson_per_point(e, pts1[subset], pts2[subset])
      inliers_arr[i] = np.sum(scores < threshold)
      sampson_arr[i] = np.sum(scores)

    n_keep: int = max(1, int(len(active) * keep_frac))
    order: Arr = np.lexsort((sampson_arr, -inliers_arr))
    active = [active[i] for i in order[:n_keep]]

    if min_inliers is not None and inliers_arr[order[0]] >= min_inliers:
      break

  # Final scoring on all points
  best_R: Arr = np.eye(3)
  best_t: Arr = np.zeros(3)
  best_score: int = -1
  best_sampson: float = float('inf')

  for hyp_idx in active:
    r, t = hypotheses[hyp_idx]
    e: Arr = skew(t) @ r
    scores: Arr = _sampson_per_point(e, pts1, pts2)
    inliers: int = int(np.sum(scores < threshold))
    total: float = np.sum(scores)

    if inliers > best_score or (inliers == best_score and total < best_sampson):
      best_score = inliers
      best_sampson = total
      best_R, best_t = r.copy(), t.copy()

  # Refinement on all inliers
  if refine:
    E_best: Arr = skew(best_t) @ best_R
    scores = _sampson_per_point(E_best, pts1, pts2)
    mask: Arr = scores < threshold
    if np.sum(mask) >= sample_size:
      try:
        if refine_fn is not None:
          R_ref, t_ref = refine_fn(pts1[mask], pts2[mask], best_R, best_t)
        else:
          R_ref, t_ref = solver_fn(pts1[mask], pts2[mask])
        if R_ref is not None and t_ref is not None:
          t_ref = t_ref / (np.linalg.norm(t_ref) + 1e-12)
          E_ref: Arr = skew(t_ref) @ R_ref
          scores_ref: Arr = _sampson_per_point(E_ref, pts1, pts2)
          if np.sum(scores_ref < threshold) >= best_score:
            best_R, best_t = R_ref, t_ref
      except Exception:
        pass

  return best_R, best_t


def _epipolar_inliers(E: Arr, pts1: Arr, pts2: Arr, threshold: float) -> Arr:
  """Epipolar distance inlier mask, matching OpenCV's RANSAC metric.

  d = |x2' * E * x1| / sqrt((E*x1)_1^2 + (E*x1)_2^2)
  """
  n: int = len(pts1)
  pts1_h: Arr = np.hstack([pts1, np.ones((n, 1))])
  pts2_h: Arr = np.hstack([pts2, np.ones((n, 1))])
  Ex1: Arr = (E @ pts1_h.T).T
  numerator: Arr = np.abs(np.sum(pts2_h * Ex1, axis=1))
  denominator: Arr = np.sqrt(Ex1[:, 0]**2 + Ex1[:, 1]**2)
  with np.errstate(divide='ignore', invalid='ignore'):
    dist: Arr = np.where(denominator > 1e-12, numerator / denominator,
                         float('inf'))
  return dist < threshold


def _compute_nister_candidates(pts1: Arr, pts2: Arr) -> list[Arr]:
  """Compute all E matrix candidates from 5-point Nister solver (no pose selection)."""
  Ex: Arr
  Ey: Arr
  Ez: Arr
  Ew: Arr
  Ex, Ey, Ez, Ew = _compute_nullspace_basis(pts1, pts2)
  M: Arr = _build_constraint_matrix(Ex, Ey, Ez, Ew)
  xyz_sols: list[tuple[float, float, float]] = _solve_system_nister(M)
  candidates: list[Arr] = []
  for x, y, z in xyz_sols:
    e_candidate: Arr = x * Ex + y * Ey + z * Ez + Ew
    u: Arr
    s: Arr
    vt: Arr
    u, s, vt = np.linalg.svd(e_candidate)
    e_clean: Arr = u @ np.diag([(s[0] + s[1]) / 2.0,
                                (s[0] + s[1]) / 2.0, 0.0]) @ vt
    candidates.append(e_clean)
  return candidates


def ransac_5pt_standard(
    pts1: Arr,
    pts2: Arr,
    max_iters: int = 2000,
    threshold: float = 1e-3,
    prob: float = 0.99,
) -> tuple[Arr, Arr]:
  """Standard RANSAC with the Nister 5-point solver (matching OpenCV's approach).

  For each 5-point sample, computes ALL candidate E matrices from the
  5-point polynomial solver and scores each one against the full point
  set using epipolar distance (matching OpenCV's metric).  The best E
  by inlier count is kept.

  Adaptively determines iteration count based on the best inlier ratio
  seen so far (same adaptive scheme as ``cv2.findEssentialMat`` with
  ``cv2.RANSAC``).
  """
  N: int = len(pts1)
  best_E: Optional[Arr] = None
  best_score: int = -1

  iters: int = 0
  max_attempts: int = max_iters

  while iters < max_attempts:
    idx: Arr = np.random.choice(N, 5, replace=False)
    try:
      e_candidates: list[Arr] = _compute_nister_candidates(pts1[idx], pts2[idx])
    except Exception:
      iters += 1
      continue

    if not e_candidates:
      iters += 1
      continue

    for e_candidate in e_candidates:
      inlier_mask: Arr = _epipolar_inliers(e_candidate, pts1, pts2, threshold)
      inliers: int = int(np.sum(inlier_mask))

      if inliers > best_score:
        best_score = inliers
        best_E = e_candidate.copy()

        # Adaptive iteration count (OpenCV style)
        inlier_ratio: float = inliers / N
        if inlier_ratio >= 1.0:
          max_attempts = iters + 1
        elif inlier_ratio > 0.0:
          denom: float = 1.0 - inlier_ratio**5
          if denom > 0.0:
            adapt_iters = int(np.log(1.0 - prob) / np.log(denom))
            max_attempts = min(max_iters, max(adapt_iters, 10))

    iters += 1

  if best_E is None:
    return np.eye(3), np.zeros(3)

  # Decompose best E into R, t with cheirality check
  poses: list[tuple[Arr, Arr]] = decompose_essential_matrix(best_E)
  best_R: Arr = np.eye(3)
  best_t: Arr = np.zeros(3)
  best_pts: int = -1
  for r_candidate, t_candidate in poses:
    front_pts: int = cheirality_check(r_candidate, t_candidate, pts1, pts2)
    if front_pts > best_pts:
      best_pts = front_pts
      best_R, best_t = r_candidate, t_candidate

  # Refinement on all inliers
  inlier_mask = _epipolar_inliers(best_E, pts1, pts2, threshold)
  if np.sum(inlier_mask) >= 5:
    try:
      ref_candidates: list[Arr] = _compute_nister_candidates(
          pts1[inlier_mask], pts2[inlier_mask])
      if ref_candidates:
        best_ref_E: Arr = ref_candidates[0]
        best_ref_score: int = int(
            np.sum(_epipolar_inliers(best_ref_E, pts1, pts2, threshold)))
        for e_ref in ref_candidates[1:]:
          s: int = int(np.sum(_epipolar_inliers(e_ref, pts1, pts2, threshold)))
          if s > best_ref_score:
            best_ref_score = s
            best_ref_E = e_ref
        if best_ref_score >= best_score:
          ref_poses: list[tuple[Arr,
                                Arr]] = decompose_essential_matrix(best_ref_E)
          best_ref_pts: int = -1
          for r_ref, t_ref in ref_poses:
            fp: int = cheirality_check(r_ref, t_ref, pts1, pts2)
            if fp > best_ref_pts:
              best_ref_pts = fp
              best_R, best_t = r_ref, t_ref
    except Exception:
      pass

  return best_R, best_t


def _lui_adapter(pts1_sample: Arr, pts2_sample: Arr) -> tuple[Arr, Arr]:
  """Adapter: (N,2) image points -> bearing vectors for Lui solver."""
  v1: Arr = np.column_stack([pts1_sample, np.ones(len(pts1_sample))])
  v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
  v2: Arr = np.column_stack([pts2_sample, np.ones(len(pts2_sample))])
  v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
  return lui_5point_algorithm(v1, v2)


def ransac_5pt_nister_preemptive(pts1: Arr, pts2: Arr,
                                 **kwargs: Any) -> tuple[Arr, Arr]:
  """Pre-emptive RANSAC with the Nister 5-point solver."""
  return ransac_preemptive(pts1, pts2, nister_5point_algorithm, **kwargs)


def ransac_5pt_lui_preemptive(pts1: Arr, pts2: Arr,
                              **kwargs: Any) -> tuple[Arr, Arr]:
  """Pre-emptive RANSAC with the Lui 5-point solver."""
  return ransac_preemptive(pts1, pts2, _lui_adapter, **kwargs)


def _hedborg_refine(pts1: Arr, pts2: Arr, R_init: Arr,
                    t_init: Arr) -> tuple[Arr, Arr]:
  """Refine Hedborg from an initial guess (avoids multi-start)."""
  return hedborg_5point_algorithm(pts1,
                                  pts2,
                                  max_iters=20,
                                  tol=1e-12,
                                  R_init=R_init,
                                  t_init=t_init)


def ransac_5pt_hedborg_preemptive(
    pts1: Arr,
    pts2: Arr,
    **kwargs: Any,
) -> tuple[Arr, Arr]:
  """Pre-emptive RANSAC with the Hedborg 5-point solver.

  The refinement step initialises the LM solver from the best RANSAC
  hypothesis, avoiding random multi-start on the inlier set.
  """
  kwargs.setdefault('refine_fn', _hedborg_refine)
  return ransac_preemptive(pts1, pts2, hedborg_5point_algorithm, **kwargs)


###############################################################################
# VERIFICATION & COMPARISON BENCHMARK
###############################################################################


def _run_solvers(pts1: Arr,
                 pts2: Arr,
                 R_gt: Arr,
                 t_gt: Arr,
                 ransac_iters: int = 200
                ) -> dict[str, tuple[float, float, float]]:
  """Run all solvers once and return dict of {name: (R_err, t_ang, elapsed_ms)}."""
  N: int = len(pts1)
  import time

  v1_all: Arr = np.column_stack([pts1, np.ones(N)])
  v1_all /= np.linalg.norm(v1_all, axis=1, keepdims=True)
  v2_all: Arr = np.column_stack([pts2, np.ones(N)])
  v2_all /= np.linalg.norm(v2_all, axis=1, keepdims=True)

  def _t_err(t: Arr) -> float:
    if np.dot(t, t_gt) < 0:
      t *= -1
    return float(np.degrees(np.arccos(np.clip(np.dot(t_gt, t), -1.0, 1.0))))

  def _r_err(R: Arr) -> float:
    return float(np.linalg.norm(R_gt - R))

  out: dict[str, tuple[float, float, float]] = {}

  t0: float = time.time()
  r, t = nister_5point_algorithm(pts1, pts2)
  if r is None or t is None:
    r, t = np.eye(3), np.zeros(3)
  out['Nister'] = (_r_err(r), _t_err(t), (time.time() - t0) * 1000)

  t0 = time.time()
  r, t = opencv_5point_algorithm(pts1, pts2)
  out['OpenCV'] = (_r_err(r), _t_err(t), (time.time() - t0) * 1000)

  t0 = time.time()
  r, t = lui_5point_algorithm(v1_all[:5], v2_all[:5])
  out['Lui'] = (_r_err(r), _t_err(t), (time.time() - t0) * 1000)

  t0 = time.time()
  r, t = hedborg_5point_algorithm(pts1, pts2)
  out['Hedborg'] = (_r_err(r), _t_err(t), (time.time() - t0) * 1000)

  t0 = time.time()
  r, t = ransac_5pt_standard(pts1, pts2, max_iters=ransac_iters, threshold=1e-3)
  out['Nister RANSAC'] = (_r_err(r), _t_err(t), (time.time() - t0) * 1000)

  t0 = time.time()
  r, t = ransac_5pt_nister_preemptive(pts1,
                                      pts2,
                                      num_hypotheses=ransac_iters,
                                      threshold=1e-3)
  out['Nister Preemptive'] = (_r_err(r), _t_err(t), (time.time() - t0) * 1000)

  t0 = time.time()
  r, t = ransac_5pt_lui_preemptive(pts1,
                                   pts2,
                                   num_hypotheses=ransac_iters,
                                   threshold=1e-3)
  out['Lui RANSAC'] = (_r_err(r), _t_err(t), (time.time() - t0) * 1000)

  t0 = time.time()
  r, t = ransac_5pt_hedborg(pts1, pts2, max_iters=ransac_iters, threshold=1e-3)
  out['Hedborg RANSAC'] = (_r_err(r), _t_err(t), (time.time() - t0) * 1000)

  return out


def _add_pixel_noise(pts: Arr,
                     sigma_px: float,
                     f: float = 500.0,
                     rng: Any = None) -> Arr:
  """Add zero-mean Gaussian noise in pixel space, then convert back."""
  if rng is None:
    rng = np.random
  if sigma_px <= 0:
    return pts.copy()
  return pts + rng.randn(*pts.shape) * sigma_px / f


def benchmark_summary(R_gt: Arr,
                      t_gt: Arr,
                      pts1_clean: Arr,
                      pts2_clean: Arr,
                      noise_sigmas_px: tuple[float,
                                             ...] = (0.0, 0.5, 1.0, 2.0, 4.0),
                      focal_length: float = 500.0,
                      n_trials: int = 10,
                      ransac_iters: int = 100) -> None:
  """Run multiple trials at each pixel-noise level and print statistical summary."""
  method_names: list[str] = [
      'OpenCV',
      'Nister',
      'Lui',
      'Hedborg',
      'Nister Standard',
      'Nister Preemptive',
      'Lui RANSAC',
      'Hedborg RANSAC',
  ]

  for sigma_px in noise_sigmas_px:
    r_errs: dict[str, list[float]] = {m: [] for m in method_names}
    t_errs: dict[str, list[float]] = {m: [] for m in method_names}

    for _ in range(n_trials):
      pts1: Arr = _add_pixel_noise(pts1_clean, sigma_px, focal_length)
      pts2: Arr = _add_pixel_noise(pts2_clean, sigma_px, focal_length)

      try:
        result: dict[str,
                     tuple[float, float,
                           float]] = _run_solvers(pts1,
                                                  pts2,
                                                  R_gt,
                                                  t_gt,
                                                  ransac_iters=ransac_iters)
        for m in method_names:
          if m in result:
            r_errs[m].append(result[m][0])
            t_errs[m].append(result[m][1])
      except Exception:
        pass

    def _fmt(a: list[float]) -> str:
      if len(a) == 0:
        return f"{'—':>28}"
      r_arr: Arr = np.array(a)
      return f"{r_arr.mean():>12.4e} ± {r_arr.std():>12.4e}"

    hdr: str = f"Noise {sigma_px:.1f} px"
    print(f"\n{'='*len(hdr)}")
    print(hdr)
    print(f"{'='*len(hdr)}")
    print(f"{'Method':<20} {'R_err':>28} {'t_err (deg)':>28}  {'n':>4}")
    print("-" * 84)
    for m in method_names:
      r: list[float] = r_errs[m]
      t: list[float] = t_errs[m]
      print(f"{m:<20} {_fmt(r)}  {_fmt(t)}  {len(r):>4}")


def outlier_benchmark(R_gt: Arr,
                      t_gt: Arr,
                      pts1_clean: Arr,
                      pts2_clean: Arr,
                      outlier_fracs: tuple[float, ...] = (0.3, 0.4, 0.5),
                      noise_sigma_px: float = 1.0,
                      focal_length: float = 500.0,
                      n_trials: int = 10,
                      ransac_iters: int = 100) -> None:
  """Benchmark all solvers with varying fractions of outlier correspondences.

  At each outlier fraction, a subset of point pairs are replaced with
  random uniform correspondences (outliers), and mild Gaussian noise
  (in pixels) is added to all points.
  """
  method_names: list[str] = [
      'OpenCV',
      'Nister',
      'Lui',
      'Hedborg',
      'Nister Standard',
      'Nister Preemptive',
      'Lui RANSAC',
      'Hedborg RANSAC',
  ]

  N: int = len(pts1_clean)
  for outlier_frac in outlier_fracs:
    r_errs: dict[str, list[float]] = {m: [] for m in method_names}
    t_errs: dict[str, list[float]] = {m: [] for m in method_names}

    for _ in range(n_trials):
      pts1: Arr = _add_pixel_noise(pts1_clean, noise_sigma_px, focal_length)
      pts2: Arr = _add_pixel_noise(pts2_clean, noise_sigma_px, focal_length)

      if outlier_frac > 0:
        n_out: int = max(1, int(N * outlier_frac))
        idx: Arr = np.random.choice(N, n_out, replace=False)
        bound: float = max(np.abs(pts1_clean).max(),
                           np.abs(pts2_clean).max()) * 2
        pts1[idx] = np.random.uniform(-bound, bound, (n_out, 2))
        pts2[idx] = np.random.uniform(-bound, bound, (n_out, 2))

      try:
        result: dict[str,
                     tuple[float, float,
                           float]] = _run_solvers(pts1,
                                                  pts2,
                                                  R_gt,
                                                  t_gt,
                                                  ransac_iters=ransac_iters)
        for m in method_names:
          if m in result:
            r_errs[m].append(result[m][0])
            t_errs[m].append(result[m][1])
      except Exception:
        pass

    def _fmt(a: list[float]) -> str:
      if len(a) == 0:
        return f"{'—':>28}"
      r_arr: Arr = np.array(a)
      return f"{r_arr.mean():>12.4e} ± {r_arr.std():>12.4e}"

    hdr: str = f"Outliers {outlier_frac*100:.0f}%  (noise {noise_sigma_px:.1f} px)"
    print(f"\n{'='*len(hdr)}")
    print(hdr)
    print(f"{'='*len(hdr)}")
    print(f"{'Method':<20} {'R_err':>28} {'t_err (deg)':>28}  {'n':>4}")
    print("-" * 84)
    for m in method_names:
      r: list[float] = r_errs[m]
      t: list[float] = t_errs[m]
      print(f"{m:<20} {_fmt(r)}  {_fmt(t)}  {len(r):>4}")


if __name__ == "__main__":
  np.random.seed(42)

  theta: float = np.radians(15.0)
  R_gt: Arr = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
  t_gt: Arr = np.array([0.5, -0.2, 0.84])
  t_gt /= np.linalg.norm(t_gt)

  N: int = 60
  X_3D: Arr = np.random.uniform(-1, 1, (N, 3))
  X_3D[:, 2] += 3.0

  pts1_clean: Arr = X_3D[:, :2] / X_3D[:, 2:]
  X_cam2: Arr = (R_gt @ X_3D.T).T + t_gt
  pts2_clean: Arr = X_cam2[:, :2] / X_cam2[:, 2:]

  # benchmark_summary(
  #     R_gt=R_gt,
  #     t_gt=t_gt,
  #     pts1_clean=pts1_clean,
  #     pts2_clean=pts2_clean,
  #     n_trials=10,
  #     ransac_iters=50,
  # )

  outlier_benchmark(
      R_gt=R_gt,
      t_gt=t_gt,
      pts1_clean=pts1_clean,
      pts2_clean=pts2_clean,
      n_trials=10,
      ransac_iters=50,
  )
