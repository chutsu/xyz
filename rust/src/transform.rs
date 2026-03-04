

pub struct Transform {
  parent: String,
  child: String,
  data: [f64, 16],
}


// def rotx(theta: float) -> Mat3:
//   """Form rotation matrix around x axis"""
//   row0 = [1.0, 0.0, 0.0]
//   row1 = [0.0, cos(theta), -sin(theta)]
//   row2 = [0.0, sin(theta), cos(theta)]
//   return np.array([row0, row1, row2])
//
//
// def roty(theta: float) -> Mat3:
//   """Form rotation matrix around y axis"""
//   row0 = [cos(theta), 0.0, sin(theta)]
//   row1 = [0.0, 1.0, 0.0]
//   row2 = [-sin(theta), 0.0, cos(theta)]
//   return np.array([row0, row1, row2])
//
//
// def rotz(theta: float) -> Mat3:
//   """Form rotation matrix around z axis"""
//   row0 = [cos(theta), -sin(theta), 0.0]
//   row1 = [sin(theta), cos(theta), 0.0]
//   row2 = [0.0, 0.0, 1.0]
//   return np.array([row0, row1, row2])
//
//
// def rot2euler(C: Mat3) -> Vec3:
//   """
//   Convert 3x3 rotation matrix to euler angles (yaw, pitch, roll). The result is
//   also equivalent to rotation around (z, y, x) axes.
//   """
//   assert C.shape == (3, 3)
//   q = rot2quat(C)
//   return quat2euler(q)
//
//
// def rot2quat(C: Mat3) -> Vec4:
//   """
//   Convert 3x3 rotation matrix to quaternion.
//   """
//   assert C.shape == (3, 3)
//
//   m00 = C[0, 0]
//   m01 = C[0, 1]
//   m02 = C[0, 2]
//
//   m10 = C[1, 0]
//   m11 = C[1, 1]
//   m12 = C[1, 2]
//
//   m20 = C[2, 0]
//   m21 = C[2, 1]
//   m22 = C[2, 2]
//
//   tr = m00 + m11 + m22
//
//   if tr > 0:
//     S = sqrt(tr + 1.0) * 2.0
//     # S=4*qw
//     qw = 0.25 * S
//     qx = (m21 - m12) / S
//     qy = (m02 - m20) / S
//     qz = (m10 - m01) / S
//   elif (m00 > m11) and (m00 > m22):
//     S = sqrt(1.0 + m00 - m11 - m22) * 2.0
//     # S=4*qx
//     qw = (m21 - m12) / S
//     qx = 0.25 * S
//     qy = (m01 + m10) / S
//     qz = (m02 + m20) / S
//   elif m11 > m22:
//     S = sqrt(1.0 + m11 - m00 - m22) * 2.0
//     # S=4*qy
//     qw = (m02 - m20) / S
//     qx = (m01 + m10) / S
//     qy = 0.25 * S
//     qz = (m12 + m21) / S
//   else:
//     S = sqrt(1.0 + m22 - m00 - m11) * 2.0
//     # S=4*qz
//     qw = (m10 - m01) / S
//     qx = (m02 + m20) / S
//     qy = (m12 + m21) / S
//     qz = 0.25 * S
//
//   return quat_normalize(np.array([qw, qx, qy, qz]))
//
//
// def rot_diff(C0: Mat3, C1: Mat3, tol: float = 1e-5):
//   """Difference between two rotation matrices"""
//   dC = C0.T @ C1
//   tr = np.trace(dC)
//   if tr < 0:
//     tr *= -1
//
//   if np.fabs(tr - 3.0) < tol:
//     dtheta = 0.0
//   else:
//     dtheta = acos((tr - 1.0) / 2.0)
//
//   return dtheta
//
//
// def aa2quat(axis: Vec3, angle: float) -> Vec4:
//   """
//   Convert Axis-angle to quaternion
//
//   Source:
//   Sola, Joan. "Quaternion kinematics for the error-state Kalman filter." arXiv
//   preprint arXiv:1711.02508 (2017).
//   [Page 22, eq (101), "Quaternion and rotation vector"]
//   """
//   ax, ay, az = axis
//   qw = cos(angle / 2.0)
//   qx = ax * sin(angle / 2.0)
//   qy = ay * sin(angle / 2.0)
//   qz = az * sin(angle / 2.0)
//   return np.array([qw, qx, qy, qz])
//
//
// def aa2rot(aa: Vec3) -> Mat3:
//   """Axis-angle to rotation matrix"""
//   # If small rotation
//   theta = sqrt(aa @ aa)  # = norm(aa), but faster
//   eps = 1e-8
//   if theta < eps:
//     return hat(aa)
//
//   # Convert aa to rotation matrix
//   aa = aa / theta
//   x, y, z = aa
//
//   c = cos(theta)
//   s = sin(theta)
//   C = 1 - c
//
//   xs = x * s
//   ys = y * s
//   zs = z * s
//
//   xC = x * C
//   yC = y * C
//   zC = z * C
//
//   xyC = x * yC
//   yzC = y * zC
//   zxC = z * xC
//
//   row0 = [x * xC + c, xyC - zs, zxC + ys]
//   row1 = [xyC + zs, y * yC + c, yzC - xs]
//   row2 = [zxC - ys, yzC + xs, z * zC + c]
//   return np.array([row0, row1, row2])
//
//
// def aa_vec(axis: Vec3, angle: float) -> Vec3:
//   """Form Axis-Angle Vector"""
//   assert axis.shape[0] == 3
//   return axis * angle
//
//
// def aa_decomp(aa: Vec3):
//   """Decompose an axis-angle into its components"""
//   w = aa / np.linalg.norm(aa)
//   theta = np.linalg.norm(aa)
//   return w, theta
//
//
// def vecs2aa(u: Vec3, v: Vec3) -> Vec3:
//   """From 2 vectors form an axis-angle vector"""
//   angle = math.acos(u.T * v)
//   ax = normalize(np.cross(u, v))
//   return ax * angle
//
//
// def euler321(
//   yaw: float | np.float32 | np.float64,
//   pitch: float | np.float32 | np.float64,
//   roll: float | np.float32 | np.float64,
// ) -> Mat3:
//   """
//   Convert yaw, pitch, roll in radians to a 3x3 rotation matrix.
//
//   Source:
//   Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
//   Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
//   Princeton University Press, 1999. Print.
//   [Page 85-86, "The Aerospace Sequence"]
//   """
//   psi = yaw
//   theta = pitch
//   phi = roll
//
//   cpsi = cos(psi)
//   spsi = sin(psi)
//   ctheta = cos(theta)
//   stheta = sin(theta)
//   cphi = cos(phi)
//   sphi = sin(phi)
//
//   C11 = cpsi * ctheta
//   C21 = spsi * ctheta
//   C31 = -stheta
//
//   C12 = cpsi * stheta * sphi - spsi * cphi
//   C22 = spsi * stheta * sphi + cpsi * cphi
//   C32 = ctheta * sphi
//
//   C13 = cpsi * stheta * cphi + spsi * sphi
//   C23 = spsi * stheta * cphi - cpsi * sphi
//   C33 = ctheta * cphi
//
//   return np.array([[C11, C12, C13], [C21, C22, C23], [C31, C32, C33]])
//
//
// def euler2quat(yaw: float, pitch: float, roll: float) -> Mat3:
//   """
//   Convert yaw, pitch, roll in radians to a quaternion.
//
//   Source:
//   Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
//   Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
//   Princeton University Press, 1999. Print.
//   [Page 166-167, "Euler Angles to Quaternion"]
//   """
//   psi = yaw  # Yaw
//   theta = pitch  # Pitch
//   phi = roll  # Roll
//
//   c_phi = cos(phi / 2.0)
//   c_theta = cos(theta / 2.0)
//   c_psi = cos(psi / 2.0)
//   s_phi = sin(phi / 2.0)
//   s_theta = sin(theta / 2.0)
//   s_psi = sin(psi / 2.0)
//
//   qw = c_psi * c_theta * c_phi + s_psi * s_theta * s_phi
//   qx = c_psi * c_theta * s_phi - s_psi * s_theta * c_phi
//   qy = c_psi * s_theta * c_phi + s_psi * c_theta * s_phi
//   qz = s_psi * c_theta * c_phi - c_psi * s_theta * s_phi
//
//   mag = sqrt(qw**2 + qx**2 + qy**2 + qz**2)
//   return np.array([qw / mag, qx / mag, qy / mag, qz / mag])
//
//
// def quat2euler(q: Vec4) -> Vec3:
//   """
//   Convert quaternion to euler angles (yaw, pitch, roll).
//
//   Source:
//   Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
//   Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
//   Princeton University Press, 1999. Print.
//   [Page 168, "Quaternion to Euler Angles"]
//   """
//   qw, qx, qy, qz = q
//
//   m11 = (2 * qw**2) + (2 * qx**2) - 1
//   m12 = 2 * (qx * qy + qw * qz)
//   m13 = 2 * qx * qz - 2 * qw * qy
//   m23 = 2 * qy * qz + 2 * qw * qx
//   m33 = (2 * qw**2) + (2 * qz**2) - 1
//
//   psi = math.atan2(m12, m11)
//   theta = math.asin(-m13)
//   phi = math.atan2(m23, m33)
//
//   ypr = np.array([psi, theta, phi])
//   return ypr
//
//
// def quat2rot(q: Vec4) -> Mat3:
//   """
//   Convert quaternion to 3x3 rotation matrix.
//
//   Source:
//   Blanco, Jose-Luis. "A tutorial on se (3) transformation parameterizations
//   and on-manifold optimization." University of Malaga, Tech. Rep 3 (2010): 6.
//   [Page 18, Equation (2.20)]
//   """
//   assert len(q) == 4
//   qw, qx, qy, qz = q
//
//   qx2 = qx**2
//   qy2 = qy**2
//   qz2 = qz**2
//   qw2 = qw**2
//
//   # Homogeneous form
//   C11 = qw2 + qx2 - qy2 - qz2
//   C12 = 2.0 * (qx * qy - qw * qz)
//   C13 = 2.0 * (qx * qz + qw * qy)
//
//   C21 = 2.0 * (qx * qy + qw * qz)
//   C22 = qw2 - qx2 + qy2 - qz2
//   C23 = 2.0 * (qy * qz - qw * qx)
//
//   C31 = 2.0 * (qx * qz - qw * qy)
//   C32 = 2.0 * (qy * qz + qw * qx)
//   C33 = qw2 - qx2 - qy2 + qz2
//
//   return np.array([[C11, C12, C13], [C21, C22, C23], [C31, C32, C33]])
