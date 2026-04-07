use std::ops::Mul;

use nalgebra::Dyn;
use nalgebra::OMatrix;
use nalgebra::SMatrix;
use nalgebra::SVector;

type Vector2d = SVector<f64, 2>;
type Vector3d = SVector<f64, 3>;
type Vector4d = SVector<f64, 4>;
type Matrix3d = SMatrix<f64, 3, 3>;
type Matrix4d = SMatrix<f64, 4, 4>;
type MatrixXd = OMatrix<f64, Dyn, Dyn>;

///////////////////////////////////////////////////////////////////////////////
// ROTATION MATRIX                                                           //
///////////////////////////////////////////////////////////////////////////////

/// Form rotation matrix around x axis
pub fn rotx(theta: f64) -> Matrix3d {
  let ctheta = theta.cos();
  let stheta = theta.sin();

  #[rustfmt::skip]
  let data = [
    1.0,    0.0,     0.0,
    0.0, ctheta, -stheta,
    0.0, stheta,  ctheta,
  ];

  Matrix3d::from_row_slice(&data)
}

/// Form rotation matrix around y axis
pub fn roty(theta: f64) -> Matrix3d {
  let ctheta = theta.cos();
  let stheta = theta.sin();

  #[rustfmt::skip]
  let data = [
     ctheta, 0.0, stheta,
        0.0, 1.0,    0.0,
    -stheta, 0.0, ctheta,
  ];

  Matrix3d::from_row_slice(&data)
}

/// Form rotation matrix around z axis
pub fn rotz(theta: f64) -> Matrix3d {
  let ctheta = theta.cos();
  let stheta = theta.sin();

  #[rustfmt::skip]
  let data = [
    ctheta, -stheta, 0.0,
    stheta,  ctheta, 0.0,
       0.0,     0.0, 1.0,
  ];

  Matrix3d::from_row_slice(&data)
}

/// Convert rotation matrix to euler angles
pub fn rot2euler(rot: &Matrix3d) -> Vector3d {
  let q = Quaternion::from_rot(rot);
  q.to_euler()
}

/// Calculate difference between rotation matrices
pub fn rot_diff(rot1: &Matrix3d, rot2: &Matrix3d, tol: f64) -> f64 {
  let drot = &rot1.transpose() * rot2;
  let mut tr = drot.trace();
  if tr < 0.0 {
    tr *= -1.0;
  }

  if (tr - 3.0).abs() < tol {
    0.0
  } else {
    ((tr - 1.0) / 2.0).acos()
  }
}

/// Convert yaw, pitch, roll in radians to a 3x3 rotation matrix.
///
/// Source:
/// Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
/// Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
/// Princeton University Press, 1999. Print.
/// [Page 85-86, "The Aerospace Sequence"]
pub fn euler321(roll: f64, yaw: f64, pitch: f64) -> Matrix3d {
  let phi = roll;
  let theta = pitch;
  let psi = yaw;

  let cpsi = (psi).cos();
  let spsi = (psi).sin();
  let ctheta = (theta).cos();
  let stheta = (theta).sin();
  let cphi = (phi).cos();
  let sphi = (phi).sin();

  let r11 = cpsi * ctheta;
  let r21 = spsi * ctheta;
  let r31 = -stheta;

  let r12 = cpsi * stheta * sphi - spsi * cphi;
  let r22 = spsi * stheta * sphi + cpsi * cphi;
  let r32 = ctheta * sphi;

  let r13 = cpsi * stheta * cphi + spsi * sphi;
  let r23 = spsi * stheta * cphi - cpsi * sphi;
  let r33 = ctheta * cphi;

  Matrix3d::from_row_slice(&[r11, r12, r13, r21, r22, r23, r31, r32, r33])
}

///////////////////////////////////////////////////////////////////////////////
// AXIS ANGLE                                                                //
///////////////////////////////////////////////////////////////////////////////

pub struct AxisAngle {
  pub axis: Vector3d,
  pub angle: f64,
}

impl AxisAngle {
  pub fn new(axis: Vector3d, angle: f64) -> Self {
    Self { axis, angle }
  }

  pub fn from_vec(&self, aa: Vector3d) -> AxisAngle {
    let angle = aa.norm();
    let axis = aa * (1.0 / angle);
    AxisAngle { axis, angle }
  }

  pub fn from_vecs(&self, u: Vector3d, v: Vector3d) -> AxisAngle {
    let angle = (u.dot(&v)).acos();
    let axis = (u.cross(&v)).normalize();
    AxisAngle { axis, angle }
  }

  pub fn to_vec(&self) -> Vector3d {
    self.axis * self.angle
  }

  pub fn to_rot(&self) -> Matrix3d {
    let x = self.axis.x;
    let y = self.axis.y;
    let z = self.axis.z;

    let norm = (x * x + y * y + z * z).sqrt();
    if norm < 1e-12 {
      return Matrix3d::identity();
    }
    let x = x / norm;
    let y = y / norm;
    let z = z / norm;

    let c = self.angle.cos();
    let s = self.angle.sin();
    let one_c = 1.0 - c;

    let r00 = c + x * x * one_c;
    let r01 = x * y * one_c - z * s;
    let r02 = x * z * one_c + y * s;

    let r10 = y * x * one_c + z * s;
    let r11 = c + y * y * one_c;
    let r12 = y * z * one_c - x * s;

    let r20 = z * x * one_c - y * s;
    let r21 = z * y * one_c + x * s;
    let r22 = c + z * z * one_c;

    Matrix3d::from_row_slice(&[r00, r01, r02, r10, r11, r12, r20, r21, r22])
  }
}

#[cfg(test)]
mod axis_angle_tests {
  use super::*;
  const EPS: f64 = 1e-6;

  fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
  }

  #[test]
  #[allow(non_snake_case)]
  fn test_axis_angle_z_90deg() {
    // Rotate 90 degrees around Z
    let axis = Vector3d::new(0.0, 0.0, 1.0);
    let angle = std::f64::consts::FRAC_PI_2;

    let aa = AxisAngle::new(axis, angle);
    let R = aa.to_rot();

    // x axis should rotate to y axis
    let v = Vector3d::new(1.0, 0.0, 0.0);
    let result = &R * &v;

    assert!(approx_eq(result.x, 0.0));
    assert!(approx_eq(result.y, 1.0));
    assert!(approx_eq(result.z, 0.0));
  }

  #[test]
  #[allow(non_snake_case)]
  fn test_axis_angle_identity() {
    let axis = Vector3d::new(1.0, 0.0, 0.0);
    let angle = 0.0;

    let aa = AxisAngle::new(axis, angle);
    let R = aa.to_rot();

    let v = Vector3d::new(1.0, 2.0, 3.0);
    let result = &R * &v;

    assert!(approx_eq(result.x, v.x));
    assert!(approx_eq(result.y, v.y));
    assert!(approx_eq(result.z, v.z));
  }

  #[test]
  #[allow(non_snake_case)]
  fn test_axis_normalization() {
    // axis does not need to be normalized
    let axis = Vector3d::new(0.0, 0.0, 10.0);
    let angle = std::f64::consts::FRAC_PI_2;

    let aa = AxisAngle::new(axis, angle);
    let R = aa.to_rot();

    let v = Vector3d::new(1.0, 0.0, 0.0);
    let result = &R * &v;

    assert!(approx_eq(result.x, 0.0));
    assert!(approx_eq(result.y, 1.0));
    assert!(approx_eq(result.z, 0.0));
  }
}

///////////////////////////////////////////////////////////////////////////////
// QUATERNION                                                                //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy)]
pub struct Quaternion {
  pub data: [f64; 4],
}

impl Default for Quaternion {
  fn default() -> Self {
    Quaternion {
      data: [0.0, 0.0, 0.0, 1.0],
    }
  }
}

impl Quaternion {
  pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
    Self { data: [x, y, z, w] }
  }

  pub fn x(&self) -> f64 {
    self.data[0]
  }

  pub fn y(&self) -> f64 {
    self.data[1]
  }

  pub fn z(&self) -> f64 {
    self.data[2]
  }

  pub fn w(&self) -> f64 {
    self.data[3]
  }

  pub fn at(&self, i: usize) -> &f64 {
    assert!(i < 3, "Index out of bounds");
    &self.data[i]
  }

  pub fn set(&mut self, i: usize, value: f64) {
    assert!(i < 3, "Index out of bounds");
    self.data[i] = value;
  }

  pub fn norm(&self) -> f64 {
    let x2 = self.x() * self.x();
    let y2 = self.y() * self.y();
    let z2 = self.z() * self.z();
    let w2 = self.w() * self.w();
    (x2 + y2 + z2 + w2).sqrt()
  }

  pub fn normalize(&self) -> Self {
    let n = self.norm();
    let x = self.x() / n;
    let y = self.y() / n;
    let z = self.z() / n;
    let w = self.w() / n;
    Self { data: [x, y, z, w] }
  }

  pub fn conj(&self) -> Self {
    let x = -self.x();
    let y = -self.y();
    let z = -self.z();
    let w = self.w();
    Self { data: [x, y, z, w] }
  }

  pub fn left_product(&self) -> Matrix4d {
    let qw = self.w();
    let qx = self.x();
    let qy = self.y();
    let qz = self.z();

    #[rustfmt::skip]
    let data = [
      qw, -qx, -qy, -qz,
      qx, qw, -qz, qy,
      qy, qz, qw, -qx,
      qz, -qy, qx, qw
    ];

    Matrix4d::from_row_slice(&data)
  }

  pub fn right_product(&self) -> Matrix4d {
    let qw = self.w();
    let qx = self.x();
    let qy = self.y();
    let qz = self.z();

    #[rustfmt::skip]
    let data = [
      qw, -qx, -qy, -qz,
      qx, qw, qz, -qy,
      qy, -qz, qw, qx,
      qz, qy, -qx, qw
    ];

    Matrix4d::from_row_slice(&data)
  }

  fn mul(&self, rhs: &Quaternion) -> Quaternion {
    let q = &self.left_product() * &rhs.to_vec();
    Quaternion { data: q.into() }
  }

  pub fn from_array(data: [f64; 4]) -> Self {
    Quaternion::new(data[3], data[0], data[1], data[2])
  }

  pub fn to_vec(&self) -> Vector4d {
    Vector4d::from(self.data)
  }

  /// Convert yaw, pitch, roll in radians to a quaternion.
  ///
  /// Source:
  /// Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
  /// Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
  /// Princeton University Press, 1999. Print.
  /// [Page 166-167, "Euler Angles to Quaternion"]
  pub fn from_euler(yaw: f64, pitch: f64, roll: f64) -> Self {
    let psi = yaw;
    let theta = pitch;
    let phi = roll;

    let c_phi = (phi / 2.0).cos();
    let c_theta = (theta / 2.0).cos();
    let c_psi = (psi / 2.0).cos();
    let s_phi = (phi / 2.0).sin();
    let s_theta = (theta / 2.0).sin();
    let s_psi = (psi / 2.0).sin();

    let mut qw = c_psi * c_theta * c_phi + s_psi * s_theta * s_phi;
    let mut qx = c_psi * c_theta * s_phi - s_psi * s_theta * c_phi;
    let mut qy = c_psi * s_theta * c_phi + s_psi * c_theta * s_phi;
    let mut qz = s_psi * c_theta * c_phi - c_psi * s_theta * s_phi;
    let n = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
    qw /= n;
    qx /= n;
    qy /= n;
    qz /= n;

    Self {
      data: [qx, qy, qz, qw],
    }
  }

  pub fn from_rot(rot: &Matrix3d) -> Self {
    let m00: f64 = rot[(0, 0)];
    let m01: f64 = rot[(0, 1)];
    let m02: f64 = rot[(0, 2)];

    let m10: f64 = rot[(1, 0)];
    let m11: f64 = rot[(1, 1)];
    let m12: f64 = rot[(1, 2)];

    let m20: f64 = rot[(2, 0)];
    let m21: f64 = rot[(2, 1)];
    let m22: f64 = rot[(2, 2)];

    let tr = m00 + m11 + m22;
    if tr > 0.0 {
      let s = (tr + 1.0).sqrt() * 2.0;
      let qw = 0.25 * s;
      let qx = (m21 - m12) / s;
      let qy = (m02 - m20) / s;
      let qz = (m10 - m01) / s;

      let q = Quaternion::new(qw, qx, qy, qz);
      q.normalize()
    } else if m00 > m11 && m00 > m22 {
      let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
      let qw = (m21 - m12) / s;
      let qx = 0.25 * s;
      let qy = (m01 + m10) / s;
      let qz = (m02 + m20) / s;

      let q = Quaternion::new(qw, qx, qy, qz);
      q.normalize()
    } else if m11 > m22 {
      let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
      let qw = (m02 - m20) / s;
      let qx = (m01 + m10) / s;
      let qy = 0.25 * s;
      let qz = (m12 + m21) / s;

      let q = Quaternion::new(qw, qx, qy, qz);
      q.normalize()
    } else {
      let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
      let qw = (m10 - m01) / s;
      let qx = (m02 + m20) / s;
      let qy = (m12 + m21) / s;
      let qz = 0.25 * s;

      let q = Quaternion::new(qw, qx, qy, qz);
      q.normalize()
    }
  }

  pub fn from_axis_angle(axis: &Vector3d, angle: f64) -> Self {
    let ax = axis.x;
    let ay = axis.y;
    let az = axis.z;
    let k = (angle / 2.0).sin();
    let qw = (angle / 2.0).cos();
    let qx = ax * k;
    let qy = ay * k;
    let qz = az * k;

    Self {
      data: [qx, qy, qz, qw],
    }
  }

  /// Convert quaternion to euler angles (yaw, pitch, roll).
  ///
  /// Source:
  /// Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
  /// Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
  /// Princeton University Press, 1999. Print.
  /// [Page 168, "Quaternion to Euler Angles"]
  pub fn to_euler(&self) -> Vector3d {
    let qw = self.w();
    let qx = self.x();
    let qy = self.y();
    let qz = self.z();

    let qw2 = qw * qw;
    let qx2 = qx * qx;
    let qz2 = qz * qz;

    let m11 = (2.0 * qw2) + (2.0 * qx2) - 1.0;
    let m12 = 2.0 * (qx * qy + qw * qz);
    let m13 = 2.0 * qx * qz - 2.0 * qw * qy;
    let m23 = 2.0 * qy * qz + 2.0 * qw * qx;
    let m33 = (2.0 * qw2) + (2.0 * qz2) - 1.0;

    let psi = (m11).atan2(m12);
    let theta = (-m13).asin();
    let phi = (m33).atan2(m23);

    Vector3d::new(psi, theta, phi)
  }

  /// Convert quaternion to 3x3 rotation matrix.
  ///
  /// Source:
  /// Blanco, Jose-Luis. "A tutorial on se (3) transformation parameterizations
  /// and on-manifold optimization." University of Malaga, Tech. Rep 3 (2010): 6.
  /// [Page 18, Equation (2.20)]
  pub fn to_rot(&self) -> Matrix3d {
    let qw = self.w();
    let qx = self.x();
    let qy = self.y();
    let qz = self.z();

    let qw2 = qw * qw;
    let qx2 = qx * qx;
    let qy2 = qy * qy;
    let qz2 = qz * qz;

    // Homogeneous form
    let r11 = qw2 + qx2 - qy2 - qz2;
    let r12 = 2.0 * (qx * qy - qw * qz);
    let r13 = 2.0 * (qx * qz + qw * qy);

    let r21 = 2.0 * (qx * qy + qw * qz);
    let r22 = qw2 - qx2 + qy2 - qz2;
    let r23 = 2.0 * (qy * qz - qw * qx);

    let r31 = 2.0 * (qx * qz - qw * qy);
    let r32 = 2.0 * (qy * qz + qw * qx);
    let r33 = qw2 - qx2 - qy2 + qz2;

    Matrix3d::from_row_slice(&[r11, r12, r13, r21, r22, r23, r31, r32, r33])
  }
}

///////////////////////////////////////////////////////////////////////////////
// TRANSFORM                                                                 //
///////////////////////////////////////////////////////////////////////////////

pub struct Transform {
  parent: String,
  child: String,
  pos: Vector3d,
  quat: Quaternion,
}

impl Transform {
  fn default() -> Self {
    Transform {
      parent: String::from("NOT_SET"),
      child: String::from("NOT_SET"),
      pos: Vector3d::default(),
      quat: Quaternion::default(),
    }
  }

  pub fn new(
    parent: String,
    child: String,
    pos: Vector3d,
    quat: Quaternion,
  ) -> Self {
    Self {
      parent,
      child,
      pos,
      quat,
    }
  }

  pub fn x(&self) -> f64 {
    self.pos.x
  }

  pub fn y(&self) -> f64 {
    self.pos.y
  }

  pub fn z(&self) -> f64 {
    self.pos.z
  }

  pub fn rot(&self) -> Matrix3d {
    self.quat.to_rot()
  }

  pub fn inv(&self) -> Self {
    #[allow(non_snake_case)]
    {
      let R = self.rot();
      let t = self.pos;

      let R_new = &R.transpose();
      let pos_new = &-R.transpose() * &t;
      let quat_new = Quaternion::from_rot(R_new);

      Transform {
        parent: self.child.clone(),
        child: self.parent.clone(),
        pos: pos_new,
        quat: quat_new,
      }
    }
  }

  // pub fn transform(&self, rhs: &Vector3d) {
  //
  // }

  pub fn from_mat(parent: String, child: String, tf: &Matrix4d) -> Self {
    // Translation
    let px = tf[(0, 3)];
    let py = tf[(1, 3)];
    let pz = tf[(2, 3)];
    let pos = Vector3d::new(px, py, pz);

    // Rotation
    let m00 = tf[(0, 0)];
    let m01 = tf[(0, 1)];
    let m02 = tf[(0, 2)];

    let m10 = tf[(1, 0)];
    let m11 = tf[(1, 1)];
    let m12 = tf[(1, 2)];

    let m20 = tf[(2, 0)];
    let m21 = tf[(2, 1)];
    let m22 = tf[(2, 2)];

    #[rustfmt::skip]
    let rot = Matrix3d::from_row_slice(&[
      m00, m01, m02,
      m10, m11, m12,
      m20, m21, m22,
    ]);
    let quat = Quaternion::from_rot(&rot);

    // Transform
    Self {
      parent: parent.clone(),
      child: child.clone(),
      pos,
      quat,
    }
  }

  pub fn to_mat(&self) -> Matrix4d {
    let rot = self.quat.to_rot();
    let pos = self.pos;

    let mut tf = Matrix4d::identity();
    tf.fixed_view_mut::<3, 3>(0, 0).copy_from(&rot);
    tf.fixed_view_mut::<3, 1>(0, 3).copy_from(&pos);
    tf
  }
}

// -- Transform * Transform
impl Mul<&Transform> for &Transform {
  type Output = Transform;
  fn mul(self, rhs: &Transform) -> Self::Output {
    assert!(self.child == rhs.parent);

    let a = self.to_mat();
    let b = rhs.to_mat();
    let c = &a * &b;

    Transform::from_mat(self.parent.clone(), rhs.child.clone(), &c)
  }
}
