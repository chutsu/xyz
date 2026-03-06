// use std::ops::Mul;

use crate::linalg::Matrix3d;
use crate::linalg::Matrix4d;
use crate::linalg::Vector3d;
use crate::linalg::Vector4d;

///////////////////////////////////////////////////////////////////////////////
// ROTATION MATRIX                                                           //
///////////////////////////////////////////////////////////////////////////////

// Form rotation matrix around x axis
fn rotx(theta: f64) -> Matrix3d {
  let ctheta = theta.cos();
  let stheta = theta.sin();

  #[rustfmt::skip]
  let data = [
    1.0,    0.0,     0.0,
    0.0, ctheta, -stheta,
    0.0, stheta,  ctheta,
  ];

  Matrix3d { data }
}

// Form rotation matrix around y axis
fn roty(theta: f64) -> Matrix3d {
  let ctheta = theta.cos();
  let stheta = theta.sin();

  #[rustfmt::skip]
  let data = [
     ctheta, 0.0, stheta,
        0.0, 1.0,    0.0,
    -stheta, 0.0, ctheta,
  ];

  Matrix3d { data }
}

// Form rotation matrix around z axis
fn rotz(theta: f64) -> Matrix3d {
  let ctheta = theta.cos();
  let stheta = theta.sin();

  #[rustfmt::skip]
  let data = [
    ctheta, -stheta, 0.0,
    stheta,  ctheta, 0.0,
       0.0,     0.0, 1.0,
  ];

  Matrix3d { data }
}

// Convert rotation matrix to euler angles
fn rot2euler(rot: &Matrix3d) -> Vector3d {
  let q = Quaternion::from_rot(rot);
  q.to_euler()
}

// Calculate difference between rotation matrices
fn rot_diff(rot1: &Matrix3d, rot2: &Matrix3d, tol: f64) -> f64 {
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

// Convert euler angles to rotation matrix
pub fn euler321(yaw: f64, pitch: f64, roll: f64) -> Matrix3d {
  // Convert yaw, pitch, roll in radians to a 3x3 rotation matrix.
  //
  // Source:
  // Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
  // Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
  // Princeton University Press, 1999. Print.
  // [Page 85-86, "The Aerospace Sequence"]

  let psi = yaw;
  let theta = pitch;
  let phi = roll;

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

  Matrix3d {
    #[rustfmt::skip]
      data: [r11, r12, r13,
             r21, r22, r23,
             r31, r32, r33],
  }
}

///////////////////////////////////////////////////////////////////////////////
// AXIS ANGLE                                                                //
///////////////////////////////////////////////////////////////////////////////

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

// // Convert axis-angle to rotation matrix
// pub fn aa2rot(aa: Vector3d) -> Matrix3d {
//   // If small rotation
//   // let theta = sqrt(aa * aa);  // = norm(aa), but faster
//   // let eps = 1e-8;
//   // if theta < eps {
//   //   return hat(aa)
//   // }
//
//   // Convert aa to rotation matrix
//   // let aa = aa / theta
//   // x, y, z = aa
//
//   //   c = cos(theta)
//   //   s = sin(theta)
//   //   C = 1 - c
//   //
//   //   xs = x * s
//   //   ys = y * s
//   //   zs = z * s
//   //
//   //   xC = x * C
//   //   yC = y * C
//   //   zC = z * C
//   //
//   //   xyC = x * yC
//   //   yzC = y * zC
//   //   zxC = z * xC
//   //
//   //   row0 = [x * xC + c, xyC - zs, zxC + ys]
//   //   row1 = [xyC + zs, y * yC + c, yzC - xs]
//   //   row2 = [zxC - ys, yzC + xs, z * zC + c]
//
//   Matrix3d {
//     data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
//   }
// }

///////////////////////////////////////////////////////////////////////////////
// QUATERNION                                                                //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy)]
pub struct Quaternion {
  pub data: [f64; 4],
}

impl Quaternion {
  pub fn new(data: [f64; 4]) -> Self {
    Self { data }
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

    Matrix4d {
      #[rustfmt::skip]
      data: [qw, -qx, -qy, -qz,
             qx, qw, -qz, qy,
             qy, qz, qw, -qx,
             qz, -qy, qx, qw],
    }
  }

  pub fn right_product(&self) -> Matrix4d {
    let qw = self.w();
    let qx = self.x();
    let qy = self.y();
    let qz = self.z();

    Matrix4d {
      #[rustfmt::skip]
      data: [qw, -qx, -qy, -qz,
             qx, qw, qz, -qy,
             qy, -qz, qw, qx,
             qz, qy, -qx, qw],
    }
  }

  fn mul(&self, rhs: &Quaternion) -> Quaternion {
    let q = &self.left_product() * &rhs.to_vec();
    Quaternion { data: q.data }
  }

  pub fn to_vec(&self) -> Vector4d {
    Vector4d { data: self.data }
  }

  pub fn from_euler(yaw: f64, pitch: f64, roll: f64) -> Self {
    // Convert yaw, pitch, roll in radians to a quaternion.
    //
    // Source:
    // Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
    // Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
    // Princeton University Press, 1999. Print.
    // [Page 166-167, "Euler Angles to Quaternion"]
    //
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
    let m00 = rot.at(0, 0);
    let m01 = rot.at(0, 1);
    let m02 = rot.at(0, 2);

    let m10 = rot.at(1, 0);
    let m11 = rot.at(1, 1);
    let m12 = rot.at(1, 2);

    let m20 = rot.at(2, 0);
    let m21 = rot.at(2, 1);
    let m22 = rot.at(2, 2);

    let tr = m00 + m11 + m22;
    if tr > 0.0 {
      let s = (tr + 1.0).sqrt() * 2.0;
      let qw = 0.25 * s;
      let qx = (m21 - m12) / s;
      let qy = (m02 - m20) / s;
      let qz = (m10 - m01) / s;

      let q = Quaternion::new([qx, qy, qz, qw]);
      q.normalize()
    } else if m00 > m11 && m00 > m22 {
      let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
      let qw = (m21 - m12) / s;
      let qx = 0.25 * s;
      let qy = (m01 + m10) / s;
      let qz = (m02 + m20) / s;

      let q = Quaternion::new([qx, qy, qz, qw]);
      q.normalize()
    } else if m11 > m22 {
      let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
      let qw = (m02 - m20) / s;
      let qx = (m01 + m10) / s;
      let qy = 0.25 * s;
      let qz = (m12 + m21) / s;

      let q = Quaternion::new([qx, qy, qz, qw]);
      q.normalize()
    } else {
      let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
      let qw = (m10 - m01) / s;
      let qx = (m02 + m20) / s;
      let qy = (m12 + m21) / s;
      let qz = 0.25 * s;

      let q = Quaternion::new([qx, qy, qz, qw]);
      q.normalize()
    }
  }

  pub fn from_axis_angle(axis: &Vector3d, angle: f64) -> Self {
    let ax = axis.x();
    let ay = axis.y();
    let az = axis.z();
    let k = (angle / 2.0).sin();
    let qw = (angle / 2.0).cos();
    let qx = ax * k;
    let qy = ay * k;
    let qz = az * k;

    Self {
      data: [qx, qy, qz, qw],
    }
  }

  pub fn to_euler(&self) -> Vector3d {
    // Convert quaternion to euler angles (yaw, pitch, roll).
    //
    // Source:
    // Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
    // Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
    // Princeton University Press, 1999. Print.
    // [Page 168, "Quaternion to Euler Angles"]

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

    Vector3d {
      data: [psi, theta, phi],
    }
  }

  pub fn to_rot(&self) -> Matrix3d {
    // Convert quaternion to 3x3 rotation matrix.
    //
    // Source:
    // Blanco, Jose-Luis. "A tutorial on se (3) transformation parameterizations
    // and on-manifold optimization." University of Malaga, Tech. Rep 3 (2010): 6.
    // [Page 18, Equation (2.20)]

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

    Matrix3d {
      #[rustfmt::skip]
      data: [r11, r12, r13,
             r21, r22, r23,
             r31, r32, r33],
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// TRANSFORM                                                                 //
///////////////////////////////////////////////////////////////////////////////

pub struct Transform {
  parent: String,
  child: String,
  data: Matrix4d,
}

impl Transform {
  fn default() -> Self {
    Transform {
      parent: String::from("NOT_SET"),
      child: String::from("NOT_SET"),
      data: Matrix4d::eye(),
    }
  }

  pub fn new(parent: String, child: String, data: Matrix4d) -> Self {
    Self {
      parent,
      child,
      data,
    }
  }

  pub fn x(&self) -> f64 {
    *self.data.at(0, 3)
  }

  pub fn y(&self) -> f64 {
    *self.data.at(1, 3)
  }

  pub fn z(&self) -> f64 {
    *self.data.at(2, 3)
  }

  // pub fn pos(&self) -> Vector3d {
  // }

  // pub fn rot(&self) -> Matrix3d {
  // }
  //
  // pub fn quat(&self) -> Quaternion{
  // }

  // pub fn transform(&self, rhs: &Vector3d) {
  //
  // }
}

// -- Transform * Transform
// impl Mul<&Transform> for &Transform {
//   type Output = Transform;
//   fn mul(self, rhs: &Transform) -> Self::Output {
//     assert!(self.child == rhs.parent);
//     Transform::new(self.parent, rhs.child, &self.data * &rhs.data)
//   }
// }
