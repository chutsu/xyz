use nalgebra::Dyn;
use nalgebra::OMatrix;
use nalgebra::SMatrix;
use nalgebra::SVector;

type Vector2d = SVector<f64, 2>;
type Vector3d = SVector<f64, 3>;
type Vector4d = SVector<f64, 4>;
type Matrix3d = SMatrix<f64, 3, 3>;
type MatrixXd = OMatrix<f64, Dyn, Dyn>;

///////////////////////////////////////////////////////////////////////////////
// RESOLUTION                                                                //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct Resolution {
  pub width: i32,
  pub height: i32,
}

///////////////////////////////////////////////////////////////////////////////
// CAMERA MODEL                                                              //
///////////////////////////////////////////////////////////////////////////////

pub trait CameraModel {
  fn project(res: &Resolution, intrinsic: &[f64], point: &Vector3d) -> bool;
  fn project_jacobian(intrinsic: &[f64], point3d: &Vector3d) -> MatrixXd;
  fn params_jacobian(intrinsic: &[f64], point3d: &Vector3d) -> MatrixXd;
  fn back_project(&self, intrinsic: &[f64], point: &Vector2d) -> Vector3d;
  // fn undistort(&self, intrinsic: &[f64], z: &Vector2d) -> Vector2d;
}

///////////////////////////////////////////////////////////////////////////////
// PINHOLE                                                                   //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct Pinhole {}

/// Estimated focal length based on `image_width` and field of fiew `fov_deg`
/// in degrees.
pub fn pinhole_focal(image_width: i32, fov_deg: f64) -> f64 {
  let deg2rad = std::f64::consts::PI / 180.0;
  ((image_width as f64) / 2.0) / f64::tan((fov_deg / 2.0) * deg2rad)
}

/// Form camera matrix K
#[allow(non_snake_case)]
pub fn pinhole_K(fx: f64, fy: f64, cx: f64, cy: f64) -> Matrix3d {
  Matrix3d::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
}

pub fn pinhole_project_jacobian(point3d: &Vector3d) -> MatrixXd {
  let x = point3d.x;
  let y = point3d.y;
  let z = point3d.z;

  let data = vec![1.0 / z, 0.0, -x / (z * z), 0.0, 1.0 / z, -y / (z * z)];
  MatrixXd::from_vec(2, 3, data)
}

pub fn pinhole_point_jacobian(intrinsic: &[f64]) -> MatrixXd {
  let fx = intrinsic[0];
  let fy = intrinsic[1];
  MatrixXd::from_vec(2, 2, vec![fx, 0.0, 0.0, fy])
}

pub fn pinhole_params_jacobian(point: &Vector2d) -> MatrixXd {
  let x = point.x;
  let y = point.y;
  MatrixXd::from_vec(2, 4, vec![x, 0.0, 1.0, 0.0, 0.0, y, 0.0, 1.0])
}

impl CameraModel for Pinhole {
  fn project(res: &Resolution, intrinsic: &[f64], point3d: &Vector3d) -> bool {
    let fx = intrinsic[0];
    let fy = intrinsic[1];
    let cx = intrinsic[2];
    let cy = intrinsic[3];

    let px = point3d.x / point3d.z;
    let py = point3d.y / point3d.z;
    let zx = fx * px + cx;
    let zy = fy * py + cy;

    let x_ok = zx >= 0.0 && zx < (res.width as f64);
    let y_ok = zy >= 0.0 && zy < (res.height as f64);
    let z_ok = point3d.z > 0.0;

    x_ok && y_ok && z_ok
  }

  fn project_jacobian(intrinsic: &[f64], point3d: &Vector3d) -> MatrixXd {
    let jac_k = pinhole_point_jacobian(intrinsic);
    let jac_p = pinhole_project_jacobian(point3d);
    &jac_k * &jac_p
  }

  fn params_jacobian(_: &[f64], point3d: &Vector3d) -> MatrixXd {
    let px = point3d.x / point3d.z;
    let py = point3d.y / point3d.z;
    let p = Vector2d::new(px, py);
    let mut jac = MatrixXd::zeros(2, 4);
    jac
      .fixed_view_mut::<2, 4>(0, 0)
      .copy_from(&pinhole_params_jacobian(&p));
    jac
  }

  fn back_project(&self, intrinsic: &[f64], point: &Vector2d) -> Vector3d {
    let fx = intrinsic[0];
    let fy = intrinsic[1];
    let cx = intrinsic[2];
    let cy = intrinsic[3];
    let rx = (point.x - cx) / fx;
    let ry = (point.y - cy) / fy;
    Vector3d::new(rx, ry, 1.0)
  }

  // fn undistort(&self, intrinsic: &[f64], z: &Vector2d) -> Vector2d {
  //   let fx = intrinsic[0];
  //   let fy = intrinsic[1];
  //   let cx = intrinsic[2];
  //   let cy = intrinsic[3];
  //
  //   let ray = self.back_project(intrinsic, z);
  //   let zx = fx * ray.x() + cx;
  //   let zy = fy * ray.y() + cy;
  //
  //   Vector2d::new([zx, zy])
  // }
}

///////////////////////////////////////////////////////////////////////////////
// PINHOLE-RADTAN4                                                           //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct PinholeRadtan4 {}

/// Distort point with Radial-Tangential distortion
pub fn radtan4_distort(
  k1: f64,
  k2: f64,
  p1: f64,
  p2: f64,
  p: &Vector2d,
) -> Vector2d {
  // Point
  let x = p.x;
  let y = p.y;

  // Apply radial distortion
  let x2 = x * x;
  let y2 = y * y;
  let r2 = x2 + y2;
  let r4 = r2 * r2;
  let radial_factor = 1.0 + (k1 * r2) + (k2 * r4);
  let x_dash = x * radial_factor;
  let y_dash = y * radial_factor;

  // Apply tangential distortion
  let xy = x * y;
  let x_ddash = x_dash + (2.0 * p1 * xy + p2 * (r2 + 2.0 * x2));
  let y_ddash = y_dash + (p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy);

  Vector2d::new(x_ddash, y_ddash)
}

pub fn radtan4_undistort(
  k1: f64,
  k2: f64,
  p1: f64,
  p2: f64,
  p0: &Vector2d,
) -> Vector2d {
  let max_iter = 5;
  let threshold = 1.0e-15;
  let mut p = *p0;

  #[allow(non_snake_case)]
  for _ in 0..max_iter {
    // Error
    let pd = radtan4_distort(k1, k2, p1, p2, &p);
    let err = p0 - pd;

    // Optimize
    let J = radtan4_point_jacobian(k1, k2, p1, p2, &p);
    let H_inv = (J.transpose() * &J).try_inverse().unwrap();
    let dp = &H_inv * &J.transpose() * err;
    p += dp;

    // Early exit
    if err.dot(&err) < threshold {
      break;
    }
  }

  p
}

/// Radial-tangential point jacobian
pub fn radtan4_point_jacobian(
  k1: f64,
  k2: f64,
  p1: f64,
  p2: f64,
  p: &Vector2d,
) -> MatrixXd {
  let x = p.x;
  let y = p.y;

  let x2 = x * x;
  let y2 = y * y;
  let r2 = x2 + y2;
  let r4 = r2 * r2;

  #[allow(non_snake_case)]
  {
    // Point Jacobian
    // Let u = [x; y] normalized point
    // Let u' be the distorted u
    // The jacobian of u' w.r.t. u (or du'/du) is:
    let mut J00 = k1 * r2 + k2 * r4 + 2.0 * p1 * y + 6.0 * p2 * x;
    J00 += x * (2.0 * k1 * x + 4.0 * k2 * x * r2) + 1.0;
    let mut J10 = 2.0 * p1 * x + 2.0 * p2 * y;
    J10 += y * (2.0 * k1 * x + 4.0 * k2 * x * r2);
    let J01 = J10;
    let mut J11 = k1 * r2 + k2 * r4 + 6.0 * p1 * y + 2.0 * p2 * x;
    J11 += y * (2.0 * k1 * y + 4.0 * k2 * y * r2) + 1.0;
    // Above is generated using sympy

    MatrixXd::from_vec(2, 2, vec![J00, J01, J10, J11])
  }
}

/// Radial-tangential parameters jacobian
pub fn radtan4_params_jacobian(p: &Vector2d) -> MatrixXd {
  let x = p.x;
  let y = p.y;

  let x2 = x * x;
  let y2 = y * y;
  let xy = x * y;
  let r2 = x2 + y2;
  let r4 = r2 * r2;

  #[allow(non_snake_case)]
  {
    let J00 = x * r2;
    let J01 = x * r4;
    let J02 = 2.0 * xy;
    let J03 = 3.0 * x2 + y2;
    let J10 = y * r2;
    let J11 = y * r4;
    let J12 = x2 + 3.0 * y2;
    let J13 = 2.0 * xy;
    MatrixXd::from_vec(2, 4, vec![J00, J01, J02, J03, J10, J11, J12, J13])
  }
}

impl CameraModel for PinholeRadtan4 {
  fn project(res: &Resolution, intrinsic: &[f64], point3d: &Vector3d) -> bool {
    let fx = intrinsic[0];
    let fy = intrinsic[1];
    let cx = intrinsic[2];
    let cy = intrinsic[3];

    let k1 = intrinsic[4];
    let k2 = intrinsic[5];
    let p1 = intrinsic[6];
    let p2 = intrinsic[7];

    let px = point3d.x / point3d.z;
    let py = point3d.y / point3d.z;
    let p = Vector2d::new(px, py);
    let pd = radtan4_distort(k1, k2, p1, p2, &p);
    let zx = fx * pd.x + cx;
    let zy = fy * pd.y + cy;

    let x_ok = zx >= 0.0 && zx < (res.width as f64);
    let y_ok = zy >= 0.0 && zy < (res.height as f64);
    let z_ok = point3d.z > 0.0;

    x_ok && y_ok && z_ok
  }

  fn project_jacobian(intrinsic: &[f64], point3d: &Vector3d) -> MatrixXd {
    let k1 = intrinsic[4];
    let k2 = intrinsic[5];
    let p1 = intrinsic[6];
    let p2 = intrinsic[7];

    let px = point3d.x / point3d.z;
    let py = point3d.y / point3d.z;
    let p = Vector2d::new(px, py);

    let jac_k = pinhole_point_jacobian(intrinsic);
    let jac_d = radtan4_point_jacobian(k1, k2, p1, p2, &p);
    let jac_p = pinhole_project_jacobian(point3d);

    &jac_k * &jac_d * &jac_p
  }

  fn params_jacobian(intrinsic: &[f64], point3d: &Vector3d) -> MatrixXd {
    let k1 = intrinsic[4];
    let k2 = intrinsic[5];
    let p1 = intrinsic[6];
    let p2 = intrinsic[7];

    let px = point3d.x / point3d.z;
    let py = point3d.y / point3d.z;
    let p = Vector2d::new(px, py);
    let pd = radtan4_distort(k1, k2, p1, p2, &p);

    let jac_proj_params = pinhole_params_jacobian(&pd);
    let jac_proj_point = pinhole_point_jacobian(intrinsic);
    let jac_dist_params = &jac_proj_point * radtan4_params_jacobian(&p);

    let mut jac = MatrixXd::zeros(2, 8);
    jac.fixed_view_mut::<2, 4>(0, 0).copy_from(&jac_proj_params);
    jac.fixed_view_mut::<2, 4>(0, 4).copy_from(&jac_dist_params);
    jac
  }

  fn back_project(&self, intrinsic: &[f64], point: &Vector2d) -> Vector3d {
    let fx = intrinsic[0];
    let fy = intrinsic[1];
    let cx = intrinsic[2];
    let cy = intrinsic[3];

    let px = (point.x - cx) / fx;
    let py = (point.y - cy) / fy;
    let p = Vector2d::new(px, py);

    Vector3d::new(p.x, p.y, 1.0)
  }

  // fn undistort(&self, intrinsic: &[f64], z: &Vector2d) -> Vector2d {
  //   let fx = intrinsic[0];
  //   let fy = intrinsic[1];
  //   let cx = intrinsic[2];
  //   let cy = intrinsic[3];
  //
  //   let ray = self.back_project(intrinsic, z);
  //   let zx = fx * ray.x() + cx;
  //   let zy = fy * ray.y() + cy;
  //
  //   Vector2d::new([zx, zy])
  // }
}

///////////////////////////////////////////////////////////////////////////////
// PINHOLE-RADTAN5                                                           //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
struct PinholeRadtan5 {}

///////////////////////////////////////////////////////////////////////////////
// PINHOLE-EQUI4                                                             //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
struct PinholeEqui4 {}

///////////////////////////////////////////////////////////////////////////////
// OMNI-RADTAN4                                                              //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
struct OmniRadtan4 {}
