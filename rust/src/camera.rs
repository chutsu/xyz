use crate::linalg::Matrix;
use crate::linalg::Vector2d;
use crate::linalg::Vector3d;

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

trait CameraModel {
  fn project(
    &self,
    res: &Resolution,
    intrinsic: Vec<f64>,
    point: &Vector3d,
  ) -> bool;

  fn project_jacobian(
    &self,
    intrinsic: Vec<f64>,
    point: &Vector3d,
  ) -> Matrix<f64>;

  fn params_jacobian(
    &self,
    intrinsic: Vec<f64>,
    point: &Vector3d,
  ) -> Matrix<f64>;

  fn back_project(&self, intrinsic: Vec<f64>, point: &Vector2d) -> Vector3d;

  fn undistort(&self, intrinsic: Vec<f64>, z: &Vector2d) -> Vector2d;
}

///////////////////////////////////////////////////////////////////////////////
// PINHOLE                                                                   //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
struct Pinhole {}

impl CameraModel for Pinhole {
  fn project(
    &self,
    res: &Resolution,
    intrinsic: Vec<f64>,
    point3d: &Vector3d,
  ) -> bool {
    let fx = intrinsic[0];
    let fy = intrinsic[1];
    let cx = intrinsic[2];
    let cy = intrinsic[3];

    let px = point3d.x() / point3d.z();
    let py = point3d.y() / point3d.z();
    let zx = fx * px + cx;
    let zy = fy * py + cy;

    let x_ok = zx >= 0.0 && zx < (res.width as f64);
    let y_ok = zy >= 0.0 && zy < (res.height as f64);
    let z_ok = point3d.z() > 0.0;

    x_ok && y_ok && z_ok
  }

  fn project_jacobian(
    &self,
    intrinsic: Vec<f64>,
    point: &Vector3d,
  ) -> Matrix<f64> {
    let fx = intrinsic[0];
    let fy = intrinsic[1];
    let x = point.x();
    let y = point.y();
    let z = point.z();

    // Point jacobian Jk
    #[rustfmt::skip]
    let data = vec![
       fx, 0.0,
      0.0,  fy,
    ];
    let jac_k = Matrix::new(2, 2, data);

    // Project jacobian Jp
    #[rustfmt::skip]
    let data = vec![
      1.0 / z,     0.0, -x / (z * z),
          0.0, 1.0 / z, -y / (z * z),
    ];
    let jac_p = Matrix::new(2, 3, data);

    &jac_k * &jac_p
  }

  fn params_jacobian(&self, _: Vec<f64>, point: &Vector3d) -> Matrix<f64> {
    let x = point.x();
    let y = point.y();

    #[rustfmt::skip]
    let data = vec![
        x, 0.0, 1.0, 0.0,
      0.0,   y, 0.0, 1.0,
    ];
    Matrix::new(2, 3, data)
  }

  fn back_project(&self, intrinsic: Vec<f64>, point: &Vector2d) -> Vector3d {
    let fx = intrinsic[0];
    let fy = intrinsic[1];
    let cx = intrinsic[2];
    let cy = intrinsic[3];
    let rx = (point.x() - cx) / fx;
    let ry = (point.y() - cy) / fy;

    Vector3d::new([rx, ry, 1.0])
  }

  fn undistort(&self, intrinsic: Vec<f64>, z: &Vector2d) -> Vector2d {
    let fx = intrinsic[0];
    let fy = intrinsic[1];
    let cx = intrinsic[2];
    let cy = intrinsic[3];

    let ray = self.back_project(intrinsic, z);
    let zx = fx * ray.x() + cx;
    let zy = fy * ray.y() + cy;

    Vector2d::new([zx, zy])
  }
}

///////////////////////////////////////////////////////////////////////////////
// PINHOLE-RADTAN4                                                           //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
struct PinholeRadtan4 {}

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

///////////////////////////////////////////////////////////////////////////////
// CAMERA GEOMETRY                                                           //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct CameraGeometry {
  pub camera_id: u8,
  pub camera_model: String,
  pub resolution: Resolution,
  pub intrinsic: Vec<f64>,
  pub extrinsic: Vec<f64>,
}
