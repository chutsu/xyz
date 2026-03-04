use matrix::Matrix;

#[derive(Debug)]
pub struct Resolution {
  pub width: u32,
  pub height: u32,
}

#[derive(Debug)]
pub struct CameraIntrinsic {
  data: Matrix<f64>,
}

#[derive(Debug)]
pub struct SensorExtrinsic {
  data: Matrix<f64>,
}

#[derive(Debug)]
pub struct CameraGeometry {
  pub camera_id: u8,
  pub camera_model: String,
  pub resolution: Resolution,
  pub intrinsic: CameraIntrinsic,
  pub extrinsic: SensorExtrinsic,
}
