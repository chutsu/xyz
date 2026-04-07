use std::f64::consts::PI;

use rand::Rng;

use nalgebra::Matrix3;
use nalgebra::Matrix4;
use nalgebra::Vector2;
use nalgebra::Vector3;

use crate::geometry::euler321;

pub type Vector2d = Vector2<f64>;
pub type Vector3d = Vector3<f64>;
pub type Matrix3d = Matrix3<f64>;
pub type Matrix4d = Matrix4<f64>;

// use rerun::RecordingStream;
// use rerun::demo_util::grid;

///////////////////////////////////////////////////////////////////////////////
// SimFeatures                                                               //
///////////////////////////////////////////////////////////////////////////////

/// Create `num_features` 3-D points sampled uniformly within the given bounds.
pub fn sim_create_3d_features(
  x_bounds: (f64, f64),
  y_bounds: (f64, f64),
  z_bounds: (f64, f64),
  num_features: usize,
) -> Vec<Vector3d> {
  let mut rng = rand::thread_rng();
  (0..num_features)
    .map(|_| {
      Vector3d::new(
        rng.gen_range(x_bounds.0..=x_bounds.1),
        rng.gen_range(y_bounds.0..=y_bounds.1),
        rng.gen_range(z_bounds.0..=z_bounds.1),
      )
    })
    .collect()
}

/// Create `num_features` 3-D points distributed along the four sides of a
/// rectangular perimeter centred at `origin` with half-extents `dim`.
pub fn sim_create_3d_features_perimeter(
  origin: Vector3d,
  dim: Vector3d,
  num_features: usize,
) -> Vec<Vector3d> {
  assert!(num_features > 0, "num_features must be > 0");
  let (w, l, h) = (dim.x, dim.y, dim.z);
  let num_features_per_side = num_features / 4;

  // East side  (y = origin.y + l)
  let east = sim_create_3d_features(
    (origin.x - w, origin.x + w),
    (origin.y + l, origin.y + l),
    (origin.z - h, origin.z + h),
    num_features_per_side,
  );

  // North side (x = origin.x + w)
  let north = sim_create_3d_features(
    (origin.x + w, origin.x + w),
    (origin.y - l, origin.y + l),
    (origin.z - h, origin.z + h),
    num_features_per_side,
  );

  // West side  (y = origin.y - l)
  let west = sim_create_3d_features(
    (origin.x - w, origin.x + w),
    (origin.y - l, origin.y - l),
    (origin.z - h, origin.z + h),
    num_features_per_side,
  );

  // South side (x = origin.x - w)
  let south = sim_create_3d_features(
    (origin.x - w, origin.x - w),
    (origin.y - l, origin.y + l),
    (origin.z - h, origin.z + h),
    num_features_per_side,
  );

  [east, north, west, south].concat()
}

///////////////////////////////////////////////////////////////////////////////
// SimCamera                                                                 //
///////////////////////////////////////////////////////////////////////////////

struct SimCamera {
  pub camera_id: usize,
  pub resolution: (u32, u32),

  pub fx: f64,
  pub fy: f64,
  pub cx: f64,
  pub cy: f64,
}

impl SimCamera {
  #[allow(non_snake_case)]
  pub fn project(&self, p_C: &Vector3d) -> (bool, Vector2d) {
    if p_C.z <= 0.0 {
      return (false, Vector2d::zeros());
    }

    let u = self.fx * (p_C.x / p_C.z) + self.cx;
    let v = self.fy * (p_C.y / p_C.z) + self.cy;

    let (w, h) = self.resolution;
    let x_ok = u >= 0.0 && u < w as f64;
    let y_ok = v >= 0.0 && v < h as f64;
    let in_bounds = x_ok && y_ok;

    (in_bounds, Vector2d::new(u, v))
  }
}

///////////////////////////////////////////////////////////////////////////////
// SimCameraFrame                                                            //
///////////////////////////////////////////////////////////////////////////////

struct SimCameraFrame {
  pub ts: u64,
  pub camera_id: usize,
  pub pose: Matrix4d,
  pub camera: SimCamera,
  pub feature_ids: Vec<usize>,
  pub keypoints: Vec<Vector2d>,
}

impl SimCameraFrame {
  pub fn new(
    ts: u64,
    camera_id: usize,
    pose: Matrix4d,
    camera: SimCamera,
    features: Vec<Vector3d>,
  ) -> Self {
    let mut feature_ids: Vec<usize> = vec![];
    let mut keypoints: Vec<Vector2d> = vec![];

    #[allow(non_snake_case)]
    for (i, feature) in features.iter().enumerate() {
      let p_C = (pose * feature.to_homogeneous()).xyz();
      let (status, z) = camera.project(&p_C);
      if status {
        feature_ids.push(i);
        keypoints.push(z);
      }
    }

    Self {
      ts,
      camera_id,
      pose,
      camera,
      feature_ids,
      keypoints,
    }
  }

  pub fn num_measurements(&self) -> usize {
    self.feature_ids.len()
  }
}

///////////////////////////////////////////////////////////////////////////////
// SimCameraData                                                             //
///////////////////////////////////////////////////////////////////////////////

struct SimCameraData {
  pub camera_id: usize,
  pub camera: SimCamera,
  pub timestamps: Vec<u64>,
  pub frames: Vec<SimCameraFrame>,
}

///////////////////////////////////////////////////////////////////////////////
// SimImuData                                                                //
///////////////////////////////////////////////////////////////////////////////

struct SimImuData {
  pub imu_id: usize,
  pub camera: SimCamera,
  pub timestamps: Vec<u64>,
  pub poses: Vec<Matrix4d>,
  pub vels: Vec<Vector3d>,
  pub accs: Vec<Vector3d>,
  pub gyrs: Vec<Vector3d>,
}

///////////////////////////////////////////////////////////////////////////////
// SimCircle                                                                 //
///////////////////////////////////////////////////////////////////////////////

struct SimCircle {
  pub circle_r: f64,
  pub circle_v: f64,
  pub circle_dist: f64,
  pub traj_duration: f64,
  pub w: f64,
  pub theta_init: f64,
  pub yaw_init: f64,
}

impl SimCircle {
  pub fn default() -> Self {
    let circle_r = 5.0;
    let circle_v = 1.0;
    let circle_dist = 2.0 * PI * circle_r;
    let traj_duration = circle_dist / circle_v;
    let w = -2.0 * PI * (1.0 / traj_duration);
    let theta_init = PI;
    let yaw_init = PI / 2.0;

    Self {
      circle_r,
      circle_v,
      circle_dist,
      traj_duration,
      w,
      theta_init,
      yaw_init,
    }
  }

  pub fn get_position(&self, ts: u64) -> Vector3d {
    let ts_s = (ts as f64) * 1e-9;
    let theta = self.theta_init + (ts_s / self.traj_duration) * PI;

    let x = self.circle_r * f64::cos(theta);
    let y = self.circle_r * f64::sin(theta);
    let z = 0.0;
    Vector3d::new(x, y, z)
  }

  pub fn get_rotation(&self, ts: u64) -> Matrix3d {
    let ts_s = (ts as f64) * 1e-9;
    let yaw = self.yaw_init + (ts_s / self.traj_duration) * PI;
    euler321(0.0, 0.0, yaw)
  }

  pub fn get_pose(&self, ts: u64) -> Matrix4d {
    let pos = self.get_position(ts);
    let rot = self.get_rotation(ts);

    let mut tf = Matrix4::<f64>::identity();
    tf.fixed_view_mut::<3, 3>(0, 0).copy_from(&rot);
    tf.fixed_view_mut::<3, 1>(0, 3).copy_from(&pos);
    tf
  }

  pub fn get_velocity(&self, ts: u64) -> Vector3d {
    let ts_s = (ts as f64) * 1e-9;
    let theta = self.theta_init + (ts_s / self.traj_duration) * PI;

    let vx = -self.circle_r * self.w * f64::sin(theta);
    let vy = self.circle_r * self.w * f64::cos(theta);
    let vz = 0.0;

    Vector3::new(vx, vy, vz)
  }

  pub fn get_acceleration(&self, ts: u64) -> Vector3d {
    let ts_s = (ts as f64) * 1e-9;
    let theta = self.theta_init + (ts_s / self.traj_duration) * PI;

    let ax = -self.circle_r * self.w * self.w * f64::cos(theta);
    let ay = -self.circle_r * self.w * self.w * f64::sin(theta);
    let az = 0.0;

    Vector3::new(ax, ay, az)
  }
}

// #[cfg(test)]
// mod sim_tests {
//   // Note this useful idiom: importing names from outer (for mod tests) scope.
//   // use super::*;
//
//   #[test]
//   #[allow(non_snake_case)]
//   fn test_sim() -> Result<(), Box<dyn std::error::Error>> {
//     // Connect to rerun viewer
//     let rec = rerun::RecordingStreamBuilder::new("test_sim").connect_grpc()?;
//
//     // Log a 3D point
//     rec.log("world/point", &rerun::Points3D::new([(1.0, 2.0, 3.0)]))?;
//
//     // Log multiple points
//     let points: Vec<[f32; 3]> =
//       vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
//     rec.log("world/points", &rerun::Points3D::new(points))?;
//
//     Ok(())
//   }
// }
