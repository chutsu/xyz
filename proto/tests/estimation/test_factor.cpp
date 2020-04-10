#include "proto/munit.hpp"
#include "proto/estimation/factor.hpp"

namespace proto {

#if PRECISION == SINGLE
  real_t step_size = 1e-3;
  real_t threshold = 0.5;
#else
  real_t step_size = 1e-6;
  real_t threshold = 1e-4;
#endif

void save_data(const std::string &save_path,
               const timestamps_t &ts,
               const vec3s_t &y) {
  std::ofstream file{save_path};
  if (file.good() != true) {
    printf("Failed to open file for output!");
    exit(-1);
  }

  for (size_t i = 0; i < ts.size(); i++) {
    file << ts[i] << ",";
    file << y[i](0) << ",";
    file << y[i](1) << ",";
    file << y[i](2) << std::endl;
  }

  file.close();
}

void save_data(const std::string &save_path,
               const timestamps_t &ts,
               const quats_t &y) {
  std::ofstream file{save_path};
  if (file.good() != true) {
    printf("Failed to open file for output!");
    exit(-1);
  }

  for (size_t i = 0; i < ts.size(); i++) {
    file << ts[i] << ",";
    file << y[i].w() << ",";
    file << y[i].x() << ",";
    file << y[i].y() << ",";
    file << y[i].z() << std::endl;
  }

  file.close();
}

template <typename CAMERA>
static int check_J_h(
    const int img_w,
    const int img_h,
    const real_t *proj_params,
    const real_t *dist_params) {
  // Calculate baseline
  const vec2_t z{0.0, 0.0};
  const CAMERA cam{img_w, img_h, proj_params, dist_params};
  const vec3_t p_C{1.0, 2.0, 10.0};
  vec2_t z_hat;
  mat_t<2, 3> J_h;
  cam.project(p_C, z_hat, J_h);
  const vec2_t e = z - z_hat;

  // Perturb camera parameters
  mat_t<2, 3> fdiff = zeros(2, 3);
  for (int i = 0; i < 3; i++) {
    vec3_t p_C_diff = p_C;
    p_C_diff(i) += step_size;
    cam.project(p_C_diff, z_hat, J_h);
    auto e_prime = z - z_hat;

    // Forward finite difference
    fdiff.block(0, i, 2, 1) = (e_prime - e) / step_size;
  }

  return check_jacobian("J_h", fdiff, -1 * J_h, threshold, true);
}

template <typename CAMERA>
static int check_J_proj_params(
    const int img_w,
    const int img_h,
    const mat4_t &T_WC,
    const vec3_t &p_W,
    const real_t *proj_params,
    const real_t *dist_params,
    const matx_t &J) {
  const vec2_t z{0.0, 0.0};
  const CAMERA cam{img_w, img_h, proj_params, dist_params};
  vec2_t z_hat;
  cam.project(tf_point(T_WC.inverse(), p_W), z_hat);
  const vec2_t e = z - z_hat;

  // Perturb camera parameters
  matx_t fdiff = zeros(2, 4);
  for (int i = 0; i < 4; i++) {
    real_t params_fd[4] = {
      proj_params[0],
      proj_params[1],
      proj_params[2],
      proj_params[3]
    };
    params_fd[i] += step_size;

    const CAMERA cam{img_w, img_h, params_fd, dist_params};
    vec2_t z_hat;
    cam.project(tf_point(T_WC.inverse(), p_W), z_hat);
    auto e_prime = z - z_hat;

    // Forward finite difference
    fdiff.block(0, i, 2, 1) = (e_prime - e) / step_size;
  }

  return check_jacobian("J_proj_params", fdiff, J, threshold, true);
}

template <typename CAMERA>
static int check_J_dist_params(
    const int img_w,
    const int img_h,
    const mat4_t &T_WC,
    const vec3_t &p_W,
    const real_t *proj_params,
    const real_t *dist_params,
    const matx_t &J) {
  const vec2_t z{0.0, 0.0};
  vec2_t z_hat;
  const CAMERA cam{img_w, img_h, proj_params, dist_params};
  cam.project(tf_point(T_WC.inverse(), p_W), z_hat);
  const vec2_t e = z - z_hat;

  // Perturb camera parameters
  matx_t fdiff = zeros(2, 4);
  for (int i = 0; i < 4; i++) {
    real_t params_fd[4] = {
      dist_params[0],
      dist_params[1],
      dist_params[2],
      dist_params[3]
    };
    params_fd[i] += step_size;

    vec2_t z_hat;
    const CAMERA cam{img_w, img_h, proj_params, params_fd};
    cam.project(tf_point(T_WC.inverse(), p_W), z_hat);
    auto e_prime = z - z_hat;

    // Forward finite difference
    fdiff.block(0, i, 2, 1) = (e_prime - e) / step_size;
  }

  return check_jacobian("J_dist_params", fdiff, J, threshold, true);
}

template <typename CAMERA>
static int check_ba_factor_J_cam_pose(
    const int img_w,
    const int img_h,
    const mat4_t &T_WC,
    const vec3_t &p_W,
    const real_t *proj_params,
    const real_t *dist_params,
    const matx_t &J) {
  // Calculate baseline
  const vec2_t z{0.0, 0.0};
  vec2_t z_hat;
  const CAMERA cam{img_w, img_h, proj_params, dist_params};
  cam.project(tf_point(T_WC.inverse(), p_W), z_hat);
  const vec2_t e = z - z_hat;

  // Perturb rotation
  matx_t fdiff = zeros(2, 6);
  for (int i = 0; i < 3; i++) {
    auto T_WC_diff = tf_perturb_rot(T_WC, step_size, i);
    vec2_t z_hat;
    cam.project(tf_point(T_WC_diff.inverse(), p_W), z_hat);
    auto e_prime = z - z_hat;

    // Forward finite difference
    fdiff.block(0, i, 2, 1) = (e_prime - e) / step_size;
  }

  // Perturb translation
  for (int i = 0; i < 3; i++) {
    auto T_WC_diff = tf_perturb_trans(T_WC, step_size, i);
    vec2_t z_hat;
    cam.project(tf_point(T_WC_diff.inverse(), p_W), z_hat);
    auto e_prime = z - z_hat;

    // Forward finite difference
    fdiff.block(0, i + 3, 2, 1) = (e_prime - e) / step_size;
  }

  return check_jacobian("J_sensor_pose", fdiff, J, threshold, true);
}

template <typename CAMERA>
static int check_ba_factor_J_landmark(
    const int img_w,
    const int img_h,
    const mat4_t &T_WC,
    const vec3_t &p_W,
    const real_t *proj_params,
    const real_t *dist_params,
    const matx_t &J) {
  const vec2_t z{0.0, 0.0};
  vec2_t z_hat;
  const CAMERA cam{img_w, img_h, proj_params, dist_params};
  cam.project(tf_point(T_WC.inverse(), p_W), z_hat);
  const vec2_t e = z - z_hat;

  // Perturb landmark
  matx_t fdiff = zeros(2, 3);
  mat3_t dr = I(3) * step_size;
  for (int i = 0; i < 3; i++) {
    auto p_W_diff = p_W + dr.col(i);
    vec2_t z_hat;
    cam.project(tf_point(T_WC.inverse(), p_W_diff), z_hat);
    auto e_prime = z - z_hat;

    // Forward finite difference
    fdiff.block(0, i, 2, 1) = (e_prime - e) / step_size;
  }

  return check_jacobian("J_landmark", fdiff, J, threshold, true);
}

int test_ba_factor_jacobians() {
  // Setup parameters
  // -- Camera
  const int img_w = 640;
  const int img_h = 480;
  const vec3_t euler{-90.0, 0.0, -90.0};
  const mat3_t C_WC = euler321(deg2rad(euler));
  const vec3_t r_WC{0.0, 0.0, 0.0};
  const mat4_t T_WC = tf(C_WC, r_WC);
  pose_t cam_pose{T_WC};
  // -- Landmark
  const vec3_t p_W{10.0, 0.0, 0.0};
  landmark_t landmark{p_W};
  // -- Camera intrinsics
  real_t proj_params[4] = {600.0, 600.0, 325.0, 240.0};
  real_t dist_params[4] = {0.15, -0.3, 0.0001, 0.001};

  // Create factor
  const timestamp_t ts = 0;
  const size_t id = 0;
  const int cam_index = 0;
  const vec2_t z{0.0, 0.0};
  ba_factor_t<pinhole_radtan4_t> factor{id, ts, cam_index, img_w, img_h, z};

  // Evaluate factor
  real_t *params[4] {
    cam_pose.data(),
    landmark.data(),
    proj_params,
    dist_params
  };
  factor.eval(params);

  // Check ba factor parameter jacobians
  int retval = 0;
  // -- Check measurement model jacobian
  retval = check_J_h<pinhole_radtan4_t>(
    img_w, img_h, proj_params, dist_params);
  MU_CHECK(retval == 0);
  // -- Check camera pose jacobian
  const mat_t<2, 6> J0 = factor.jacobians[0];
  retval = check_ba_factor_J_cam_pose<pinhole_radtan4_t>(
    img_w, img_h, T_WC, p_W, proj_params, dist_params, J0);
  MU_CHECK(retval == 0);
  // -- Check landmark jacobian
  const mat_t<2, 3> J1 = factor.jacobians[1];
  retval = check_ba_factor_J_landmark<pinhole_radtan4_t>(
    img_w, img_h, T_WC, p_W, proj_params, dist_params, J1);
  MU_CHECK(retval == 0);
  // -- Check cam params jacobian
  const mat_t<2, 4> J2 = factor.jacobians[2];
  retval = check_J_proj_params<pinhole_radtan4_t>(
    img_w, img_h, T_WC, p_W, proj_params, dist_params, J2);
  MU_CHECK(retval == 0);
  // -- Check dist params jacobian
  const mat_t<2, 4> J3 = factor.jacobians[3];
  retval = check_J_dist_params<pinhole_radtan4_t>(
    img_w, img_h, T_WC, p_W, proj_params, dist_params, J3);
  MU_CHECK(retval == 0);

  return 0;
}

template <typename CAMERA>
static int check_cam_factor_J_sensor_pose(
    const int img_w,
    const int img_h,
    const mat4_t &T_WS,
    const mat4_t &T_SC,
    const vec3_t &p_W,
    const real_t *proj_params,
    const real_t *dist_params,
    const matx_t &J) {
  // Calculate baseline
  const vec2_t z{0.0, 0.0};
  vec2_t z_hat;
  const CAMERA cam{img_w, img_h, proj_params, dist_params};
  cam.project(tf_point((T_WS * T_SC).inverse(), p_W), z_hat);
  const vec2_t e = z - z_hat;

  // Perturb rotation
  matx_t fdiff = zeros(2, 6);
  for (int i = 0; i < 3; i++) {
    auto T_WS_diff = tf_perturb_rot(T_WS, step_size, i);
    vec2_t z_hat;
    const CAMERA cam{img_w, img_h, proj_params, dist_params};
    cam.project(tf_point((T_WS_diff * T_SC).inverse(), p_W), z_hat);
    auto e_prime = z - z_hat;

    // Forward finite difference
    fdiff.block(0, i, 2, 1) = (e_prime - e) / step_size;
  }

  // Perturb translation
  for (int i = 0; i < 3; i++) {
    auto T_WS_diff = tf_perturb_trans(T_WS, step_size, i);
    vec2_t z_hat;
    const CAMERA cam{img_w, img_h, proj_params, dist_params};
    cam.project(tf_point((T_WS_diff * T_SC).inverse(), p_W), z_hat);
    auto e_prime = z - z_hat;

    // Forward finite difference
    fdiff.block(0, i + 3, 2, 1) = (e_prime - e) / step_size;
  }

  return check_jacobian("J_sensor_pose", fdiff, J, threshold, true);
}

template <typename CAMERA>
static int check_cam_factor_J_sensor_camera_pose(
    const int img_w,
    const int img_h,
    const mat4_t &T_WS,
    const mat4_t &T_SC,
    const vec3_t &p_W,
    const real_t *proj_params,
    const real_t *dist_params,
    const matx_t &J) {
  // Calculate baseline
  const vec2_t z{0.0, 0.0};
  vec2_t z_hat;
  const CAMERA cam{img_w, img_h, proj_params, dist_params};
  cam.project(tf_point((T_WS * T_SC).inverse(), p_W), z_hat);
  const vec2_t e = z - z_hat;

  // Perturb rotation
  matx_t fdiff = zeros(2, 6);
  for (int i = 0; i < 3; i++) {
    auto T_SC_diff = tf_perturb_rot(T_SC, step_size, i);
    vec2_t z_hat;
    const CAMERA cam{img_w, img_h, proj_params, dist_params};
    cam.project(tf_point((T_WS * T_SC_diff).inverse(), p_W), z_hat);
    auto e_prime = z - z_hat;
    fdiff.block(0, i, 2, 1) = (e_prime - e) / step_size;
  }

  // Perturb translation
  for (int i = 0; i < 3; i++) {
    auto T_SC_diff = tf_perturb_trans(T_SC, step_size, i);
    vec2_t z_hat;
    const CAMERA cam{img_w, img_h, proj_params, dist_params};
    cam.project(tf_point((T_WS * T_SC_diff).inverse(), p_W), z_hat);
    auto e_prime = z - z_hat;

    // Forward finite difference
    fdiff.block(0, i + 3, 2, 1) = (e_prime - e) / step_size;
  }

  return check_jacobian("J_sensor_camera_pose", fdiff, J, threshold, true);
}

template <typename CAMERA>
static int check_cam_factor_J_landmark(
    const int img_w,
    const int img_h,
    const mat4_t &T_WS,
    const mat4_t &T_SC,
    const vec3_t &p_W,
    const real_t *proj_params,
    const real_t *dist_params,
    const matx_t &J) {
  const vec2_t z{0.0, 0.0};
  vec2_t z_hat;
  const CAMERA cam{img_w, img_h, proj_params, dist_params};
  cam.project(tf_point((T_WS * T_SC).inverse(), p_W), z_hat);
  const vec2_t e = z - z_hat;

  // Perturb landmark
  matx_t fdiff = zeros(2, 3);
  mat3_t dr = I(3) * step_size;
  for (int i = 0; i < 3; i++) {
    auto p_W_diff = p_W + dr.col(i);
    vec2_t z_hat;
    const CAMERA cam{img_w, img_h, proj_params, dist_params};
    cam.project(tf_point((T_WS * T_SC).inverse(), p_W_diff), z_hat);
    auto e_prime = z - z_hat;

    // Forward finite difference
    fdiff.block(0, i, 2, 1) = (e_prime - e) / step_size;
  }

  return check_jacobian("J_landmark", fdiff, J, threshold, true);
}

int test_cam_factor_jacobians() {
  // Setup parameters
  // clang-format off
  const int img_w = 640;
  const int img_h = 480;

  mat4_t T_WS;
  T_WS << 0.0, 0.0, 1.0, 0.0,
          0.0, -1.0, 0.0, 0.0,
          1.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 1.0;
  pose_t sensor_pose{T_WS};

  mat4_t T_SC;
  T_SC << 0.0, -1.0, 0.0, -0.02,
          1.0, 0.0,  0.0, -0.06,
          0.0, 0.0, 1.0, 0.00,
          0.0, 0.0, 0.0, 1.0;
  pose_t imu_cam_extrinsics{T_SC};

  vec3_t p_W{10.0, 0.0, 0.0};
  landmark_t landmark{p_W};

  real_t proj_params[4] = {600.0, 600.0, 325.0, 240.0};
  real_t dist_params[4] = {0.15, -0.3, 0.0001, 0.001};
  // clang-format on

  // Create factor
  const timestamp_t ts = 0;
  const size_t id = 0;
  const int cam_index = 0;
  const vec2_t z{0.0, 0.0};
  cam_factor_t<pinhole_radtan4_t> factor{id, ts, cam_index, img_w, img_h, z};

  // Evaluate factor
  real_t *params[5] {
    sensor_pose.data(),
    imu_cam_extrinsics.data(),
    landmark.data(),
    proj_params,
    dist_params
  };
  factor.eval(params);

  // Check camera factor parameter jacobians
  int retval = 0;
  // -- Check measurement model jacobian
  retval = check_J_h<pinhole_radtan4_t>(
    img_w, img_h, proj_params, dist_params);
  MU_CHECK(retval == 0);
  // -- Check sensor pose jacobian
  const mat_t<2, 6> J0 = factor.jacobians[0];
  retval = check_cam_factor_J_sensor_pose<pinhole_radtan4_t>(
    img_w, img_h, T_WS, T_SC, p_W, proj_params, dist_params, J0);
  MU_CHECK(retval == 0);
  // -- Check sensor camera pose jacobian
  const mat_t<2, 6> J1 = factor.jacobians[1];
  retval = check_cam_factor_J_sensor_camera_pose<pinhole_radtan4_t>(
    img_w, img_h, T_WS, T_SC, p_W, proj_params, dist_params, J1);
  MU_CHECK(retval == 0);
  // -- Check landmark jacobian
  const mat_t<2, 3> J2 = factor.jacobians[2];
  retval = check_cam_factor_J_landmark<pinhole_radtan4_t>(
    img_w, img_h, T_WS, T_SC, p_W, proj_params, dist_params, J2);
  MU_CHECK(retval == 0);
  // -- Check cam params jacobian
  const mat_t<2, 4> J3 = factor.jacobians[3];
  retval = check_J_proj_params<pinhole_radtan4_t>(
    img_w, img_h, T_WS * T_SC, p_W, proj_params, dist_params, J3);
  MU_CHECK(retval == 0);
  // -- Check dist params jacobian
  const mat_t<2, 4> J4 = factor.jacobians[4];
  retval = check_J_dist_params<pinhole_radtan4_t>(
    img_w, img_h, T_WS * T_SC, p_W, proj_params, dist_params, J4);
  MU_CHECK(retval == 0);

  return 0;
}

static int check_imu_factor_J_sensor_pose_i(
    imu_factor_t &factor,
    mat4_t &T_WS_i, vec_t<9> &sb_i,
    mat4_t &T_WS_j, vec_t<9> &sb_j,
    matx_t &J) {
  imu_factor_t imu_factor = factor;
  pose_t sensor_pose_i{T_WS_i};
  pose_t sensor_pose_j{T_WS_j};
  real_t *params[4] = {
    sensor_pose_i.data(),
    sb_i.data(),
    sensor_pose_j.data(),
    sb_j.data()
  };
  factor.eval(params);
  const auto e = factor.residuals;

  // Perturb rotation
  matx_t fdiff = zeros(15, 6);
  for (int i = 0; i < 3; i++) {
    auto T_WS_i_diff = tf_perturb_rot(T_WS_i, step_size, i);
    pose_t sensor_pose_i{T_WS_i_diff};
    real_t *params[4] = {
      sensor_pose_i.data(),
      sb_i.data(),
      sensor_pose_j.data(),
      sb_j.data()
    };
    factor.eval(params);
    const auto e_prime = factor.residuals;

    fdiff.block(0, i, 15, 1) = (e_prime - e) / step_size;
  }

  // Perturb translation
  for (int i = 0; i < 3; i++) {
    auto T_WS_i_diff = tf_perturb_trans(T_WS_i, step_size, i);
    pose_t sensor_pose_i{T_WS_i_diff};
    real_t *params[4] = {
      sensor_pose_i.data(),
      sb_i.data(),
      sensor_pose_j.data(),
      sb_j.data()
    };
    factor.eval(params);
    const auto e_prime = factor.residuals;

    fdiff.block(0, i + 3, 15, 1) = (e_prime - e) / step_size;
  }

  return check_jacobian("J_sensor_pose_i", fdiff, J, threshold, true);
}

static int check_imu_factor_J_speed_bias_i(
    imu_factor_t &factor,
    mat4_t &T_WS_i, vec_t<9> &sb_i,
    mat4_t &T_WS_j, vec_t<9> &sb_j,
    matx_t &J) {
  imu_factor_t imu_factor = factor;
  pose_t sensor_pose_i{T_WS_i};
  pose_t sensor_pose_j{T_WS_j};
  real_t *params[4] = {
    sensor_pose_i.data(),
    sb_i.data(),
    sensor_pose_j.data(),
    sb_j.data()
  };
  factor.eval(params);
  const auto e = factor.residuals;

  // Perturb
  matx_t fdiff = zeros(15, 9);
  for (int i = 0; i < 9; i++) {
    auto sb_i_diff = sb_i;
    sb_i_diff(i) += step_size;
    real_t *params[4] = {
      sensor_pose_i.data(),
      sb_i_diff.data(),
      sensor_pose_j.data(),
      sb_j.data()
    };
    factor.eval(params);
    const auto e_prime = factor.residuals;

    fdiff.block(0, i, 15, 1) = (e_prime - e) / step_size;
  }

  return check_jacobian("J_speed_bias_i", fdiff, J, threshold, true);
}

static int check_imu_factor_J_sensor_pose_j(
    imu_factor_t &factor,
    mat4_t &T_WS_i, vec_t<9> &sb_i,
    mat4_t &T_WS_j, vec_t<9> &sb_j,
    matx_t &J) {
  imu_factor_t imu_factor = factor;
  pose_t sensor_pose_i{T_WS_i};
  pose_t sensor_pose_j{T_WS_j};
  real_t *params[4] = {
    sensor_pose_i.data(),
    sb_i.data(),
    sensor_pose_j.data(),
    sb_j.data()
  };
  factor.eval(params);
  const auto e = factor.residuals;

  // Perturb rotation
  matx_t fdiff = zeros(15, 6);
  for (int i = 0; i < 3; i++) {
    auto T_WS_j_diff = tf_perturb_rot(T_WS_j, step_size, i);
    pose_t sensor_pose_j{T_WS_j_diff};
    real_t *params[4] = {
      sensor_pose_i.data(),
      sb_i.data(),
      sensor_pose_j.data(),
      sb_j.data()
    };
    factor.eval(params);
    const auto e_prime = factor.residuals;

    fdiff.block(0, i, 15, 1) = (e_prime - e) / step_size;
  }

  // Perturb translation
  for (int i = 0; i < 3; i++) {
    auto T_WS_j_diff = tf_perturb_trans(T_WS_j, step_size, i);
    pose_t sensor_pose_j{T_WS_j_diff};
    real_t *params[4] = {
      sensor_pose_i.data(),
      sb_i.data(),
      sensor_pose_j.data(),
      sb_j.data()
    };
    factor.eval(params);
    const auto e_prime = factor.residuals;

    fdiff.block(0, i + 3, 15, 1) = (e_prime - e) / step_size;
  }

  return check_jacobian("J_sensor_pose_j", fdiff, J, threshold, true);
}

static int check_imu_factor_J_speed_bias_j(
    imu_factor_t &factor,
    mat4_t &T_WS_i, vec_t<9> &sb_i,
    mat4_t &T_WS_j, vec_t<9> &sb_j,
    matx_t &J) {
  imu_factor_t imu_factor = factor;
  pose_t sensor_pose_i{T_WS_i};
  pose_t sensor_pose_j{T_WS_j};
  real_t *params[4] = {
    sensor_pose_i.data(),
    sb_i.data(),
    sensor_pose_j.data(),
    sb_j.data()
  };
  factor.eval(params);
  const auto e = factor.residuals;

  // Perturb
  matx_t fdiff = zeros(15, 9);
  for (int i = 0; i < 9; i++) {
    auto sb_j_diff = sb_j;
    sb_j_diff(i) += step_size;
    real_t *params[4] = {
      sensor_pose_i.data(),
      sb_i.data(),
      sensor_pose_j.data(),
      sb_j_diff.data()
    };
    factor.eval(params);
    const auto e_prime = factor.residuals;

    fdiff.block(0, i, 15, 1) = (e_prime - e) / step_size;
  }

  return check_jacobian("J_speed_bias_i", fdiff, J, threshold, true);
}

int test_imu_factor_jacobians() {
  // Generate trajectory
  timestamps_t timestamps;
  vec3s_t positions;
  quats_t orientations;
  for (int i = 0; i <= 5; i++) {
    timestamps.push_back(i * 1e8);
    positions.emplace_back(i, i, 0.0);
    orientations.emplace_back(1.0, 0.0, 0.0, 0.0);
  }
  ctraj_t ctraj(timestamps, positions, orientations);
  save_data("/tmp/pos_data.csv", timestamps, positions);
  save_data("/tmp/att_data.csv", timestamps, orientations);

  // Setup imu sim
  sim_imu_t imu;
  imu.rate = 400;
  imu.tau_a = 3600;
  imu.tau_g = 3600;
  imu.sigma_g_c = 0.00275;
  imu.sigma_a_c = 0.0250;
  imu.sigma_gw_c = 1.65e-05;
  imu.sigma_aw_c = 0.000441;
  imu.g = 9.81007;

  // Simulate IMU measurements
  std::default_random_engine rndeng;
  timestamps_t imu_ts;
  vec3s_t imu_accel;
  vec3s_t imu_gyro;
  vec3s_t pos_prop;
  vec3s_t vel_prop;
  quats_t att_prop;

  timestamp_t ts_k = 0;
  const timestamp_t ts_end = timestamps.back();
  const timestamp_t dt = (1 / imu.rate) * 1e9;

  // -- Initialize position, velocity and attidue
  auto T_WS = ctraj_get_pose(ctraj, 0.0);
  vec3_t r_WS = tf_trans(T_WS);
  mat3_t C_WS = tf_rot(T_WS);
  vec3_t v_WS = ctraj_get_velocity(ctraj, 0.0);

  // -- Simulate imu measurements
  while (ts_k <= ts_end) {
    const auto T_WS_W = ctraj_get_pose(ctraj, ts_k);
    const auto w_WS_W = ctraj_get_angular_velocity(ctraj, ts_k);
    const auto a_WS_W = ctraj_get_acceleration(ctraj, ts_k);
    vec3_t a_WS_S;
    vec3_t w_WS_S;
    sim_imu_measurement(imu,
                        rndeng,
                        ts_k,
                        T_WS_W,
                        w_WS_W,
                        a_WS_W,
                        a_WS_S,
                        w_WS_S);

    // Propagate simulated IMU measurements
    const real_t dt_k = ts2sec(dt);
    const real_t dt_k_sq = dt_k * dt_k;
    const vec3_t g{0.0, 0.0, -imu.g};
    // -- Position at time k
    const vec3_t b_a = ones(3, 1) * imu.b_a;
    const vec3_t n_a = ones(3, 1) * imu.sigma_a_c;
    r_WS += v_WS * dt_k;
    r_WS += 0.5 * g * dt_k_sq;
    r_WS += 0.5 * C_WS * (a_WS_S - b_a - n_a) * dt_k_sq;
    // -- velocity at time k
    v_WS += C_WS * (a_WS_S - b_a - n_a) * dt_k + g * dt_k;
    // -- Attitude at time k
    const vec3_t b_g = ones(3, 1) * imu.b_g;
    const vec3_t n_g = ones(3, 1) * imu.sigma_g_c;
    C_WS = C_WS * lie::Exp((w_WS_S - b_g - n_g) * ts2sec(dt));

    // Reocord IMU measurments
    pos_prop.push_back(r_WS);
    vel_prop.push_back(v_WS);
    att_prop.emplace_back(quat_t{C_WS});
    imu_ts.push_back(ts_k);
    imu_accel.push_back(a_WS_S);
    imu_gyro.push_back(w_WS_S);

    ts_k += dt;
  }
  save_data("/tmp/att_prop.csv", imu_ts, att_prop);
  save_data("/tmp/pos_prop.csv", imu_ts, pos_prop);
  save_data("/tmp/imu_accel.csv", imu_ts, imu_accel);
  save_data("/tmp/imu_gyro.csv", imu_ts, imu_gyro);

  // Create factor
  imu_factor_t factor(0, imu_ts, imu_accel, imu_gyro);
  factor.propagate(imu_ts, imu_accel, imu_gyro);

  // Evaluate factor
  // -- Sensor pose at i
  mat4_t T_WS_i = tf(orientations.front(), positions.front());
  pose_t pose_i{T_WS_i};
  // -- Speed and bias at i
  const vec3_t v_WS_i = ctraj_get_velocity(ctraj, 0.0);
  vec_t<9> sb_i;
  sb_i << v_WS_i, zeros(3, 1), zeros(3, 1);
  // -- Sensor pose at j
  mat4_t T_WS_j = tf(orientations.back(), positions.back());
  pose_t pose_j{T_WS_j};
  // -- Speed and bias at j
  const vec3_t v_WS_j = ctraj_get_velocity(ctraj, 10.0);
  vec_t<9> sb_j;
  sb_j << v_WS_j, zeros(3, 1), zeros(3, 1);
  // -- Evaluate
  real_t *params[4] = {
    pose_i.data(),
    sb_i.data(),
    pose_j.data(),
    sb_j.data(),
  };
  factor.eval(params);

  // Check jacobians
  int retval = 0;
  // -- Check jacobian of sensor pose at i
  retval = check_imu_factor_J_sensor_pose_i(
    factor, T_WS_i, sb_i, T_WS_j, sb_j, factor.jacobians[0]);
  MU_CHECK(retval == 0);
  // -- Check jacobian of speed and bias at i
  retval = check_imu_factor_J_speed_bias_i(
    factor, T_WS_i, sb_i, T_WS_j, sb_j, factor.jacobians[1]);
  MU_CHECK(retval == 0);
  // -- Check jacobian of sensor pose at j
  retval = check_imu_factor_J_sensor_pose_j(
    factor, T_WS_i, sb_i, T_WS_j, sb_j, factor.jacobians[2]);
  MU_CHECK(retval == 0);
  // -- Check jacobian of speed and bias at j
  retval = check_imu_factor_J_speed_bias_j(
    factor, T_WS_i, sb_i, T_WS_j, sb_j, factor.jacobians[3]);
  MU_CHECK(retval == 0);

  // Debug
  // const bool debug = true;
  const bool debug = false;
  if (debug) {
    OCTAVE_SCRIPT("scripts/core/plot_imu_measurements.m "
                  "/tmp/pos_data.csv "
                  "/tmp/pos_prop.csv "
                  "/tmp/att_data.csv "
                  "/tmp/att_prop.csv "
                  "/tmp/imu_accel.csv "
                  "/tmp/imu_gyro.csv ");
  }

  return 0;
}

int test_graph() {
  graph_t graph;

  MU_CHECK(graph.params.size() == 0);
  MU_CHECK(graph.params.size() == 0);
  MU_CHECK(graph.factors.size() == 0);

  return 0;
}

int test_graph_add_pose() {
  graph_t graph;

  timestamp_t ts = 0;
  mat4_t T_WS = I(4);
  graph_add_pose(graph, ts, T_WS);

  MU_CHECK(graph.params.size() == 1);
  MU_CHECK(graph.params[0] != nullptr);

  return 0;
}

int test_graph_add_landmark() {
  graph_t graph;

  vec3_t landmark = zeros(3, 1);
  graph_add_landmark(graph, landmark);

  MU_CHECK(graph.params.size() == 1);
  MU_CHECK(graph.params[0] != nullptr);

  return 0;
}

int test_graph_add_cam_params() {
  graph_t graph;

  int cam_index = 0;
  vec4_t params{1.0, 2.0, 3.0, 4.0};
  graph_add_cam_params(graph, cam_index, params);

  MU_CHECK(graph.params.size() == 1);
  MU_CHECK(graph.params[0] != nullptr);

  return 0;
}

int test_graph_add_dist_params() {
  graph_t graph;

  int cam_index = 0;
  vec4_t params{1.0, 2.0, 3.0, 4.0};
  graph_add_dist_params(graph, cam_index, params);

  MU_CHECK(graph.params.size() == 1);
  MU_CHECK(graph.params[0] != nullptr);

  return 0;
}

int test_graph_add_sb_params() {
  graph_t graph;

  int imu_index = 0;
  vec3_t v{1.0, 2.0, 3.0};
  vec3_t ba{4.0, 5.0, 6.0};
  vec3_t bg{7.0, 8.0, 9.0};
  graph_add_sb_params(graph, imu_index, v, ba, bg);

  MU_CHECK(graph.params.size() == 1);
  MU_CHECK(graph.params[0] != nullptr);

  return 0;
}

int test_graph_add_ba_factor() {
  graph_t graph;

  // Camera pose
  const timestamp_t ts = 0;
  const vec3_t euler{-90.0, 0.0, -90.0};
  const mat3_t C_WC = euler321(deg2rad(euler));
  const vec3_t r_WC = zeros(3, 1);
  const mat4_t T_WC = tf(C_WC, r_WC);
  const auto cam_pose_id = graph_add_pose(graph, ts, T_WC);

  // Landmark
  const vec3_t p_W{10.0, 0.0, 0.0};
  const auto landmark_id = graph_add_landmark(graph, p_W);

  // Camera and distortion parameters
  const int cam_index = 0;
  const vec4_t proj_params{640, 480, 320, 240};
  const vec4_t dist_params{0.01, 0.001, 0.001, 0.001};
  const auto cam_param_id = graph_add_cam_params(graph, cam_index, proj_params);
  const auto dist_param_id = graph_add_dist_params(graph, cam_index, dist_params);

  // BA factor
  const int img_w = 640;
  const int img_h = 480;
  const pinhole_radtan4_t cm{img_w, img_h, proj_params, dist_params};

  vec2_t z;
  cm.project(tf_point(T_WC.inverse(), p_W), z);
  graph_add_ba_factor<pinhole_radtan4_t>(
    graph,
    ts,
    cam_index,
    img_w,
    img_h,
    cam_pose_id,
    landmark_id,
    cam_param_id,
    dist_param_id,
    z
  );

  MU_CHECK(graph.params.size() == 4);
  MU_CHECK(graph.params[0] != nullptr);
  MU_CHECK(graph.params[1] != nullptr);
  MU_CHECK(graph.params[2] != nullptr);
  MU_CHECK(graph.params[3] != nullptr);

  MU_CHECK(graph.factors.size() == 1);

  return 0;
}

int test_graph_add_cam_factor() {
  graph_t graph;

  // Sensor pose
  const timestamp_t ts = 0;
  const mat3_t C_WS = I(3);
  const vec3_t r_WS = zeros(3, 1);
  const mat4_t T_WS = tf(C_WS, r_WS);
  const auto sensor_pose_id = graph_add_pose(graph, ts, T_WS);

  // IMU-Camera pose
  const vec3_t euler{-90.0, 0.0, -90.0};
  const mat3_t C_SC = euler321(deg2rad(euler));
  const vec3_t r_SC = zeros(3, 1);
  const mat4_t T_SC = tf(C_SC, r_SC);
  const auto imucam_pose_id = graph_add_pose(graph, ts, T_SC);

  // Landmark
  const vec3_t p_W{10.0, 0.0, 0.0};
  const auto landmark_id = graph_add_landmark(graph, p_W);

  // Camera and distortion parameters
  const int cam_index = 0;
  const vec4_t proj_params{640, 480, 320, 240};
  const vec4_t dist_params{0.01, 0.001, 0.001, 0.001};
  const auto cam_param_id = graph_add_cam_params(graph, cam_index, proj_params);
  const auto dist_param_id = graph_add_dist_params(graph, cam_index, dist_params);

  // BA factor
  const int img_w = 640;
  const int img_h = 480;
  const pinhole_radtan4_t cm{img_w, img_h, proj_params, dist_params};
  vec2_t z;
  cm.project(tf_point((T_WS * T_SC).inverse(), p_W), z);
  graph_add_cam_factor<pinhole_radtan4_t>(
    graph,
    ts,
    cam_index,
    img_w,
    img_h,
    sensor_pose_id,
    imucam_pose_id,
    landmark_id,
    cam_param_id,
    dist_param_id,
    z
  );

  MU_CHECK(graph.params.size() == 5);
  MU_CHECK(graph.params[0] != nullptr);
  MU_CHECK(graph.params[1] != nullptr);
  MU_CHECK(graph.params[2] != nullptr);
  MU_CHECK(graph.params[3] != nullptr);
  MU_CHECK(graph.params[4] != nullptr);

  MU_CHECK(graph.factors.size() == 1);

  return 0;
}

int test_graph_add_imu_factor() {
  // Generate trajectory
  timestamps_t timestamps;
  vec3s_t positions;
  quats_t orientations;
  for (int i = 0; i <= 5; i++) {
    timestamps.push_back(i * 1e8);
    positions.emplace_back(i, i, 0.0);
    orientations.emplace_back(1.0, 0.0, 0.0, 0.0);
  }
  ctraj_t ctraj(timestamps, positions, orientations);

  // Setup imu sim
  sim_imu_t imu;
  imu.rate = 400;
  imu.tau_a = 3600;
  imu.tau_g = 3600;
  imu.sigma_g_c = 0.00275;
  imu.sigma_a_c = 0.0250;
  imu.sigma_gw_c = 1.65e-05;
  imu.sigma_aw_c = 0.000441;
  imu.g = 9.81007;

  // Simulate IMU measurements
  std::default_random_engine rndeng;
  timestamps_t imu_ts;
  vec3s_t imu_accel;
  vec3s_t imu_gyro;

  timestamp_t ts_k = 0;
  const timestamp_t ts_end = timestamps.back();
  const timestamp_t dt = (1 / imu.rate) * 1e9;

  // -- Simulate imu measurements
  while (ts_k <= ts_end) {
    const auto T_WS_W = ctraj_get_pose(ctraj, ts_k);
    const auto w_WS_W = ctraj_get_angular_velocity(ctraj, ts_k);
    const auto a_WS_W = ctraj_get_acceleration(ctraj, ts_k);
    vec3_t a_WS_S;
    vec3_t w_WS_S;
    sim_imu_measurement(imu,
                        rndeng,
                        ts_k,
                        T_WS_W,
                        w_WS_W,
                        a_WS_W,
                        a_WS_S,
                        w_WS_S);

    // Reocord IMU measurments
    imu_ts.push_back(ts_k);
    imu_accel.push_back(a_WS_S);
    imu_gyro.push_back(w_WS_S);

    ts_k += dt;
  }

  // Create graph
  graph_t graph;
  timestamp_t t0 = 0;
  timestamp_t t1 = ts_end;
  // -- Add sensor pose at i
  const mat4_t T_WS_i = tf(orientations.front(), positions.front());
  auto pose0_id = graph_add_pose(graph, t0, T_WS_i);
  // -- Add speed and bias at i
  const vec3_t v_WS_i = ctraj_get_velocity(ctraj, ns2sec(t0));
  const vec3_t ba_i{0.0, 0.0, 0.0};
  const vec3_t bg_i{0.0, 0.0, 0.0};
  auto sb0_id = graph_add_sb_params(graph, t0, v_WS_i, ba_i, bg_i);
  // -- Add sensor pose at j
  const mat4_t T_WS_j = tf(orientations.back(), positions.back());
  auto pose1_id = graph_add_pose(graph, t1, T_WS_j);
  // -- Add speed and bias at j
  const vec3_t v_WS_j = ctraj_get_velocity(ctraj, ns2sec(t1));
  const vec3_t ba_j{0.0, 0.0, 0.0};
  const vec3_t bg_j{0.0, 0.0, 0.0};
  auto sb1_id = graph_add_sb_params(graph, t1, v_WS_j, ba_j, bg_j);
  // -- Add imu factor
  const int imu_index = 0;
  graph_add_imu_factor(graph,
                       imu_index,
                       imu_ts,
                       imu_accel,
                       imu_gyro,
                       pose0_id,
                       sb0_id,
                       pose1_id,
                       sb1_id);

  real_t *params[4] = {
    graph.params[0]->data(),
    graph.params[1]->data(),
    graph.params[2]->data(),
    graph.params[3]->data()
  };
  graph.factors[0]->eval(params);
  std::cout << graph.factors[0]->residuals << std::endl;

  // Asserts
  MU_CHECK(graph.params.size() == 4);
  MU_CHECK(graph.factors.size() == 1);

  return 0;
}

static vec3s_t create_3d_features(const real_t *x_bounds,
                                  const real_t *y_bounds,
                                  const real_t *z_bounds,
                                  const size_t nb_features) {
  vec3s_t features;
  for (size_t i = 0; i < nb_features; i++) {
    real_t x = randf(x_bounds[0], x_bounds[1]);
    real_t y = randf(y_bounds[0], y_bounds[1]);
    real_t z = randf(z_bounds[0], z_bounds[1]);
    features.emplace_back(x, y, z);
  }
  return features;
}

static vec3s_t create_3d_features_perimeter(const vec3_t &origin,
                                            const vec3_t &dim,
                                            const size_t nb_features) {
  // Dimension of the outskirt
  const real_t w = dim(0);
  const real_t l = dim(1);
  const real_t h = dim(2);

  // Features per side
  size_t nb_fps = nb_features / 4.0;
  vec3s_t features;

  // Features in the east side
  {
    const real_t x_bounds[2] = {origin(0) - w, origin(0) + w};
    const real_t y_bounds[2] = {origin(1) + l, origin(1) + l};
    const real_t z_bounds[2] = {origin(2) - h, origin(2) + h};
    auto f = create_3d_features(x_bounds, y_bounds, z_bounds, nb_fps);
    features.reserve(features.size() + std::distance(f.begin(), f.end()));
    features.insert(features.end(), f.begin(), f.end());
  }

  // Features in the north side
  {
    const real_t x_bounds[2] = {origin(0) + w, origin(0) + w};
    const real_t y_bounds[2] = {origin(1) - l, origin(1) + l};
    const real_t z_bounds[2] = {origin(2) - h, origin(2) + h};
    auto f = create_3d_features(x_bounds, y_bounds, z_bounds, nb_fps);
    features.reserve(features.size() + std::distance(f.begin(), f.end()));
    features.insert(features.end(), f.begin(), f.end());
  }

  // Features in the west side
  {
    const real_t x_bounds[2] = {origin(0) - w, origin(0) + w};
    const real_t y_bounds[2] = {origin(1) - l, origin(1) - l};
    const real_t z_bounds[2] = {origin(2) - h, origin(2) + h};
    auto f = create_3d_features(x_bounds, y_bounds, z_bounds, nb_fps);
    features.reserve(features.size() + std::distance(f.begin(), f.end()));
    features.insert(features.end(), f.begin(), f.end());
  }

  // Features in the south side
  {
    const real_t x_bounds[2] = {origin(0) - w, origin(0) - w};
    const real_t y_bounds[2] = {origin(1) - l, origin(1) + l};
    const real_t z_bounds[2] = {origin(2) - h, origin(2) + h};
    auto f = create_3d_features(x_bounds, y_bounds, z_bounds, nb_fps);
    features.reserve(features.size() + std::distance(f.begin(), f.end()));
    features.insert(features.end(), f.begin(), f.end());
  }

  return features;
}

void save_features(const std::string &path, const vec3s_t &features) {
  FILE *csv = fopen(path.c_str(), "w");
  for (const auto &f : features) {
    fprintf(csv, "%f,%f,%f\n", f(0), f(1), f(2));
  }
  fflush(csv);
  fclose(csv);
}

void save_poses(const std::string &path,
                const timestamps_t &timestamps,
                const vec3s_t &positions,
                const quats_t &orientations) {
  FILE *csv = fopen(path.c_str(), "w");
  for (size_t i = 0; i < timestamps.size(); i++) {
    const timestamp_t ts = timestamps[i];
    const vec3_t pos = positions[i];
    const quat_t rot = orientations[i];
    fprintf(csv, "%ld,", ts);
    fprintf(csv, "%f,%f,%f,", pos(0), pos(1), pos(2));
    fprintf(csv, "%f,%f,%f,%f\n", rot.w(), rot.x(), rot.y(), rot.z());
  }
  fflush(csv);
  fclose(csv);
}

void save_imu_data(const std::string &imu_data_path,
                   const std::string &imu_poses_path,
                   const timestamps_t &imu_ts,
                   const vec3s_t &imu_accel,
                   const vec3s_t &imu_gyro,
                   const vec3s_t &imu_pos,
                   const quats_t &imu_rot) {
  {
    FILE *csv = fopen(imu_data_path.c_str(), "w");
    for (size_t i = 0; i < imu_ts.size(); i++) {
      const timestamp_t ts = imu_ts[i];
      const vec3_t acc = imu_accel[i];
      const vec3_t gyr = imu_gyro[i];
      fprintf(csv, "%ld,", ts);
      fprintf(csv, "%f,%f,%f,", acc(0), acc(1), acc(2));
      fprintf(csv, "%f,%f,%f\n", gyr(0), gyr(1), gyr(2));
    }
    fflush(csv);
    fclose(csv);
  }

  {
    FILE *csv = fopen(imu_poses_path.c_str(), "w");
    for (size_t i = 0; i < imu_ts.size(); i++) {
      const timestamp_t ts = imu_ts[i];
      const vec3_t pos = imu_pos[i];
      const quat_t rot = imu_rot[i];
      fprintf(csv, "%ld,", ts);
      fprintf(csv, "%f,%f,%f,", pos(0), pos(1), pos(2));
      fprintf(csv, "%f,%f,%f,%f\n", rot.w(), rot.x(), rot.y(), rot.z());
    }
    fflush(csv);
    fclose(csv);
  }
}

void simulate_trajectory() {
  const std::string features_path = "/tmp/features.csv";
  const std::string cam_poses_path = "/tmp/cam_poses.csv";
  const std::string imu_data_path = "/tmp/imu.csv";
  const std::string imu_poses_path = "/tmp/imu_poses.csv";

  // Create features
  const vec3_t origin{0.0, 0.0, 0.0};
  const vec3_t dim{5.0, 5.0, 5.0};
  const size_t nb_features = 1000;
  const vec3s_t features = create_3d_features_perimeter(origin, dim, nb_features);
  save_features(features_path, features);

  // // Generate circle trajectory
  // const int nb_poses = 20;
  // const real_t r = 3.0;
  // const real_t dtheta = deg2rad(360.0) / nb_poses;
  // real_t theta = deg2rad(180.0);
  // real_t yaw = deg2rad(0.0);
  // const real_t t_end = 20.0;
  // const real_t dt = t_end / nb_poses;
  // real_t t = 0.0;
  //
  // timestamps_t timestamps;
  // vec3s_t positions;
  // quats_t orientations;

  // // while (theta > deg2rad(-180)) {
  // // while (theta > deg2rad(-90)) {
  // while (theta > deg2rad(50)) {
  //   const real_t x = r * cos(theta);
  //   const real_t y = r * sin(theta);
  //   const real_t z = 0.0;
  //   const vec3_t rpy{deg2rad(-90.0 + randf(-1.0, 1.0)),
  //                    deg2rad(randf(-1.0, 1.0)),
  //                    wrapPi(yaw)};
  //
  //   timestamps.push_back(t * 1e9);
  //   positions.emplace_back(x, y, z);
  //   orientations.emplace_back(euler321(rpy));
  //
  //   t += dt;
  //   theta -= dtheta;
  //   yaw -= dtheta;
  // }

  timestamps_t timestamps;
  vec3s_t positions;
  quats_t orientations;

  timestamps.push_back(0.0 * 1e9);
  positions.emplace_back(-3.5, 3.5, 0.0);
  orientations.emplace_back(1.0, 0.0, 0.0, 0.0);

  timestamps.push_back(2.5 * 1e9);
  positions.emplace_back(-2.0, -2.0, 0.0);
  orientations.emplace_back(1.0, 0.0, 0.0, 0.0);

  timestamps.push_back(5.0 * 1e9);
  positions.emplace_back(0.0, 0.0, 0.0);
  orientations.emplace_back(1.0, 0.0, 0.0, 0.0);

  timestamps.push_back(7.5 * 1e9);
  positions.emplace_back(2.0, 2.0, 0.0);
  orientations.emplace_back(1.0, 0.0, 0.0, 0.0);

  timestamps.push_back(10 * 1e9);
  positions.emplace_back(3.5, -3.5, 0.0);
  orientations.emplace_back(1.0, 0.0, 0.0, 0.0);

  save_poses(cam_poses_path, timestamps, positions, orientations);

  // Simulate IMU measurements
  ctraj_t ctraj(timestamps, positions, orientations);
  sim_imu_t imu;
  imu.rate = 400;
  imu.tau_a = 3600;
  imu.tau_g = 3600;
  imu.sigma_g_c = 0.00275;
  imu.sigma_a_c = 0.0250;
  imu.sigma_gw_c = 1.65e-05;
  imu.sigma_aw_c = 0.000441;
  imu.g = 9.81007;

  std::default_random_engine rndeng;
  timestamps_t imu_ts;
  vec3s_t imu_pos;
  quats_t imu_rot;
  vec3s_t imu_accel;
  vec3s_t imu_gyro;

  timestamps_t cam_ts;
  vec3s_t cam_pos;
  quats_t cam_rot;

  const real_t cam_rate = 20.0;
  timestamp_t ts_k = 0;
  const timestamp_t ts_end = timestamps.back();
  const timestamp_t dt_imu = (1 / imu.rate) * 1e9;
  const timestamp_t imu_period = (1.0 / imu.rate) * 1e9;
  const timestamp_t cam_period = (1.0 / cam_rate) * 1e9;

  while (ts_k <= ts_end) {
    if ((ts_k % imu_period) == 0) {
      const auto T_WS_W = ctraj_get_pose(ctraj, ts_k);
      const auto w_WS_W = ctraj_get_angular_velocity(ctraj, ts_k);
      const auto a_WS_W = ctraj_get_acceleration(ctraj, ts_k);
      vec3_t a_WS_S;
      vec3_t w_WS_S;
      sim_imu_measurement(imu,
                          rndeng,
                          ts_k,
                          T_WS_W,
                          w_WS_W,
                          a_WS_W,
                          a_WS_S,
                          w_WS_S);

      // Record IMU measurments
      imu_ts.push_back(ts_k);
      imu_accel.push_back(a_WS_S);
      imu_gyro.push_back(w_WS_S);

      imu_pos.push_back(tf_trans(T_WS_W));
      imu_rot.push_back(tf_quat(T_WS_W));
    }

    if ((ts_k % cam_period) == 0) {
      const auto C_CS = euler321(deg2rad(vec3_t{-90.0, 0.0, -90.0}));
      const vec3_t r_CS = zeros(3, 1);
      const auto T_SC = tf(C_CS, r_CS);
      const auto T_WS = ctraj_get_pose(ctraj, ts_k);
      const auto T_WC = T_WS * T_SC;

      cam_ts.push_back(ts_k);
      cam_pos.push_back(tf_trans(T_WC));
      cam_rot.push_back(tf_quat(T_WC));
    }

    ts_k += dt_imu;
  }
  save_imu_data(imu_data_path, imu_poses_path,
                imu_ts, imu_accel, imu_gyro, imu_pos, imu_rot);
  save_poses(cam_poses_path, cam_ts, cam_pos, cam_rot);
}

int test_graph_eval() {
  graph_t graph;
  simulate_trajectory();

  // Debug
  const bool debug = true;
  // const bool debug = false;
  if (debug) {
    OCTAVE_SCRIPT("scripts/estimation/plot_sim.m");
  }

  return 0;
}

// int test_graph_solve() {
//   graph_t graph;
//
//   // Add landmarks
//   const vec3_t p0_W{1.0, 2.0, 3.0};
//   const vec3_t p1_W{4.0, 5.0, 6.0};
//   const vec3_t p2_W{7.0, 8.0, 9.0};
//   const size_t p0_id = graph_add_landmark(graph, p0_W);
//   const size_t p1_id = graph_add_landmark(graph, p1_W);
//   const size_t p2_id = graph_add_landmark(graph, p2_W);
//
//   // Add pose
//   const timestamp_t ts = 0;
//   const vec2_t z{0.0, 0.0};
//   const vec3_t rpy_WC{-M_PI / 2.0, 0.0, -M_PI / 2.0};
//   const mat3_t C_WC = euler321(rpy_WC);
//   const vec3_t r_WC = zeros(3, 1);
//   const mat4_t T_WC = tf(C_WC, r_WC);
//   const size_t pose_id = graph_add_pose(graph, ts, T_WC);
//
//   // Add Factors
//   // graph_add_factor(graph, ts, z, p0_id, pose_id);
//   // graph_add_factor(graph, ts, z, p1_id, pose_id);
//   // graph_add_factor(graph, ts, z, p2_id, pose_id);
//
//   // graph_solve(graph);
//   // graph_free(graph);
//
//   return 0;
// }

void test_suite() {
  MU_ADD_TEST(test_ba_factor_jacobians);
  MU_ADD_TEST(test_cam_factor_jacobians);
  MU_ADD_TEST(test_imu_factor_jacobians);

  MU_ADD_TEST(test_graph);
  MU_ADD_TEST(test_graph_add_pose);
  MU_ADD_TEST(test_graph_add_landmark);
  MU_ADD_TEST(test_graph_add_cam_params);
  MU_ADD_TEST(test_graph_add_dist_params);
  MU_ADD_TEST(test_graph_add_sb_params);
  MU_ADD_TEST(test_graph_add_ba_factor);
  MU_ADD_TEST(test_graph_add_cam_factor);
  MU_ADD_TEST(test_graph_add_imu_factor);
  MU_ADD_TEST(test_graph_eval);
  // MU_ADD_TEST(test_graph_solve);
}

} // namespace proto

MU_RUN_TESTS(proto::test_suite);
