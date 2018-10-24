#include "sim/camera.hpp"

namespace prototype {

VirtualCamera::VirtualCamera() {}

VirtualCamera::VirtualCamera(const int image_width,
                             const int image_height,
                             const double fx,
                             const double fy,
                             const double cx,
                             const double cy)
    : camera_model{image_width, image_height, fx, fy, cx, cy} {}

VirtualCamera::~VirtualCamera() {}

MatX VirtualCamera::observedFeatures(const MatX &features,
                                     const Vec3 &rpy_G,
                                     const Vec3 &t_G,
                                     std::vector<int> &mask) {
  // Rotation matrix
  // Convert from Global frame NWU to Camera frame EDN
  // NWU: (x - forward, y - left, z - up)
  // EDN: (x - right, y - down, z - forward)
  const Vec3 rpy_C{-rpy_G(1), -rpy_G(2), rpy_G(0)};
  const Mat3 R = euler123ToRot(rpy_C);

  // Check which features are observable from camera
  std::vector<Vec2> observed;
  for (int i = 0; i < features.rows(); i++) {
    // Transform point from global frame to camera frame
    const Vec3 pt_G = features.row(i).transpose();
    const Vec3 pt_C = rotx(-M_PI / 2.0) * rotz(-M_PI / 2.0) * pt_G;

    // Transform translation from global frame to camera frame
    const Vec3 t_C = rotx(-M_PI / 2.0) * rotz(-M_PI / 2.0) * t_G;

    // Project 3D world point to 2D image plane
    Vec3 img_pt = this->camera_model.project(homogeneous(pt_C), R, t_C);

    // Check to see if feature is valid and infront of camera
    if (img_pt(2) < 1.0) {
      continue; // skip this feature! It is not infront of camera
    }

    // Normalize pixels
    img_pt(0) = img_pt(0) / img_pt(2);
    img_pt(1) = img_pt(1) / img_pt(2);
    img_pt(2) = img_pt(2) / img_pt(2);

    // Check to see if feature observed is within image plane
    const int image_width = this->camera_model.image_width;
    const int image_height = this->camera_model.image_height;
    const bool x_ok = (img_pt(0) < image_width) && (img_pt(0) > 0.0);
    const bool y_ok = (img_pt(1) < image_height) && (img_pt(1) > 0.0);
    if (x_ok && y_ok) {
      observed.emplace_back(img_pt(0), img_pt(1));
      mask.push_back(i);
    }
  }

  // Convert vector of Vec2 to MatX
  MatX result = zeros(observed.size(), 2);
  for (size_t i = 0; i < observed.size(); i++) {
    result.block(i, 0, 1, 2) = observed[i].transpose();
  }

  return result;
}

VirtualStereoCamera::VirtualStereoCamera() {}

VirtualStereoCamera::VirtualStereoCamera(const int image_width,
                                         const int image_height,
                                         const double fx,
                                         const double fy,
                                         const double cx,
                                         const double cy,
                                         const Mat4 &T_cam1_cam0)
    : type{"STATIC"}, camera_model{image_width, image_height, fx, fy, cx, cy},
      T_cam1_cam0{T_cam1_cam0} {}

VirtualStereoCamera::VirtualStereoCamera(const int image_width,
                                         const int image_height,
                                         const double fx,
                                         const double fy,
                                         const double cx,
                                         const double cy,
                                         const GimbalModel &gimbal_model)
    : type{"DYNAMIC"}, camera_model{image_width, image_height, fx, fy, cx, cy},
      gimbal_model{gimbal_model} {}

VirtualStereoCamera::~VirtualStereoCamera() {}

MatX VirtualStereoCamera::observedFeatures(const MatX &features,
                                           const Vec3 &rpy_BG,
                                           const Vec3 &t_G_B,
                                           std::vector<int> &mask) {
  // Build transform from global to body frame
  const Mat3 R_BG = euler123ToRot(rpy_BG);
  const Mat4 T_BG = transformation_matrix(R_BG, -R_BG * t_G_B);

  // Build transform from body to camera0 frame
  const Mat3 R_C0B = euler123ToRot(Vec3{-M_PI / 2.0, 0.0, -M_PI / 2.0});
  const Vec3 t_B_C0 = zeros(3, 1);
  const Mat4 T_C0B = transformation_matrix(R_C0B, -R_C0B * t_B_C0);

  // Build transform from camera0 to camera1 frame
  Mat4 T_C1_C0;
  if (this->type == "STATIC") {
    T_C1_C0 = this->T_cam1_cam0;
  } else if (this->type == "DYNAMIC") {
    T_C1_C0 = this->gimbal_model.T_ds();
  } else {
    FATAL("Invalid stereo camera type [%s]\n", this->type.c_str());
  }

  // Check which features are observable from camera
  std::vector<Vec2> observed;
  for (int i = 0; i < features.rows(); i++) {
    // Transform point from global frame to camera frame
    const Vec3 pt_G = features.row(i).transpose();
    const Vec3 pt_C0 = (T_C0B * T_BG * pt_G.homogeneous()).head(3);
    const Vec3 pt_C1 = (T_C1_C0 * T_C0B * T_BG * pt_G.homogeneous()).head(3);

    // Project 3D world point to 2D image plane
    Vec3 p0 = this->camera_model.K * pt_C0;
    Vec3 p1 = this->camera_model.K * pt_C1;
    // -- Check to see if feature is valid and infront of camera
    if (p0(2) < 1.0 || p1(2) < 1.0) {
      continue; // skip this feature! It is not infront of camera
    }
    // -- Normalize pixels
    p0(0) = p0(0) / p0(2);
    p0(1) = p0(1) / p0(2);
    p1(0) = p1(0) / p1(2);
    p1(1) = p1(1) / p1(2);

    // Check to see if feature observed is within image plane
    const int image_width = this->camera_model.image_width;
    const int image_height = this->camera_model.image_height;
    const bool x0_ok = (p0(0) < image_width) && (p0(0) > 0.0);
    const bool y0_ok = (p0(1) < image_height) && (p0(1) > 0.0);
    const bool x1_ok = (p1(0) < image_width) && (p1(0) > 0.0);
    const bool y1_ok = (p1(1) < image_height) && (p1(1) > 0.0);
    if (x0_ok && y0_ok && x1_ok && y1_ok) {
      observed.emplace_back(p0(0), p0(1));
      observed.emplace_back(p1(0), p1(1));
      mask.push_back(i);
    }
  }

  // Convert vector of Vec2 to MatX
  MatX result = zeros(observed.size(), 2);
  for (size_t i = 0; i < observed.size(); i++) {
    result.block(i, 0, 1, 2) = observed[i].transpose();
  }

  return result;
}

void VirtualStereoCamera::setGimbalAttitude(const double roll,
                                            const double pitch) {
  this->gimbal_model.setAttitude(roll, pitch);
}

} //  namespace prototype
