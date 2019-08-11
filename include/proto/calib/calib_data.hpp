#ifndef PROTO_CALIB_CALIB_DATA_HPP
#define PROTO_CALIB_CALIB_DATA_HPP

#include <string>

#include <opencv2/calib3d/calib3d.hpp>

#include "proto/core/math.hpp"
#include "proto/core/time.hpp"
#include "proto/calib/aprilgrid.hpp"
#include "proto/vision/vision_common.hpp"
#include "proto/vision/camera/camera_geometry.hpp"
#include "proto/vision/camera/pinhole.hpp"
#include "proto/vision/camera/radtan.hpp"
#include "proto/vision/camera/equi.hpp"

namespace proto {

/**
 * Pose parameter block
 */
struct calib_pose_param_t {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  quat_t q;
  vec3_t r;

  calib_pose_param_t(const mat4_t &T);
  calib_pose_param_t(const mat3_t &C, const vec3_t &r);
  calib_pose_param_t(const quat_t &q, const vec3_t &r);
  ~calib_pose_param_t();
};

/**
 * Calibration target.
 */
struct calib_target_t {
  std::string target_type;
  int tag_rows = 0;
  int tag_cols = 0;
  double tag_size = 0.0;
  double tag_spacing = 0.0;

  calib_target_t();
  ~calib_target_t();
};

/**
 * Load calibration target.
 * @returns 0 or -1 for success or failure
 */
int calib_target_load(calib_target_t &ct,
                      const std::string &target_file,
                      const std::string &prefix = "");

/**
 * Detect AprilGrid specified by `target` and given a directory of images in
 * `image_dir`.
 * @returns 0 for Success, -1 for failure.
 */
int detect_calib_data(const calib_target_t &target,
                      const std::string &image_dir,
                      const bool imshow = true);

/**
 * Preprocess camera image data and output AprilGrid detection data as
 * csv. The AprilGrid tag corners are estimated using the camera intrinsics
 * matrix `cam_K` and distortion vector `cam_D`. Once the data is preprocessed
 * the data is saved to `output_dir`.
 *
 * @returns 0 for Success, -1 for failure, and 1 where the output directory
 * contains data.
 */
int preprocess_camera_data(const calib_target_t &target,
                           const std::string &image_dir,
                           const mat3_t &cam_K,
                           const vec4_t &cam_D,
                           const std::string &output_dir,
                           const bool imshow = false,
                           const bool show_progress = true);

/**
 * Preprocess camera image data and output AprilGrid detection data as
 * csv. The data is initialized with `image_size` in pixels, the horizontal
 * lens fov `lens_hfov` and vertical lens fov `lens_vfov` in degrees. Once the
 * data is preprocessed the data is saved to `output_dir`.
 *
 * @returns 0 for Success, -1 for failure, and 1 where the output directory
 * contains data.
 */
int preprocess_camera_data(const calib_target_t &target,
                           const std::string &image_dir,
                           const vec2_t &image_size,
                           const double lens_hfov,
                           const double lens_vfov,
                           const std::string &output_dir,
                           const bool imshow = false,
                           const bool show_progress = true);

/**
 * Load preprocess-ed camera calibration data located in `data_dir` where the
 * data will be loaded in `aprilgrids`. By default, this function will only
 * return aprilgrids that are detected. To return all calibration data
 * including camera frames where aprilgrids were not detected, change
 * `detected_only` to false.
 *
 * @returns 0 or -1 for success or failure
 */
int load_camera_calib_data(const std::string &data_dir,
                           aprilgrids_t &aprilgrids,
                           timestamps_t &timestamps,
                           bool detected_only = true);

/**
 * Preprocess stereo image data and output AprilGrid detection data as
 * csv. The data is initialized with `image_size` in pixels, the horizontal
 * lens fov `lens_hfov` and vertical lens fov `lens_vfov` in degrees.
 *
 * This function assumes:
 *
 * - Stereo camera images are synchronized
 * - Number of images observed by both cameras are the same
 *
 * @returns 0 for Success, -1 for failure, and 1 where the output directory
 * contains data.
 */
int preprocess_stereo_data(const calib_target_t &target,
                           const std::string &cam0_image_dir,
                           const std::string &cam1_image_dir,
                           const vec2_t &cam0_image_size,
                           const vec2_t &cam1_image_size,
                           const double cam0_lens_hfov,
                           const double cam0_lens_vfov,
                           const double cam1_lens_hfov,
                           const double cam1_lens_vfov,
                           const std::string &cam0_output_dir,
                           const std::string &cam1_output_dir);

/**
 * Extract and only keep common aprilgrid corners between `grids0` and `grids1`.
 */
void extract_common_calib_data(aprilgrids_t &grids0, aprilgrids_t &grids1);

/**
 * Load preprocessed stereo calibration data, where `cam0_data_dir` and
 * `cam1_data_dir` are preprocessed calibration data observed from cam0 and
 * cam1. The preprocessed calibration data will be loaded into
 * `cam0_aprilgrids` and `cam1_aprilgrids` respectively, where the data
 * contains AprilGrids observed by both cameras at the same timestamp.**
 *
 * This function assumes:
 *
 * - Stereo camera images are synchronized
 * - Images that are synchronized are expected to have the **same exact
 *   timestamp**
 *
 * @returns 0 or -1 for success or failure
 */
int load_stereo_calib_data(const std::string &cam0_data_dir,
                           const std::string &cam1_data_dir,
                           aprilgrids_t &cam0_aprilgrids,
                           aprilgrids_t &cam1_aprilgrids);

/**
 * Load preprocessed multi-camera calibration data, where each data path in
 * `data_dirs` are the preprocessed calibration data observed by each camera,
 * and the preprocessed calibration data will be loaded into `calib_data` where
 * the key is the camera index and the value is the detected aprilgrids. The
 * data in `calib_data` contains AprilGrids observed by all cameras at the same
 * timestamp.
 *
 * This function assumes:
 *
 * - Camera images are synchronized
 * - Images that are synchronized are expected to have the **same exact
 *   timestamp**
 *
 * @returns 0 or -1 for success or failure
 */
int load_multicam_calib_data(const int nb_cams,
                             const std::vector<std::string> &data_dirs,
                             std::map<int, aprilgrids_t> &calib_data);

/**
 * Draw measured and projected pixel points.
 * @returns Image
 */
cv::Mat draw_calib_validation(const cv::Mat &image,
                              const vec2s_t &measured,
                              const vec2s_t &projected,
                              const cv::Scalar &measured_color,
                              const cv::Scalar &projected_color);

/**
 * Validate calibration.
 * @returns Validation image for visual inspection
 */
template <typename CM, typename DM>
cv::Mat validate_intrinsics(const cv::Mat &image,
                            const vec2s_t &keypoints,
                            const vec3s_t &points,
                            const camera_geometry_t<CM, DM> &camera_geometry);

/**
 * Validate stereo extrinsics.
 * @returns Validation image for visual inspection
 */
template <typename CM, typename DM>
cv::Mat validate_stereo(const cv::Mat &image0,
                        const cv::Mat &image1,
                        const vec2s_t &kps0,
                        const vec3s_t &points0,
                        const vec2s_t &kps1,
                        const vec3s_t &points1,
                        const camera_geometry_t<CM, DM> &cam0,
                        const camera_geometry_t<CM, DM> &cam1,
                        const mat4_t &T_C1C0);

/**
 * `calib_target_t` to output stream.
 */
std::ostream &operator<<(std::ostream &os, const calib_target_t &target);

} // namespace proto
#include "calib_data_impl.hpp"
#endif // PROTO_CALIB_CALIB_DATA_HPP
