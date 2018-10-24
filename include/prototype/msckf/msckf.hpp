/**
 * @file
 * @defgroup msckf msckf
 */
#ifndef PROTOTYPE_VISION_MSCKF_MSCKF_HPP
#define PROTOTYPE_VISION_MSCKF_MSCKF_HPP

#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
#include <boost/math/distributions/chi_squared.hpp>

#include "prototype/core.hpp"
#include "prototype/core/quaternion/jpl.hpp"
#include "prototype/vision/feature2d/feature_tracker.hpp"

#include "prototype/msckf/camera_state.hpp"
#include "prototype/msckf/feature_estimator.hpp"
#include "prototype/msckf/imu_state.hpp"

namespace prototype {
/**
 * @addtogroup msckf
 * @{
 */

enum class MSCKFState {
  IDLE,
  CONFIGURED,
  INITIALIZED,
};

/**
 * Multi-State Constraint Kalman Filter
 *
 * This class implements the MSCKF based on:
 *
 * Mourikis, Anastasios I., and Stergios I. Roumeliotis. "A multi-state
 * constraint Kalman filter for vision-aided inertial navigation." Robotics
 * and automation, 2007 IEEE international conference on. IEEE, 2007.  APA
 *
 * A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Kalman Filter for
 * Vision-Aided Inertial Navigation," Technical Report, September 2006
 * [http://www.ee.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf]
 */
class MSCKF {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Covariance matrices
  MatX P_cam = zeros(1, 1);
  MatX P_imu_cam = zeros(1, 1);

  // IMU
  IMUState imu_state;

  // Camera
  // -- State
  CameraStates cam_states;
  CameraStates old_cam_states;
  FrameID counter_frame_id = 0;
  // -- Extrinsics
  Vec3 ext_p_IC = zeros(3, 1);
  Vec4 ext_q_CI = Vec4{0.0, 0.0, 0.0, 1.0};
  // -- Measurement noise
  double img_var = 1e-1;

  // Misc
  std::map<int, double> chi_squared_table;
  long timestamp_first = 0;
  long last_updated = 0;

  // Settings
  int max_window_size = 30;
  int max_nb_tracks = 10;
  bool enable_ns_trick = true;
  bool enable_qr_trick = true;
  std::string qr_mode = "dense";

  MSCKF();

  /**
   * Configure
   *
   * @param config_file Path to configuration file
   * @returns 0 for success, -1 for failure
   */
  int configure(const std::string &config_file);

  /**
   * Get State
   *
   * @returns State (p_G, v_G, rpy_G)
   */
  VecX getState();

  /**
   * Return covariance matrix P
   */
  MatX P();

  /**
   * Jacobian J matrix
   *
   * @param cam_q_CI Rotation from IMU to camera frame in quaternion (x,y,z,w)
   * @param cam_p_IC Position of camera in IMU frame
   * @param q_hat_IG Rotation from global to IMU frame
   * @param N Number of camera states
   */
  MatX J(const Vec4 &cam_q_CI,
         const Vec3 &cam_p_IC,
         const Vec4 &q_hat_IG,
         const int N);

  /**
   * Return length of sliding window
   */
  int N();

  /**
   * Return measurement matrix H
   *
   * @param track Feature track
   * @param track_cam_states List of camera states
   * @param p_G_f Feature position in the global frame
   * @param H_f_j Measurement matrix
   * @param H_x_j Measurement matrix
   */
  void H(const FeatureTrack &track,
         const CameraStates &track_cam_states,
         const Vec3 &p_G_f,
         MatX &H_f_j,
         MatX &H_x_j);

  /**
   * Initialize
   *
   * @param ts timestamp in nano-seconds
   * @param q_IG Initial quaternion
   * @param v_G Initial velocity
   * @param p_G Initial position
   *
   * @returns 0 for success, -1 for failure
   */
  int initialize(const long ts,
                 const Vec4 &q_IG = Vec4{0.0, 0.0, 0.0, 1.0},
                 const Vec3 &v_G = Vec3::Zero(),
                 const Vec3 &p_G = Vec3::Zero());

  /**
   * Initialize
   *
   * @param ts timestamp in nano-seconds
   * @param imu_gyro_buffer IMU gyro buffer
   * @param imu_accel_buffer IMU accel buffer
	 *
   * @returns 0 for success, -1 for failure
   */
  int initialize(const long ts,
                 const std::vector<Vec3> &imu_gyro_buffer,
                 const std::vector<Vec3> &imu_accel_buffer);

  /**
   * Augment state vector with new camera state
   *
   * Augment state and covariance matrix with a copy of the current camera
   * pose estimate when a new image is recorded
   */
  void augmentState();

  /**
   * Get camera states the feature track was observed in
   *
   * @param track Feature track
   * @returns Camera states where feature track was observed
   */
  CameraStates getTrackCameraStates(const FeatureTrack &track);

  /**
   * Get all camera states including old and current
   *
   * @returns All camera states (old + current)
   */
  CameraStates getAllCameraStates();

  /**
   * Prediction update
   *
   * @param a_m Measured acceleration in body frame
   * @param w_m Measured angular velicty in body frame
   * @param ts Timestamp in nano-seconds
   *
   * @returns 0 for success, -1 for failure
   */
  int predictionUpdate(const Vec3 &a_m, const Vec3 &w_m, const long ts);

  /**
   * Chi-squared test
   */
  int chiSquaredTest(const MatX &H, const VecX &r, const int dof);

  /**
   * Residualize track
   *
   * @param track Feature track
   * @param H_j Measurement jacobian matrix
   * @param r_j Residuals vector
   *
   * @returns
   *  - -1: Track length < Min track length
   *  - -2: Failed to estimate feature position
   */
  int residualizeTrack(const FeatureTrack &track, MatX &H_j, VecX &r_j);

  /**
   * Calculate residuals
   *
   * @param tracks Feature tracks
   * @param T_H
   * @param r_n Residuals vector
   *
   * @returns
   *  - -1: Track length < Min track length
   *  - -2: Failed to estimate feature position
   */
  int calcResiduals(const FeatureTracks &tracks, MatX &T_H, VecX &r_n);

  /**
   * Correct IMU state
   *
   * @param dx Correction vector
   */
  void correctIMUState(const VecX &dx);

  /**
   * Correct camera states
   *
   * @param dx Correction vector
   */
  void correctCameraStates(const VecX &dx);

  /**
   * Prune old camera state to maintain sliding window size
   */
  CameraStates pruneCameraState();

  /**
   * Filter tracks
   *
   * @param tracks Feature tracks
   * @returns 0 for success, -1 for failure
   */
  FeatureTracks filterTracks(const FeatureTracks &tracks);

  /**
   * Measurmement update
   *
   * @param tracks Feature tracks
   * @returns 0 for success, -1 for failure
   */
  int measurementUpdate(const FeatureTracks &tracks);
};

/** @} group msckf */
} //  namespace prototype
#endif // PROTOTYPE_VISION_MSCKF_MSCKF_HPP
