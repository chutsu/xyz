#ifndef PROTO_VISION_CAMERA_EQUI_HPP
#define PROTO_VISION_CAMERA_EQUI_HPP

#include <opencv2/calib3d/calib3d.hpp>

#include "proto/core/core.hpp"

namespace proto {

/**
 * Equi-distant distortion
 */
struct equi4_t {
  double k1 = 0.0;
  double k2 = 0.0;
  double k3 = 0.0;
  double k4 = 0.0;

  equi4_t(const double k1_,
          const double k2_,
          const double k3_,
          const double k4_);
  ~equi4_t();
};

/**
 * Create Equidistant distortion vector
 */
template <typename T>
Eigen::Matrix<T, 4, 1> equi4_D(const T *distortion) {
  const T k1 = distortion[0];
  const T k2 = distortion[1];
  const T k3 = distortion[2];
  const T k4 = distortion[3];
  Eigen::Matrix<T, 4, 1> D{k1, k2, k3, k4};
  return D;
}

/**
 * Type to output stream.
 */
std::ostream &operator<<(std::ostream &os, const equi4_t &equi4);

/**
 * Distort point with equi-distant distortion model.
 *
 * @param[in] equi Equi-distance parameters
 * @param[in] point Point
 * @returns Distorted point
 */
vec2_t distort(const equi4_t &equi, const vec2_t &point);

/**
 * Distort point with equi-distant distortion model.
 *
 * @param[in] equi Equi-distance parameters
 * @param[in] point Point
 * @param[out] J_point Jacobian of equi w.r.t. point
 * @returns Distorted point
 */
vec2_t distort(const equi4_t &equi, const vec2_t &point, mat2_t &J_point);

/**
 * Distort point with equi-distant distortion model.
 *
 * @param[in] equi Equi-distance parameters
 * @param[in] points Points
 * @returns Distorted points
 */
matx_t distort(const equi4_t &equi, const matx_t &points);

/**
 * Un-distort a 2D point with the equi-distant distortion model.
 */
vec2_t undistort(const equi4_t &equi, const vec2_t &p);

} //  namespace proto
#endif // PROTO_VISION_CAMERA_EQUI_HPP
