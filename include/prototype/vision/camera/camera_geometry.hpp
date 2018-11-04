/**
 * @file
 * @ingroup calibration
 */
#ifndef PROTOTYPE_CALIB_CAMERA_CAMERA_GEOMETRY_HPP
#define PROTOTYPE_CALIB_CAMERA_CAMERA_GEOMETRY_HPP

#include <iostream>

#include "prototype/core.hpp"
#include "prototype/vision/camera/pinhole.hpp"

namespace prototype {
/**
 * @addtogroup calibration
 * @{
 */

/**
 * Camera geometry
 */
template <typename CM, typename DM>
struct camera_geometry_t {
  int camera_index = 0;
  CM camera_model;
  DM distortion_model;

  camera_geometry_t(const CM &camera_model_,
                    const DM &distortion_model_);
  ~camera_geometry_t();
};

/**
 * Project point to image plane in pixels
 *
 * @param[in] cam Camera geometry
 * @param[in] point Point
 * @returns Point to image plane projection in pixel coordinates
 */
template <typename CM, typename DM>
vec2_t camera_geometry_project( const camera_geometry_t<CM, DM>
    &cam, const vec3_t &point);

/** @} group calibration */
} //  namespace prototype
#include "camera_geometry_impl.hpp"
#endif // PROTOTYPE_CALIB_CAMERA_CAMERA_GEOMETRY_HPP
