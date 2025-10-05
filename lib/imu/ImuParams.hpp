#pragma once
#include "Core.hpp"

namespace xyz {

struct ImuParams {
  double noise_acc;
  double noise_gyr;
  double noise_ba;
  double noise_bg;
  Vec3 g{0.0, 0.0, 9.81};
};

} // namespace xyz
