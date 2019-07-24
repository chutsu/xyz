#ifndef PROTOTYPE_CORE_MATH_IMPL_HPP
#define PROTOTYPE_CORE_MATH_IMPL_HPP

namespace proto {

template <typename T>
T lerp(const T &a, const T &b, const double t) {
  return a * (1.0 - t) + b * t;
}

} //  namespace proto
#endif // PROTOTYPE_CORE_MATH_HPP