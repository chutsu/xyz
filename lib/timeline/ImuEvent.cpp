#include "ImuEvent.hpp"

namespace xyz {

ImuEvent::ImuEvent(const timestamp_t ts_,
                   const Vec3 &acc_,
                   const Vec3 &gyr_)
    : TimelineEvent{"ImuEvent", ts_}, acc{acc_}, gyr{gyr_} {}

} // namespace xyz
