#pragma once

#include "coordinate_frame.hpp"

#include <mp-units/systems/si.h>
#include <mp-units/framework.h>
#include <type_traits>

namespace tsm {

struct CartesianXAxis {};
struct CartesianYAxis {};
struct CartesianZAxis {};
struct GenericAxis {};

struct IndexTypeBase {};

template <IsCoordinateFrame Frame, typename Axis, auto Unit>
struct IndexType : IndexTypeBase {
    using frame_type = Frame;
    using axis_type = Axis;
    static constexpr auto unit = Unit;
};

template <typename T>
concept IsIndexType = std::is_base_of_v<IndexTypeBase, T>;

struct NoIdx : IndexTypeBase {
    using frame_type = void;
    using axis_type = void;
    static constexpr auto unit = mp_units::one;
};

template <IsCoordinateFrame Frame, typename Axis>
struct DistanceIdx : IndexType<Frame, Axis, mp_units::si::metre> {};

template <IsCoordinateFrame Frame, typename Axis>
struct VelocityIdx : IndexType<Frame, Axis, mp_units::si::metre / mp_units::si::second> {};

template <IsCoordinateFrame Frame, typename Axis>
struct AccelerationIdx : IndexType<Frame, Axis, mp_units::si::metre / (mp_units::si::second * mp_units::si::second)> {};

template <IsCoordinateFrame Frame>
struct AngleIdx : IndexType<Frame, GenericAxis, mp_units::si::radian> {};

template <IsCoordinateFrame Frame, typename Axis, auto Unit>
struct GenericIdx : IndexType<Frame, Axis, Unit> {};

} // namespace tsm
