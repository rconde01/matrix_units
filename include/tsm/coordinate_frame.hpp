#pragma once

#include <type_traits>

namespace tsm {

struct CoordinateFrameBase {};

template <bool IsMoving = false>
struct CoordinateFrame : CoordinateFrameBase {
    static constexpr bool is_moving = IsMoving;
};

template <typename T>
concept IsCoordinateFrame = std::is_base_of_v<CoordinateFrameBase, T>;

} // namespace tsm
