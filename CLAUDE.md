# Type-Safe Matrix Library (TSM)

## Project Overview
A C++20 header-only library for compile-time type-safe linear algebra with physical units.
Uses mp-units for physical quantities and provides compile-time safety for matrix operations
involving typed rows, columns, coordinate frames, and isometries.

## Key Design Decisions
- **Header-only**: All code in `include/tsm/`
- **C++20 concepts** for constraining matrix operations at compile time
- **mp-units integration** for physical unit safety (meters, seconds, radians, etc.)
- **MatrixTag**: Core type that encodes row/column types as template parameters
- **IndexType**: Associates dimension indices with physical unit types
- **CoordinateFrame**: Named frames (e.g., WorldFrame, BodyFrame) for transformation safety
- **Isometry**: Type-safe rigid body transformations between coordinate frames
- **Eigen backend**: Optional `backends/eigen.hpp` wraps Eigen::Matrix with type safety

## Build
mkdir build && cd build
cmake .. && make
Requires: CMake 3.20+, C++20 compiler, mp-units, gsl-lite

## Architecture
- `include/tsm/type_safe_matrix.hpp` - Core TypeSafeMatrix class
- `include/tsm/matrix_tag.hpp` - MatrixTag encoding row/column types
- `include/tsm/index_type.hpp` - IndexType mapping dimensions to units
- `include/tsm/coordinate_frame.hpp` - CoordinateFrame concept and helpers
- `include/tsm/isometry.hpp` - Isometry (rotation + translation) between frames
- `include/tsm/detail/` - Implementation details (concepts, storage, type_list)
- `include/tsm/backends/eigen.hpp` - Eigen integration
- `include/tsm/tsm.hpp` - Convenience header (includes everything)
- `tests/tests.cpp` - Compile-time and runtime tests
- `examples/vehicle_tracking.cpp` - Vehicle tracking example with GPS/IMU fusion

## Status
- Core library implemented and compiling
- Tests pass (compile-time static_asserts + runtime checks)
- Vehicle tracking example demonstrates real-world usage
- Needs: CI setup, package manager support (vcpkg/conan), documentation
