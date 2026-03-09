cmake_minimum_required(VERSION 3.20)
project(typesafe-matrix VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(gsl-lite CONFIG REQUIRED)
find_package(mp-units CONFIG REQUIRED)

# Header-only library
add_library(tsm INTERFACE)
target_include_directories(tsm INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tsm INTERFACE mp-units::mp-units)
target_compile_features(tsm INTERFACE cxx_std_20)

# Tests
option(TSM_BUILD_TESTS "Build tests" ON)
if(TSM_BUILD_TESTS)
    enable_testing()
    add_executable(tsm_tests tests/tests.cpp)
    target_link_libraries(tsm_tests PRIVATE tsm)
    add_test(NAME tsm_tests COMMAND tsm_tests)
endif()

# Examples
option(TSM_BUILD_EXAMPLES "Build examples" ON)
if(TSM_BUILD_EXAMPLES)
    add_executable(tsm_example examples/vehicle_tracking.cpp)
    target_link_libraries(tsm_example PRIVATE tsm)
endif()
