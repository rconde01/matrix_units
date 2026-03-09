// TypeSafeMatrix test suite
// Uses static_assert for compile-time checks + runtime assertions for values.

#include <tsm/tsm.hpp>

#include <mp-units/systems/si.h>
#include <mp-units/framework.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <type_traits>

using namespace mp_units;
using namespace mp_units::si::unit_symbols;

struct VehicleFrame : tsm::CoordinateFrame<false> {};
struct SensorFrame  : tsm::CoordinateFrame<false> {};
struct OdomFrame    : tsm::CoordinateFrame<false> {};

struct DX   : tsm::DistanceIdx<VehicleFrame, tsm::CartesianXAxis> {};
struct DY   : tsm::DistanceIdx<VehicleFrame, tsm::CartesianYAxis> {};
struct DZ   : tsm::DistanceIdx<VehicleFrame, tsm::CartesianZAxis> {};

struct VX   : tsm::VelocityIdx<VehicleFrame, tsm::CartesianXAxis> {};
struct VY   : tsm::VelocityIdx<VehicleFrame, tsm::CartesianYAxis> {};

struct DX_S : tsm::DistanceIdx<SensorFrame, tsm::CartesianXAxis> {};
struct DY_S : tsm::DistanceIdx<SensorFrame, tsm::CartesianYAxis> {};
struct DZ_S : tsm::DistanceIdx<SensorFrame, tsm::CartesianZAxis> {};

struct VX_S : tsm::VelocityIdx<SensorFrame, tsm::CartesianXAxis> {};
struct VY_S : tsm::VelocityIdx<SensorFrame, tsm::CartesianYAxis> {};

struct MeasDR    : tsm::GenericIdx<SensorFrame, tsm::GenericAxis, mp_units::si::metre> {};
struct MeasVR    : tsm::GenericIdx<SensorFrame, tsm::GenericAxis, mp_units::si::metre / mp_units::si::second> {};
struct MeasAngle : tsm::GenericIdx<SensorFrame, tsm::GenericAxis, mp_units::si::radian> {};

using PosVec3Vehicle = tsm::TypeSafeVector<double,
    tsm::detail::TypeList<DX, DY, DZ>, tsm::VectorTag>;

using PosVec2Vehicle = tsm::TypeSafeVector<double,
    tsm::detail::TypeList<DX, DY>, tsm::VectorTag>;

using PosVec3Sensor = tsm::TypeSafeVector<double,
    tsm::detail::TypeList<DX_S, DY_S, DZ_S>, tsm::VectorTag>;

using DeltaPosVec3Vehicle = tsm::TypeSafeVector<double,
    tsm::detail::TypeList<DX, DY, DZ>, tsm::DeltaVectorTag>;

using DeltaPosVec2Vehicle = tsm::TypeSafeVector<double,
    tsm::detail::TypeList<DX, DY>, tsm::DeltaVectorTag>;

using DeltaPosVec3Sensor = tsm::TypeSafeVector<double,
    tsm::detail::TypeList<DX_S, DY_S, DZ_S>, tsm::DeltaVectorTag>;

using VelVec2Vehicle = tsm::TypeSafeVector<double,
    tsm::detail::TypeList<VX, VY>, tsm::VectorTag>;

using CovPos3Vehicle = tsm::TypeSafeMatrix<double,
    tsm::detail::TypeList<DX, DY, DZ>,
    tsm::detail::TypeList<DX, DY, DZ>,
    tsm::CovarianceMatrixTag>;

using StateVec = tsm::TypeSafeVector<double,
    tsm::detail::TypeList<DX, DY, VX, VY>, tsm::VectorTag>;

using StateCov = tsm::TypeSafeMatrix<double,
    tsm::detail::TypeList<DX, DY, VX, VY>,
    tsm::detail::TypeList<DX, DY, VX, VY>,
    tsm::CovarianceMatrixTag>;

using MeasJacobian = tsm::TypeSafeMatrix<double,
    tsm::detail::TypeList<MeasDR, MeasVR, MeasAngle>,
    tsm::detail::TypeList<DX, DY, DZ>,
    tsm::JacobianMatrixTag>;

using SensorToVehicleJac = tsm::TypeSafeMatrix<double,
    tsm::detail::TypeList<DX_S, DY_S>,
    tsm::detail::TypeList<DX, DY>,
    tsm::JacobianMatrixTag>;

using CovPos2Sensor = tsm::TypeSafeMatrix<double,
    tsm::detail::TypeList<DX_S, DY_S>,
    tsm::detail::TypeList<DX_S, DY_S>,
    tsm::CovarianceMatrixTag>;

using CovPos2Vehicle = tsm::TypeSafeMatrix<double,
    tsm::detail::TypeList<DX, DY>,
    tsm::detail::TypeList<DX, DY>,
    tsm::CovarianceMatrixTag>;

constexpr double eps = 1e-10;

bool approxEq(double a, double b) {
    return std::abs(a - b) < eps;
}

void test_vector_construction_and_access() {
    std::printf("  vector construction and access... ");

    PosVec3Vehicle pos(
        tsm::wrapCoeff<DX>(1.0),
        tsm::wrapCoeff<DY>(2.0),
        tsm::wrapCoeff<DZ>(3.0)
    );

    assert(pos.at<DX>() == 1.0);
    assert(pos.at<DY>() == 2.0);
    assert(pos.at<DZ>() == 3.0);

    PosVec3Vehicle pos2(PosVec3Vehicle::Unchecked{}, 4.0, 5.0, 6.0);
    assert(pos2.at<DX>() == 4.0);
    assert(pos2.at<DY>() == 5.0);
    assert(pos2.at<DZ>() == 6.0);

    std::printf("OK\n");
}

void test_entry_transfer() {
    std::printf("  entry transfer... ");

    PosVec3Vehicle src(PosVec3Vehicle::Unchecked{}, 10.0, 20.0, 30.0);
    PosVec3Vehicle dst;

    dst.assignEntry(src.entry<DX>());
    dst.assignEntry(src.entry<DY>());
    dst.assignEntry(src.entry<DZ>());

    assert(dst.at<DX>() == 10.0);
    assert(dst.at<DY>() == 20.0);
    assert(dst.at<DZ>() == 30.0);

    std::printf("OK\n");
}

void test_coeffSi() {
    std::printf("  coeffSi unit inference... ");

    PosVec3Vehicle pos(PosVec3Vehicle::Unchecked{}, 5.0, 0.0, 0.0);

    auto q = pos.coeffSi<DX>();
    auto val = q.numerical_value_ref_in(si::metre);
    assert(approxEq(val, 5.0));

    std::printf("OK\n");
}

void test_delta_vector_addition() {
    std::printf("  delta vector addition... ");

    DeltaPosVec3Vehicle d1(DeltaPosVec3Vehicle::Unchecked{}, 1.0, 2.0, 3.0);
    DeltaPosVec3Vehicle d2(DeltaPosVec3Vehicle::Unchecked{}, 4.0, 5.0, 6.0);

    auto sum = d1 + d2;
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(sum)>::tag_type, tsm::DeltaVectorTag>);
    assert(approxEq(sum.at<DX>(), 5.0));
    assert(approxEq(sum.at<DY>(), 7.0));
    assert(approxEq(sum.at<DZ>(), 9.0));

    std::printf("OK\n");
}

void test_point_plus_delta() {
    std::printf("  point + delta = point... ");

    PosVec3Vehicle pos(PosVec3Vehicle::Unchecked{}, 1.0, 0.0, 0.0);
    DeltaPosVec3Vehicle delta(DeltaPosVec3Vehicle::Unchecked{}, 0.5, 1.0, 1.5);

    auto result = pos + delta;
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(result)>::tag_type, tsm::VectorTag>);
    assert(approxEq(result.at<DX>(), 1.5));
    assert(approxEq(result.at<DY>(), 1.0));
    assert(approxEq(result.at<DZ>(), 1.5));

    std::printf("OK\n");
}

void test_point_minus_point() {
    std::printf("  point - point = delta... ");

    PosVec3Vehicle p1(PosVec3Vehicle::Unchecked{}, 5.0, 3.0, 1.0);
    PosVec3Vehicle p2(PosVec3Vehicle::Unchecked{}, 2.0, 1.0, 0.0);

    auto diff = p1 - p2;
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(diff)>::tag_type, tsm::DeltaVectorTag>);
    assert(approxEq(diff.at<DX>(), 3.0));
    assert(approxEq(diff.at<DY>(), 2.0));
    assert(approxEq(diff.at<DZ>(), 1.0));

    std::printf("OK\n");
}

void test_scalar_multiplication() {
    std::printf("  scalar multiplication... ");

    DeltaPosVec3Vehicle d(DeltaPosVec3Vehicle::Unchecked{}, 1.0, 2.0, 3.0);
    auto scaled = d * 2.0;
    assert(approxEq(scaled.at<DX>(), 2.0));
    assert(approxEq(scaled.at<DY>(), 4.0));
    assert(approxEq(scaled.at<DZ>(), 6.0));

    auto scaled2 = 3.0 * d;
    assert(approxEq(scaled2.at<DX>(), 3.0));

    std::printf("OK\n");
}

void test_matrix_construction() {
    std::printf("  matrix construction... ");

    CovPos2Vehicle cov(
        tsm::wrapCoeff<DX, DX>(4.0), tsm::wrapCoeff<DX, DY>(1.0),
        tsm::wrapCoeff<DY, DX>(1.0), tsm::wrapCoeff<DY, DY>(9.0)
    );

    assert(approxEq(cov.at<DX, DX>(), 4.0));
    assert(approxEq(cov.at<DX, DY>(), 1.0));
    assert(approxEq(cov.at<DY, DX>(), 1.0));
    assert(approxEq(cov.at<DY, DY>(), 9.0));

    std::printf("OK\n");
}

void test_matrix_multiplication_jac_cov_jacT() {
    std::printf("  J * Cov * J^T... ");

    SensorToVehicleJac jac(SensorToVehicleJac::Unchecked{},
        1.0, 0.0,
        0.0, 1.0
    );

    CovPos2Vehicle cov_vehicle(CovPos2Vehicle::Unchecked{},
        3.1, 0.0,
        0.0, 2.4
    );

    auto jac_cov = jac * cov_vehicle;
    auto cov_sensor = jac_cov * jac.transpose();

    static_assert(std::is_same_v<std::remove_cvref_t<decltype(cov_sensor)>::tag_type, tsm::CovarianceMatrixTag>);
    assert(approxEq(cov_sensor.at<DX_S, DX_S>(), 3.1));
    assert(approxEq(cov_sensor.at<DY_S, DY_S>(), 2.4));
    assert(approxEq(cov_sensor.at<DX_S, DY_S>(), 0.0));

    std::printf("OK\n");
}

void test_transpose() {
    std::printf("  transpose... ");

    SensorToVehicleJac jac(SensorToVehicleJac::Unchecked{},
        1.0, 2.0,
        3.0, 4.0
    );

    auto jac_t = jac.transpose();
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(jac_t)>::tag_type, tsm::TransposedJacobianMatrixTag>);

    assert(approxEq(jac_t.at<DX, DX_S>(), 1.0));
    assert(approxEq(jac_t.at<DX, DY_S>(), 3.0));
    assert(approxEq(jac_t.at<DY, DX_S>(), 2.0));
    assert(approxEq(jac_t.at<DY, DY_S>(), 4.0));

    std::printf("OK\n");
}

void test_head_and_tail() {
    std::printf("  head and tail... ");

    PosVec3Vehicle pos(PosVec3Vehicle::Unchecked{}, 1.0, 2.0, 3.0);

    auto h = pos.head<2>();
    static_assert(h.num_rows == 2);
    assert(approxEq(h.at<DX>(), 1.0));
    assert(approxEq(h.at<DY>(), 2.0));

    auto t = pos.tail<2>();
    static_assert(t.num_rows == 2);
    assert(approxEq(t.at<DY>(), 2.0));
    assert(approxEq(t.at<DZ>(), 3.0));

    std::printf("OK\n");
}

void test_identity_and_zero() {
    std::printf("  identity and zero... ");

    auto id = CovPos2Vehicle::identity();
    assert(approxEq(id.at<DX, DX>(), 1.0));
    assert(approxEq(id.at<DX, DY>(), 0.0));
    assert(approxEq(id.at<DY, DX>(), 0.0));
    assert(approxEq(id.at<DY, DY>(), 1.0));

    auto z = PosVec3Vehicle::zero();
    assert(approxEq(z.at<DX>(), 0.0));

    std::printf("OK\n");
}

void test_squared_norm() {
    std::printf("  squared norm... ");

    DeltaPosVec3Vehicle d(DeltaPosVec3Vehicle::Unchecked{}, 3.0, 4.0, 0.0);
    assert(approxEq(d.squaredNorm(), 25.0));

    std::printf("OK\n");
}

void test_dot_product() {
    std::printf("  dot product... ");

    DeltaPosVec3Vehicle a(DeltaPosVec3Vehicle::Unchecked{}, 1.0, 0.0, 0.0);
    DeltaPosVec3Vehicle b(DeltaPosVec3Vehicle::Unchecked{}, 0.0, 1.0, 0.0);

    assert(approxEq(a.dot(b), 0.0));

    DeltaPosVec3Vehicle c(DeltaPosVec3Vehicle::Unchecked{}, 2.0, 3.0, 4.0);
    DeltaPosVec3Vehicle d(DeltaPosVec3Vehicle::Unchecked{}, 1.0, 1.0, 1.0);
    assert(approxEq(c.dot(d), 9.0));

    std::printf("OK\n");
}

using VehicleIdxList3 = tsm::detail::TypeList<DX, DY, DZ>;
using SensorIdxList3 = tsm::detail::TypeList<DX_S, DY_S, DZ_S>;

struct DX_O : tsm::DistanceIdx<OdomFrame, tsm::CartesianXAxis> {};
struct DY_O : tsm::DistanceIdx<OdomFrame, tsm::CartesianYAxis> {};
struct DZ_O : tsm::DistanceIdx<OdomFrame, tsm::CartesianZAxis> {};
using OdomIdxList3 = tsm::detail::TypeList<DX_O, DY_O, DZ_O>;
using PosVec3Odom = tsm::TypeSafeVector<double, OdomIdxList3, tsm::VectorTag>;

void test_isometry_position_transform() {
    std::printf("  isometry position transform... ");

    using Iso = tsm::Isometry<double, VehicleIdxList3, SensorIdxList3>;

    typename Iso::RotationMatrix rot(typename Iso::RotationMatrix::Unchecked{},
        1, 0, 0,
        0, 1, 0,
        0, 0, 1);
    typename Iso::TranslationVector trans(typename Iso::TranslationVector::Unchecked{},
        10.0, 0.0, 0.0);

    Iso vehicle_T_sensor(rot, trans);

    PosVec3Sensor sensor_pos(PosVec3Sensor::Unchecked{}, 1.0, 2.0, 3.0);

    auto vehicle_pos = vehicle_T_sensor * sensor_pos;
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(vehicle_pos)>::tag_type, tsm::VectorTag>);

    assert(approxEq(vehicle_pos.at<DX>(), 11.0));
    assert(approxEq(vehicle_pos.at<DY>(), 2.0));
    assert(approxEq(vehicle_pos.at<DZ>(), 3.0));

    std::printf("OK\n");
}

void test_isometry_delta_transform() {
    std::printf("  isometry delta transform... ");

    using Iso = tsm::Isometry<double, VehicleIdxList3, SensorIdxList3>;

    typename Iso::RotationMatrix rot(typename Iso::RotationMatrix::Unchecked{},
        1, 0, 0,
        0, 1, 0,
        0, 0, 1);
    typename Iso::TranslationVector trans(typename Iso::TranslationVector::Unchecked{},
        10.0, 20.0, 30.0);

    Iso vehicle_T_sensor(rot, trans);

    DeltaPosVec3Sensor delta(DeltaPosVec3Sensor::Unchecked{}, 1.0, 2.0, 3.0);

    auto vehicle_delta = vehicle_T_sensor * delta;
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(vehicle_delta)>::tag_type, tsm::DeltaVectorTag>);

    assert(approxEq(vehicle_delta.at<DX>(), 1.0));
    assert(approxEq(vehicle_delta.at<DY>(), 2.0));
    assert(approxEq(vehicle_delta.at<DZ>(), 3.0));

    std::printf("OK\n");
}

void test_isometry_composition() {
    std::printf("  isometry composition... ");

    using IsoVS = tsm::Isometry<double, VehicleIdxList3, SensorIdxList3>;
    using IsoSO = tsm::Isometry<double, SensorIdxList3, OdomIdxList3>;

    typename IsoVS::RotationMatrix rot1(typename IsoVS::RotationMatrix::Unchecked{},
        1, 0, 0, 0, 1, 0, 0, 0, 1);
    typename IsoVS::TranslationVector trans1(typename IsoVS::TranslationVector::Unchecked{},
        10.0, 0.0, 0.0);
    IsoVS v_T_s(rot1, trans1);

    typename IsoSO::RotationMatrix rot2(typename IsoSO::RotationMatrix::Unchecked{},
        1, 0, 0, 0, 1, 0, 0, 0, 1);
    typename IsoSO::TranslationVector trans2(typename IsoSO::TranslationVector::Unchecked{},
        0.0, 5.0, 0.0);
    IsoSO s_T_o(rot2, trans2);

    auto v_T_o = v_T_s * s_T_o;

    PosVec3Odom odom_pos(PosVec3Odom::Unchecked{}, 0.0, 0.0, 0.0);
    auto vehicle_pos = v_T_o * odom_pos;

    assert(approxEq(vehicle_pos.at<DX>(), 10.0));
    assert(approxEq(vehicle_pos.at<DY>(), 5.0));
    assert(approxEq(vehicle_pos.at<DZ>(), 0.0));

    std::printf("OK\n");
}

void test_isometry_inverse() {
    std::printf("  isometry inverse... ");

    using Iso = tsm::Isometry<double, VehicleIdxList3, SensorIdxList3>;

    typename Iso::RotationMatrix rot(typename Iso::RotationMatrix::Unchecked{},
        1, 0, 0, 0, 1, 0, 0, 0, 1);
    typename Iso::TranslationVector trans(typename Iso::TranslationVector::Unchecked{},
        10.0, 20.0, 30.0);

    Iso v_T_s(rot, trans);
    auto s_T_v = v_T_s.inverse();

    PosVec3Vehicle vehicle_pos(PosVec3Vehicle::Unchecked{}, 10.0, 20.0, 30.0);
    auto sensor_pos = s_T_v * vehicle_pos;

    assert(approxEq(sensor_pos.at<DX_S>(), 0.0));
    assert(approxEq(sensor_pos.at<DY_S>(), 0.0));
    assert(approxEq(sensor_pos.at<DZ_S>(), 0.0));

    std::printf("OK\n");
}

// =========================================================================
// Compile-time negative tests: verify that invalid operations are rejected
// =========================================================================
// GCC 13 has a bug where requires-expressions containing constrained template
// operators emit hard errors instead of evaluating to false. Use void_t-based
// SFINAE traits as a portable workaround.

namespace neg_tests {

template <typename A, typename B, typename = void>
struct can_add : std::false_type {};
template <typename A, typename B>
struct can_add<A, B, std::void_t<decltype(std::declval<A>() + std::declval<B>())>> : std::true_type {};

template <typename A, typename B, typename = void>
struct can_sub : std::false_type {};
template <typename A, typename B>
struct can_sub<A, B, std::void_t<decltype(std::declval<A>() - std::declval<B>())>> : std::true_type {};

template <typename A, typename B, typename = void>
struct can_mul : std::false_type {};
template <typename A, typename B>
struct can_mul<A, B, std::void_t<decltype(std::declval<A>() * std::declval<B>())>> : std::true_type {};

template <typename V, typename Idx, typename = void>
struct can_at : std::false_type {};
template <typename V, typename Idx>
struct can_at<V, Idx, std::void_t<decltype(std::declval<V>().template at<Idx>())>> : std::true_type {};

template <typename V, typename RowIdx, typename ColIdx, typename = void>
struct can_at2 : std::false_type {};
template <typename V, typename RowIdx, typename ColIdx>
struct can_at2<V, RowIdx, ColIdx, std::void_t<decltype(std::declval<V>().template at<RowIdx, ColIdx>())>> : std::true_type {};

template <typename V, typename E, typename = void>
struct can_assign_entry : std::false_type {};
template <typename V, typename E>
struct can_assign_entry<V, E, std::void_t<decltype(std::declval<V>().assignEntry(std::declval<E>()))>> : std::true_type {};

template <typename A, typename B, typename = void>
struct can_dot : std::false_type {};
template <typename A, typename B>
struct can_dot<A, B, std::void_t<decltype(std::declval<A>().dot(std::declval<B>()))>> : std::true_type {};

} // namespace neg_tests

// --- Vector addition / subtraction constraints ---

// Cannot add two position vectors (point + point is meaningless)
static_assert(!neg_tests::can_add<PosVec3Vehicle, PosVec3Vehicle>::value,
    "Adding two position vectors (VectorTag + VectorTag) must not compile");

// Cannot add vectors from different coordinate frames
static_assert(!neg_tests::can_add<DeltaPosVec3Vehicle, DeltaPosVec3Sensor>::value,
    "Adding vectors from different frames must not compile");

static_assert(!neg_tests::can_add<PosVec3Vehicle, DeltaPosVec3Sensor>::value,
    "Adding point + delta from different frames must not compile");

// Cannot subtract vectors from different frames
static_assert(!neg_tests::can_sub<PosVec3Vehicle, PosVec3Sensor>::value,
    "Subtracting vectors from different frames must not compile");

// Cannot add vectors with different dimensions
static_assert(!neg_tests::can_add<PosVec3Vehicle, PosVec2Vehicle>::value,
    "Adding vectors with different dimensions must not compile");

// Cannot add Jacobian to Covariance (different tag exponents)
static_assert(!neg_tests::can_add<SensorToVehicleJac, CovPos2Vehicle>::value,
    "Adding Jacobian + Covariance must not compile");

// --- Matrix multiplication constraints ---

// Cannot multiply matrices with mismatched inner index lists
static_assert(!neg_tests::can_mul<SensorToVehicleJac, CovPos3Vehicle>::value,
    "Multiplying matrices with mismatched inner dimensions must not compile");

// Cannot multiply Covariance * Covariance (col_exponent + row_exponent != 0)
static_assert(!neg_tests::can_mul<CovPos2Vehicle, CovPos2Vehicle>::value,
    "Multiplying Covariance * Covariance must not compile");

// Cannot multiply Covariance * Jacobian (wrong exponent pairing)
static_assert(!neg_tests::can_mul<CovPos2Vehicle, SensorToVehicleJac>::value,
    "Multiplying Covariance * Jacobian must not compile");

// --- Index access constraints ---

// Cannot access a sensor-frame index on a vehicle-frame vector
static_assert(!neg_tests::can_at<PosVec3Vehicle, DX_S>::value,
    "Accessing sensor index on vehicle vector must not compile");

// Cannot access a vehicle-frame index on a sensor-frame vector
static_assert(!neg_tests::can_at<PosVec3Sensor, DX>::value,
    "Accessing vehicle index on sensor vector must not compile");

// Cannot access velocity index on a position vector
static_assert(!neg_tests::can_at<PosVec3Vehicle, VX>::value,
    "Accessing velocity index on position vector must not compile");

// Cannot access wrong column index on a matrix
static_assert(!neg_tests::can_at2<CovPos2Vehicle, DX, DX_S>::value,
    "Accessing wrong column index on matrix must not compile");

// --- Entry type safety ---

// Cannot assign a sensor-frame entry to a vehicle-frame vector
static_assert(!neg_tests::can_assign_entry<PosVec3Vehicle, tsm::Entry<DX_S, double>>::value,
    "Assigning sensor entry to vehicle vector must not compile");

// --- Isometry constraints ---

using IsoVehicleFromSensor = tsm::Isometry<double, VehicleIdxList3, SensorIdxList3>;
using IsoSensorFromOdom = tsm::Isometry<double, SensorIdxList3, OdomIdxList3>;

// Cannot apply vehicle<-sensor isometry to a vehicle-frame position
static_assert(!neg_tests::can_mul<IsoVehicleFromSensor, PosVec3Vehicle>::value,
    "Applying vehicle<-sensor isometry to vehicle position must not compile");

// Cannot apply vehicle<-sensor isometry to an odom-frame position
static_assert(!neg_tests::can_mul<IsoVehicleFromSensor, PosVec3Odom>::value,
    "Applying vehicle<-sensor isometry to odom position must not compile");

// Cannot compose isometries with mismatched intermediate frames
// vehicle<-sensor * vehicle<-sensor (sensor != vehicle)
static_assert(!neg_tests::can_mul<IsoVehicleFromSensor, IsoVehicleFromSensor>::value,
    "Composing isometries with mismatched intermediate frames must not compile");

// Cannot compose sensor<-odom * vehicle<-sensor (wrong order)
static_assert(!neg_tests::can_mul<IsoSensorFromOdom, IsoVehicleFromSensor>::value,
    "Composing isometries in wrong order must not compile");

// --- Dot product / norm constraints ---

// Cannot dot product vectors from different frames
static_assert(!neg_tests::can_dot<DeltaPosVec3Vehicle, DeltaPosVec3Sensor>::value,
    "Dot product of vectors from different frames must not compile");

// Cannot take dot product of matrices (not vectors)
static_assert(!neg_tests::can_dot<CovPos2Vehicle, CovPos2Vehicle>::value,
    "Dot product on matrices must not compile");

// =========================================================================

int main() {
    std::printf("TypeSafeMatrix tests:\n");

    test_vector_construction_and_access();
    test_entry_transfer();
    test_coeffSi();
    test_delta_vector_addition();
    test_point_plus_delta();
    test_point_minus_point();
    test_scalar_multiplication();
    test_matrix_construction();
    test_matrix_multiplication_jac_cov_jacT();
    test_transpose();
    test_head_and_tail();
    test_identity_and_zero();
    test_squared_norm();
    test_dot_product();
    test_isometry_position_transform();
    test_isometry_delta_transform();
    test_isometry_composition();
    test_isometry_inverse();

    std::printf("\nAll tests passed!\n");
    return 0;
}
