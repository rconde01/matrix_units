// Example: Vehicle tracking with TypeSafeMatrix
//
// Demonstrates a simplified Kalman-filter-style state prediction:
//   x_{k+1} = F * x_k
//   P_{k+1} = F * P_k * F^T + Q

#include <tsm/tsm.hpp>

#include <mp-units/systems/si.h>
#include <cstdio>

using namespace mp_units::si::unit_symbols;

struct EgoFrame : tsm::CoordinateFrame<false> {};

struct PosX : tsm::DistanceIdx<EgoFrame, tsm::CartesianXAxis> {};
struct PosY : tsm::DistanceIdx<EgoFrame, tsm::CartesianYAxis> {};
struct VelX : tsm::VelocityIdx<EgoFrame, tsm::CartesianXAxis> {};
struct VelY : tsm::VelocityIdx<EgoFrame, tsm::CartesianYAxis> {};

using IdxList = tsm::detail::TypeList<PosX, PosY, VelX, VelY>;

using StateVec = tsm::TypeSafeVector<double, IdxList, tsm::VectorTag>;

using StateCov = tsm::TypeSafeMatrix<double, IdxList, IdxList,
                                     tsm::CovarianceMatrixTag>;

using StateTransitionJac = tsm::TypeSafeMatrix<double, IdxList, IdxList,
                                               tsm::JacobianMatrixTag>;

int main() {
    const double dt = 0.1;

    StateVec state(
        tsm::wrapCoeff<PosX>(0.0),
        tsm::wrapCoeff<PosY>(0.0),
        tsm::wrapCoeff<VelX>(10.0),
        tsm::wrapCoeff<VelY>(5.0)
    );

    StateCov cov(StateCov::Unchecked{},
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.0, 0.5
    );

    StateTransitionJac F(StateTransitionJac::Unchecked{},
        1.0, 0.0, dt,  0.0,
        0.0, 1.0, 0.0, dt,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    StateCov Q(StateCov::Unchecked{},
        0.01, 0.0,  0.0,  0.0,
        0.0,  0.01, 0.0,  0.0,
        0.0,  0.0,  0.1,  0.0,
        0.0,  0.0,  0.0,  0.1
    );

    std::printf("Vehicle tracking prediction (constant velocity model)\n");
    std::printf("%-6s  %10s  %10s  %10s  %10s\n",
                "step", "px [m]", "py [m]", "vx [m/s]", "vy [m/s]");
    std::printf("------  ----------  ----------  ----------  ----------\n");

    for (int step = 0; step <= 10; ++step) {
        auto px = state.coeffSi<PosX>();
        auto py = state.coeffSi<PosY>();
        auto vx = state.coeffSi<VelX>();
        auto vy = state.coeffSi<VelY>();

        std::printf("%-6d  %10.3f  %10.3f  %10.3f  %10.3f\n",
                    step,
                    px.numerical_value_ref_in(m),
                    py.numerical_value_ref_in(m),
                    vx.numerical_value_ref_in(m / s),
                    vy.numerical_value_ref_in(m / s));

        state = F * state;
        cov = F * cov * F.transpose() + Q;
    }

    std::printf("\nFinal position uncertainty (std dev):\n");
    std::printf("  sigma_px = %.4f m\n", std::sqrt(cov.at<PosX, PosX>()));
    std::printf("  sigma_py = %.4f m\n", std::sqrt(cov.at<PosY, PosY>()));
    std::printf("  sigma_vx = %.4f m/s\n", std::sqrt(cov.at<VelX, VelX>()));
    std::printf("  sigma_vy = %.4f m/s\n", std::sqrt(cov.at<VelY, VelY>()));

    return 0;
}
