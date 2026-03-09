#pragma once

#include <Eigen/Core>
#include <cstddef>

namespace tsm::detail {

template <typename Scalar_, std::size_t Rows_, std::size_t Cols_>
class EigenStorage {
public:
    using Scalar = Scalar_;
    static constexpr std::size_t Rows = Rows_;
    static constexpr std::size_t Cols = Cols_;
    using EigenMatrix = Eigen::Matrix<Scalar_, static_cast<int>(Rows_), static_cast<int>(Cols_)>;

private:
    EigenMatrix matrix_ = EigenMatrix::Zero();

public:
    EigenStorage() = default;

    explicit EigenStorage(const EigenMatrix& m) : matrix_(m) {}
    explicit EigenStorage(EigenMatrix&& m) : matrix_(std::move(m)) {}

    [[nodiscard]] Scalar_& operator()(std::size_t row, std::size_t col) {
        return matrix_(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
    }

    [[nodiscard]] const Scalar_& operator()(std::size_t row, std::size_t col) const {
        return matrix_(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
    }

    [[nodiscard]] const EigenMatrix& eigen() const { return matrix_; }
    [[nodiscard]] EigenMatrix& eigen() { return matrix_; }

    [[nodiscard]] bool operator==(const EigenStorage& other) const {
        return matrix_ == other.matrix_;
    }
};

struct EigenStoragePolicy {
    template <typename Scalar, std::size_t Rows, std::size_t Cols>
    using type = EigenStorage<Scalar, Rows, Cols>;
};

} // namespace tsm::detail

namespace tsm {
    using EigenStoragePolicy = detail::EigenStoragePolicy;
}
