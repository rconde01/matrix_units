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

    [[nodiscard]] EigenStorage operator+(const EigenStorage& other) const {
        return EigenStorage{(matrix_ + other.matrix_).eval()};
    }

    [[nodiscard]] EigenStorage operator-(const EigenStorage& other) const {
        return EigenStorage{(matrix_ - other.matrix_).eval()};
    }

    [[nodiscard]] EigenStorage operator-() const {
        return EigenStorage{(-matrix_).eval()};
    }

    [[nodiscard]] EigenStorage operator*(Scalar_ s) const {
        return EigenStorage{(matrix_ * s).eval()};
    }

    friend EigenStorage operator*(Scalar_ s, const EigenStorage& m) {
        return m * s;
    }

    [[nodiscard]] EigenStorage operator/(Scalar_ s) const {
        return EigenStorage{(matrix_ / s).eval()};
    }

    template <std::size_t OtherCols>
    [[nodiscard]] EigenStorage<Scalar_, Rows_, OtherCols>
    multiply(const EigenStorage<Scalar_, Cols_, OtherCols>& other) const {
        return EigenStorage<Scalar_, Rows_, OtherCols>{(matrix_ * other.eigen()).eval()};
    }

    [[nodiscard]] EigenStorage<Scalar_, Cols_, Rows_> transpose() const {
        return EigenStorage<Scalar_, Cols_, Rows_>{matrix_.transpose().eval()};
    }

    [[nodiscard]] Scalar_ squaredNorm() const
        requires (Cols_ == 1)
    {
        return matrix_.squaredNorm();
    }

    [[nodiscard]] Scalar_ dot(const EigenStorage& other) const
        requires (Cols_ == 1)
    {
        return matrix_.dot(other.matrix_);
    }

    [[nodiscard]] static EigenStorage identity()
        requires (Rows_ == Cols_)
    {
        return EigenStorage{EigenMatrix::Identity()};
    }

    template <std::size_t StartRow, std::size_t StartCol, std::size_t BlockRows, std::size_t BlockCols>
        requires (StartRow + BlockRows <= Rows_) && (StartCol + BlockCols <= Cols_)
    [[nodiscard]] EigenStorage<Scalar_, BlockRows, BlockCols> block() const {
        return EigenStorage<Scalar_, BlockRows, BlockCols>{
            matrix_.template block<static_cast<int>(BlockRows), static_cast<int>(BlockCols)>(
                static_cast<int>(StartRow), static_cast<int>(StartCol)).eval()};
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
