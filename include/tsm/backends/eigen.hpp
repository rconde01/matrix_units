#pragma once

#include <Eigen/Core>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace tsm::detail {

template <typename EigenType_>
class EigenStorage {
    EigenType_ data_;

public:
    using Scalar = typename EigenType_::Scalar;
    using EigenType = EigenType_;
    static constexpr std::size_t Rows = static_cast<std::size_t>(EigenType_::RowsAtCompileTime);
    static constexpr std::size_t Cols = static_cast<std::size_t>(EigenType_::ColsAtCompileTime);

    EigenStorage()
        requires requires { EigenType_::Zero(); }
        : data_(EigenType_::Zero()) {}

    explicit EigenStorage(const EigenType_& e) : data_(e) {}
    explicit EigenStorage(EigenType_&& e) : data_(std::move(e)) {}

    // Converting constructor: evaluates expression-typed storage into concrete storage
    template <typename OtherEigenType>
        requires (!std::same_as<std::remove_const_t<EigenType_>,
                                std::remove_const_t<OtherEigenType>>)
    EigenStorage(const EigenStorage<OtherEigenType>& other) : data_(other.eigen()) {}

    [[nodiscard]] Scalar& operator()(std::size_t row, std::size_t col) {
        return data_(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
    }

    [[nodiscard]] const Scalar& operator()(std::size_t row, std::size_t col) const {
        return data_(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
    }

    [[nodiscard]] const EigenType_& eigen() const { return data_; }
    [[nodiscard]] EigenType_& eigen() { return data_; }

    [[nodiscard]] bool operator==(const EigenStorage& other) const {
        return data_ == other.data_;
    }

    // Arithmetic operations return expression-typed EigenStorage (no eager evaluation)

    template <typename OtherType>
    [[nodiscard]] auto operator+(const EigenStorage<OtherType>& other) const {
        return EigenStorage<decltype(data_ + other.eigen())>{data_ + other.eigen()};
    }

    template <typename OtherType>
    [[nodiscard]] auto operator-(const EigenStorage<OtherType>& other) const {
        return EigenStorage<decltype(data_ - other.eigen())>{data_ - other.eigen()};
    }

    [[nodiscard]] auto operator-() const {
        return EigenStorage<decltype(-data_)>{-data_};
    }

    [[nodiscard]] auto operator*(Scalar s) const {
        return EigenStorage<decltype(data_ * s)>{data_ * s};
    }

    friend auto operator*(Scalar s, const EigenStorage& m) {
        return EigenStorage<decltype(s * m.data_)>{s * m.data_};
    }

    [[nodiscard]] auto operator/(Scalar s) const {
        return EigenStorage<decltype(data_ / s)>{data_ / s};
    }

    template <typename OtherType>
    [[nodiscard]] auto multiply(const EigenStorage<OtherType>& other) const {
        return EigenStorage<decltype(data_ * other.eigen())>{data_ * other.eigen()};
    }

    [[nodiscard]] auto transpose() const {
        return EigenStorage<decltype(data_.transpose())>{data_.transpose()};
    }

    [[nodiscard]] Scalar squaredNorm() const
        requires (Cols == 1)
    {
        return data_.squaredNorm();
    }

    template <typename OtherType>
    [[nodiscard]] Scalar dot(const EigenStorage<OtherType>& other) const
        requires (Cols == 1)
    {
        return data_.dot(other.eigen());
    }

    [[nodiscard]] static EigenStorage identity()
        requires (Rows == Cols) && requires { EigenType_::Identity(); }
    {
        return EigenStorage{EigenType_::Identity()};
    }

    template <std::size_t StartRow, std::size_t StartCol, std::size_t BlockRows, std::size_t BlockCols>
        requires (StartRow + BlockRows <= Rows) && (StartCol + BlockCols <= Cols)
    [[nodiscard]] auto block() const {
        auto expr = data_.template block<static_cast<int>(BlockRows), static_cast<int>(BlockCols)>(
            static_cast<int>(StartRow), static_cast<int>(StartCol));
        return EigenStorage<decltype(expr)>{expr};
    }
};

struct EigenStoragePolicy {
    template <typename Scalar, std::size_t Rows, std::size_t Cols>
    using type = EigenStorage<Eigen::Matrix<Scalar, static_cast<int>(Rows), static_cast<int>(Cols)>>;
};

} // namespace tsm::detail

namespace tsm {
    using EigenStoragePolicy = detail::EigenStoragePolicy;
}
