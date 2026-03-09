#pragma once

#include "type_list.hpp"

#include <array>
#include <concepts>
#include <cstddef>

namespace tsm::detail {

template <typename S>
concept StorageBackend = requires(S s, const S cs, std::size_t i, std::size_t j) {
    { cs(i, j) } -> std::convertible_to<typename S::Scalar>;
    { s(i, j) } -> std::same_as<typename S::Scalar&>;
    { S::Rows } -> std::convertible_to<std::size_t>;
    { S::Cols } -> std::convertible_to<std::size_t>;
};

template <typename Scalar_, std::size_t Rows_, std::size_t Cols_>
class ArrayStorage {
    std::array<Scalar_, Rows_ * Cols_> data_{};

public:
    using Scalar = Scalar_;
    static constexpr std::size_t Rows = Rows_;
    static constexpr std::size_t Cols = Cols_;

    ArrayStorage() = default;

    [[nodiscard]] Scalar_& operator()(std::size_t row, std::size_t col) {
        return data_[col * Rows_ + row]; // column-major
    }

    [[nodiscard]] const Scalar_& operator()(std::size_t row, std::size_t col) const {
        return data_[col * Rows_ + row];
    }

    [[nodiscard]] bool operator==(const ArrayStorage&) const = default;

    [[nodiscard]] ArrayStorage operator+(const ArrayStorage& other) const {
        ArrayStorage result;
        for (std::size_t i = 0; i < Rows_ * Cols_; ++i)
            result.data_[i] = data_[i] + other.data_[i];
        return result;
    }

    [[nodiscard]] ArrayStorage operator-(const ArrayStorage& other) const {
        ArrayStorage result;
        for (std::size_t i = 0; i < Rows_ * Cols_; ++i)
            result.data_[i] = data_[i] - other.data_[i];
        return result;
    }

    [[nodiscard]] ArrayStorage operator-() const {
        ArrayStorage result;
        for (std::size_t i = 0; i < Rows_ * Cols_; ++i)
            result.data_[i] = -data_[i];
        return result;
    }

    [[nodiscard]] ArrayStorage operator*(Scalar_ s) const {
        ArrayStorage result;
        for (std::size_t i = 0; i < Rows_ * Cols_; ++i)
            result.data_[i] = data_[i] * s;
        return result;
    }

    friend ArrayStorage operator*(Scalar_ s, const ArrayStorage& m) {
        return m * s;
    }

    [[nodiscard]] ArrayStorage operator/(Scalar_ s) const {
        ArrayStorage result;
        for (std::size_t i = 0; i < Rows_ * Cols_; ++i)
            result.data_[i] = data_[i] / s;
        return result;
    }

    template <std::size_t OtherCols>
    [[nodiscard]] ArrayStorage<Scalar_, Rows_, OtherCols>
    multiply(const ArrayStorage<Scalar_, Cols_, OtherCols>& other) const {
        ArrayStorage<Scalar_, Rows_, OtherCols> result;
        for (std::size_t r = 0; r < Rows_; ++r)
            for (std::size_t c = 0; c < OtherCols; ++c) {
                Scalar_ sum{};
                for (std::size_t k = 0; k < Cols_; ++k)
                    sum += (*this)(r, k) * other(k, c);
                result(r, c) = sum;
            }
        return result;
    }

    [[nodiscard]] ArrayStorage<Scalar_, Cols_, Rows_> transpose() const {
        ArrayStorage<Scalar_, Cols_, Rows_> result;
        for (std::size_t r = 0; r < Rows_; ++r)
            for (std::size_t c = 0; c < Cols_; ++c)
                result(c, r) = (*this)(r, c);
        return result;
    }

    [[nodiscard]] Scalar_ squaredNorm() const
        requires (Cols_ == 1)
    {
        Scalar_ sum{};
        for (std::size_t i = 0; i < Rows_; ++i)
            sum += (*this)(i, 0) * (*this)(i, 0);
        return sum;
    }

    [[nodiscard]] Scalar_ dot(const ArrayStorage& other) const
        requires (Cols_ == 1)
    {
        Scalar_ sum{};
        for (std::size_t i = 0; i < Rows_; ++i)
            sum += (*this)(i, 0) * other(i, 0);
        return sum;
    }

    [[nodiscard]] static ArrayStorage identity()
        requires (Rows_ == Cols_)
    {
        ArrayStorage result;
        for (std::size_t i = 0; i < Rows_; ++i)
            result(i, i) = Scalar_{1};
        return result;
    }

    template <std::size_t StartRow, std::size_t StartCol, std::size_t BlockRows, std::size_t BlockCols>
        requires (StartRow + BlockRows <= Rows_) && (StartCol + BlockCols <= Cols_)
    [[nodiscard]] ArrayStorage<Scalar_, BlockRows, BlockCols> block() const {
        ArrayStorage<Scalar_, BlockRows, BlockCols> result;
        for (std::size_t r = 0; r < BlockRows; ++r)
            for (std::size_t c = 0; c < BlockCols; ++c)
                result(r, c) = (*this)(StartRow + r, StartCol + c);
        return result;
    }
};

struct ArrayStoragePolicy {
    template <typename Scalar, std::size_t Rows, std::size_t Cols>
    using type = ArrayStorage<Scalar, Rows, Cols>;
};

} // namespace tsm::detail
