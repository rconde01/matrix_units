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
};

struct ArrayStoragePolicy {
    template <typename Scalar, std::size_t Rows, std::size_t Cols>
    using type = ArrayStorage<Scalar, Rows, Cols>;
};

} // namespace tsm::detail
