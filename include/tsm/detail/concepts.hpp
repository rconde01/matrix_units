#pragma once

#include "../matrix_tag.hpp"
#include "type_list.hpp"

#include <concepts>
#include <type_traits>

namespace tsm {

template <typename Scalar, typename RowIdxList, typename ColIdxList,
          IsMatrixTag MatrixTag, typename StoragePolicy>
class TypeSafeMatrix;

namespace detail {

template <typename T>
struct IsTypeSafeMatrixImpl : std::false_type {};

template <typename S, typename R, typename C, IsMatrixTag M, typename SP>
struct IsTypeSafeMatrixImpl<TypeSafeMatrix<S, R, C, M, SP>> : std::true_type {};

template <typename T>
concept IsTypeSafeMatrix = IsTypeSafeMatrixImpl<std::remove_cvref_t<T>>::value;

template <typename A, typename B>
concept Addable = IsTypeSafeMatrix<A> && IsTypeSafeMatrix<B> &&
    std::same_as<typename std::remove_cvref_t<A>::scalar_type,
                 typename std::remove_cvref_t<B>::scalar_type> &&
    are_identical_v<typename std::remove_cvref_t<A>::row_idx_list,
                    typename std::remove_cvref_t<B>::row_idx_list> &&
    are_identical_v<typename std::remove_cvref_t<A>::col_idx_list,
                    typename std::remove_cvref_t<B>::col_idx_list> &&
    (std::remove_cvref_t<A>::tag_type::row_exponent ==
     std::remove_cvref_t<B>::tag_type::row_exponent) &&
    (std::remove_cvref_t<A>::tag_type::col_exponent ==
     std::remove_cvref_t<B>::tag_type::col_exponent);

template <typename A, typename B>
concept Subtractable = Addable<A, B>;

template <typename A, typename B>
concept Multipliable = IsTypeSafeMatrix<A> && IsTypeSafeMatrix<B> &&
    std::same_as<typename std::remove_cvref_t<A>::scalar_type,
                 typename std::remove_cvref_t<B>::scalar_type> &&
    are_identical_v<typename std::remove_cvref_t<A>::col_idx_list,
                    typename std::remove_cvref_t<B>::row_idx_list> &&
    (std::remove_cvref_t<A>::tag_type::col_exponent +
     std::remove_cvref_t<B>::tag_type::row_exponent == 0);

} // namespace detail
} // namespace tsm
