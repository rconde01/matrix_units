#pragma once

#include "coordinate_frame.hpp"
#include "index_type.hpp"
#include "matrix_tag.hpp"
#include "detail/type_list.hpp"
#include "detail/storage.hpp"
#include "detail/concepts.hpp"

#include <mp-units/systems/si.h>
#include <mp-units/framework.h>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace tsm {

namespace detail {

template <typename = void>
struct DefaultStoragePolicyHelper {
    using type = ArrayStoragePolicy;
};

} // namespace detail

using DefaultStoragePolicy = detail::ArrayStoragePolicy;

template <IsIndexType RowIdx, typename Scalar, IsIndexType ColIdx = NoIdx>
struct Entry {
    using row_idx = RowIdx;
    using col_idx = ColIdx;
    using scalar_type = Scalar;
    Scalar value;

    constexpr explicit Entry(Scalar v) : value(v) {}
};

template <IsIndexType RowIdx, IsIndexType ColIdx = NoIdx, typename Scalar = double>
constexpr auto wrapCoeff(Scalar value) {
    return Entry<RowIdx, Scalar, ColIdx>{value};
}

template <IsIndexType RowIdx, IsIndexType ColIdx = NoIdx, typename Q>
constexpr auto wrapCoeffSi(Q quantity) {
    auto val = quantity.numerical_value_ref_in(decltype(quantity)::unit);
    using Scalar = std::remove_cvref_t<decltype(val)>;
    return Entry<RowIdx, Scalar, ColIdx>{val};
}

template <typename Scalar_,
          typename RowIdxList_,
          typename ColIdxList_,
          IsMatrixTag MatrixTag_,
          typename StoragePolicy_ = DefaultStoragePolicy>
class TypeSafeMatrix {
public:
    using scalar_type    = Scalar_;
    using row_idx_list   = RowIdxList_;
    using col_idx_list   = ColIdxList_;
    using tag_type       = MatrixTag_;
    using storage_policy = StoragePolicy_;

    static constexpr std::size_t num_rows = detail::size_of_v<RowIdxList_>;
    static constexpr std::size_t num_cols = detail::size_of_v<ColIdxList_>;

    using storage_type = typename StoragePolicy_::template type<Scalar_, num_rows, num_cols>;

private:
    storage_type storage_{};

public:
    TypeSafeMatrix() = default;

    explicit TypeSafeMatrix(storage_type s) : storage_(std::move(s)) {}

    template <typename... Entries>
        requires (sizeof...(Entries) == num_rows * num_cols) &&
                 (std::is_same_v<typename Entries::scalar_type, Scalar_> && ...)
    TypeSafeMatrix(Entries... entries) : storage_{} {
        initFromEntries<0>(entries...);
    }

    struct Unchecked {};
    template <typename... Args>
        requires (sizeof...(Args) == num_rows * num_cols) &&
                 (std::convertible_to<Args, Scalar_> && ...)
    TypeSafeMatrix(Unchecked, Args... args) : storage_{} {
        Scalar_ values[] = {static_cast<Scalar_>(args)...};
        for (std::size_t r = 0; r < num_rows; ++r)
            for (std::size_t c = 0; c < num_cols; ++c)
                storage_(r, c) = values[r * num_cols + c];
    }

    template <IsIndexType RowIdx, IsIndexType ColIdx = NoIdx>
        requires detail::contains_v<RowIdxList_, RowIdx> &&
                 detail::contains_v<ColIdxList_, ColIdx>
    [[nodiscard]] Scalar_& at() {
        constexpr auto r = detail::index_of_v<RowIdxList_, RowIdx>;
        constexpr auto c = detail::index_of_v<ColIdxList_, ColIdx>;
        return storage_(r, c);
    }

    template <IsIndexType RowIdx, IsIndexType ColIdx = NoIdx>
        requires detail::contains_v<RowIdxList_, RowIdx> &&
                 detail::contains_v<ColIdxList_, ColIdx>
    [[nodiscard]] const Scalar_& at() const {
        constexpr auto r = detail::index_of_v<RowIdxList_, RowIdx>;
        constexpr auto c = detail::index_of_v<ColIdxList_, ColIdx>;
        return storage_(r, c);
    }

    template <IsIndexType RowIdx, IsIndexType ColIdx = NoIdx>
        requires detail::contains_v<RowIdxList_, RowIdx> &&
                 detail::contains_v<ColIdxList_, ColIdx>
    [[nodiscard]] auto coeffSi() const {
        const Scalar_ val = at<RowIdx, ColIdx>();
        return inferUnit<RowIdx, ColIdx>(val);
    }

    template <IsIndexType RowIdx, IsIndexType ColIdx = NoIdx>
        requires detail::contains_v<RowIdxList_, RowIdx> &&
                 detail::contains_v<ColIdxList_, ColIdx>
    [[nodiscard]] Entry<RowIdx, Scalar_, ColIdx> entry() const {
        return Entry<RowIdx, Scalar_, ColIdx>{at<RowIdx, ColIdx>()};
    }

    template <IsIndexType RowIdx, IsIndexType ColIdx = NoIdx>
        requires detail::contains_v<RowIdxList_, RowIdx> &&
                 detail::contains_v<ColIdxList_, ColIdx>
    void assignEntry(Entry<RowIdx, Scalar_, ColIdx> e) {
        at<RowIdx, ColIdx>() = e.value;
    }

    [[nodiscard]] const storage_type& storage() const { return storage_; }
    [[nodiscard]] storage_type& storage() { return storage_; }

    [[nodiscard]] Scalar_& rawAt(std::size_t row, std::size_t col) {
        return storage_(row, col);
    }
    [[nodiscard]] const Scalar_& rawAt(std::size_t row, std::size_t col) const {
        return storage_(row, col);
    }

    template <std::size_t N>
        requires (N <= num_rows) && (num_cols == 1)
    [[nodiscard]] auto head() const {
        using SubRows = detail::head_t<RowIdxList_, N>;
        TypeSafeMatrix<Scalar_, SubRows, ColIdxList_, MatrixTag_, StoragePolicy_> result;
        for (std::size_t i = 0; i < N; ++i)
            result.rawAt(i, 0) = storage_(i, 0);
        return result;
    }

    template <std::size_t N>
        requires (N <= num_rows) && (num_cols == 1)
    [[nodiscard]] auto tail() const {
        using SubRows = detail::tail_t<RowIdxList_, num_rows - N>;
        TypeSafeMatrix<Scalar_, SubRows, ColIdxList_, MatrixTag_, StoragePolicy_> result;
        for (std::size_t i = 0; i < N; ++i)
            result.rawAt(i, 0) = storage_(num_rows - N + i, 0);
        return result;
    }

    template <std::size_t RowStart, std::size_t RowCount,
              std::size_t ColStart, std::size_t ColCount>
        requires (RowStart + RowCount <= num_rows) &&
                 (ColStart + ColCount <= num_cols)
    [[nodiscard]] auto block() const {
        using SubRows = detail::sub_list_t<RowIdxList_, RowStart, RowCount>;
        using SubCols = detail::sub_list_t<ColIdxList_, ColStart, ColCount>;
        TypeSafeMatrix<Scalar_, SubRows, SubCols, MatrixTag_, StoragePolicy_> result;
        for (std::size_t r = 0; r < RowCount; ++r)
            for (std::size_t c = 0; c < ColCount; ++c)
                result.rawAt(r, c) = storage_(RowStart + r, ColStart + c);
        return result;
    }

    template <typename Other>
        requires detail::Addable<TypeSafeMatrix, Other>
    [[nodiscard]] auto operator+(const Other& other) const {
        using ResultTag = addition_result_tag_t<MatrixTag_, typename Other::tag_type>;
        TypeSafeMatrix<Scalar_, RowIdxList_, ColIdxList_, ResultTag, StoragePolicy_> result;
        for (std::size_t r = 0; r < num_rows; ++r)
            for (std::size_t c = 0; c < num_cols; ++c)
                result.rawAt(r, c) = storage_(r, c) + other.rawAt(r, c);
        return result;
    }

    template <typename Other>
        requires detail::Subtractable<TypeSafeMatrix, Other>
    [[nodiscard]] auto operator-(const Other& other) const {
        using ResultTag = subtraction_result_tag_t<MatrixTag_, typename Other::tag_type>;
        TypeSafeMatrix<Scalar_, RowIdxList_, ColIdxList_, ResultTag, StoragePolicy_> result;
        for (std::size_t r = 0; r < num_rows; ++r)
            for (std::size_t c = 0; c < num_cols; ++c)
                result.rawAt(r, c) = storage_(r, c) - other.rawAt(r, c);
        return result;
    }

    [[nodiscard]] TypeSafeMatrix operator*(Scalar_ s) const {
        TypeSafeMatrix result;
        for (std::size_t r = 0; r < num_rows; ++r)
            for (std::size_t c = 0; c < num_cols; ++c)
                result.rawAt(r, c) = storage_(r, c) * s;
        return result;
    }

    friend TypeSafeMatrix operator*(Scalar_ s, const TypeSafeMatrix& m) {
        return m * s;
    }

    [[nodiscard]] TypeSafeMatrix operator/(Scalar_ s) const {
        TypeSafeMatrix result;
        for (std::size_t r = 0; r < num_rows; ++r)
            for (std::size_t c = 0; c < num_cols; ++c)
                result.rawAt(r, c) = storage_(r, c) / s;
        return result;
    }

    template <typename Other>
        requires detail::Multipliable<TypeSafeMatrix, Other>
    [[nodiscard]] auto operator*(const Other& other) const {
        using ResultTag = multiplication_result_tag_t<MatrixTag_, typename Other::tag_type>;
        using ResultCols = typename Other::col_idx_list;
        constexpr auto other_cols = Other::num_cols;

        TypeSafeMatrix<Scalar_, RowIdxList_, ResultCols, ResultTag, StoragePolicy_> result;
        for (std::size_t r = 0; r < num_rows; ++r)
            for (std::size_t c = 0; c < other_cols; ++c) {
                Scalar_ sum{};
                for (std::size_t k = 0; k < num_cols; ++k)
                    sum += storage_(r, k) * other.rawAt(k, c);
                result.rawAt(r, c) = sum;
            }
        return result;
    }

    [[nodiscard]] TypeSafeMatrix operator-() const {
        TypeSafeMatrix result;
        for (std::size_t r = 0; r < num_rows; ++r)
            for (std::size_t c = 0; c < num_cols; ++c)
                result.rawAt(r, c) = -storage_(r, c);
        return result;
    }

    [[nodiscard]] auto transpose() const {
        using TTag = transpose_tag_t<MatrixTag_>;
        TypeSafeMatrix<Scalar_, ColIdxList_, RowIdxList_, TTag, StoragePolicy_> result;
        for (std::size_t r = 0; r < num_rows; ++r)
            for (std::size_t c = 0; c < num_cols; ++c)
                result.rawAt(c, r) = storage_(r, c);
        return result;
    }

    [[nodiscard]] bool operator==(const TypeSafeMatrix& other) const {
        for (std::size_t r = 0; r < num_rows; ++r)
            for (std::size_t c = 0; c < num_cols; ++c)
                if (storage_(r, c) != other.storage_(r, c))
                    return false;
        return true;
    }

    [[nodiscard]] static TypeSafeMatrix zero() {
        return TypeSafeMatrix{};
    }

    [[nodiscard]] static TypeSafeMatrix identity()
        requires (num_rows == num_cols)
    {
        TypeSafeMatrix result;
        for (std::size_t i = 0; i < num_rows; ++i)
            result.rawAt(i, i) = Scalar_{1};
        return result;
    }

    [[nodiscard]] Scalar_ squaredNorm() const
        requires (num_cols == 1)
    {
        Scalar_ sum{};
        for (std::size_t i = 0; i < num_rows; ++i)
            sum += storage_(i, 0) * storage_(i, 0);
        return sum;
    }

    template <typename Other>
        requires detail::Addable<TypeSafeMatrix, Other> && (num_cols == 1)
    [[nodiscard]] Scalar_ dot(const Other& other) const {
        Scalar_ sum{};
        for (std::size_t i = 0; i < num_rows; ++i)
            sum += storage_(i, 0) * other.rawAt(i, 0);
        return sum;
    }

private:
    template <IsIndexType RowIdx, IsIndexType ColIdx>
    auto inferUnit(Scalar_ val) const {
        constexpr auto row_unit = RowIdx::unit;

        if constexpr (std::is_same_v<ColIdx, NoIdx>) {
            if constexpr (MatrixTag_::row_exponent == 1)
                return val * row_unit;
            else if constexpr (MatrixTag_::row_exponent == -1)
                return val / (Scalar_{1} * row_unit);
            else
                static_assert(MatrixTag_::row_exponent == 1 ||
                              MatrixTag_::row_exponent == -1,
                              "Unsupported row exponent for vector");
        } else {
            constexpr auto col_unit = ColIdx::unit;

            if constexpr (MatrixTag_::row_exponent == 1 && MatrixTag_::col_exponent == 1)
                return val * row_unit * col_unit;
            else if constexpr (MatrixTag_::row_exponent == 1 && MatrixTag_::col_exponent == -1)
                return val * row_unit / (Scalar_{1} * col_unit);
            else if constexpr (MatrixTag_::row_exponent == -1 && MatrixTag_::col_exponent == -1)
                return val / (Scalar_{1} * row_unit * col_unit);
            else if constexpr (MatrixTag_::row_exponent == -1 && MatrixTag_::col_exponent == 1)
                return val * col_unit / (Scalar_{1} * row_unit);
            else
                static_assert(MatrixTag_::row_exponent >= -1 &&
                              MatrixTag_::row_exponent <= 1,
                              "Unsupported exponent combination");
        }
    }

    template <std::size_t>
    void initFromEntries() {}

    template <std::size_t Idx, typename E, typename... Rest>
    void initFromEntries(E e, Rest... rest) {
        if constexpr (num_cols == 1) {
            using ExpectedRow = detail::type_at_t<RowIdxList_, Idx>;
            static_assert(std::is_same_v<typename E::row_idx, ExpectedRow>,
                "Entry row index does not match expected index at this position");
            storage_(Idx, 0) = e.value;
        } else {
            constexpr std::size_t row = Idx / num_cols;
            constexpr std::size_t col = Idx % num_cols;
            using ExpectedRow = detail::type_at_t<RowIdxList_, row>;
            using ExpectedCol = detail::type_at_t<ColIdxList_, col>;
            static_assert(std::is_same_v<typename E::row_idx, ExpectedRow>,
                "Entry row index does not match expected index at this position");
            static_assert(std::is_same_v<typename E::col_idx, ExpectedCol>,
                "Entry column index does not match expected index at this position");
            storage_(row, col) = e.value;
        }
        initFromEntries<Idx + 1>(rest...);
    }
};

template <typename Scalar, typename RowIdxList, IsMatrixTag Tag = VectorTag,
          typename StoragePolicy = DefaultStoragePolicy>
using TypeSafeVector = TypeSafeMatrix<Scalar, RowIdxList,
                                      detail::TypeList<NoIdx>, Tag, StoragePolicy>;

} // namespace tsm
