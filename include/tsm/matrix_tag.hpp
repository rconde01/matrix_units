#pragma once

#include <type_traits>

namespace tsm {

struct MatrixTagBase {};

struct CovarianceMatrixTag : MatrixTagBase {
    static constexpr int row_exponent = 1;
    static constexpr int col_exponent = 1;
};

struct JacobianMatrixTag : MatrixTagBase {
    static constexpr int row_exponent = 1;
    static constexpr int col_exponent = -1;
};

struct InformationMatrixTag : MatrixTagBase {
    static constexpr int row_exponent = -1;
    static constexpr int col_exponent = -1;
};

struct VectorTag : MatrixTagBase {
    static constexpr int row_exponent = 1;
    static constexpr int col_exponent = 0;
};

struct DeltaVectorTag : MatrixTagBase {
    static constexpr int row_exponent = 1;
    static constexpr int col_exponent = 0;
};

struct InformationVectorTag : MatrixTagBase {
    static constexpr int row_exponent = -1;
    static constexpr int col_exponent = 0;
};

struct TransposedCovarianceMatrixTag : MatrixTagBase {
    static constexpr int row_exponent = 1;
    static constexpr int col_exponent = 1;
};

struct TransposedJacobianMatrixTag : MatrixTagBase {
    static constexpr int row_exponent = -1;
    static constexpr int col_exponent = 1;
};

template <typename T>
concept IsMatrixTag = std::is_base_of_v<MatrixTagBase, T>;

template <IsMatrixTag Tag1, IsMatrixTag Tag2>
struct AdditionResultTag {
    static_assert(std::is_same_v<Tag1, Tag2>,
                  "Incompatible matrix tags for addition");
    static_assert(!std::is_same_v<Tag1, VectorTag>,
                  "Cannot add two position vectors (points). "
                  "Subtract them to get a displacement vector, or use DeltaVectorTag.");
    using type = Tag1;
};

template <>
struct AdditionResultTag<VectorTag, DeltaVectorTag> {
    using type = VectorTag;
};

template <>
struct AdditionResultTag<DeltaVectorTag, VectorTag> {
    using type = VectorTag;
};

template <>
struct AdditionResultTag<DeltaVectorTag, DeltaVectorTag> {
    using type = DeltaVectorTag;
};

template <>
struct AdditionResultTag<CovarianceMatrixTag, CovarianceMatrixTag> {
    using type = CovarianceMatrixTag;
};

template <>
struct AdditionResultTag<InformationMatrixTag, InformationMatrixTag> {
    using type = InformationMatrixTag;
};

template <IsMatrixTag Tag1, IsMatrixTag Tag2>
using addition_result_tag_t = typename AdditionResultTag<Tag1, Tag2>::type;

template <IsMatrixTag Tag1, IsMatrixTag Tag2>
struct SubtractionResultTag {
    using type = addition_result_tag_t<Tag1, Tag2>;
};

template <>
struct SubtractionResultTag<VectorTag, VectorTag> {
    using type = DeltaVectorTag;
};

template <IsMatrixTag Tag1, IsMatrixTag Tag2>
using subtraction_result_tag_t = typename SubtractionResultTag<Tag1, Tag2>::type;

template <IsMatrixTag TagA, IsMatrixTag TagB>
struct MultiplicationResultTag;

template <>
struct MultiplicationResultTag<JacobianMatrixTag, CovarianceMatrixTag> {
    using type = CovarianceMatrixTag;
};

template <>
struct MultiplicationResultTag<CovarianceMatrixTag, TransposedJacobianMatrixTag> {
    using type = CovarianceMatrixTag;
};

template <>
struct MultiplicationResultTag<JacobianMatrixTag, VectorTag> {
    using type = VectorTag;
};

template <>
struct MultiplicationResultTag<JacobianMatrixTag, DeltaVectorTag> {
    using type = DeltaVectorTag;
};

template <>
struct MultiplicationResultTag<JacobianMatrixTag, JacobianMatrixTag> {
    using type = JacobianMatrixTag;
};

template <IsMatrixTag TagA, IsMatrixTag TagB>
using multiplication_result_tag_t = typename MultiplicationResultTag<TagA, TagB>::type;

template <IsMatrixTag Tag>
struct TransposeTag;

template <>
struct TransposeTag<CovarianceMatrixTag> {
    using type = CovarianceMatrixTag;
};

template <>
struct TransposeTag<JacobianMatrixTag> {
    using type = TransposedJacobianMatrixTag;
};

template <>
struct TransposeTag<TransposedJacobianMatrixTag> {
    using type = JacobianMatrixTag;
};

template <>
struct TransposeTag<InformationMatrixTag> {
    using type = InformationMatrixTag;
};

template <IsMatrixTag Tag>
using transpose_tag_t = typename TransposeTag<Tag>::type;

} // namespace tsm
