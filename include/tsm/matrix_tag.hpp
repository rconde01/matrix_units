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

// Primary template: no `type` member — invalid combinations are SFINAE-rejected
template <IsMatrixTag Tag1, IsMatrixTag Tag2>
struct AdditionResultTag {};

template <> struct AdditionResultTag<VectorTag, DeltaVectorTag>             { using type = VectorTag; };
template <> struct AdditionResultTag<DeltaVectorTag, VectorTag>             { using type = VectorTag; };
template <> struct AdditionResultTag<DeltaVectorTag, DeltaVectorTag>        { using type = DeltaVectorTag; };
template <> struct AdditionResultTag<CovarianceMatrixTag, CovarianceMatrixTag>   { using type = CovarianceMatrixTag; };
template <> struct AdditionResultTag<InformationMatrixTag, InformationMatrixTag> { using type = InformationMatrixTag; };

template <IsMatrixTag Tag1, IsMatrixTag Tag2>
using addition_result_tag_t = typename AdditionResultTag<Tag1, Tag2>::type;

// Primary template: no `type` member — invalid combinations are SFINAE-rejected
template <IsMatrixTag Tag1, IsMatrixTag Tag2>
struct SubtractionResultTag {};

template <> struct SubtractionResultTag<VectorTag, VectorTag>               { using type = DeltaVectorTag; };
template <> struct SubtractionResultTag<VectorTag, DeltaVectorTag>          { using type = VectorTag; };
template <> struct SubtractionResultTag<DeltaVectorTag, VectorTag>          { using type = VectorTag; };
template <> struct SubtractionResultTag<DeltaVectorTag, DeltaVectorTag>     { using type = DeltaVectorTag; };
template <> struct SubtractionResultTag<CovarianceMatrixTag, CovarianceMatrixTag>   { using type = CovarianceMatrixTag; };
template <> struct SubtractionResultTag<InformationMatrixTag, InformationMatrixTag> { using type = InformationMatrixTag; };

template <IsMatrixTag Tag1, IsMatrixTag Tag2>
using subtraction_result_tag_t = typename SubtractionResultTag<Tag1, Tag2>::type;

template <IsMatrixTag TagA, IsMatrixTag TagB>
struct MultiplicationResultTag;

template <> struct MultiplicationResultTag<JacobianMatrixTag, CovarianceMatrixTag>          { using type = CovarianceMatrixTag; };
template <> struct MultiplicationResultTag<CovarianceMatrixTag, TransposedJacobianMatrixTag> { using type = CovarianceMatrixTag; };
template <> struct MultiplicationResultTag<JacobianMatrixTag, VectorTag>                    { using type = VectorTag; };
template <> struct MultiplicationResultTag<JacobianMatrixTag, DeltaVectorTag>               { using type = DeltaVectorTag; };
template <> struct MultiplicationResultTag<JacobianMatrixTag, JacobianMatrixTag>            { using type = JacobianMatrixTag; };

template <IsMatrixTag TagA, IsMatrixTag TagB>
using multiplication_result_tag_t = typename MultiplicationResultTag<TagA, TagB>::type;

template <IsMatrixTag Tag>
struct TransposeTag;

template <> struct TransposeTag<CovarianceMatrixTag>          { using type = CovarianceMatrixTag; };
template <> struct TransposeTag<JacobianMatrixTag>            { using type = TransposedJacobianMatrixTag; };
template <> struct TransposeTag<TransposedJacobianMatrixTag>  { using type = JacobianMatrixTag; };
template <> struct TransposeTag<InformationMatrixTag>         { using type = InformationMatrixTag; };

template <IsMatrixTag Tag>
using transpose_tag_t = typename TransposeTag<Tag>::type;

} // namespace tsm
