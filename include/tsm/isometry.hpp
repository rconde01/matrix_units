#pragma once

#include "type_safe_matrix.hpp"
#include "coordinate_frame.hpp"
#include "detail/type_list.hpp"

#include <cstddef>

namespace tsm {

template <typename Scalar,
          typename DestIdxList,
          typename SourceIdxList,
          typename StoragePolicy = DefaultStoragePolicy>
class Isometry {
    static constexpr std::size_t Dim = detail::size_of_v<DestIdxList>;
    static_assert(detail::size_of_v<SourceIdxList> == Dim,
                  "Source and destination index lists must have the same dimension");

public:
    using dest_idx_list = DestIdxList;
    using source_idx_list = SourceIdxList;

    using RotationMatrix = TypeSafeMatrix<Scalar, DestIdxList, SourceIdxList,
                                          JacobianMatrixTag, StoragePolicy>;

    using TranslationVector = TypeSafeVector<Scalar, DestIdxList, DeltaVectorTag, StoragePolicy>;

private:
    RotationMatrix rotation_;
    TranslationVector translation_;

public:
    Isometry() = default;

    Isometry(RotationMatrix rot, TranslationVector trans)
        : rotation_(std::move(rot)), translation_(std::move(trans)) {}

    [[nodiscard]] const RotationMatrix& linear() const { return rotation_; }
    [[nodiscard]] RotationMatrix& linear() { return rotation_; }

    [[nodiscard]] const TranslationVector& translation() const { return translation_; }
    [[nodiscard]] TranslationVector& translation() { return translation_; }

    [[nodiscard]] auto operator*(
        const TypeSafeVector<Scalar, SourceIdxList, VectorTag, StoragePolicy>& pos) const
    {
        TypeSafeVector<Scalar, DestIdxList, VectorTag, StoragePolicy> result;
        for (std::size_t r = 0; r < Dim; ++r) {
            Scalar sum{};
            for (std::size_t k = 0; k < Dim; ++k)
                sum += rotation_.rawAt(r, k) * pos.rawAt(k, 0);
            result.rawAt(r, 0) = sum + translation_.rawAt(r, 0);
        }
        return result;
    }

    [[nodiscard]] auto operator*(
        const TypeSafeVector<Scalar, SourceIdxList, DeltaVectorTag, StoragePolicy>& delta) const
    {
        TypeSafeVector<Scalar, DestIdxList, DeltaVectorTag, StoragePolicy> result;
        for (std::size_t r = 0; r < Dim; ++r) {
            Scalar sum{};
            for (std::size_t k = 0; k < Dim; ++k)
                sum += rotation_.rawAt(r, k) * delta.rawAt(k, 0);
            result.rawAt(r, 0) = sum;
        }
        return result;
    }

    template <typename OtherSourceIdxList>
    [[nodiscard]] auto operator*(
        const Isometry<Scalar, SourceIdxList, OtherSourceIdxList, StoragePolicy>& other) const
    {
        using ResultRotation = TypeSafeMatrix<Scalar, DestIdxList, OtherSourceIdxList,
                                              JacobianMatrixTag, StoragePolicy>;
        ResultRotation result_rot;
        for (std::size_t r = 0; r < Dim; ++r)
            for (std::size_t c = 0; c < Dim; ++c) {
                Scalar sum{};
                for (std::size_t k = 0; k < Dim; ++k)
                    sum += rotation_.rawAt(r, k) * other.linear().rawAt(k, c);
                result_rot.rawAt(r, c) = sum;
            }

        using ResultTranslation = TypeSafeVector<Scalar, DestIdxList, DeltaVectorTag, StoragePolicy>;
        ResultTranslation result_trans;
        for (std::size_t r = 0; r < Dim; ++r) {
            Scalar sum{};
            for (std::size_t k = 0; k < Dim; ++k)
                sum += rotation_.rawAt(r, k) * other.translation().rawAt(k, 0);
            result_trans.rawAt(r, 0) = sum + translation_.rawAt(r, 0);
        }

        return Isometry<Scalar, DestIdxList, OtherSourceIdxList, StoragePolicy>{
            result_rot, result_trans};
    }

    [[nodiscard]] auto inverse() const {
        using InvRotation = TypeSafeMatrix<Scalar, SourceIdxList, DestIdxList,
                                           JacobianMatrixTag, StoragePolicy>;
        InvRotation inv_rot;
        for (std::size_t r = 0; r < Dim; ++r)
            for (std::size_t c = 0; c < Dim; ++c)
                inv_rot.rawAt(r, c) = rotation_.rawAt(c, r); // R^T

        using InvTranslation = TypeSafeVector<Scalar, SourceIdxList, DeltaVectorTag, StoragePolicy>;
        InvTranslation inv_trans;
        for (std::size_t r = 0; r < Dim; ++r) {
            Scalar sum{};
            for (std::size_t k = 0; k < Dim; ++k)
                sum += inv_rot.rawAt(r, k) * translation_.rawAt(k, 0);
            inv_trans.rawAt(r, 0) = -sum;
        }

        return Isometry<Scalar, SourceIdxList, DestIdxList, StoragePolicy>{
            inv_rot, inv_trans};
    }
};

} // namespace tsm
