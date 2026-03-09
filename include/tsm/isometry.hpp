#pragma once

#include "type_safe_matrix.hpp"
#include "coordinate_frame.hpp"
#include "detail/type_list.hpp"

namespace tsm {

template <typename Scalar,
          typename DestIdxList,
          typename SourceIdxList,
          typename StoragePolicy = DefaultStoragePolicy>
class Isometry {
    static constexpr std::size_t Dim = detail::size_of_v<DestIdxList>;
    static_assert(detail::size_of_v<SourceIdxList> == Dim,
                  "Source and destination index lists must have the same dimension");

    using SquareStorage = typename StoragePolicy::template type<Scalar, Dim, Dim>;
    using VecStorage = typename StoragePolicy::template type<Scalar, Dim, 1>;

public:
    using dest_idx_list = DestIdxList;
    using source_idx_list = SourceIdxList;

    using RotationMatrix = TypeSafeMatrix<Scalar, DestIdxList, SourceIdxList,
                                          JacobianMatrixTag, SquareStorage>;

    using TranslationVector = TypeSafeVector<Scalar, DestIdxList, DeltaVectorTag, VecStorage>;

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

    template <typename PosStorage>
    [[nodiscard]] auto operator*(
        const TypeSafeVector<Scalar, SourceIdxList, VectorTag, PosStorage>& pos) const
    {
        return rotation_ * pos + translation_;
    }

    template <typename DeltaStorage>
    [[nodiscard]] auto operator*(
        const TypeSafeVector<Scalar, SourceIdxList, DeltaVectorTag, DeltaStorage>& delta) const
    {
        return rotation_ * delta;
    }

    template <typename OtherSourceIdxList>
    [[nodiscard]] auto operator*(
        const Isometry<Scalar, SourceIdxList, OtherSourceIdxList, StoragePolicy>& other) const
    {
        using ResultSquareStorage = typename StoragePolicy::template type<Scalar, Dim, Dim>;
        using ResultRotMatrix = TypeSafeMatrix<Scalar, DestIdxList, OtherSourceIdxList,
                                                JacobianMatrixTag, ResultSquareStorage>;
        // Evaluate into concrete types to avoid dangling expression references
        ResultRotMatrix result_rot = rotation_ * other.linear();
        TranslationVector result_trans = rotation_ * other.translation() + translation_;

        return Isometry<Scalar, DestIdxList, OtherSourceIdxList, StoragePolicy>{
            result_rot, result_trans};
    }

    [[nodiscard]] auto inverse() const {
        // R^T has TransposedJacobianMatrixTag, but the inverse rotation is semantically
        // a Jacobian (maps from dest frame back to source frame). Construct from the
        // transposed storage directly with the correct tag.
        using InvRotMatrix = TypeSafeMatrix<Scalar, SourceIdxList, DestIdxList,
                                            JacobianMatrixTag, SquareStorage>;
        InvRotMatrix inv_rot{SquareStorage{rotation_.storage().transpose()}};

        using InvTransVector = TypeSafeVector<Scalar, SourceIdxList, DeltaVectorTag, VecStorage>;
        InvTransVector inv_trans = -(inv_rot * translation_);

        return Isometry<Scalar, SourceIdxList, DestIdxList, StoragePolicy>{
            inv_rot, inv_trans};
    }
};

} // namespace tsm
