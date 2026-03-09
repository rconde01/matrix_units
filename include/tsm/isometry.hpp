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
        auto rotated = rotation_ * pos;
        return rotated + translation_;
    }

    [[nodiscard]] auto operator*(
        const TypeSafeVector<Scalar, SourceIdxList, DeltaVectorTag, StoragePolicy>& delta) const
    {
        return rotation_ * delta;
    }

    template <typename OtherSourceIdxList>
    [[nodiscard]] auto operator*(
        const Isometry<Scalar, SourceIdxList, OtherSourceIdxList, StoragePolicy>& other) const
    {
        auto result_rot = rotation_ * other.linear();
        auto result_trans = rotation_ * other.translation() + translation_;

        return Isometry<Scalar, DestIdxList, OtherSourceIdxList, StoragePolicy>{
            result_rot, result_trans};
    }

    [[nodiscard]] auto inverse() const {
        auto inv_rot = rotation_.transpose();
        auto inv_trans = -(inv_rot * translation_);

        return Isometry<Scalar, SourceIdxList, DestIdxList, StoragePolicy>{
            inv_rot, inv_trans};
    }
};

} // namespace tsm
