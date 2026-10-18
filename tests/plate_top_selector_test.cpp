#include "det-src/PlateTopSelector.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace
{
bool near(const cv::Point2f& actual, const cv::Point2f& expected)
{
    return cv::norm(actual - expected) < 0.01f;
}

bool expect(bool condition, const std::string& message)
{
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}
}

int main()
{
    const cv::Size image_size(1280, 800);
    const PlateTopCandidate center_cell{0, {275, 55, 688, 704}};
    const PlateTopCandidate adjacent_wheel{0, {955, 52, 324, 678}};
    const PlateTopCandidate center_hole{0, {598, 382, 50, 50}};
    const PlateTopCandidate adjacent_hole{1, {1099, 172, 180, 430}};
    bool ok = true;

    // 00146：中央 cell 必须压过相邻轮子。
    auto selection = selectPlateTopTarget(
        image_size, { center_cell }, {}, { adjacent_wheel });
    ok &= expect(selection.kind == PlateTopSelectionKind::Cell,
                 "00146 should select the center plate_cell");
    ok &= expect(near(selection.target_px, { 619.0f, 407.0f }),
                 "00146 should use the center plate_cell geometry center");
    ok &= expect(selection.cell_bbox == center_cell.bbox,
                 "pts_cell_rect source must be the selected center plate_cell");

    // 只有中央 cell 仍应成功。
    selection = selectPlateTopTarget(image_size, { center_cell }, {}, {});
    ok &= expect(selection.kind == PlateTopSelectionKind::Cell,
                 "center plate_cell without holes should remain valid");

    // 同格位孔可以修正点，但外框仍来自 cell。
    selection = selectPlateTopTarget(
        image_size, { center_cell }, { center_hole }, {});
    ok &= expect(selection.matched_hole_index == center_hole.index,
                 "center plate_hole should refine the center point");
    ok &= expect(near(selection.target_px, { 623.0f, 407.0f }),
                 "matched plate_hole center should be used");
    ok &= expect(selection.cell_bbox == center_cell.bbox,
                 "matched hole must not replace the cell rectangle");

    // 相邻孔和相邻轮子不能取代中央 cell。
    selection = selectPlateTopTarget(
        image_size, { center_cell }, { adjacent_hole }, { adjacent_wheel });
    ok &= expect(selection.kind == PlateTopSelectionKind::Cell,
                 "adjacent evidence must not override the center cell");
    ok &= expect(selection.matched_hole_index < 0,
                 "adjacent plate_hole must not refine the center cell");

    // 只有边缘/相邻证据时安全失败。
    selection = selectPlateTopTarget(
        image_size, {}, { adjacent_hole }, { adjacent_wheel });
    ok &= expect(selection.kind == PlateTopSelectionKind::None,
                 "adjacent-only evidence must return no reliable target");

    // 新模型可能只给中心 plate_hole；足够靠近图像中心时允许作为空位兜底。
    selection = selectPlateTopTarget(image_size, {}, { center_hole }, {});
    ok &= expect(selection.kind == PlateTopSelectionKind::Hole,
                 "center plate_hole alone should be accepted as fine empty evidence");
    ok &= expect(selection.matched_hole_index == center_hole.index,
                 "center hole-only selection should record the hole index");
    ok &= expect(near(selection.target_px, { 623.0f, 407.0f }),
                 "center hole-only selection should use the hole center");

    // 只有中央轮子时返回 wheel，不伪造成 cell。
    const PlateTopCandidate center_wheel{2, {460, 220, 360, 360}};
    selection = selectPlateTopTarget(image_size, {}, {}, { center_wheel });
    ok &= expect(selection.kind == PlateTopSelectionKind::Wheel,
                 "center wheel should remain wheel evidence");
    ok &= expect(selection.cell_index < 0,
                 "center wheel must not be represented as pts_cell");

    if (!ok)
        return 1;
    std::cout << "plate top selector tests passed\n";
    return 0;
}
