#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

struct PlateTopCandidate
{
    int index = -1;
    cv::Rect2f bbox;
};

enum class PlateTopSelectionKind
{
    None,
    Cell,
    Hole,
    Wheel
};

struct PlateTopSelection
{
    PlateTopSelectionKind kind = PlateTopSelectionKind::None;
    int cell_index = -1;
    int matched_hole_index = -1;
    int wheel_index = -1;
    cv::Point2f target_px;
    cv::Rect2f cell_bbox;
    const char* source = "none";
};

inline cv::Point2f plateTopBboxCenter(const cv::Rect2f& bbox)
{
    return (bbox.tl() + bbox.br()) * 0.5f;
}

inline const PlateTopCandidate* findPlateTopCenterCandidate(
    const cv::Size& image_size,
    const std::vector<PlateTopCandidate>& candidates)
{
    const cv::Point2f image_center(
        image_size.width * 0.5f, image_size.height * 0.5f);
    const float center_distance_limit =
        std::min(image_size.width, image_size.height) * 0.20f;

    const PlateTopCandidate* best_containing = nullptr;
    const PlateTopCandidate* best_nearby = nullptr;
    float best_containing_dist = FLT_MAX;
    float best_nearby_dist = FLT_MAX;

    for (const auto& candidate : candidates)
    {
        if (candidate.index < 0 || candidate.bbox.width <= 0.0f ||
            candidate.bbox.height <= 0.0f)
            continue;

        const cv::Point2f center = plateTopBboxCenter(candidate.bbox);
        const float distance = cv::norm(center - image_center);
        if (candidate.bbox.contains(image_center))
        {
            if (distance < best_containing_dist)
            {
                best_containing_dist = distance;
                best_containing = &candidate;
            }
        }
        else if (distance <= center_distance_limit &&
                 distance < best_nearby_dist)
        {
            best_nearby_dist = distance;
            best_nearby = &candidate;
        }
    }

    return best_containing != nullptr ? best_containing : best_nearby;
}

inline const PlateTopCandidate* findPlateTopCenterHoleOnlyCandidate(
    const cv::Size& image_size,
    const std::vector<PlateTopCandidate>& candidates)
{
    const cv::Point2f image_center(
        image_size.width * 0.5f, image_size.height * 0.5f);
    const float center_distance_limit = std::max(
        35.0f,
        std::min(image_size.width, image_size.height) * 0.08f);

    const PlateTopCandidate* best = nullptr;
    float best_dist = FLT_MAX;

    for (const auto& candidate : candidates)
    {
        if (candidate.index < 0 || candidate.bbox.width <= 0.0f ||
            candidate.bbox.height <= 0.0f)
            continue;

        const cv::Point2f center = plateTopBboxCenter(candidate.bbox);
        const float distance = cv::norm(center - image_center);
        if (distance > center_distance_limit)
            continue;

        const float max_side = std::max(candidate.bbox.width, candidate.bbox.height);
        const float min_side = std::min(candidate.bbox.width, candidate.bbox.height);
        if (min_side <= 0.0f || max_side / min_side > 3.0f)
            continue;

        if (distance < best_dist)
        {
            best_dist = distance;
            best = &candidate;
        }
    }

    return best;
}

inline PlateTopSelection selectPlateTopTarget(
    const cv::Size& image_size,
    const std::vector<PlateTopCandidate>& cells,
    const std::vector<PlateTopCandidate>& plate_holes,
    const std::vector<PlateTopCandidate>& wheels)
{
    PlateTopSelection result;
    const PlateTopCandidate* center_cell =
        findPlateTopCenterCandidate(image_size, cells);

    if (center_cell != nullptr)
    {
        result.kind = PlateTopSelectionKind::Cell;
        result.cell_index = center_cell->index;
        result.cell_bbox = center_cell->bbox;
        result.target_px = plateTopBboxCenter(center_cell->bbox);
        result.source = "plate_cell";

        const cv::Point2f cell_center = result.target_px;
        const float hole_distance_limit = std::max(
            20.0f,
            0.25f * std::min(
                center_cell->bbox.width, center_cell->bbox.height));
        float best_hole_distance = FLT_MAX;
        for (const auto& hole : plate_holes)
        {
            if (hole.index < 0 || hole.bbox.width <= 0.0f ||
                hole.bbox.height <= 0.0f)
                continue;

            const cv::Point2f hole_center = plateTopBboxCenter(hole.bbox);
            if (!center_cell->bbox.contains(hole_center))
                continue;

            const float distance = cv::norm(hole_center - cell_center);
            if (distance <= hole_distance_limit &&
                distance < best_hole_distance)
            {
                best_hole_distance = distance;
                result.matched_hole_index = hole.index;
                result.target_px = hole_center;
                result.source = "matched_plate_hole";
            }
        }
        return result;
    }

    const PlateTopCandidate* center_hole =
        findPlateTopCenterHoleOnlyCandidate(image_size, plate_holes);
    if (center_hole != nullptr)
    {
        result.kind = PlateTopSelectionKind::Hole;
        result.matched_hole_index = center_hole->index;
        result.target_px = plateTopBboxCenter(center_hole->bbox);
        result.source = "center_plate_hole_only";
        return result;
    }

    const PlateTopCandidate* center_wheel =
        findPlateTopCenterCandidate(image_size, wheels);
    if (center_wheel != nullptr)
    {
        result.kind = PlateTopSelectionKind::Wheel;
        result.wheel_index = center_wheel->index;
        result.target_px = plateTopBboxCenter(center_wheel->bbox);
        result.source = "lunzi";
    }

    return result;
}
