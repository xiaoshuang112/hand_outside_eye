#include "top_face.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    using json = nlohmann::json;
    if (argc != 2) return 2;
    const auto source = fs::absolute(argv[1]) / "config" / "top_face";
    const auto app = fs::temp_directory_path() / ("top-plane-snapshot-" + std::to_string(getpid()));
    try {
        const auto dir = app / "config" / "top_face";
        fs::create_directories(dir);
        json algorithm, mapping;
        std::ifstream(source / "algorithm.json") >> algorithm;
        std::ifstream(source / "tcp_xy_mapping.json") >> mapping;
        algorithm["use_center_model"] = false;
        algorithm["background_image"] = (source / "background.bmp").string();
        std::ofstream(dir / "algorithm.json") << algorithm;
        std::ofstream(dir / "tcp_xy_mapping.json") << mapping;
        if (top_face_init(app.string()) != 0) throw std::runtime_error(top_face_init_error());
        for (const auto& row : mapping.at("training_samples")) {
            for (double height : {0., 35., 60.}) {
                double x = 0, y = 0;
                if (!top_face_tcp_xy(row.at("uv_px")[0], row.at("uv_px")[1], {1280, 1024},
                                     x, y, height, cv::Matx33d::eye()))
                    throw std::runtime_error("Mapping unexpectedly disabled");
                const auto& expected = row.at("fitted_top_base_xyz_mm");
                if (std::hypot(x - expected[0].get<double>(), y - expected[1].get<double>()) > 1e-7)
                    throw std::runtime_error("Snapshot mapping mismatch or height applied twice");
            }
        }
        const auto image = cv::imread((fs::absolute(argv[1]) / "tests/data/00218.jpg").string());
        json detected;
        if (image.empty() || top_face_detect(image, json::object(), detected) != 0 ||
            detected.value("top_face_count", 0) == 0)
            throw std::runtime_error("Angle reference image did not produce a top face");
        if (detected.value("angle_definition", std::string()) !=
            "projected_long_axis_counterclockwise_from_image_positive_x_mod_180")
            throw std::runtime_error("Angle direction metadata mismatch");
        for (const auto& object : detected.at("objects")) {
            if (object.at("status") != "ok") continue;
            const auto& corners = object.at("corners_px");
            auto point = [&](int i) {
                return cv::Point2d(corners[i][0].get<double>(), corners[i][1].get<double>());
            };
            const auto axis = (point(1) - point(0)) + (point(2) - point(3));
            const double expected = std::fmod(std::atan2(-axis.y, axis.x) * 180 / CV_PI + 180, 180);
            double error = std::abs(object.at("angle_deg").get<double>() - expected);
            error = std::min(error, 180 - error);
            if (error > 1)
                throw std::runtime_error("Reported angle disagrees with counter-clockwise image axis");
        }
        top_face_close();
        fs::remove_all(app);
        std::cout << "Production mapping, height independence and counter-clockwise angles passed\n";
        return 0;
    } catch (const std::exception& e) {
        top_face_close();
        fs::remove_all(app);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
