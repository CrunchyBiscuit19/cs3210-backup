#include <iostream>
#include <string>

#include "graphics.cuh"
#include "utils.cuh"

int main(int argc, char** argv) try {
    std::string output_path;
    std::string correctness_reference;
    const Scene scene = load_scene(argc, argv, output_path, correctness_reference);

    auto start_time = std::chrono::high_resolution_clock::now();
    const std::vector<Pixel> pixels = render_scene(scene.settings, scene.objects);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    
    std::cout << "Time taken: " << elapsed_time.count() << " seconds\n";

    if (!write_ppm(output_path.c_str(), scene.settings.width, scene.settings.height, pixels)) {
        std::cerr << "failed to write " << output_path << '\n';
        return 1;
    }

    std::cout << "Wrote " << output_path << '\n';
    if (!correctness_reference.empty()) {
        double error_value = 0;
        double max_error = 0;
        bool has_max_error = false;
        std::string error;
        if (!calculate_reference_error(pixels, scene.settings.width, scene.settings.height,
                                       correctness_reference.c_str(), error_value, max_error,
                                       has_max_error, error)) {
            throw std::runtime_error(error);
        }
        std::cout << std::setprecision(10) << "Error: " << error_value << '\n';
        if (has_max_error) {
            std::cout << "Max error: " << max_error << '\n';
            const bool correct = error_value <= max_error;
            std::cout << "Correctness check: " << (correct ? "PASS" : "FAIL") << '\n';
            if (!correct) return 1;
        } else {
            std::cout << "Max error: not provided; skipping PASS/FAIL\n";
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
