#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "graphics.cuh"

struct Scene {
    RenderSettings settings;
    std::vector<Object> objects;
};

namespace rtscene_detail {
inline bool parse_cli_args(int argc, char* const* argv,
                           RenderSettings& settings, std::string& output_filename,
                           std::string& correctness_reference,
                           std::string& error) {
    RenderSettings parsed = settings;
    std::string parsed_output = output_filename;
    std::string parsed_reference = correctness_reference;
    error.clear();
    for (int i = 0; i < argc; ++i) {
        const std::string option = argv[i];
        auto read = [&](auto& value) {
            if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) {
                error = "missing value for " + option;
                return false;
            }
            std::istringstream stream(argv[++i]);
            if (!(stream >> value) || !(stream >> std::ws).eof()) {
                error = "invalid value for " + option + ": " + argv[i];
                return false;
            }
            return true;
        };
        int* integer = nullptr;
        FP* scalar = nullptr;
        if (option == "--output") {
            if (i + 1 >= argc || std::string(argv[i + 1]).empty() ||
                std::string(argv[i + 1]).rfind("--", 0) == 0) {
                error = "missing value for --output";
                return false;
            }
            parsed_output = argv[++i];
        }
        else if (option == "--correctness-check") {
            if (i + 1 >= argc || std::string(argv[i + 1]).empty() ||
                std::string(argv[i + 1]).rfind("--", 0) == 0) {
                error = "missing value for --correctness-check";
                return false;
            }
            parsed_reference = argv[++i];
        }
        else if (option == "--width") integer = &parsed.width;
        else if (option == "--height") integer = &parsed.height;
        else if (option == "--samples_per_pixel") integer = &parsed.samples_per_pixel;
        else if (option == "--supersampling_factor") integer = &parsed.supersampling_factor;
        else if (option == "--max_bounces") integer = &parsed.max_bounces;
        else if (option == "--camera_position") scalar = &parsed.camera_position;
        else if (option == "--screen_width") scalar = &parsed.screen_width;
        else if (option == "--screen_height") scalar = &parsed.screen_height;
        else if (option == "--seed") {
            long long seed;
            if (!read(seed)) return false;
            if (seed < 0 || seed > std::numeric_limits<std::uint32_t>::max()) {
                error = "--seed must be between 0 and 4294967295";
                return false;
            }
            parsed.seed = static_cast<std::uint32_t>(seed);
        } else if (option == "--background_light") {
            for (FP* channel : {&parsed.background_light.x,
                                &parsed.background_light.y,
                                &parsed.background_light.z}) {
                if (!read(*channel)) return false;
                if (!std::isfinite(*channel) || *channel < 0) {
                    error = "--background_light needs three finite nonnegative values";
                    return false;
                }
            }
        } else {
            error = "unknown option: " + option;
            return false;
        }
        if (integer) {
            if (!read(*integer)) return false;
            if (*integer < 0 || (*integer == 0 && option != "--max_bounces")) {
                error = option + (option == "--max_bounces"
                    ? " must be non-negative" : " must be positive");
                return false;
            }
        }
        if (scalar) {
            if (!read(*scalar)) return false;
            if (!std::isfinite(*scalar) ||
                (option != "--camera_position" && *scalar <= 0)) {
                error = option + (option == "--camera_position"
                    ? " must be finite" : " must be finite and positive");
                return false;
            }
        }
    }
    settings = parsed;
    output_filename = parsed_output;
    correctness_reference = parsed_reference;
    return true;
}

inline std::string trim(const std::string& text) {
    const std::size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }

    const std::size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

inline Vec3 read_vec3(std::istringstream& stream) {
    Vec3 value;
    stream >> value.x >> value.y >> value.z;
    return value;
}

inline Sphere make_bounds(const std::vector<Triangle>& triangles) {
    Sphere bounds;
    if (triangles.empty()) {
        bounds.center = Vec3();
        bounds.radius = 0.0;
        return bounds;
    }

    Vec3 minimum = triangles.front().p0;
    Vec3 maximum = minimum;
    for (const Triangle& triangle : triangles) {
        const Vec3 points[] = {triangle.p0, triangle.p1, triangle.p2};
        for (const Vec3& point : points) {
            minimum.x = std::min(minimum.x, point.x);
            minimum.y = std::min(minimum.y, point.y);
            minimum.z = std::min(minimum.z, point.z);
            maximum.x = std::max(maximum.x, point.x);
            maximum.y = std::max(maximum.y, point.y);
            maximum.z = std::max(maximum.z, point.z);
        }
    }

    bounds.center = (minimum + maximum) * 0.5;
    for (const Triangle& triangle : triangles) {
        const Vec3 points[] = {triangle.p0, triangle.p1, triangle.p2};
        for (const Vec3& point : points) {
            bounds.radius = std::max(
                bounds.radius,
                length(point - bounds.center)
            );
        }
    }
    bounds.radius += EPSILON;
    return bounds;
}

inline Object make_mesh_object(
    const std::vector<Triangle>& triangles,
    const Vec3& albedo,
    const Vec3& emission = Vec3()
) {
    Object object;
    object.bound = make_bounds(triangles);
    object.triangles = triangles;
    object.albedo = albedo;
    object.emission = emission;
    return object;
}

struct FaceVertex {
    int position_index;
};

inline bool parse_face_vertex(const std::string& token, std::size_t count,
                              FaceVertex& vertex) {
    const std::string index = token.substr(0, token.find('/'));
    try {
        std::size_t consumed;
        const long long value = std::stoll(index, &consumed);
        if (consumed != index.size() || value == 0) return false;
        const long long resolved = value > 0 ? value - 1 : static_cast<long long>(count) + value;
        if (resolved < 0 || resolved >= static_cast<long long>(count)) return false;
        vertex.position_index = static_cast<int>(resolved);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

inline bool append_face(
    const std::vector<Vec3>& positions,
    const std::vector<FaceVertex>& face,
    std::vector<Triangle>& triangles,
    std::string& error
) {
    if (face.size() < 3) {
        error = "face needs at least three vertices";
        return false;
    }

    for (const FaceVertex& vertex : face) {
        if (vertex.position_index < 0 ||
            vertex.position_index >= int(positions.size())) {
            error = "face references an unknown vertex";
            return false;
        }
    }

    for (int index = 1; index + 1 < int(face.size()); ++index) {
        triangles.push_back(Triangle{
            positions[face[0].position_index],
            positions[face[index].position_index],
            positions[face[index + 1].position_index]
        });
    }
    return true;
}

inline bool load_obj(const std::filesystem::path& path, const Vec3& position,
                     float scale, const Vec3& rotation, int max_triangles,
                     const Vec3& albedo, const Vec3& emission,
                     std::vector<Object>& objects, std::string& error,
                     int selected_group = -1) {
    std::ifstream in(path);
    if (!in) {
        error = "failed to open OBJ " + path.string();
        return false;
    }
    const Vec3 radians = rotation * (PI / 180.0f);
    const float cx = std::cos(radians.x), sx = std::sin(radians.x);
    const float cy = std::cos(radians.y), sy = std::sin(radians.y);
    const float cz = std::cos(radians.z), sz = std::sin(radians.z);
    std::vector<Vec3> positions;
    std::vector<Triangle> pending;
    std::vector<Object> imported;
    int group_index = 0;
    bool group_has_faces = false;
    auto flush = [&]() {
        if (!pending.empty()) {
            imported.push_back(make_mesh_object(pending, albedo, emission));
            pending.clear();
        }
    };
    std::string line;
    int line_number = 0;
    auto fail = [&](const std::string& message) {
        error = path.string() + ":" + std::to_string(line_number) + ": " + message;
        return false;
    };
    while (std::getline(in, line)) {
        ++line_number;
        line = line.substr(0, line.find('#'));
        std::istringstream stream(line);
        std::string tag;
        if (!(stream >> tag)) continue;
        if (tag == "v") {
            Vec3 v;
            if (!(stream >> v.x >> v.y >> v.z) ||
                !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
                return fail("vertex needs three finite coordinates");
            }
            v = v * scale;
            v = Vec3(v.x, cx * v.y - sx * v.z, sx * v.y + cx * v.z);
            v = Vec3(cy * v.x + sy * v.z, v.y, -sy * v.x + cy * v.z);
            v = Vec3(cz * v.x - sz * v.y, sz * v.x + cz * v.y, v.z);
            v += position;
            if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
                return fail("transformed vertex is not finite");
            }
            positions.push_back(v);
        } else if (tag == "o" || tag == "g") {
            flush();
            if (group_has_faces) ++group_index;
            group_has_faces = false;
        } else if (tag == "f") {
            std::vector<FaceVertex> face;
            std::string token;
            while (stream >> token) {
                FaceVertex vertex;
                if (!parse_face_vertex(token, positions.size(), vertex)) {
                    return fail("invalid face vertex " + token);
                }
                face.push_back(vertex);
            }
            std::vector<Triangle> triangles;
            std::string face_error;
            if (!append_face(positions, face, triangles, face_error)) return fail(face_error);
            group_has_faces = true;
            if (selected_group >= 0 && group_index != selected_group) continue;
            for (const Triangle& triangle : triangles) {
                pending.push_back(triangle);
                if (pending.size() == static_cast<std::size_t>(max_triangles)) flush();
            }
        } else if (tag != "vn" && tag != "vt" && tag != "s" &&
                   tag != "mtllib" && tag != "usemtl") {
            return fail("unsupported OBJ directive " + tag);
        }
    }
    if (in.bad()) return fail("failed to read OBJ");
    flush();
    if (imported.empty()) return fail(selected_group < 0 ? "OBJ contains no faces" : "OBJ group contains no faces or does not exist");
    objects.insert(objects.end(), imported.begin(), imported.end());
    return true;
}

struct ObjectBuilder {
    bool active;
    Vec3 albedo;
    Vec3 emission;
    std::vector<Vec3> positions;
    std::vector<Triangle> triangles;

    ObjectBuilder(): active(false), albedo(0.7, 0.7, 0.7), emission() {}

    void reset(const Vec3& new_albedo, const Vec3& new_emission) {
        active = true;
        albedo = new_albedo;
        emission = new_emission;
        positions.clear();
        triangles.clear();
    }
};

inline void flush_object(ObjectBuilder& builder, std::vector<Object>& objects) {
    if (builder.active && !builder.triangles.empty()) {
        objects.push_back(make_mesh_object(
            builder.triangles,
            builder.albedo,
            builder.emission
        ));
    }
    builder.active = false;
    builder.positions.clear();
    builder.triangles.clear();
}

}

inline Scene make_default_scene() {
    Scene scene;
    scene.settings.width = 640;
    scene.settings.height = 360;
    scene.settings.camera_position = -4.0;
    scene.settings.screen_width = 3.2;
    scene.settings.screen_height = 1.8;
    scene.settings.samples_per_pixel = 1024;
    scene.settings.supersampling_factor = 2;
    scene.settings.max_bounces = 5;
    scene.settings.seed = 1;
    scene.settings.background_light = Vec3(0.7, 0.8, 1.0);
    return scene;
}

inline bool load_rtscene(const char* path, Scene& scene, std::string& error) {
    using namespace rtscene_detail;

    std::ifstream in(path);
    if (!in) {
        error = std::string("failed to open ") + path;
        return false;
    }

    scene = make_default_scene();
    Vec3 current_albedo(0.7, 0.7, 0.7);
    Vec3 current_emission;
    ObjectBuilder object_builder;
    std::string line;
    int line_number = 0;

    while (std::getline(in, line)) {
        ++line_number;
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        std::istringstream stream(line);
        std::string tag;
        stream >> tag;

        if (tag == "render") {
            stream >> scene.settings.width >> scene.settings.height;
        } else if (tag == "samples_per_pixel") {
            stream >> scene.settings.samples_per_pixel;
        } else if (tag == "supersampling_factor") {
            stream >> scene.settings.supersampling_factor;
        } else if (tag == "max_bounces") {
            stream >> scene.settings.max_bounces;
        } else if (tag == "screen") {
            stream >> scene.settings.screen_width >> scene.settings.screen_height;
        } else if (tag == "camera") {
            Vec3 camera;
            camera = read_vec3(stream);
            scene.settings.camera_position = camera.z;
        } else if (tag == "background") {
            scene.settings.background_light = read_vec3(stream);
        } else if (tag == "albedo") {
            flush_object(object_builder, scene.objects);
            current_albedo = read_vec3(stream);
        } else if (tag == "emission") {
            flush_object(object_builder, scene.objects);
            current_emission = read_vec3(stream);
        } else if (tag == "o") {
            flush_object(object_builder, scene.objects);
            object_builder.reset(current_albedo, current_emission);
        } else if (tag == "v") {
            if (!object_builder.active) {
                object_builder.reset(current_albedo, current_emission);
            }
            object_builder.positions.push_back(read_vec3(stream));
        } else if (tag == "vn") {
            // nothing
        } else if (tag == "f") {
            if (!object_builder.active) {
                object_builder.reset(current_albedo, current_emission);
            }
            std::vector<FaceVertex> face;
            std::string token;
            while (stream >> token) {
                FaceVertex vertex;
                if (!parse_face_vertex(token, object_builder.positions.size(), vertex)) {
                    error = "line " + std::to_string(line_number) + ": invalid face vertex " + token;
                    return false;
                }
                face.push_back(vertex);
            }
            if (!append_face(
                    object_builder.positions,
                    face,
                    object_builder.triangles,
                    error
                )) {
                error = "line " + std::to_string(line_number) + ": " + error;
                return false;
            }
        } else if (tag == "obj") {
            flush_object(object_builder, scene.objects);
            std::string filename, position_tag, scale_tag, rotation_tag, limit_tag, extra;
            Vec3 position, rotation;
            float scale = 0;
            int max_triangles = 0;
            if (!(stream >> std::quoted(filename) >> position_tag
                         >> position.x >> position.y >> position.z
                         >> scale_tag >> scale >> rotation_tag
                         >> rotation.x >> rotation.y >> rotation.z
                         >> limit_tag >> max_triangles) ||
                filename.empty() || position_tag != "position" || scale_tag != "scale" ||
                rotation_tag != "rotation" || limit_tag != "max_triangles" ||
                !std::isfinite(scale) || scale == 0 || max_triangles < 1 ||
                !std::isfinite(position.x) || !std::isfinite(position.y) ||
                !std::isfinite(position.z) || !std::isfinite(rotation.x) ||
                !std::isfinite(rotation.y) || !std::isfinite(rotation.z)) {
                error = "line " + std::to_string(line_number) +
                    ": expected obj path position x y z scale s rotation rx ry rz max_triangles X"
                    " (finite coordinates, nonzero scale and integer X >= 1)";
                return false;
            }
            int selected_group = -1;
            if (stream >> extra) {
                if (extra != "group" || !(stream >> selected_group) || selected_group < 0 || (stream >> extra)) {
                    error = "line " + std::to_string(line_number) + ": expected optional group followed by a nonnegative integer";
                    return false;
                }
            }
            const auto obj_path = std::filesystem::path(path).parent_path() / filename;
            if (!load_obj(obj_path, position, scale, rotation, max_triangles,
                          current_albedo, current_emission, scene.objects, error, selected_group)) {
                error = "line " + std::to_string(line_number) + ": " + error;
                return false;
            }
        } else {
            error = "line " + std::to_string(line_number) +
                ": unknown directive " + tag;
            return false;
        }
    }

    flush_object(object_builder, scene.objects);
    if (scene.settings.width <= 0 || scene.settings.height <= 0 ||
        scene.settings.samples_per_pixel <= 0 || scene.settings.max_bounces < 0) {
        error = "render and samples_per_pixel must be positive; max_bounces must be non-negative";
        return false;
    }
    return true;
}

// Usage: program [scene.rtscene] [--output output.ppm]
//                [--correctness-check reference.ppm] [render options...]
inline Scene load_scene(int argc, char* const* argv, std::string& output_filename,
                        std::string& correctness_reference) {
    int first_option = 1;
    const char* scene_path = "scene.rtscene";
    std::string output_path = "output.ppm";
    std::string reference_path;
    if (first_option < argc && std::string(argv[first_option]).rfind("--", 0) != 0)
        scene_path = argv[first_option++];

    Scene scene;
    std::string error;
    if (!load_rtscene(scene_path, scene, error) ||
        !rtscene_detail::parse_cli_args(argc - first_option, argv + first_option,
                                      scene.settings, output_path, reference_path, error)) {
        throw std::runtime_error(error);
    }
    output_filename = output_path;
    correctness_reference = reference_path;
    return scene;
}

inline bool write_ppm(
    const char* path,
    int width,
    int height,
    const std::vector<Pixel>& pixels
) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }

    out << "P3\n" << width << " " << height << "\n255\n";
    for (const Pixel& pixel : pixels) {
        out << (int)pixel.r << ' '
            << (int)pixel.g << ' '
            << (int)pixel.b << '\n';
    }
    return true;
}

namespace rtscene_detail {

struct PpmMetadata {
    bool has_max_error = false;
    double max_error = 0;
};

inline bool parse_ppm_comment(const std::string& comment, PpmMetadata& metadata) {
    const std::string text = trim(comment);
    const std::string prefix = "Max error:";
    if (text.rfind(prefix, 0) != 0) return true;

    double value = 0;
    std::istringstream stream(text.substr(prefix.size()));
    if (!(stream >> value) || !(stream >> std::ws).eof() ||
        !std::isfinite(value) || value < 0) {
        metadata.has_max_error = false;
        return true;
    }
    metadata.has_max_error = true;
    metadata.max_error = value;
    return true;
}

inline bool read_ppm_token(std::istream& in, std::string& token,
                           PpmMetadata& metadata) {
    token.clear();
    char character;
    while (in.get(character)) {
        if (character == '#') {
            std::string comment;
            std::getline(in, comment);
            if (!parse_ppm_comment(comment, metadata)) return false;
        } else if (!std::isspace(static_cast<unsigned char>(character))) {
            token.push_back(character);
            break;
        }
    }
    if (token.empty()) return false;

    while (in.get(character)) {
        if (character == '#') {
            std::string comment;
            std::getline(in, comment);
            return parse_ppm_comment(comment, metadata);
        }
        if (std::isspace(static_cast<unsigned char>(character))) {
            if (character == '\r' && in.peek() == '\n') in.get();
            return true;
        }
        token.push_back(character);
    }
    return true;
}

inline bool parse_ppm_integer(const std::string& token, int& value) {
    std::istringstream stream(token);
    return (stream >> value) && (stream >> std::ws).eof();
}

inline bool read_ppm(const char* path, int& width, int& height,
                     std::vector<Pixel>& pixels, PpmMetadata& metadata,
                     std::string& error) {
    metadata = PpmMetadata{};
    error.clear();
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = std::string("failed to open reference image ") + path;
        return false;
    }

    std::string token;
    if (!read_ppm_token(in, token, metadata) || (token != "P3" && token != "P6")) {
        if (!error.empty()) return false;
        error = std::string("reference image is not a P3 or P6 PPM: ") + path;
        return false;
    }
    const bool binary = token == "P6";
    int max_value = 0;
    if (!read_ppm_token(in, token, metadata) || !parse_ppm_integer(token, width) || width <= 0 ||
        !read_ppm_token(in, token, metadata) || !parse_ppm_integer(token, height) || height <= 0 ||
        !read_ppm_token(in, token, metadata) || !parse_ppm_integer(token, max_value) ||
        max_value <= 0 || max_value > 65535) {
        if (!error.empty()) return false;
        error = std::string("invalid PPM header in reference image ") + path;
        return false;
    }
    const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
    if (pixel_count > std::numeric_limits<std::size_t>::max() / 3) {
        error = std::string("reference image is too large: ") + path;
        return false;
    }
    pixels.resize(pixel_count);

    auto read_sample = [&](int& sample) {
        if (!binary) {
            return read_ppm_token(in, token, metadata) && parse_ppm_integer(token, sample) &&
                   sample >= 0 && sample <= max_value;
        }
        const int high = in.get();
        if (high == std::char_traits<char>::eof()) return false;
        if (max_value < 256) {
            sample = static_cast<unsigned char>(high);
            return true;
        }
        const int low = in.get();
        if (low == std::char_traits<char>::eof()) return false;
        sample = (static_cast<unsigned char>(high) << 8) |
                 static_cast<unsigned char>(low);
        return sample <= max_value;
    };

    for (Pixel& pixel : pixels) {
        int samples[3];
        if (!read_sample(samples[0]) || !read_sample(samples[1]) || !read_sample(samples[2])) {
            if (!error.empty()) return false;
            error = std::string("invalid or truncated pixel data in reference image ") + path;
            return false;
        }
        pixel.r = static_cast<std::uint8_t>((samples[0] * 255LL + max_value / 2) / max_value);
        pixel.g = static_cast<std::uint8_t>((samples[1] * 255LL + max_value / 2) / max_value);
        pixel.b = static_cast<std::uint8_t>((samples[2] * 255LL + max_value / 2) / max_value);
    }
    return true;
}

}

inline bool calculate_reference_error(const std::vector<Pixel>& output,
                                      int width, int height,
                                      const char* reference_path,
                                      double& mse, double& max_error,
                                      bool& has_max_error,
                                      std::string& error) {
    int reference_width = 0;
    int reference_height = 0;
    std::vector<Pixel> reference;
    rtscene_detail::PpmMetadata metadata;
    if (!rtscene_detail::read_ppm(reference_path, reference_width, reference_height,
                                  reference, metadata, error)) {
        return false;
    }
    if (reference_width != width || reference_height != height ||
        output.size() != reference.size()) {
        error = "reference image dimensions do not match the rendered image";
        return false;
    }

    long double squared_error = 0;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const int red = int(output[index].r) - int(reference[index].r);
        const int green = int(output[index].g) - int(reference[index].g);
        const int blue = int(output[index].b) - int(reference[index].b);
        squared_error += static_cast<long double>(red) * red;
        squared_error += static_cast<long double>(green) * green;
        squared_error += static_cast<long double>(blue) * blue;
    }
    mse = static_cast<double>(squared_error / (reference.size() * 3));
    max_error = metadata.max_error;
    has_max_error = metadata.has_max_error;
    return true;
}

#endif
