#ifndef GRAPHICS_CUH
#define GRAPHICS_CUH

#include <cstdint>
#include <sys/types.h>
#include <vector>
#include <cuda_runtime.h>
#include <stdio.h>

#include "geometry.cuh"

struct RenderSettings {
    float camera_position;
    Vec3 background_light;
    int width;
    int height;
    float screen_width;
    float screen_height;
    int samples_per_pixel;
    int supersampling_factor;
    int max_bounces;
    std::uint32_t seed;
};

struct Pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;

    __host__ __device__ Pixel() : r(0), g(0), b(0) {}
    __host__ __device__ Pixel(float r, float g, float b) : r(uint8_t(r * 255.0f + 0.5f)), g(uint8_t(g * 255.0f + 0.5f)), b(uint8_t(b * 255.0f + 0.5f)) {}
};

__host__ __device__ inline Pixel to_pixel(const Vec3& color) {
    // reinhard tone mapping
    const float r = color.x;
    const float g = color.y;
    const float b = color.z;
    const float tone_mapped_r = r / (r + 1.0);
    const float tone_mapped_g = g / (g + 1.0);
    const float tone_mapped_b = b / (b + 1.0);

    // gamma correction
    const float gamma = 1. / 2.2;
    const float corrected_r = std::pow(tone_mapped_r, gamma);
    const float corrected_g = std::pow(tone_mapped_g, gamma);
    const float corrected_b = std::pow(tone_mapped_b, gamma);

    return Pixel(corrected_r, corrected_g, corrected_b);
}

/*
Given the render settings and pixel (x, y). Return a ray that originates from the camera position and passes through the pixel (x, y) on the screen. The ray's direction is normalized.

Pixels are indexed from the top-left corner of the screen, with (0, 0) being the top-left pixel and (width * supersampling_factor - 1, height * supersampling_factor - 1) being the bottom-right pixel.
*/
__host__ __device__ inline Ray make_camera_ray(const RenderSettings& settings, int x, int y) {
    int width = settings.width * settings.supersampling_factor;
    int height = settings.height * settings.supersampling_factor;
    
    const float px = ((float)x + 0.5f) / width * settings.screen_width - settings.screen_width * 0.5;
    const float py = settings.screen_height * 0.5 - (float(y) + 0.5f) / height * settings.screen_height;

    return make_ray(
        Vec3(0.0, 0.0, settings.camera_position),
        Vec3(px, py, 0.0)
    );
}

std::vector<Pixel> render_scene(
    const RenderSettings& settings,
    const std::vector<Object>& objects
);

#endif
