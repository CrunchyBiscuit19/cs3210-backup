#include <cstdint>
#include <iostream>


#include "graphics.cuh"

struct Hit {
    HitRecord record;
    int object_index;
    int triangle_index;

    Hit(): object_index(-1), triangle_index(-1) {}
    bool hit() const { return object_index != -1; }
};

Hit find_closest_hit(const Ray& source_ray, const std::vector<Object>& objects, int last_hit = -1) {
    Hit closest;

    for (int object_index = 0, triangle_offset = 0; object_index < int(objects.size());
         triangle_offset += int(objects[object_index].triangles.size()), ++object_index) {
        const Object& scene_object = objects[object_index];

        for (int i = 0; i < int(scene_object.triangles.size()); ++i) {
            const int triangle_index = triangle_offset + i;
            if (triangle_index == last_hit) continue;
            const Triangle& triangle = scene_object.triangles[i];
            HitRecord record;
            if (intersect_triangle(source_ray, triangle, &record) && record.t < closest.record.t) {
                closest.record = record;
                closest.object_index = object_index;
                closest.triangle_index = triangle_index;
            }
        }
    }

    return closest;
}

Vec3 trace_ray(
    const Ray& source_ray,
    const RenderSettings& settings,
    const std::vector<Object>& objects,
    int depth,
    RandomCPU& random,
    int last_hit = -1
) {
    const Hit hit = find_closest_hit(source_ray, objects, last_hit);
    if (!hit.hit()) {
        return settings.background_light;
    }

    const Object& object = objects[hit.object_index];
    if (depth >= settings.max_bounces) {
        return object.emission;
    }

    const Ray scattered_ray = cosine_weighted_sampling(hit.record, random);
    return object.emission + multiply(
        object.albedo,
        trace_ray(scattered_ray, settings, objects, depth + 1, random, hit.triangle_index)
    );
}

std::vector<Pixel> render_scene(
    const RenderSettings& settings,
    const std::vector<Object>& objects
) {
    std::vector<Pixel> pixels(settings.width * settings.height);
    const int samples_per_pixel = settings.samples_per_pixel;
    const int supersampling_factor = settings.supersampling_factor;

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < settings.height * supersampling_factor; y += supersampling_factor) {
        for (int x = 0; x < settings.width * supersampling_factor; x += supersampling_factor) {
            Vec3 color;
            RandomCPU random(y * settings.width * supersampling_factor + x);

            for (int sy = 0; sy < supersampling_factor; ++sy) {
                for (int sx = 0; sx < supersampling_factor; ++sx) {
                    for (int s = 0; s < samples_per_pixel; ++s) {
                        const Ray sample_ray = make_camera_ray(settings, x + sx, y + sy);
                        color += trace_ray(sample_ray, settings, objects, 0, random);
                    }
                }
            }

            pixels[(y / supersampling_factor) * settings.width + x / supersampling_factor] = to_pixel(color / FP(samples_per_pixel * supersampling_factor * supersampling_factor));
        }
    }

    return pixels;
}
