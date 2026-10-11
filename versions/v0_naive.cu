// v0 naive: a direct CUDA port of cpu_path_tracing.cu.
//
// - One thread per output pixel (16x16 blocks). Each thread loops over its
//   subpixels and samples, exactly like one OpenMP iteration of the CPU code.
// - The recursive trace_ray becomes a loop (brief, recommendation 2).
// - The scene is flattened into plain global-memory arrays, and every ray
//   tests every triangle in the scene (no bounding-sphere culling).
// - The provided geometry.cuh helpers and RandomGPU (curand XORWOW) are used unchanged.

#include "versions/common.cuh"

namespace {

struct Hit {
    HitRecord record;
    int object_index;
    int triangle_index;
};

__device__ Hit find_closest_hit(
    const Ray& ray,
    const Triangle* triangles,
    const FlatObject* objects,
    int n_objects,
    int last_hit
) {
    Hit closest;
    closest.object_index = -1;
    closest.triangle_index = -1;

    for (int object_index = 0; object_index < n_objects; ++object_index) {
        const int end = objects[object_index].tri_end;
        for (int i = objects[object_index].tri_begin; i < end; ++i) {
            if (i == last_hit) continue;
            HitRecord record;
            if (intersect_triangle(ray, triangles[i], &record) && record.t < closest.record.t) {
                closest.record = record;
                closest.object_index = object_index;
                closest.triangle_index = i;
            }
        }
    }
    return closest;
}

// Iterative form of the CPU trace_ray:
//   L = e0 + a0*(e1 + a1*(e2 + ...)) = e0 + a0*e1 + (a0*a1)*e2 + ...
// `throughput` holds the product of the albedos along the path so far.
__device__ Vec3 trace_path(
    Ray ray,
    const RenderSettings& settings,
    const Triangle* triangles,
    const FlatObject* objects,
    int n_objects,
    RandomGPU& random
) {
    Vec3 color;
    Vec3 throughput(1.0f, 1.0f, 1.0f);
    int last_hit = -1;

    for (int depth = 0;; ++depth) {
        const Hit hit = find_closest_hit(ray, triangles, objects, n_objects, last_hit);
        if (hit.object_index < 0) {
            color += multiply(throughput, settings.background_light);
            break;
        }

        const FlatObject& object = objects[hit.object_index];
        color += multiply(throughput, object.emission);
        if (depth >= settings.max_bounces) break;

        throughput = multiply(throughput, object.albedo);
        ray = cosine_weighted_sampling(hit.record, random);
        last_hit = hit.triangle_index;
    }
    return color;
}

__global__ void render_kernel(
    const RenderSettings settings,
    const Triangle* triangles,
    const FlatObject* objects,
    int n_objects,
    Pixel* pixels
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= settings.width || y >= settings.height) return;

    const int pixel_index = y * settings.width + x;
    const int s = settings.supersampling_factor;
    RandomGPU random(hash_u32(std::uint32_t(pixel_index) ^ hash_u32(settings.seed)));

    Vec3 color;
    for (int sy = 0; sy < s; ++sy) {
        for (int sx = 0; sx < s; ++sx) {
            for (int sample = 0; sample < settings.samples_per_pixel; ++sample) {
                const Ray camera_ray = make_camera_ray(settings, x * s + sx, y * s + sy);
                color += trace_path(camera_ray, settings, triangles, objects, n_objects, random);
            }
        }
    }

    pixels[pixel_index] = to_pixel(color / float(settings.samples_per_pixel * s * s));
}

}  // namespace

std::vector<Pixel> render_scene(
    const RenderSettings& settings,
    const std::vector<Object>& objects
) {
    const FlatScene scene = flatten_scene(objects);
    const DeviceBuffer<Triangle> d_triangles(scene.triangles);
    const DeviceBuffer<FlatObject> d_objects(scene.objects);
    DeviceBuffer<Pixel> d_pixels(std::size_t(settings.width) * settings.height);

    const dim3 block(16, 16);
    const dim3 grid(
        (settings.width + block.x - 1) / block.x,
        (settings.height + block.y - 1) / block.y
    );
    render_kernel<<<grid, block>>>(
        settings, d_triangles.get(), d_objects.get(), int(scene.objects.size()), d_pixels.get());
    CUDA_CHECK(cudaGetLastError());

    return d_pixels.download();
}
