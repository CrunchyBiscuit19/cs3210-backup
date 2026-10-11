// v2 ray cost: v1 with cheaper per-ray work (see versions/fast_geometry.cuh).
//
// Same work decomposition as v0/v1 (one thread per pixel, 16x16 blocks, curand
// RNG), so the v1 -> v2 difference isolates per-ray arithmetic and memory:
// - precomputed triangle edges in aligned float4 loads, t-only intersection tests,
//   normal and position computed once per bounce;
// - single-precision math only (no FP64 from double literals), fast intrinsics;
// - culling data in __constant__ memory, and objects that start beyond the
//   current closest hit are skipped;
// - the camera ray is built once per subpixel instead of once per sample;
// - paths stop early once the throughput is exactly black.

#include "versions/fast_geometry.cuh"

namespace {

__device__ Vec3 trace_path(
    Vec3 origin,
    Vec3 direction,
    const RenderSettings& settings,
    const TriangleGPU* __restrict__ triangles,
    const MaterialGPU* __restrict__ materials,
    int n_objects,
    RandomGPU& random
) {
    Vec3 color;
    Vec3 throughput(1.0f, 1.0f, 1.0f);
    int last_hit = -1;

    for (int depth = 0;; ++depth) {
        const HitInfo hit = find_closest_hit(origin, direction, triangles, n_objects, last_hit);
        if (hit.triangle < 0) {
            color += multiply(throughput, settings.background_light);
            break;
        }

        const MaterialGPU& material = materials[hit.object];
        color += multiply(throughput, xyz(__ldg(&material.emission)));
        if (depth >= settings.max_bounces) break;

        throughput = multiply(throughput, xyz(__ldg(&material.albedo)));
        if (is_black(throughput)) break;

        const Vec3 normal = facing_normal(triangles + hit.triangle, direction);
        origin = origin + hit.t * direction;
        direction = sample_cosine_direction(normal, random);
        last_hit = hit.triangle;
    }
    return color;
}

__global__ void render_kernel(
    const RenderSettings settings,
    const TriangleGPU* __restrict__ triangles,
    const MaterialGPU* __restrict__ materials,
    int n_objects,
    Pixel* __restrict__ pixels
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
            const Ray camera_ray = make_camera_ray(settings, x * s + sx, y * s + sy);
            for (int sample = 0; sample < settings.samples_per_pixel; ++sample) {
                color += trace_path(camera_ray.origin, camera_ray.direction, settings,
                                    triangles, materials, n_objects, random);
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
    const FastScene scene = build_fast_scene(objects);
    upload_objects_to_constant(scene.objects);
    const DeviceBuffer<TriangleGPU> d_triangles(scene.triangles);
    const DeviceBuffer<MaterialGPU> d_materials(scene.materials);
    DeviceBuffer<Pixel> d_pixels(std::size_t(settings.width) * settings.height);

    const dim3 block(16, 16);
    const dim3 grid(
        (settings.width + block.x - 1) / block.x,
        (settings.height + block.y - 1) / block.y
    );
    render_kernel<<<grid, block>>>(
        settings, d_triangles.get(), d_materials.get(), int(scene.objects.size()), d_pixels.get());
    CUDA_CHECK(cudaGetLastError());

    return d_pixels.download();
}
