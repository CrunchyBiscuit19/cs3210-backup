// v3 occupancy: v2 with a work decomposition that fills the GPU.
//
// - Each pixel's s^2 * spp paths are split into chunks, one thread per (pixel, chunk),
//   so the grid has several waves of threads even for small images (choose_launch_shape).
// - Threads accumulate privately, then atomicAdd into a per-pixel float sum buffer;
//   finalize_kernel divides and tone-maps on the GPU.
// - 1D blocks (default 128 threads, PT_BLOCK_SIZE to override). __launch_bounds__ caps
//   registers so any block size up to kMaxBlockSize can launch.
// - PCG32 replaces curand XORWOW: less RNG state, no curand_init, fewer registers.

#include "versions/fast_geometry.cuh"

namespace {

constexpr int kDefaultBlockSize = 128;
constexpr int kMaxBlockSize = 512;

__device__ Vec3 trace_path(
    Vec3 origin,
    Vec3 direction,
    const RenderSettings& settings,
    const TriangleGPU* __restrict__ triangles,
    const MaterialGPU* __restrict__ materials,
    int n_objects,
    Pcg32& random
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

__global__ void __launch_bounds__(kMaxBlockSize) render_kernel(
    const RenderSettings settings,
    const TriangleGPU* __restrict__ triangles,
    const MaterialGPU* __restrict__ materials,
    int n_objects,
    int samples_per_thread,
    int chunks,
    float* __restrict__ sums
) {
    const long long thread = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const int n_pixels = settings.width * settings.height;
    if (thread >= (long long)n_pixels * chunks) return;

    // Consecutive threads take consecutive pixels of the same chunk, so a warp traces
    // 32 neighbouring pixels of one image row.
    const int pixel = int(thread % n_pixels);
    const int chunk = int(thread / n_pixels);
    const int x = pixel % settings.width;
    const int y = pixel / settings.width;

    const int s = settings.supersampling_factor;
    const int spp = settings.samples_per_pixel;
    const int paths = spp * s * s;
    const int begin = chunk * samples_per_thread;
    const int end = min(begin + samples_per_thread, paths);

    Pcg32 random(mix64((std::uint64_t(settings.seed) << 32) ^ std::uint64_t(thread)), std::uint64_t(thread));

    // Path k of the pixel belongs to subpixel k / spp; rebuild the camera ray when it changes.
    Vec3 color;
    Ray camera_ray;
    int current_subpixel = -1;
    for (int k = begin; k < end; ++k) {
        const int subpixel = k / spp;
        if (subpixel != current_subpixel) {
            current_subpixel = subpixel;
            camera_ray = make_camera_ray(settings, x * s + subpixel % s, y * s + subpixel / s);
        }
        color += trace_path(camera_ray.origin, camera_ray.direction, settings,
                            triangles, materials, n_objects, random);
    }

    atomicAdd(&sums[3 * pixel + 0], color.x);
    atomicAdd(&sums[3 * pixel + 1], color.y);
    atomicAdd(&sums[3 * pixel + 2], color.z);
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

    const int n_pixels = settings.width * settings.height;
    const int s = settings.supersampling_factor;
    const int paths_per_pixel = settings.samples_per_pixel * s * s;
    DeviceBuffer<float> d_sums(std::size_t(n_pixels) * 3);
    d_sums.zero();

    const LaunchShape shape =
        choose_launch_shape(n_pixels, paths_per_pixel, kDefaultBlockSize, kMaxBlockSize);
    const long long threads = (long long)n_pixels * shape.chunks;
    const int blocks = int((threads + shape.block_size - 1) / shape.block_size);
    render_kernel<<<blocks, shape.block_size>>>(
        settings, d_triangles.get(), d_materials.get(), int(scene.objects.size()),
        shape.samples_per_thread, shape.chunks, d_sums.get());
    CUDA_CHECK(cudaGetLastError());

    return finalize_on_gpu(d_sums.get(), n_pixels, paths_per_pixel);
}
