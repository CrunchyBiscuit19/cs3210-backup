// v4 coherence: v3 with less warp divergence.
//
// - Path regeneration: the nested sample/bounce loops become a single loop in which
//   each iteration is one closest-hit query. When a lane's path ends (miss, max bounces
//   or black throughput) it starts its next sample straight away, instead of idling
//   until the longest path in its warp finishes.
// - Warp tiling: each warp traces an 8x4 block of pixels rather than 32 pixels of one
//   row, so the rays in a warp are closer together and tend to visit the same objects.
//
// Work split, RNG, accumulation and finalisation are the same as v3.

#include "versions/fast_geometry.cuh"

namespace {

constexpr int kDefaultBlockSize = 128;
constexpr int kMaxBlockSize = 512;
constexpr int kWarpSize = 32;
constexpr int kTileWidth = 8;
constexpr int kTileHeight = kWarpSize / kTileWidth;

__global__ void __launch_bounds__(kMaxBlockSize) render_kernel(
    const RenderSettings settings,
    const TriangleGPU* __restrict__ triangles,
    const MaterialGPU* __restrict__ materials,
    int n_objects,
    int tiles_x,
    int n_tiles,
    int samples_per_thread,
    int chunks,
    float* __restrict__ sums
) {
    const long long thread = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long warp = thread / kWarpSize;
    const int lane = int(thread % kWarpSize);
    if (warp >= (long long)n_tiles * chunks) return;

    const int tile = int(warp % n_tiles);
    const int chunk = int(warp / n_tiles);
    const int x = (tile % tiles_x) * kTileWidth + lane % kTileWidth;
    const int y = (tile / tiles_x) * kTileHeight + lane / kTileWidth;
    if (x >= settings.width || y >= settings.height) return;  // partial tiles at the image edge

    const int s = settings.supersampling_factor;
    const int spp = settings.samples_per_pixel;
    const int paths = spp * s * s;
    int k = chunk * samples_per_thread;  // index of the current path within the pixel
    const int end = min(k + samples_per_thread, paths);
    if (k >= end) return;

    Pcg32 random(mix64((std::uint64_t(settings.seed) << 32) ^ std::uint64_t(thread)), std::uint64_t(thread));

    // Path k belongs to subpixel k / spp; the camera ray is rebuilt only when that changes.
    int current_subpixel = k / spp;
    Ray camera_ray = make_camera_ray(settings, x * s + current_subpixel % s, y * s + current_subpixel / s);

    Vec3 color;
    Vec3 origin = camera_ray.origin;
    Vec3 direction = camera_ray.direction;
    Vec3 throughput(1.0f, 1.0f, 1.0f);
    int depth = 0;
    int last_hit = -1;

    while (true) {
        const HitInfo hit = find_closest_hit(origin, direction, triangles, n_objects, last_hit);

        bool path_done;
        if (hit.triangle < 0) {
            color += multiply(throughput, settings.background_light);
            path_done = true;
        } else {
            const MaterialGPU& material = materials[hit.object];
            color += multiply(throughput, xyz(__ldg(&material.emission)));
            throughput = multiply(throughput, xyz(__ldg(&material.albedo)));
            path_done = depth >= settings.max_bounces || is_black(throughput);
            if (!path_done) {
                const Vec3 normal = facing_normal(triangles + hit.triangle, direction);
                origin = origin + hit.t * direction;
                direction = sample_cosine_direction(normal, random);
                last_hit = hit.triangle;
                ++depth;
            }
        }

        if (path_done) {
            if (++k >= end) break;
            const int subpixel = k / spp;
            if (subpixel != current_subpixel) {
                current_subpixel = subpixel;
                camera_ray = make_camera_ray(settings, x * s + subpixel % s, y * s + subpixel / s);
            }
            origin = camera_ray.origin;
            direction = camera_ray.direction;
            throughput = Vec3(1.0f, 1.0f, 1.0f);
            depth = 0;
            last_hit = -1;
        }
    }

    const int pixel = y * settings.width + x;
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

    const int tiles_x = (settings.width + kTileWidth - 1) / kTileWidth;
    const int tiles_y = (settings.height + kTileHeight - 1) / kTileHeight;
    const int n_tiles = tiles_x * tiles_y;

    // Warps must not straddle tiles, so the block size stays a multiple of 32
    // (choose_launch_shape rounds it).
    const LaunchShape shape = choose_launch_shape(
        (long long)n_tiles * kWarpSize, paths_per_pixel, kDefaultBlockSize, kMaxBlockSize);
    const long long threads = (long long)n_tiles * kWarpSize * shape.chunks;
    const int blocks = int((threads + shape.block_size - 1) / shape.block_size);
    render_kernel<<<blocks, shape.block_size>>>(
        settings, d_triangles.get(), d_materials.get(), int(scene.objects.size()),
        tiles_x, n_tiles, shape.samples_per_thread, shape.chunks, d_sums.get());
    CUDA_CHECK(cudaGetLastError());

    return finalize_on_gpu(d_sums.get(), n_pixels, paths_per_pixel);
}
