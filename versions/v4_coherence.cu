// v4 coherence: v3 with less warp divergence.
//
// - Path regeneration: the nested sample/bounce loops become a single loop in which
//   each iteration is one closest-hit query. When a lane's path ends (miss, max bounces
//   or black throughput) it starts its next sample straight away, instead of idling
//   until the longest path in its warp finishes.
// - Warp tiling: each warp traces an 8x4 block of pixels rather than 32 pixels of one
//   row, so the rays in a warp are closer together and tend to visit the same objects.
//
// Work split, RNG, accumulation, finalisation and per-ray code are the same as v3.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "graphics.cuh"
#include "geometry.cuh"

// main.cu catches std::exception and prints what(), so CUDA failures are reported cleanly.
#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t cuda_error_ = (call);                                            \
        if (cuda_error_ != cudaSuccess) {                                                  \
            throw std::runtime_error(std::string(#call) + " failed at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + ": " +                     \
                                     cudaGetErrorString(cuda_error_));                     \
        }                                                                                  \
    } while (0)

constexpr int kMaxObjects = 128;  // brief: the number of objects does not exceed 128

struct alignas(16) TriangleGPU {
    float4 p0;  // xyz = first vertex
    float4 e1;  // xyz = p1 - p0
    float4 e2;  // xyz = p2 - p0
};

struct alignas(16) ObjectGPU {
    float4 sphere;  // xyz = bounding sphere center, w = radius^2
    int tri_begin;
    int tri_end;
    int padding[2];
};

struct alignas(16) MaterialGPU {
    float4 albedo;    // xyz
    float4 emission;  // xyz
};

__constant__ ObjectGPU c_objects[kMaxObjects];

namespace {

constexpr int kDefaultBlockSize = 128;
constexpr int kMaxBlockSize = 512;
constexpr int kWarpSize = 32;
constexpr int kTileWidth = 8;
constexpr int kTileHeight = kWarpSize / kTileWidth;

// Copies a host vector into a new device allocation (nullptr when empty).
template <typename T>
T* to_device(const std::vector<T>& host) {
    T* device = nullptr;
    if (!host.empty()) {
        CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return device;
}

// Positive integer tuning override from the environment (e.g. PT_BLOCK_SIZE=256),
// used for parameter sweeps. Unset or invalid values fall back to the default.
int getenv_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (*end != '\0' || parsed <= 0 || parsed > INT32_MAX) return fallback;
    return int(parsed);
}

// How the paths of each pixel are split across threads.
struct LaunchShape {
    int block_size;
    int samples_per_thread;  // consecutive paths traced by one thread
    int chunks;              // threads per pixel = ceil(paths_per_pixel / samples_per_thread)
};

// Splits each pixel's paths into chunks so the grid holds about kTargetWaves times as many
// threads as the GPU can keep resident. That fills the GPU even for small images (ufo has only
// 60k pixels) and keeps the final partial wave short. PT_BLOCK_SIZE and PT_SAMPLES_PER_THREAD
// override the defaults; PT_VERBOSE=1 prints the chosen shape.
LaunchShape choose_launch_shape(long long threads_per_chunk, int paths_per_pixel) {
    constexpr long long kTargetWaves = 4;

    int device = 0;
    int sm_count = 0;
    int threads_per_sm = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device));

    LaunchShape shape;
    const int requested_block = getenv_int("PT_BLOCK_SIZE", kDefaultBlockSize);
    shape.block_size = std::min(kMaxBlockSize, std::max(32, requested_block / 32 * 32));

    const long long target_threads = kTargetWaves * sm_count * threads_per_sm;
    long long chunks = (target_threads + threads_per_chunk - 1) / threads_per_chunk;
    chunks = std::max(1LL, std::min<long long>(chunks, paths_per_pixel));
    const int default_samples = int((paths_per_pixel + chunks - 1) / chunks);
    shape.samples_per_thread =
        std::min(paths_per_pixel, getenv_int("PT_SAMPLES_PER_THREAD", default_samples));
    shape.chunks = (paths_per_pixel + shape.samples_per_thread - 1) / shape.samples_per_thread;

    if (getenv_int("PT_VERBOSE", 0) > 0) {
        std::fprintf(stderr,
                     "launch: %d SMs x %d threads/SM, block %d, %d samples/thread, %d chunks/pixel, "
                     "%lld threads\n",
                     sm_count, threads_per_sm, shape.block_size, shape.samples_per_thread,
                     shape.chunks, threads_per_chunk * shape.chunks);
    }
    return shape;
}

// 64-bit mixer (splitmix64 finalizer) for seeding per-thread generators.
__host__ __device__ inline std::uint64_t mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// PCG32 (O'Neill, pcg-random.org): 16 bytes of state versus 48 for curand's XORWOW, and no
// curand_init. Each thread gets its own stream.
class Pcg32 {
public:
    __device__ Pcg32(std::uint64_t seed, std::uint64_t stream) : state_(0), increment_((stream << 1) | 1) {
        next();
        state_ += seed;
        next();
    }

    __device__ __forceinline__ std::uint32_t next() {
        const std::uint64_t old = state_;
        state_ = old * 6364136223846793005ULL + increment_;
        const std::uint32_t xorshifted = std::uint32_t(((old >> 18) ^ old) >> 27);
        const std::uint32_t rotation = std::uint32_t(old >> 59);
        return (xorshifted >> rotation) | (xorshifted << ((32 - rotation) & 31));
    }

    // Uniform float in [0, 1) from the top 24 bits.
    __device__ __forceinline__ float uniform() {
        return float(next() >> 8) * (1.0f / 16777216.0f);
    }

private:
    std::uint64_t state_;
    std::uint64_t increment_;
};

__device__ __forceinline__ Vec3 xyz(const float4& v) {
    return Vec3(v.x, v.y, v.z);
}

__device__ __forceinline__ bool is_black(const Vec3& v) {
    return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f;
}

// Distance at which the ray enters the sphere (0 if it starts inside), or INF if it misses.
// Hits exactly when geometry.cuh's intersect_sphere returns true (far root > EPSILON).
__device__ __forceinline__ float sphere_entry(const Vec3& origin, const Vec3& direction, const float4& sphere) {
    const Vec3 offset = origin - xyz(sphere);
    const float half_b = dot(offset, direction);
    const float c = dot(offset, offset) - sphere.w;
    const float discriminant = half_b * half_b - c;
    if (discriminant < 0.0f) return INF;

    const float root = sqrtf(discriminant);
    if (-half_b + root <= EPSILON) return INF;
    return fmaxf(-half_b - root, 0.0f);
}

// Moller-Trumbore with the same tests as geometry.cuh's intersect_triangle, returning only t
// (INF on a miss).
__device__ __forceinline__ float intersect_triangle_t(
    const Vec3& origin,
    const Vec3& direction,
    const TriangleGPU* __restrict__ triangle
) {
    const Vec3 p0 = xyz(__ldg(&triangle->p0));
    const Vec3 e1 = xyz(__ldg(&triangle->e1));
    const Vec3 e2 = xyz(__ldg(&triangle->e2));

    const Vec3 p = cross(direction, e2);
    const float determinant = dot(e1, p);
    if (fabsf(determinant) <= EPSILON) return INF;

    const float inverse_determinant = __fdividef(1.0f, determinant);
    const Vec3 offset = origin - p0;
    const float u = dot(offset, p) * inverse_determinant;
    if (u < 0.0f || u > 1.0f) return INF;

    const Vec3 q = cross(offset, e1);
    const float v = dot(direction, q) * inverse_determinant;
    if (v < 0.0f || u + v > 1.0f) return INF;

    const float t = dot(e2, q) * inverse_determinant;
    return t > EPSILON ? t : INF;
}

// Unit normal of the triangle, flipped to face against the incoming ray (as geometry.cuh does).
__device__ __forceinline__ Vec3 facing_normal(const TriangleGPU* __restrict__ triangle, const Vec3& direction) {
    const Vec3 normal = cross(xyz(__ldg(&triangle->e1)), xyz(__ldg(&triangle->e2)));
    const Vec3 unit = normal * rsqrtf(dot(normal, normal));
    return dot(unit, direction) > 0.0f ? -unit : unit;
}

struct HitInfo {
    float t;
    int triangle;  // -1 on a miss
    int object;
};

// Closest hit over the whole scene, skipping `last_hit` (brief, recommendation 4). An object is
// skipped when the ray misses its bounding sphere or enters it beyond the closest hit so far.
__device__ __forceinline__ HitInfo find_closest_hit(
    const Vec3& origin,
    const Vec3& direction,
    const TriangleGPU* __restrict__ triangles,
    int n_objects,
    int last_hit
) {
    HitInfo closest{INF, -1, -1};
    for (int object = 0; object < n_objects; ++object) {
        if (sphere_entry(origin, direction, c_objects[object].sphere) >= closest.t) continue;

        const int end = c_objects[object].tri_end;
        for (int i = c_objects[object].tri_begin; i < end; ++i) {
            if (i == last_hit) continue;
            const float t = intersect_triangle_t(origin, direction, triangles + i);
            if (t < closest.t) {
                closest.t = t;
                closest.triangle = i;
                closest.object = object;
            }
        }
    }
    return closest;
}

// Cosine-weighted hemisphere direction around `normal`, with the same construction as
// geometry.cuh's cosine_weighted_sampling but in single precision with fast intrinsics.
template <typename Random>
__device__ __forceinline__ Vec3 sample_cosine_direction(const Vec3& normal, Random& random) {
    const float z = 1.0f - 2.0f * random.uniform();
    const float angle = (2.0f * PI) * random.uniform();
    const float radius = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float sin_angle;
    float cos_angle;
    __sincosf(angle, &sin_angle, &cos_angle);

    const Vec3 direction = normal + Vec3(radius * cos_angle, radius * sin_angle, z);
    const float length_squared = dot(direction, direction);
    if (length_squared < EPSILON * EPSILON) return normal;
    return direction * rsqrtf(length_squared);
}

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

// Converts per-pixel radiance sums (3 floats per pixel) into tone-mapped pixels.
__global__ void finalize_kernel(
    const float* __restrict__ sums,
    Pixel* __restrict__ pixels,
    int n_pixels,
    float paths_per_pixel
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;
    const Vec3 sum(sums[3 * i], sums[3 * i + 1], sums[3 * i + 2]);
    pixels[i] = to_pixel(sum / paths_per_pixel);
}

}  // namespace

std::vector<Pixel> render_scene(
    const RenderSettings& settings,
    const std::vector<Object>& objects
) {
    if (objects.size() > std::size_t(kMaxObjects)) {
        throw std::runtime_error("scene has " + std::to_string(objects.size()) +
                                 " objects; at most " + std::to_string(kMaxObjects) + " are supported");
    }

    std::vector<TriangleGPU> triangles;
    std::vector<ObjectGPU> gpu_objects;
    std::vector<MaterialGPU> materials;
    for (const Object& object : objects) {
        ObjectGPU record{};
        const Sphere& bound = object.bound;
        record.sphere = make_float4(bound.center.x, bound.center.y, bound.center.z,
                                    bound.radius * bound.radius);
        record.tri_begin = int(triangles.size());
        for (const Triangle& triangle : object.triangles) {
            const Vec3 e1 = triangle.p1 - triangle.p0;
            const Vec3 e2 = triangle.p2 - triangle.p0;
            triangles.push_back(TriangleGPU{
                make_float4(triangle.p0.x, triangle.p0.y, triangle.p0.z, 0.0f),
                make_float4(e1.x, e1.y, e1.z, 0.0f),
                make_float4(e2.x, e2.y, e2.z, 0.0f),
            });
        }
        record.tri_end = int(triangles.size());
        gpu_objects.push_back(record);

        materials.push_back(MaterialGPU{
            make_float4(object.albedo.x, object.albedo.y, object.albedo.z, 0.0f),
            make_float4(object.emission.x, object.emission.y, object.emission.z, 0.0f),
        });
    }

    if (!gpu_objects.empty()) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_objects, gpu_objects.data(), gpu_objects.size() * sizeof(ObjectGPU)));
    }
    TriangleGPU* d_triangles = to_device(triangles);
    MaterialGPU* d_materials = to_device(materials);

    const int n_pixels = settings.width * settings.height;
    const int s = settings.supersampling_factor;
    const int paths_per_pixel = settings.samples_per_pixel * s * s;
    float* d_sums = nullptr;
    Pixel* d_pixels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sums, std::size_t(n_pixels) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sums, 0, std::size_t(n_pixels) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pixels, std::size_t(n_pixels) * sizeof(Pixel)));

    const int tiles_x = (settings.width + kTileWidth - 1) / kTileWidth;
    const int tiles_y = (settings.height + kTileHeight - 1) / kTileHeight;
    const int n_tiles = tiles_x * tiles_y;

    // One warp per tile; choose_launch_shape keeps the block size a multiple of 32,
    // so warps never straddle blocks.
    const LaunchShape shape = choose_launch_shape((long long)n_tiles * kWarpSize, paths_per_pixel);
    const long long threads = (long long)n_tiles * kWarpSize * shape.chunks;
    const int blocks = int((threads + shape.block_size - 1) / shape.block_size);
    render_kernel<<<blocks, shape.block_size>>>(
        settings, d_triangles, d_materials, int(gpu_objects.size()),
        tiles_x, n_tiles, shape.samples_per_thread, shape.chunks, d_sums);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kFinalizeBlock = 256;
    finalize_kernel<<<(n_pixels + kFinalizeBlock - 1) / kFinalizeBlock, kFinalizeBlock>>>(
        d_sums, d_pixels, n_pixels, float(paths_per_pixel));
    CUDA_CHECK(cudaGetLastError());

    std::vector<Pixel> pixels(n_pixels);
    CUDA_CHECK(cudaMemcpy(pixels.data(), d_pixels, std::size_t(n_pixels) * sizeof(Pixel), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pixels));
    CUDA_CHECK(cudaFree(d_sums));
    CUDA_CHECK(cudaFree(d_materials));
    CUDA_CHECK(cudaFree(d_triangles));
    return pixels;
}
