#ifndef PT_COMMON_CUH
#define PT_COMMON_CUH

// Host and device utilities shared by every render_scene version in versions/.
// Include paths are relative to this file, so it works both from versions/*.cu
// (built with -I.) and from the top-level path_tracing.cu.
//
// Defines a __global__ kernel, so include it from exactly one .cu file per executable.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../geometry.cuh"
#include "../graphics.cuh"

// main.cu catches std::exception and prints what(), so CUDA failures are reported cleanly.
#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t cuda_check_error_ = (call);                                      \
        if (cuda_check_error_ != cudaSuccess) {                                            \
            throw std::runtime_error(std::string(#call) + " failed at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + ": " +                     \
                                     cudaGetErrorString(cuda_check_error_));               \
        }                                                                                  \
    } while (0)

// Positive integer tuning override from the environment (e.g. PT_BLOCK_SIZE=256),
// used for parameter sweeps. Unset or invalid values fall back to the default.
inline int getenv_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (*end != '\0' || parsed <= 0 || parsed > INT32_MAX) return fallback;
    return int(parsed);
}

// Prints launch configuration to stderr when PT_VERBOSE is set.
inline bool verbose() {
    const char* value = std::getenv("PT_VERBOSE");
    return value != nullptr && *value != '\0' && *value != '0';
}

// Bijective 32-bit integer hash (lowbias32), for decorrelating RNG seeds of neighbouring pixels.
__host__ __device__ inline std::uint32_t hash_u32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// Owning device allocation of `count` elements of T.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }

    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMemcpy(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }
    std::size_t size() const { return count_; }

    void zero() {
        if (count_ > 0) CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
    }

    // Blocking copy back to the host; also surfaces errors from earlier kernel launches.
    std::vector<T> download() const {
        std::vector<T> host(count_);
        if (count_ > 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
        }
        return host;
    }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

// The scene flattened into the layout of the CPU reference: triangles stored contiguously in
// object order, so a triangle's index here equals the CPU code's triangle_index.
struct FlatObject {
    Sphere bound;
    Vec3 albedo;
    Vec3 emission;
    int tri_begin;
    int tri_end;
};

struct FlatScene {
    std::vector<Triangle> triangles;
    std::vector<FlatObject> objects;
};

inline FlatScene flatten_scene(const std::vector<Object>& objects) {
    FlatScene scene;
    scene.objects.reserve(objects.size());
    for (const Object& object : objects) {
        FlatObject flat;
        flat.bound = object.bound;
        flat.albedo = object.albedo;
        flat.emission = object.emission;
        flat.tri_begin = int(scene.triangles.size());
        scene.triangles.insert(scene.triangles.end(), object.triangles.begin(), object.triangles.end());
        flat.tri_end = int(scene.triangles.size());
        scene.objects.push_back(flat);
    }
    return scene;
}

// How the paths of each pixel are split across threads (v3 and later).
struct LaunchShape {
    int block_size;
    int samples_per_thread;  // consecutive paths traced by one thread
    int chunks;              // threads per pixel = ceil(paths_per_pixel / samples_per_thread)
};

// Splits each pixel's paths into chunks so the grid holds about kTargetWaves times as many
// threads as the GPU can keep resident. That fills the GPU even for small images (ufo has only
// 60k pixels) and keeps the final partial wave short. PT_BLOCK_SIZE and PT_SAMPLES_PER_THREAD
// override the defaults for parameter sweeps.
inline LaunchShape choose_launch_shape(
    long long threads_per_chunk,
    int paths_per_pixel,
    int default_block_size,
    int max_block_size
) {
    constexpr long long kTargetWaves = 4;

    int device = 0;
    int sm_count = 0;
    int threads_per_sm = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device));

    LaunchShape shape;
    const int requested_block = getenv_int("PT_BLOCK_SIZE", default_block_size);
    shape.block_size = std::min(max_block_size, std::max(32, requested_block / 32 * 32));

    const long long target_threads = kTargetWaves * sm_count * threads_per_sm;
    long long chunks = (target_threads + threads_per_chunk - 1) / threads_per_chunk;
    chunks = std::max(1LL, std::min<long long>(chunks, paths_per_pixel));
    const int default_samples = int((paths_per_pixel + chunks - 1) / chunks);
    shape.samples_per_thread =
        std::min(paths_per_pixel, getenv_int("PT_SAMPLES_PER_THREAD", default_samples));
    shape.chunks = (paths_per_pixel + shape.samples_per_thread - 1) / shape.samples_per_thread;

    if (verbose()) {
        std::fprintf(stderr,
                     "launch: %d SMs x %d threads/SM, block %d, %d samples/thread, %d chunks/pixel, "
                     "%lld threads\n",
                     sm_count, threads_per_sm, shape.block_size, shape.samples_per_thread,
                     shape.chunks, threads_per_chunk * shape.chunks);
    }
    return shape;
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

// Runs finalize_kernel on the GPU and downloads the image.
inline std::vector<Pixel> finalize_on_gpu(const float* d_sums, int n_pixels, int paths_per_pixel) {
    DeviceBuffer<Pixel> d_pixels(n_pixels);
    constexpr int kBlock = 256;
    finalize_kernel<<<(n_pixels + kBlock - 1) / kBlock, kBlock>>>(
        d_sums, d_pixels.get(), n_pixels, float(paths_per_pixel));
    CUDA_CHECK(cudaGetLastError());
    return d_pixels.download();
}

#endif
