#include <cstdint>
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

namespace {

// An object's triangles are stored contiguously in one array, in object order,
// so a triangle's index matches the CPU code's triangle_index.
struct FlatObject {
    Sphere bound;
    Vec3 albedo;
    Vec3 emission;
    int tri_begin;
    int tri_end;
};

struct Hit {
    HitRecord record;
    int object_index;
    int triangle_index;
};

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

// Bijective 32-bit hash (lowbias32), so neighbouring pixels get unrelated RNG seeds.
__host__ __device__ inline std::uint32_t hash_u32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

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
    std::vector<Triangle> triangles;
    std::vector<FlatObject> flat_objects;
    for (const Object& object : objects) {
        FlatObject flat;
        flat.bound = object.bound;
        flat.albedo = object.albedo;
        flat.emission = object.emission;
        flat.tri_begin = int(triangles.size());
        triangles.insert(triangles.end(), object.triangles.begin(), object.triangles.end());
        flat.tri_end = int(triangles.size());
        flat_objects.push_back(flat);
    }

    Triangle* d_triangles = to_device(triangles);
    FlatObject* d_objects = to_device(flat_objects);
    const std::size_t n_pixels = std::size_t(settings.width) * settings.height;
    Pixel* d_pixels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pixels, n_pixels * sizeof(Pixel)));

    const dim3 block(16, 16);
    const dim3 grid(
        (settings.width + block.x - 1) / block.x,
        (settings.height + block.y - 1) / block.y
    );
    render_kernel<<<grid, block>>>(
        settings, d_triangles, d_objects, int(flat_objects.size()), d_pixels);
    CUDA_CHECK(cudaGetLastError());

    std::vector<Pixel> pixels(n_pixels);
    CUDA_CHECK(cudaMemcpy(pixels.data(), d_pixels, n_pixels * sizeof(Pixel), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pixels));
    CUDA_CHECK(cudaFree(d_objects));
    CUDA_CHECK(cudaFree(d_triangles));
    return pixels;
}
