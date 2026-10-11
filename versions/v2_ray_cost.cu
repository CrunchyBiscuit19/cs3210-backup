// v2 ray cost: v1 with cheaper per-ray work.
//
// Same work decomposition as v0/v1 (one thread per pixel, 16x16 blocks, curand
// RNG), so the v1 -> v2 difference isolates per-ray arithmetic and memory:
// - Triangles store p0 and the precomputed edges e1 = p1 - p0, e2 = p2 - p0 as
//   float4s: three aligned 16-byte loads per triangle and no subtractions.
// - The intersection loop computes only the hit distance t; the position and
//   normal are computed once per bounce, for the closest hit only.
// - Single-precision math only. The geometry.cuh helpers use double literals
//   (e.g. `1.0 / determinant`), which compile to slow FP64 instructions.
// - Culling data lives in __constant__ memory: all lanes of a warp read the same
//   object at the same time, so each read is one broadcast. Materials are read with
//   a divergent index (each lane hit a different object), so they stay in global memory.
// - Objects whose bounding sphere starts beyond the current closest hit are skipped.
// - The camera ray is built once per subpixel instead of once per sample.
// - Paths stop early once the throughput is exactly black.

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
    const std::size_t n_pixels = std::size_t(settings.width) * settings.height;
    Pixel* d_pixels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pixels, n_pixels * sizeof(Pixel)));

    const dim3 block(16, 16);
    const dim3 grid(
        (settings.width + block.x - 1) / block.x,
        (settings.height + block.y - 1) / block.y
    );
    render_kernel<<<grid, block>>>(
        settings, d_triangles, d_materials, int(gpu_objects.size()), d_pixels);
    CUDA_CHECK(cudaGetLastError());

    std::vector<Pixel> pixels(n_pixels);
    CUDA_CHECK(cudaMemcpy(pixels.data(), d_pixels, n_pixels * sizeof(Pixel), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pixels));
    CUDA_CHECK(cudaFree(d_materials));
    CUDA_CHECK(cudaFree(d_triangles));
    return pixels;
}
