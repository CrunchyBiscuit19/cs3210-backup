#ifndef PT_FAST_GEOMETRY_CUH
#define PT_FAST_GEOMETRY_CUH

// Cheaper ray primitives used by v2 and later, replacing the geometry.cuh helpers in the hot loop:
//
// - Triangles store p0 and the precomputed edges e1 = p1 - p0, e2 = p2 - p0 as float4s, so each
//   triangle is three aligned 16-byte loads and no subtractions.
// - The intersection loop computes only the hit distance t. Position and normal are computed once
//   per bounce, for the closest hit only.
// - All arithmetic is single precision. The geometry.cuh helpers use double literals
//   (e.g. `1.0 / determinant`), which compile to slow FP64 instructions on the GPU.
// - Per-object culling data lives in __constant__ memory: every lane in a warp reads the same
//   object at the same time, so each read is a single broadcast. Materials are read with a
//   divergent index (each lane hit a different object), so they stay in global memory.
//
// Defines __constant__ storage, so include it from exactly one .cu file per executable.

#include "common.cuh"

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

struct FastScene {
    std::vector<TriangleGPU> triangles;
    std::vector<ObjectGPU> objects;
    std::vector<MaterialGPU> materials;
};

inline FastScene build_fast_scene(const std::vector<Object>& objects) {
    if (objects.size() > std::size_t(kMaxObjects)) {
        throw std::runtime_error("scene has " + std::to_string(objects.size()) +
                                 " objects; at most " + std::to_string(kMaxObjects) + " are supported");
    }

    FastScene scene;
    for (const Object& object : objects) {
        ObjectGPU record{};
        const Sphere& bound = object.bound;
        record.sphere = make_float4(bound.center.x, bound.center.y, bound.center.z,
                                    bound.radius * bound.radius);
        record.tri_begin = int(scene.triangles.size());
        for (const Triangle& triangle : object.triangles) {
            const Vec3 e1 = triangle.p1 - triangle.p0;
            const Vec3 e2 = triangle.p2 - triangle.p0;
            scene.triangles.push_back(TriangleGPU{
                make_float4(triangle.p0.x, triangle.p0.y, triangle.p0.z, 0.0f),
                make_float4(e1.x, e1.y, e1.z, 0.0f),
                make_float4(e2.x, e2.y, e2.z, 0.0f),
            });
        }
        record.tri_end = int(scene.triangles.size());
        scene.objects.push_back(record);

        scene.materials.push_back(MaterialGPU{
            make_float4(object.albedo.x, object.albedo.y, object.albedo.z, 0.0f),
            make_float4(object.emission.x, object.emission.y, object.emission.z, 0.0f),
        });
    }
    return scene;
}

inline void upload_objects_to_constant(const std::vector<ObjectGPU>& objects) {
    if (!objects.empty()) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_objects, objects.data(), objects.size() * sizeof(ObjectGPU)));
    }
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

#endif
