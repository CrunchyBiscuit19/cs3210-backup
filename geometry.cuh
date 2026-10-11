#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <curand_kernel.h>
#include <random>

typedef float FP;

constexpr float EPSILON = 1.0e-5;
constexpr float INF = std::numeric_limits<FP>::infinity();
constexpr float PI = 3.14159265358979323846;

struct Vec3 {
    float x;
    float y;
    float z;

    __host__ __device__ Vec3(): x(0.0), y(0.0), z(0.0) {}
    __host__ __device__ Vec3(float x_value, float y_value, float z_value)
        : x(x_value), y(y_value), z(z_value) {}

    __host__ __device__ Vec3 operator+(const Vec3& rhs) const {
        return Vec3(x + rhs.x, y + rhs.y, z + rhs.z);
    }

    __host__ __device__ Vec3 operator-(const Vec3& rhs) const {
        return Vec3(x - rhs.x, y - rhs.y, z - rhs.z);
    }

    __host__ __device__ Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    __host__ __device__ Vec3 operator*(float scale) const {
        return Vec3(x * scale, y * scale, z * scale);
    }

    __host__ __device__ Vec3 operator/(float scale) const {
        return Vec3(x / scale, y / scale, z / scale);
    }

    __host__ __device__ Vec3& operator+=(const Vec3& rhs) {
        return *this = *this + rhs;
    }

    __host__ __device__ Vec3& operator-=(const Vec3& rhs) {
        return *this = *this - rhs;
    }

    __host__ __device__ Vec3& operator*=(float scale) {
        return *this = *this * scale;
    }

    __host__ __device__ Vec3& operator/=(float scale) {
        return *this = *this / scale;
    }
};

__host__ __device__ inline Vec3 operator*(float scale, const Vec3& vector) {
    return vector * scale;
}

__host__ __device__ inline Vec3 multiply(const Vec3& a, const Vec3& b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ inline float length(const Vec3& vector) {
    return std::sqrt(dot(vector, vector));
}

__host__ __device__ inline Vec3 normalize(const Vec3& vector) {
    return vector / length(vector);
}

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

struct Sphere {
    Vec3 center;
    float radius;
};

struct Triangle {
    Vec3 p0;
    Vec3 p1;
    Vec3 p2;
};

struct Object {
    Sphere bound;
    std::vector<Triangle> triangles;
    Vec3 albedo;
    Vec3 emission;
};

struct HitRecord {
    float t;
    Vec3 position;
    Vec3 unit_normal;

    __device__ __host__ HitRecord(): t(INF) {}
};

__host__ __device__ inline Ray make_ray(const Vec3& from, const Vec3& to) {
    return Ray{from, normalize(to - from)};
}

/*
Return true if the ray intersects the sphere, false otherwise.
The ray's direction is assumed to be normalized.
*/
__host__ __device__ inline bool intersect_sphere(
    const Ray& ray,
    const Sphere& sphere
) {
    const Vec3 offset = ray.origin - sphere.center;
    const float half_b = dot(offset, ray.direction);
    const float c = dot(offset, offset) - sphere.radius * sphere.radius;
    const float discriminant = half_b * half_b - c;
    if (discriminant < 0.0) {
        return false;
    }

    const float root = std::sqrt(discriminant);
    float t = -half_b - root;
    if (t <= EPSILON) {
        t = -half_b + root;
        if (t <= EPSILON) {
            return false;
        }
    }

    return true;
}


/*
Return true and fill the hit record if the ray intersects the triangle, false otherwise.
The ray's direction is assumed to be normalized.

For HitRecord:
- The ray intersection point is stored in `position`.
- The unit normal of the triangle is stored in `unit_normal`.
- The distance from the ray origin to the intersection point is `t`.
- We can represent the intersection point as `ray.origin + t * ray.direction`.
*/
__host__ __device__ inline bool intersect_triangle(
    const Ray& ray,
    const Triangle& triangle,
    HitRecord* record
) {
    const Vec3 edge_1 = triangle.p1 - triangle.p0;
    const Vec3 edge_2 = triangle.p2 - triangle.p0;
    const Vec3 p = cross(ray.direction, edge_2);
    const float determinant = dot(edge_1, p);
    if (std::fabs(determinant) <= EPSILON) {
        return false;
    }

    const float inverse_determinant = 1.0 / determinant;
    const Vec3 offset = ray.origin - triangle.p0;
    const float u = dot(offset, p) * inverse_determinant;
    if (u < 0.0 || u > 1.0) {
        return false;
    }

    const Vec3 q = cross(offset, edge_1);
    const float v = dot(ray.direction, q) * inverse_determinant;
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }

    const float t = dot(edge_2, q) * inverse_determinant;
    if (t <= EPSILON) {
        return false;
    }

    Vec3 normal = normalize(cross(edge_1, edge_2));
    if (dot(normal, ray.direction) > 0.0) {
        normal = -normal;
    }
    record->t = t;
    record->position = ray.origin + t * ray.direction;
    record->unit_normal = normal;
    
    return true;
}

class RandomCPU {
private:
    std::mt19937 generator_;
    std::uniform_real_distribution<FP> distribution_;

public:
    RandomCPU(std::uint32_t seed) : generator_(seed), distribution_(0.0, 1.0) {}

    float uniform() {
        return distribution_(generator_);
    }
};

class RandomGPU {
public:
    __device__
    RandomGPU(uint32_t seed)
    {
        curand_init(seed, 0, 0, &state_);
    }

    __device__ __forceinline__
    float uniform()
    {
        return curand_uniform(&state_);
    }

private:
    curandStateXORWOW_t state_;
};



#pragma hd_warning_disable
/*
Create a reflected ray originating from the hit point.
Its direction is chosen randomly using cosine-weighted hemisphere sampling around the surface normal at the hit point.

hit.unit_normal is assumed to be normalized.
*/
template<typename Random>
__host__ __device__ inline Ray cosine_weighted_sampling(const HitRecord& hit, Random& random) {
    const float z = 1.0 - 2.0 * random.uniform();
    const float angle = 2.0 * PI * random.uniform();
    const float radius = std::sqrt(1.0 - z * z);
    const Vec3 random_unit(
        radius * std::cos(angle),
        radius * std::sin(angle),
        z
    );

    Vec3 direction = hit.unit_normal + random_unit;
    if (dot(direction, direction) < EPSILON * EPSILON) {
        direction = hit.unit_normal;
    }
    return Ray{
        hit.position,
        normalize(direction)
    };
}

#endif
