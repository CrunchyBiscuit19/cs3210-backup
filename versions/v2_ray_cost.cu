// v2: v1 + cheaper rays.
// - Triangles are stored as p0 and the precomputed edges p1 - p0, p2 - p0, in float4s
//   so each one is loaded with aligned 16-byte loads.
// - The triangle loop only computes t. The normal is computed once, for the closest hit.
// - Float-only math. The geometry.cuh helpers use double literals (e.g. 1.0 / det),
//   which compile to slow FP64 instructions on the GPU.
// - Object bounds are in constant memory, since every thread reads the same object at once.
// - Objects whose bounding sphere starts beyond the closest hit so far are skipped.
// - The camera ray is computed once per subpixel instead of once per sample.

#include <vector>

#include "graphics.cuh"
#include "geometry.cuh"

#define MAX_OBJECTS 128

struct GPUTriangle {
    float4 p0;  // xyz = first vertex
    float4 e1;  // xyz = p1 - p0
    float4 e2;  // xyz = p2 - p0
};

// The triangles of all objects are stored in one array, object by object.
// Object i owns triangles [tri_begin, tri_end).
struct GPUObject {
    float4 sphere;  // xyz = bounding sphere center, w = radius squared
    int tri_begin;
    int tri_end;
};

struct GPUMaterial {
    float4 albedo;
    float4 emission;
};

struct Hit {
    float t;
    int triangle;  // -1 if nothing was hit
    int object;
};

__constant__ GPUObject c_objects[MAX_OBJECTS];

__device__ Vec3 to_vec3(float4 v) {
    return Vec3(v.x, v.y, v.z);
}

// Distance to where the ray enters the sphere (0 if it starts inside), or INF if it misses.
__device__ float sphere_entry(Vec3 origin, Vec3 direction, float4 sphere) {
    Vec3 offset = origin - to_vec3(sphere);
    float half_b = dot(offset, direction);
    float c = dot(offset, offset) - sphere.w;
    float discriminant = half_b * half_b - c;
    if (discriminant < 0.0f) return INF;

    float root = sqrtf(discriminant);
    if (-half_b + root <= EPSILON) return INF;
    return fmaxf(-half_b - root, 0.0f);
}

// Same test as intersect_triangle in geometry.cuh, but only returns t (INF if no hit).
__device__ float intersect_triangle_t(Vec3 origin, Vec3 direction, const GPUTriangle& triangle) {
    Vec3 e1 = to_vec3(triangle.e1);
    Vec3 e2 = to_vec3(triangle.e2);
    Vec3 p = cross(direction, e2);
    float det = dot(e1, p);
    if (fabsf(det) <= EPSILON) return INF;

    float inv_det = 1.0f / det;
    Vec3 offset = origin - to_vec3(triangle.p0);
    float u = dot(offset, p) * inv_det;
    if (u < 0.0f || u > 1.0f) return INF;

    Vec3 q = cross(offset, e1);
    float v = dot(direction, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return INF;

    float t = dot(e2, q) * inv_det;
    return t > EPSILON ? t : INF;
}

// Unit normal of the triangle, facing against the ray (as in intersect_triangle).
__device__ Vec3 triangle_normal(const GPUTriangle& triangle, Vec3 direction) {
    Vec3 normal = normalize(cross(to_vec3(triangle.e1), to_vec3(triangle.e2)));
    if (dot(normal, direction) > 0.0f) normal = -normal;
    return normal;
}

// Same as cosine_weighted_sampling in geometry.cuh, in single precision.
__device__ Vec3 sample_direction(Vec3 normal, RandomGPU& random) {
    float z = 1.0f - 2.0f * random.uniform();
    float angle = 2.0f * PI * random.uniform();
    float radius = sqrtf(1.0f - z * z);
    float sin_angle, cos_angle;
    sincosf(angle, &sin_angle, &cos_angle);

    Vec3 direction = normal + Vec3(radius * cos_angle, radius * sin_angle, z);
    if (dot(direction, direction) < EPSILON * EPSILON) return normal;
    return normalize(direction);
}

__device__ Hit find_closest_hit(Vec3 origin, Vec3 direction, const GPUTriangle* triangles,
                                int num_objects, int last_hit) {
    Hit closest = {INF, -1, -1};

    for (int o = 0; o < num_objects; o++) {
        // Skip the object if the ray misses its bounding sphere, or only reaches it
        // further away than the closest hit so far.
        if (sphere_entry(origin, direction, c_objects[o].sphere) >= closest.t) continue;

        for (int i = c_objects[o].tri_begin; i < c_objects[o].tri_end; i++) {
            if (i == last_hit) continue;
            float t = intersect_triangle_t(origin, direction, triangles[i]);
            if (t < closest.t) {
                closest.t = t;
                closest.triangle = i;
                closest.object = o;
            }
        }
    }
    return closest;
}

// Same as trace_ray in cpu_path_tracing.cu, but as a loop instead of recursion.
// throughput is the product of the albedos of the surfaces hit so far.
__device__ Vec3 trace_path(Ray ray, const RenderSettings& settings, const GPUTriangle* triangles,
                           const GPUMaterial* materials, int num_objects, RandomGPU& random) {
    Vec3 color;
    Vec3 throughput(1.0f, 1.0f, 1.0f);
    int last_hit = -1;

    for (int depth = 0;; depth++) {
        Hit hit = find_closest_hit(ray.origin, ray.direction, triangles, num_objects, last_hit);
        if (hit.triangle < 0) {
            color += multiply(throughput, settings.background_light);
            break;
        }

        GPUMaterial material = materials[hit.object];
        color += multiply(throughput, to_vec3(material.emission));
        if (depth >= settings.max_bounces) break;

        throughput = multiply(throughput, to_vec3(material.albedo));
        if (throughput.x == 0.0f && throughput.y == 0.0f && throughput.z == 0.0f) break;

        Vec3 normal = triangle_normal(triangles[hit.triangle], ray.direction);
        ray.origin = ray.origin + hit.t * ray.direction;
        ray.direction = sample_direction(normal, random);
        last_hit = hit.triangle;
    }
    return color;
}

// One thread per output pixel.
__global__ void render_kernel(RenderSettings settings, const GPUTriangle* __restrict__ triangles,
                              const GPUMaterial* __restrict__ materials, int num_objects,
                              Pixel* pixels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= settings.width || y >= settings.height) return;

    int pixel_index = y * settings.width + x;
    int s = settings.supersampling_factor;
    RandomGPU random(settings.seed + pixel_index);

    Vec3 color;
    for (int sy = 0; sy < s; sy++) {
        for (int sx = 0; sx < s; sx++) {
            Ray camera_ray = make_camera_ray(settings, x * s + sx, y * s + sy);
            for (int sample = 0; sample < settings.samples_per_pixel; sample++) {
                color += trace_path(camera_ray, settings, triangles, materials, num_objects, random);
            }
        }
    }

    pixels[pixel_index] = to_pixel(color / float(settings.samples_per_pixel * s * s));
}

std::vector<Pixel> render_scene(
    const RenderSettings& settings,
    const std::vector<Object>& objects
) {
    std::vector<GPUTriangle> triangles;
    std::vector<GPUObject> gpu_objects;
    std::vector<GPUMaterial> materials;
    for (const Object& object : objects) {
        GPUObject gpu_object;
        const Sphere& bound = object.bound;
        gpu_object.sphere = make_float4(bound.center.x, bound.center.y, bound.center.z,
                                        bound.radius * bound.radius);
        gpu_object.tri_begin = triangles.size();
        for (const Triangle& t : object.triangles) {
            Vec3 e1 = t.p1 - t.p0;
            Vec3 e2 = t.p2 - t.p0;
            GPUTriangle triangle;
            triangle.p0 = make_float4(t.p0.x, t.p0.y, t.p0.z, 0.0f);
            triangle.e1 = make_float4(e1.x, e1.y, e1.z, 0.0f);
            triangle.e2 = make_float4(e2.x, e2.y, e2.z, 0.0f);
            triangles.push_back(triangle);
        }
        gpu_object.tri_end = triangles.size();
        gpu_objects.push_back(gpu_object);

        GPUMaterial material;
        material.albedo = make_float4(object.albedo.x, object.albedo.y, object.albedo.z, 0.0f);
        material.emission = make_float4(object.emission.x, object.emission.y, object.emission.z, 0.0f);
        materials.push_back(material);
    }
    int num_objects = gpu_objects.size();
    int num_pixels = settings.width * settings.height;

    GPUTriangle* d_triangles;
    GPUMaterial* d_materials;
    Pixel* d_pixels;
    cudaMalloc(&d_triangles, triangles.size() * sizeof(GPUTriangle));
    cudaMalloc(&d_materials, num_objects * sizeof(GPUMaterial));
    cudaMalloc(&d_pixels, num_pixels * sizeof(Pixel));
    cudaMemcpy(d_triangles, triangles.data(), triangles.size() * sizeof(GPUTriangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_materials, materials.data(), num_objects * sizeof(GPUMaterial), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_objects, gpu_objects.data(), num_objects * sizeof(GPUObject));

    dim3 block(16, 16);
    dim3 grid((settings.width + block.x - 1) / block.x, (settings.height + block.y - 1) / block.y);
    render_kernel<<<grid, block>>>(settings, d_triangles, d_materials, num_objects, d_pixels);

    std::vector<Pixel> pixels(num_pixels);
    cudaMemcpy(pixels.data(), d_pixels, num_pixels * sizeof(Pixel), cudaMemcpyDeviceToHost);

    cudaFree(d_triangles);
    cudaFree(d_materials);
    cudaFree(d_pixels);
    return pixels;
}
