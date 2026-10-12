// - Object bounds are in constant memory, since every thread reads the same object at once.
// - Objects whose bounding sphere starts beyond the closest hit so far are skipped.
// - The camera ray is computed once per subpixel instead of once per sample.

#include <vector>

#include "graphics.cuh"
#include "geometry.cuh"

#define MAX_OBJECTS 128

struct GPUObject {
    float4 sphere;
    int tri_begin;
    int tri_end;
};

struct GPUMaterial {
    float4 albedo;
    float4 emission;
};

struct Hit {
    HitRecord record;
    int object_index;  // -1 if nothing was hit
    int triangle_index;
};

__constant__ GPUObject c_objects[MAX_OBJECTS];

__device__ Vec3 to_vec3(float4 v) {
    return Vec3(v.x, v.y, v.z);
}

// Distance to where ray enters the sphere. 0 if it starts inside, INF if it misses.
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

__device__ Hit find_closest_hit(const Ray& ray, const Triangle* triangles, int num_objects, int last_hit) {
    Hit closest;
    closest.object_index = -1;
    closest.triangle_index = -1;

    for (int o = 0; o < num_objects; o++) {
        if (sphere_entry(ray.origin, ray.direction, c_objects[o].sphere) >= closest.record.t) continue;
        for (int i = c_objects[o].tri_begin; i < c_objects[o].tri_end; i++) {
            if (i == last_hit) continue;
            HitRecord record;
            if (intersect_triangle(ray, triangles[i], &record) && record.t < closest.record.t) {
                closest.record = record;
                closest.object_index = o;
                closest.triangle_index = i;
            }
        }
    }
    return closest;
}

__device__ Vec3 trace_path(Ray ray, const RenderSettings& settings, const Triangle* triangles,
                           const GPUMaterial* materials, int num_objects, RandomGPU& random) {
    Vec3 color;
    Vec3 throughput(1.0f, 1.0f, 1.0f);
    int last_hit = -1;

    for (int depth = 0;; depth++) {
        Hit hit = find_closest_hit(ray, triangles, num_objects, last_hit);
        if (hit.object_index < 0) {
            color += multiply(throughput, settings.background_light);
            break;
        }

        GPUMaterial material = materials[hit.object_index];
        color += multiply(throughput, to_vec3(material.emission));
        if (depth >= settings.max_bounces) break;

        throughput = multiply(throughput, to_vec3(material.albedo));
        if (throughput.x == 0.0f && throughput.y == 0.0f && throughput.z == 0.0f) break;

        ray = cosine_weighted_sampling(hit.record, random);
        last_hit = hit.triangle_index;
    }
    return color;
}

__global__ void render_kernel(RenderSettings settings, const Triangle* __restrict__ triangles,
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
    std::vector<Triangle> triangles;
    std::vector<GPUObject> gpu_objects;
    std::vector<GPUMaterial> materials;
    for (const Object& object : objects) {
        GPUObject gpu_object;
        const Sphere& bound = object.bound;
        gpu_object.sphere = make_float4(bound.center.x, bound.center.y, bound.center.z,
                                        bound.radius * bound.radius);
        gpu_object.tri_begin = triangles.size();
        triangles.insert(triangles.end(), object.triangles.begin(), object.triangles.end());
        gpu_object.tri_end = triangles.size();
        gpu_objects.push_back(gpu_object);

        GPUMaterial material;
        material.albedo = make_float4(object.albedo.x, object.albedo.y, object.albedo.z, 0.0f);
        material.emission = make_float4(object.emission.x, object.emission.y, object.emission.z, 0.0f);
        materials.push_back(material);
    }
    int num_objects = gpu_objects.size();
    int num_pixels = settings.width * settings.height;

    Triangle* d_triangles;
    GPUMaterial* d_materials;
    Pixel* d_pixels;
    cudaMalloc(&d_triangles, triangles.size() * sizeof(Triangle));
    cudaMalloc(&d_materials, num_objects * sizeof(GPUMaterial));
    cudaMalloc(&d_pixels, num_pixels * sizeof(Pixel));
    cudaMemcpy(d_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
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
