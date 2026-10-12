// v1: v0 + bounding sphere culling.
// Skip all triangles of an object when the ray misses its bounding sphere.

#include <vector>

#include "graphics.cuh"
#include "geometry.cuh"

// The triangles of all objects are stored in one array, object by object.
// Object i owns triangles [tri_begin, tri_end).
struct GPUObject {
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

__device__ Hit find_closest_hit(const Ray& ray, const Triangle* triangles, const GPUObject* objects,
                                int num_objects, int last_hit) {
    Hit closest;
    closest.object_index = -1;
    closest.triangle_index = -1;

    for (int o = 0; o < num_objects; o++) {
        if (!intersect_sphere(ray, objects[o].bound)) continue;

        for (int i = objects[o].tri_begin; i < objects[o].tri_end; i++) {
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

// Same as trace_ray in cpu_path_tracing.cu, but as a loop instead of recursion.
// throughput is the product of the albedos of the surfaces hit so far.
__device__ Vec3 trace_path(Ray ray, const RenderSettings& settings, const Triangle* triangles,
                           const GPUObject* objects, int num_objects, RandomGPU& random) {
    Vec3 color;
    Vec3 throughput(1.0f, 1.0f, 1.0f);
    int last_hit = -1;

    for (int depth = 0;; depth++) {
        Hit hit = find_closest_hit(ray, triangles, objects, num_objects, last_hit);
        if (hit.object_index < 0) {
            color += multiply(throughput, settings.background_light);
            break;
        }

        const GPUObject& object = objects[hit.object_index];
        color += multiply(throughput, object.emission);
        if (depth >= settings.max_bounces) break;

        throughput = multiply(throughput, object.albedo);
        ray = cosine_weighted_sampling(hit.record, random);
        last_hit = hit.triangle_index;
    }
    return color;
}

// One thread per output pixel.
__global__ void render_kernel(RenderSettings settings, const Triangle* triangles,
                              const GPUObject* objects, int num_objects, Pixel* pixels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= settings.width || y >= settings.height) return;

    int pixel_index = y * settings.width + x;
    int s = settings.supersampling_factor;
    RandomGPU random(settings.seed + pixel_index);

    Vec3 color;
    for (int sy = 0; sy < s; sy++) {
        for (int sx = 0; sx < s; sx++) {
            for (int sample = 0; sample < settings.samples_per_pixel; sample++) {
                Ray camera_ray = make_camera_ray(settings, x * s + sx, y * s + sy);
                color += trace_path(camera_ray, settings, triangles, objects, num_objects, random);
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
    for (const Object& object : objects) {
        GPUObject gpu_object;
        gpu_object.bound = object.bound;
        gpu_object.albedo = object.albedo;
        gpu_object.emission = object.emission;
        gpu_object.tri_begin = triangles.size();
        triangles.insert(triangles.end(), object.triangles.begin(), object.triangles.end());
        gpu_object.tri_end = triangles.size();
        gpu_objects.push_back(gpu_object);
    }
    int num_objects = gpu_objects.size();
    int num_pixels = settings.width * settings.height;

    Triangle* d_triangles;
    GPUObject* d_objects;
    Pixel* d_pixels;
    cudaMalloc(&d_triangles, triangles.size() * sizeof(Triangle));
    cudaMalloc(&d_objects, num_objects * sizeof(GPUObject));
    cudaMalloc(&d_pixels, num_pixels * sizeof(Pixel));
    cudaMemcpy(d_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_objects, gpu_objects.data(), num_objects * sizeof(GPUObject), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((settings.width + block.x - 1) / block.x, (settings.height + block.y - 1) / block.y);
    render_kernel<<<grid, block>>>(settings, d_triangles, d_objects, num_objects, d_pixels);

    std::vector<Pixel> pixels(num_pixels);
    cudaMemcpy(pixels.data(), d_pixels, num_pixels * sizeof(Pixel), cudaMemcpyDeviceToHost);

    cudaFree(d_triangles);
    cudaFree(d_objects);
    cudaFree(d_pixels);
    return pixels;
}
