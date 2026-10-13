// v4: v3 + less warp divergence.
// - Path regeneration: one loop iteration per ray instead of nested sample and bounce
//   loops. When a thread's path ends it starts its next path straight away, instead of
//   waiting for the longest path in its warp to finish.
// - Each warp traces an 8x4 tile of pixels instead of 32 pixels of one row, so the rays
//   in a warp are closer together and tend to hit the same objects.
// PT_BLOCK_SIZE and PT_SAMPLES_PER_THREAD override the defaults (for parameter sweeps).
// PT_BLOCK_SIZE should be a multiple of 32 so that every warp covers exactly one tile.

#include <cstdlib>
#include <vector>

#include "graphics.cuh"
#include "geometry.cuh"

#define MAX_OBJECTS 128
#define MAX_BLOCK_SIZE 512
#define TILE_WIDTH 8
#define TILE_HEIGHT 4

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
    HitRecord record;
    int object_index;  // -1 if nothing was hit
    int triangle_index;
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

__device__ Hit find_closest_hit(const Ray& ray, const Triangle* triangles, int num_objects, int last_hit) {
    Hit closest;
    closest.object_index = -1;
    closest.triangle_index = -1;

    for (int o = 0; o < num_objects; o++) {
        // Skip the object if the ray misses its bounding sphere, or only reaches it
        // further away than the closest hit so far.
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

// One warp per (tile, chunk): each thread traces samples_per_thread of its pixel's paths
// and adds the result to sums[pixel].
__global__ void __launch_bounds__(MAX_BLOCK_SIZE)
render_kernel(RenderSettings settings, const Triangle* __restrict__ triangles,
              const GPUMaterial* __restrict__ materials, int num_objects, int tiles_x,
              int num_tiles, int samples_per_thread, int chunks, float* sums) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = thread / 32;
    int lane = thread % 32;
    if (warp >= num_tiles * chunks) return;

    int tile = warp % num_tiles;
    int chunk = warp / num_tiles;
    int x = (tile % tiles_x) * TILE_WIDTH + lane % TILE_WIDTH;
    int y = (tile / tiles_x) * TILE_HEIGHT + lane / TILE_WIDTH;
    if (x >= settings.width || y >= settings.height) return;
    int s = settings.supersampling_factor;
    int spp = settings.samples_per_pixel;

    // The pixel has s*s*spp paths; path k goes through subpixel k / spp.
    int k = chunk * samples_per_thread;
    int end = min(k + samples_per_thread, spp * s * s);
    if (k >= end) return;

    RandomGPU random(settings.seed + thread);
    int current_subpixel = k / spp;
    Ray camera_ray = make_camera_ray(settings, x * s + current_subpixel % s, y * s + current_subpixel / s);

    Vec3 color;
    Ray ray = camera_ray;
    Vec3 throughput(1.0f, 1.0f, 1.0f);
    int depth = 0;
    int last_hit = -1;

    while (true) {
        Hit hit = find_closest_hit(ray, triangles, num_objects, last_hit);

        bool path_done;
        if (hit.object_index < 0) {
            color += multiply(throughput, settings.background_light);
            path_done = true;
        } else {
            GPUMaterial material = materials[hit.object_index];
            color += multiply(throughput, to_vec3(material.emission));
            throughput = multiply(throughput, to_vec3(material.albedo));
            path_done = depth >= settings.max_bounces ||
                        (throughput.x == 0.0f && throughput.y == 0.0f && throughput.z == 0.0f);
            if (!path_done) {
                ray = cosine_weighted_sampling(hit.record, random);
                last_hit = hit.triangle_index;
                depth++;
            }
        }

        // Start the next path.
        if (path_done) {
            k++;
            if (k >= end) break;

            int subpixel = k / spp;
            if (subpixel != current_subpixel) {
                current_subpixel = subpixel;
                camera_ray = make_camera_ray(settings, x * s + subpixel % s, y * s + subpixel / s);
            }
            ray = camera_ray;
            throughput = Vec3(1.0f, 1.0f, 1.0f);
            depth = 0;
            last_hit = -1;
        }
    }

    int pixel = y * settings.width + x;
    atomicAdd(&sums[3 * pixel + 0], color.x);
    atomicAdd(&sums[3 * pixel + 1], color.y);
    atomicAdd(&sums[3 * pixel + 2], color.z);
}

// Averages each pixel's sum and converts it to a pixel.
__global__ void finalize_kernel(const float* sums, Pixel* pixels, int num_pixels, int paths_per_pixel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pixels) return;

    Vec3 color(sums[3 * i], sums[3 * i + 1], sums[3 * i + 2]);
    pixels[i] = to_pixel(color / float(paths_per_pixel));
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
    int s = settings.supersampling_factor;
    int paths_per_pixel = settings.samples_per_pixel * s * s;

    Triangle* d_triangles;
    GPUMaterial* d_materials;
    float* d_sums;
    Pixel* d_pixels;
    cudaMalloc(&d_triangles, triangles.size() * sizeof(Triangle));
    cudaMalloc(&d_materials, num_objects * sizeof(GPUMaterial));
    cudaMalloc(&d_sums, num_pixels * 3 * sizeof(float));
    cudaMalloc(&d_pixels, num_pixels * sizeof(Pixel));
    cudaMemcpy(d_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_materials, materials.data(), num_objects * sizeof(GPUMaterial), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_objects, gpu_objects.data(), num_objects * sizeof(GPUObject));
    cudaMemset(d_sums, 0, num_pixels * 3 * sizeof(float));

    // One warp per 8x4 tile of pixels.
    int tiles_x = (settings.width + TILE_WIDTH - 1) / TILE_WIDTH;
    int tiles_y = (settings.height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int num_tiles = tiles_x * tiles_y;

    // Split each pixel's paths into chunks so that there are about 4 times as many
    // threads as the GPU can run at once.
    int num_sms, threads_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    int target_threads = 4 * num_sms * threads_per_sm;
    int chunks = (target_threads + num_tiles * 32 - 1) / (num_tiles * 32);
    int samples_per_thread = (paths_per_pixel + chunks - 1) / chunks;

    const char* env = getenv("PT_SAMPLES_PER_THREAD");
    if (env) samples_per_thread = atoi(env);
    chunks = (paths_per_pixel + samples_per_thread - 1) / samples_per_thread;

    env = getenv("PT_BLOCK_SIZE");
    int block_size = env ? atoi(env) : 128;
    int num_threads = num_tiles * 32 * chunks;
    int num_blocks = (num_threads + block_size - 1) / block_size;
    render_kernel<<<num_blocks, block_size>>>(settings, d_triangles, d_materials, num_objects,
                                              tiles_x, num_tiles, samples_per_thread, chunks, d_sums);
    finalize_kernel<<<(num_pixels + 255) / 256, 256>>>(d_sums, d_pixels, num_pixels, paths_per_pixel);

    std::vector<Pixel> pixels(num_pixels);
    cudaMemcpy(pixels.data(), d_pixels, num_pixels * sizeof(Pixel), cudaMemcpyDeviceToHost);

    cudaFree(d_triangles);
    cudaFree(d_materials);
    cudaFree(d_sums);
    cudaFree(d_pixels);
    return pixels;
}
