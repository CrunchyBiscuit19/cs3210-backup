// v3: v2 + enough threads to fill the GPU.
// - Each pixel's s*s*spp paths are split over several threads, so there are about
//   4 waves of threads even for small images (ufo has only 60k pixels).
//   Each thread adds its result into a per-pixel sum with atomicAdd, and a second
//   kernel averages the sums and converts them to pixels.
// - 1D blocks of 128 threads.
// - PCG32 instead of curand: much smaller state and no expensive curand_init.
// PT_BLOCK_SIZE and PT_SAMPLES_PER_THREAD override the defaults (for parameter sweeps).

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "graphics.cuh"
#include "geometry.cuh"

#define MAX_OBJECTS 128
#define MAX_BLOCK_SIZE 512

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

// PCG32 random number generator (pcg-random.org). Each thread uses its own stream.
struct Pcg32 {
    uint64_t state;
    uint64_t inc;

    __device__ Pcg32(uint64_t seed, uint64_t stream) {
        state = 0;
        inc = (stream << 1) | 1;
        next();
        state += seed;
        next();
    }

    __device__ uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = ((old >> 18) ^ old) >> 27;
        uint32_t rot = old >> 59;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    // Uniform float in [0, 1).
    __device__ float uniform() {
        return (next() >> 8) * (1.0f / 16777216.0f);
    }
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

// Same as trace_ray in cpu_path_tracing.cu, but as a loop instead of recursion.
// throughput is the product of the albedos of the surfaces hit so far.
__device__ Vec3 trace_path(Ray ray, const RenderSettings& settings, const Triangle* triangles,
                           const GPUMaterial* materials, int num_objects, Pcg32& random) {
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

// One thread per (pixel, chunk). The thread traces samples_per_thread of the pixel's paths
// and adds the result to sums[pixel].
__global__ void __launch_bounds__(MAX_BLOCK_SIZE)
render_kernel(RenderSettings settings, const Triangle* __restrict__ triangles,
              const GPUMaterial* __restrict__ materials, int num_objects,
              int samples_per_thread, int chunks, float* sums) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = settings.width * settings.height;
    if (thread >= num_pixels * chunks) return;

    // Neighbouring threads take neighbouring pixels of the same chunk.
    int pixel = thread % num_pixels;
    int chunk = thread / num_pixels;
    int x = pixel % settings.width;
    int y = pixel / settings.width;
    int s = settings.supersampling_factor;
    int spp = settings.samples_per_pixel;

    // The pixel has s*s*spp paths; path k goes through subpixel k / spp.
    int begin = chunk * samples_per_thread;
    int end = min(begin + samples_per_thread, spp * s * s);

    Pcg32 random(settings.seed, thread);
    Vec3 color;
    Ray camera_ray;
    int current_subpixel = -1;
    for (int k = begin; k < end; k++) {
        int subpixel = k / spp;
        if (subpixel != current_subpixel) {
            current_subpixel = subpixel;
            camera_ray = make_camera_ray(settings, x * s + subpixel % s, y * s + subpixel / s);
        }
        color += trace_path(camera_ray, settings, triangles, materials, num_objects, random);
    }

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

    // Split each pixel's paths into chunks so that there are about 4 times as many
    // threads as the GPU can run at once.
    int num_sms, threads_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    int target_threads = 4 * num_sms * threads_per_sm;
    int chunks = (target_threads + num_pixels - 1) / num_pixels;
    int samples_per_thread = (paths_per_pixel + chunks - 1) / chunks;

    const char* env = getenv("PT_SAMPLES_PER_THREAD");
    if (env) samples_per_thread = atoi(env);
    chunks = (paths_per_pixel + samples_per_thread - 1) / samples_per_thread;

    env = getenv("PT_BLOCK_SIZE");
    int block_size = env ? atoi(env) : 128;
    int num_threads = num_pixels * chunks;
    int num_blocks = (num_threads + block_size - 1) / block_size;
    render_kernel<<<num_blocks, block_size>>>(settings, d_triangles, d_materials, num_objects,
                                              samples_per_thread, chunks, d_sums);
    finalize_kernel<<<(num_pixels + 255) / 256, 256>>>(d_sums, d_pixels, num_pixels, paths_per_pixel);

    std::vector<Pixel> pixels(num_pixels);
    cudaMemcpy(pixels.data(), d_pixels, num_pixels * sizeof(Pixel), cudaMemcpyDeviceToHost);

    cudaFree(d_triangles);
    cudaFree(d_materials);
    cudaFree(d_sums);
    cudaFree(d_pixels);
    return pixels;
}
