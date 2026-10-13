#include <vector>

#include "graphics.cuh"
#include "geometry.cuh"

#define MAX_OBJECTS 128
#define BLOCK_SIZE 128  // multiple of 32, so that every warp covers exactly one tile
#define TILE_WIDTH 8
#define TILE_HEIGHT 4

struct GPUObject {
    float4 boundingSphere;  // w = radius squared
    int triangleStart;
    int triangleEnd;
};

struct GPUMaterial {
    float4 albedo;
    float4 emission;
};

struct Hit {
    HitRecord record;
    int objectIndex;  // -1 if nothing was hit
    int triangleIndex;
};

__constant__ GPUObject constantObjects[MAX_OBJECTS];

__device__ Vec3 toVec3(float4 v) {
    return Vec3(v.x, v.y, v.z);
}

// Distance to where the ray enters the sphere (0 if it starts inside), or INF if it misses.
__device__ float sphereEntryDistance(Vec3 origin, Vec3 direction, float4 boundingSphere) {
    Vec3 centerToOrigin = origin - toVec3(boundingSphere);
    float halfB = dot(centerToOrigin, direction);
    float c = dot(centerToOrigin, centerToOrigin) - boundingSphere.w;
    float discriminant = halfB * halfB - c;
    if (discriminant < 0.0f) return INF;

    float sqrtDiscriminant = sqrtf(discriminant);
    if (-halfB + sqrtDiscriminant <= EPSILON) return INF;
    return fmaxf(-halfB - sqrtDiscriminant, 0.0f);
}

__device__ Hit findClosestHit(const Ray& ray, const Triangle* triangles, int numObjects, int lastHitTriangle) {
    Hit closestHit;
    closestHit.objectIndex = -1;
    closestHit.triangleIndex = -1;

    for (int objectIndex = 0; objectIndex < numObjects; objectIndex++) {
        if (sphereEntryDistance(ray.origin, ray.direction, constantObjects[objectIndex].boundingSphere) >=
            closestHit.record.t) continue;

        for (int triangleIndex = constantObjects[objectIndex].triangleStart;
             triangleIndex < constantObjects[objectIndex].triangleEnd; triangleIndex++) {
            if (triangleIndex == lastHitTriangle) continue;
            HitRecord record;
            if (intersect_triangle(ray, triangles[triangleIndex], &record) && record.t < closestHit.record.t) {
                closestHit.record = record;
                closestHit.objectIndex = objectIndex;
                closestHit.triangleIndex = triangleIndex;
            }
        }
    }
    return closestHit;
}

__global__ void renderKernel(RenderSettings settings, const Triangle* triangles,
                             const GPUMaterial* materials, int numObjects, int tilesPerRow,
                             int numTiles, int pathsPerChunk, int chunksPerPixel, float* pixelColorSums) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIndex = threadIndex / 32;
    int laneIndex = threadIndex % 32;
    if (warpIndex >= numTiles * chunksPerPixel) return;

    int tileIndex = warpIndex % numTiles;
    int chunkIndex = warpIndex / numTiles;
    int pixelX = (tileIndex % tilesPerRow) * TILE_WIDTH + laneIndex % TILE_WIDTH;
    int pixelY = (tileIndex / tilesPerRow) * TILE_HEIGHT + laneIndex / TILE_WIDTH;
    if (pixelX >= settings.width || pixelY >= settings.height) return;
    int subpixelsPerSide = settings.supersampling_factor;
    int pathsPerSubpixel = settings.samples_per_pixel;
    int pathsPerPixel = pathsPerSubpixel * subpixelsPerSide * subpixelsPerSide;

    int pathIndex = chunkIndex * pathsPerChunk;
    int endPath = min(pathIndex + pathsPerChunk, pathsPerPixel);
    if (pathIndex >= endPath) return;

    RandomGPU random(settings.seed + threadIndex);
    int currentSubpixel = pathIndex / pathsPerSubpixel;
    Ray cameraRay = make_camera_ray(settings, pixelX * subpixelsPerSide + currentSubpixel % subpixelsPerSide,
                                    pixelY * subpixelsPerSide + currentSubpixel / subpixelsPerSide);

    Vec3 chunkColorSum;
    Ray ray = cameraRay;
    Vec3 albedoProduct(1.0f, 1.0f, 1.0f);
    int bounce = 0;
    int lastHitTriangle = -1;

    while (true) {
        Hit hit = findClosestHit(ray, triangles, numObjects, lastHitTriangle);

        bool pathDone;
        if (hit.objectIndex < 0) {
            chunkColorSum += multiply(albedoProduct, settings.background_light);
            pathDone = true;
        } else {
            GPUMaterial material = materials[hit.objectIndex];
            chunkColorSum += multiply(albedoProduct, toVec3(material.emission));
            albedoProduct = multiply(albedoProduct, toVec3(material.albedo));
            pathDone = bounce >= settings.max_bounces ||
                       (albedoProduct.x == 0.0f && albedoProduct.y == 0.0f && albedoProduct.z == 0.0f);
            if (!pathDone) {
                ray = cosine_weighted_sampling(hit.record, random);
                lastHitTriangle = hit.triangleIndex;
                bounce++;
            }
        }

        // Start the next path.
        if (pathDone) {
            pathIndex++;
            if (pathIndex >= endPath) break;

            int subpixelIndex = pathIndex / pathsPerSubpixel;
            if (subpixelIndex != currentSubpixel) {
                currentSubpixel = subpixelIndex;
                cameraRay = make_camera_ray(settings, pixelX * subpixelsPerSide + subpixelIndex % subpixelsPerSide,
                                            pixelY * subpixelsPerSide + subpixelIndex / subpixelsPerSide);
            }
            ray = cameraRay;
            albedoProduct = Vec3(1.0f, 1.0f, 1.0f);
            bounce = 0;
            lastHitTriangle = -1;
        }
    }

    int pixelIndex = pixelY * settings.width + pixelX;
    atomicAdd(&pixelColorSums[3 * pixelIndex + 0], chunkColorSum.x);
    atomicAdd(&pixelColorSums[3 * pixelIndex + 1], chunkColorSum.y);
    atomicAdd(&pixelColorSums[3 * pixelIndex + 2], chunkColorSum.z);
}

// Averages each pixel's sum and converts it to a pixel.
__global__ void averageAndConvertKernel(const float* pixelColorSums, Pixel* pixels, int numPixels,
                                        int pathsPerPixel) {
    int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixelIndex >= numPixels) return;

    Vec3 colorSum(pixelColorSums[3 * pixelIndex], pixelColorSums[3 * pixelIndex + 1],
                  pixelColorSums[3 * pixelIndex + 2]);
    pixels[pixelIndex] = to_pixel(colorSum / float(pathsPerPixel));
}

std::vector<Pixel> render_scene(
    const RenderSettings& settings,
    const std::vector<Object>& objects
) {
    std::vector<Triangle> triangles;
    std::vector<GPUObject> gpuObjects;
    std::vector<GPUMaterial> materials;
    for (const Object& object : objects) {
        GPUObject gpuObject;
        const Sphere& boundingSphere = object.bound;
        gpuObject.boundingSphere = make_float4(boundingSphere.center.x, boundingSphere.center.y,
                                               boundingSphere.center.z,
                                               boundingSphere.radius * boundingSphere.radius);
        gpuObject.triangleStart = triangles.size();
        triangles.insert(triangles.end(), object.triangles.begin(), object.triangles.end());
        gpuObject.triangleEnd = triangles.size();
        gpuObjects.push_back(gpuObject);

        GPUMaterial material;
        material.albedo = make_float4(object.albedo.x, object.albedo.y, object.albedo.z, 0.0f);
        material.emission = make_float4(object.emission.x, object.emission.y, object.emission.z, 0.0f);
        materials.push_back(material);
    }
    int numObjects = gpuObjects.size();
    int numPixels = settings.width * settings.height;
    int subpixelsPerSide = settings.supersampling_factor;
    int pathsPerPixel = settings.samples_per_pixel * subpixelsPerSide * subpixelsPerSide;

    Triangle* deviceTriangles;
    GPUMaterial* deviceMaterials;
    float* devicePixelColorSums;
    Pixel* devicePixels;
    cudaMalloc(&deviceTriangles, triangles.size() * sizeof(Triangle));
    cudaMalloc(&deviceMaterials, numObjects * sizeof(GPUMaterial));
    cudaMalloc(&devicePixelColorSums, numPixels * 3 * sizeof(float));
    cudaMalloc(&devicePixels, numPixels * sizeof(Pixel));
    cudaMemcpy(deviceTriangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaterials, materials.data(), numObjects * sizeof(GPUMaterial), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constantObjects, gpuObjects.data(), numObjects * sizeof(GPUObject));
    cudaMemset(devicePixelColorSums, 0, numPixels * 3 * sizeof(float));

    int tilesPerRow = (settings.width + TILE_WIDTH - 1) / TILE_WIDTH;
    int tilesPerColumn = (settings.height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int numTiles = tilesPerRow * tilesPerColumn;

    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);
    int targetThreads = 4 * deviceProperties.multiProcessorCount * deviceProperties.maxThreadsPerMultiProcessor;
    int chunksPerPixel = (targetThreads + numTiles * 32 - 1) / (numTiles * 32);
    int pathsPerChunk = (pathsPerPixel + chunksPerPixel - 1) / chunksPerPixel;
    chunksPerPixel = (pathsPerPixel + pathsPerChunk - 1) / pathsPerChunk;

    int numThreads = numTiles * 32 * chunksPerPixel;
    int numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    renderKernel<<<numBlocks, BLOCK_SIZE>>>(settings, deviceTriangles, deviceMaterials, numObjects,
                                            tilesPerRow, numTiles, pathsPerChunk, chunksPerPixel,
                                            devicePixelColorSums);
    averageAndConvertKernel<<<(numPixels + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        devicePixelColorSums, devicePixels, numPixels, pathsPerPixel);

    std::vector<Pixel> pixels(numPixels);
    cudaMemcpy(pixels.data(), devicePixels, numPixels * sizeof(Pixel), cudaMemcpyDeviceToHost);

    cudaFree(deviceTriangles);
    cudaFree(deviceMaterials);
    cudaFree(devicePixelColorSums);
    cudaFree(devicePixels);
    return pixels;
}
