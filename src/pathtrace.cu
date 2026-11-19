#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "main.h"
#include "preview.h"
#include "denoise.h"
#include "boundingbox.h"
#include "capture.h"
#include <iomanip>
#include "denoise_sequence.h"
#include "display_transform.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
#ifndef NDEBUG
    cudaError_t executionError = cudaDeviceSynchronize();
    if (executionError != cudaSuccess) {
        fprintf(stderr, "CUDA execution error: %s\n", cudaGetErrorString(executionError));
        exit(EXIT_FAILURE);
    }
#endif
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
#endif
}

// The three panels use identical packing and display conversion.
__global__ void sendThreeImagesToPBO(uchar4* pbo, glm::ivec2 resolution,
                                    const glm::vec3* raw, const glm::vec3* original,
                                    const glm::vec3* optimized, bool toneMap, float exposureEV) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        const glm::vec3 samples[3] = {raw[index], original[index], optimized[index]};
        for (int panel = 0; panel < 3; ++panel) {
            glm::vec3 pix = displayColor(samples[panel], toneMap, exposureEV);
            pbo[x + y * resolution.x * 3 + panel * resolution.x] =
                make_uchar4((unsigned char)(pix.x * 255), (unsigned char)(pix.y * 255),
                            (unsigned char)(pix.z * 255), 255);
        }
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;

static Triangle * dev_triangles = NULL;                           // triangles
static AccelTriangle *dev_accel_triangles = nullptr;
static GBufferTexel * dev_gbuffer = NULL;                         // G-buffer for normal and depth
static glm::vec3 * dev_denoised_image = NULL;
static glm::vec3 * dev_raw_image = NULL;
static glm::vec3 * dev_reference_image = nullptr;
static glm::vec3 * dev_optimized_image = NULL;

/* BVH */
static int Box_count = 0;
static int BVH_node_count = 0;
static BoundingBox * dev_bounding = NULL;
static BVH_ArrNode* dev_bvh_nodes = NULL;

/* Texture */
static Texture* dev_texts = NULL;                                 // STORE GPU TEXTURES
static Texture* dev_nor_map = NULL;                               // STORE GPU NORMAL MAP

static Light * dev_lights = NULL;                                 // LIGHT VECTOR
FramePerformance framePerformance;
static cudaEvent_t stageEvents[5] = {};
static double accumulatedStages[5] = {};
static int profiledFrames = 0;

void waitForFrame() {
    checkCudaInterop(cudaEventSynchronize(stageEvents[4]), "Completing the rendered frame");
}

void finishFrameProfiling(float transferMilliseconds) {
    float ms[5] = {};
    for (int i = 0; i < 4; ++i) cudaEventElapsedTime(&ms[i], stageEvents[i], stageEvents[i + 1]);
    ms[4] = transferMilliseconds;
    framePerformance = {ms[0], ms[1], ms[2], ms[3], ms[4]};
    if (++profiledFrames > 8) for (int i = 0; i < 5; ++i) accumulatedStages[i] += ms[i];
}

void printFrameProfiling() {
    if (profiledFrames <= 8) return;
    const double count = profiledFrames - 8;
    printf("Stage averages after 8 warm-up frames (ms): trace=%.3f original=%.3f optimized=%.3f pack=%.3f display-transfer=%.3f\n",
           accumulatedStages[0] / count, accumulatedStages[1] / count, accumulatedStages[2] / count,
           accumulatedStages[3] / count, accumulatedStages[4] / count);
}

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    for (auto &event : stageEvents) cudaEventCreate(&event);
    profiledFrames = 0;
    for (double &value : accumulatedStages) value = 0;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

      cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

      cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
      cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

      cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
      cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

      cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
      cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    if (!scene->triangles.empty()) {
        std::vector<AccelTriangle> cache;
        cache.reserve(scene->triangles.size());
        for (const auto &triangle : scene->triangles) cache.emplace_back(triangle);
        cudaMalloc(&dev_accel_triangles, cache.size() * sizeof(AccelTriangle));
        cudaMemcpy(dev_accel_triangles, cache.data(), cache.size() * sizeof(AccelTriangle), cudaMemcpyHostToDevice);
    }
    
    cudaMalloc(&dev_gbuffer, pixelcount * sizeof(GBufferTexel));

    cudaMalloc(&dev_denoised_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_image, 0, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_raw_image, pixelcount * sizeof(glm::vec3));
    if (referenceRender) cudaMalloc(&dev_reference_image, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_optimized_image, pixelcount * sizeof(glm::vec3));
    optimizedDenoiseInit(pixelcount);

    /* Texture */
    if (scene->textures.size() > 0) {
        cudaMalloc(&dev_texts, scene->textures.size() * sizeof(Texture));
        for (int i = 0; i < scene->textures.size(); i++) {
            int w = scene->textures[i].width, h = scene->textures[i].height, com = scene->textures[i].components;
            int total_size = w * h * com;
            cudaMalloc(&(scene->textures[i].dev_image), total_size * sizeof(unsigned char));	// ALLOCATE THE TEXTURE DATA
            cudaMemcpy(scene->textures[i].dev_image, scene->textures[i].image, total_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(dev_texts, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);
    }

    /* BVH */
    Box_count = scene->BoudningBoxs.size();
    if (scene->BoudningBoxs.size() > 0) {// World bounds of mesh
        cudaMalloc(&dev_bounding, Box_count * sizeof(BoundingBox));
        cudaMemcpy(dev_bounding, scene->BoudningBoxs.data(), Box_count * sizeof(BoundingBox), cudaMemcpyHostToDevice);
    }
    BVH_node_count = scene->Node_count;
    if (BVH_node_count > 0) {//BVH Nodes
        cudaMalloc(&dev_bvh_nodes, BVH_node_count * sizeof(BVH_ArrNode));
        cudaMemcpy(dev_bvh_nodes, scene->bvh_nodes, BVH_node_count * sizeof(BVH_ArrNode), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));          // LIGHT BUFFER ALLOCATE MEMORY
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    for (auto &event : stageEvents) { if (event) cudaEventDestroy(event); event = nullptr; }
    cudaFree(dev_image);  // no-op if dev_image is null
      cudaFree(dev_paths);
      cudaFree(dev_geoms);
      cudaFree(dev_materials);
      cudaFree(dev_intersections);

    cudaFree(dev_triangles);
    cudaFree(dev_accel_triangles); dev_accel_triangles = nullptr;
    cudaFree(dev_gbuffer);
    cudaFree(dev_denoised_image);
    cudaFree(dev_raw_image);
    cudaFree(dev_reference_image); dev_reference_image = nullptr;
    cudaFree(dev_optimized_image);
    optimizedDenoiseFree();

#if USE_KDTREE
    if (Box_count>0) { cudaFree(dev_bounding); }
    if (BVH_node_count>0) { cudaFree(dev_bvh_nodes); }
#endif

#if SHOW_TEXTURE
    if (hst_scene) {
        for (auto &texture : hst_scene->textures) {
            cudaFree(texture.dev_image);
            texture.dev_image = nullptr;
        }
    }
    cudaFree(dev_texts);
    cudaFree(dev_nor_map);
#endif
    cudaFree(dev_lights);

    checkCUDAError("pathtraceFree");
}

void pathtraceReset() {
    const int count = hst_scene->state.camera.resolution.x * hst_scene->state.camera.resolution.y;
    cudaMemset(dev_image, 0, count * sizeof(glm::vec3));
    optimizedDenoiseReset();
}

void retrieveOriginalImage() {
    if (!hst_scene || !dev_denoised_image) return;
    const int count = hst_scene->state.camera.resolution.x * hst_scene->state.camera.resolution.y;
    checkCudaInterop(cudaMemcpy(hst_scene->state.image.data(), dev_denoised_image,
                     count * sizeof(glm::vec3), cudaMemcpyDeviceToHost), "Reading the image for saving");
}


// Generate PathSegments with rays from the camera through the screen into the scene, which is the first bounce of rays.
__global__ void generateRayFromCamera(Camera cam, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);

        // initial ray
        PathSegment & segment = pathSegments[index];
        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f);
        segment.ray.direction = glm::normalize(cam.view
        - cam.right * cam.pixelLength.x * ((float)x - (float)(cam.resolution.x * 0.5f - 0.5f))
        - cam.up * cam.pixelLength.y * ((float)y - (float)(cam.resolution.y * 0.5f - 0.5f))
        );
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.diffuse = false;
        segment.specular = false;
    }
}

__host__ __device__
bool computeIntersection(Ray& ray, ShadeableIntersection& intersection, Geom * geoms, int geoms_size,
                         Triangle* triangles, BoundingBox * worldBounds, BVH_ArrNode * nodes,
                         const Material *materials, const Texture *textures, const AccelTriangle *accel)
{
    // closest hit
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    int hit_tri_index = -1;
    int hit_material = -1;
    glm::vec3 normal;
    glm::vec2 uv;
    float t = -1.0f;
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    glm::vec2 tmp_uv(0);
    bool tmp_outside;

    for (int i = 0; i < geoms_size; i++) 
    {
        Geom & geom = geoms[i];
        if (geom.type == MESH) continue; // All meshes share one BVH traversal below.
        if (geom.type == CUBE) t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, tmp_outside);
        else if (geom.type == SPHERE) t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, tmp_outside);
        // update closest hit
        if (t > 0.0f && t < t_min) {
            t_min = t;
            hit_geom_index = i;
            hit_material = geom.materialid;
            normal = tmp_normal;
            uv = tmp_uv;
        }
    }

    ShadeableIntersection meshHit;
    meshHit.t = t_min;
    if (IntersectBVH(ray, &meshHit, hit_tri_index, nodes, triangles, materials, textures, false, accel) &&
        hit_tri_index >= 0 && meshHit.t > 0 && meshHit.t < t_min) {
        const Triangle &triangle = triangles[hit_tri_index];
        t_min = meshHit.t;
        hit_geom_index = triangle.geomId;
        hit_material = triangle.materialId;
        normal = meshHit.surfaceNormal;
        if (glm::dot(normal, ray.direction) > 0) normal = -normal;
        uv = meshHit.uv;
    }

    if (hit_geom_index == -1) {
        // The ray hits nothing
        intersection.t = -1.0f;
        intersection.geomId = -1;
        return false;
    } else {
        //The ray hits something
        intersection.t = t_min;
        intersection.materialId = hit_material;
        intersection.surfaceNormal = normal;
        intersection.uv = uv;
        intersection.geomId = hit_geom_index;
        return true;
    }
}

// compute shadow ray by randomly sampling in a unit circle centered at the light source
__host__ __device__
bool shadowOccluded(const Ray &ray, float lightDistance, const Geom *geoms, int count,
                    const Triangle *triangles, const BVH_ArrNode *nodes,
                    const Material *materials, const Texture *textures, const AccelTriangle *accel) {
    for (int i = 1; i < count; ++i) {
        if (geoms[i].type == MESH) continue;
        glm::vec3 point, normal;
        bool outside;
        float t = geoms[i].type == CUBE ? boxIntersectionTest(geoms[i], ray, point, normal, outside) :
                                        sphereIntersectionTest(geoms[i], ray, point, normal, outside);
        if (t > 0 && t < lightDistance) return true;
    }
    ShadeableIntersection blocker;
    blocker.t = lightDistance;
    int triangle = -1;
    return IntersectBVH(ray, &blocker, triangle, nodes, triangles, materials, textures, true, accel);
}

__host__ __device__
void computeShadowRay(Ray& shadowRay, glm::vec3 originPos, Geom& light, float lightRadius, float& shadowRayExpectDist, unsigned int& seed) {
    glm::vec3 directionToCenter = glm::normalize(light.translation - originPos);
    glm::quat rot = glm::rotation(glm::vec3(0.0f, 0.0f, 1.0f), directionToCenter);
    float theta = 2 * PI * nextRand(seed);
    glm::vec3 sampleDirection = glm::rotate(rot, glm::vec3(cosf(theta), sinf(theta), 0.0f));
    float sampleRadius = nextRand(seed) * lightRadius;

    glm::vec3 samplePoint = light.translation + sampleDirection * sampleRadius;
    shadowRayExpectDist = glm::l2Norm(samplePoint - originPos);

    shadowRay.origin = originPos;
    shadowRay.direction = glm::normalize(samplePoint - originPos);
}

// do ray tracing kernel
__global__ void rt(int frame, int num_paths, int max_depth,
    PathSegment * pathSegments, ShadeableIntersection * intersections, 
    Geom * geoms, int geoms_size, Triangle* triangles, Material * materials, GBufferTexel * gbuffer, glm::vec3 * image, glm::vec3 *raw,
    bool trace_shadowray, bool reduce_var, float sintensity, float lightSampleRadius, float lightGain, glm::vec3 skyRadiance, bool denoise, bool sepcolor,
    /* BVH & Texture */ BoundingBox * boudings, BVH_ArrNode * bvhnodes, Texture* texts, const AccelTriangle *accel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& segment = pathSegments[idx];
        ShadeableIntersection& intersection = intersections[idx];
        glm::vec3 accumulatedColor(0.0f);

        // compute first intersection and populate g-buffer
        bool hit = computeIntersection(segment.ray, intersection, geoms, geoms_size, triangles,
                                       /* BVH */ boudings, bvhnodes, materials, texts, accel);
        gbuffer[idx].position = hit ? segment.ray.origin + intersection.t * segment.ray.direction : glm::vec3(0);
        gbuffer[idx].normal = hit ? intersection.surfaceNormal : glm::vec3(0);
        gbuffer[idx].geomId = hit ? intersection.geomId : -1;
        gbuffer[idx].albedo = glm::vec3(1);
        gbuffer[idx].ialbedo = glm::vec3(1.0f);
        gbuffer[idx].emission = glm::vec3(0);
        bool separated = false;
        if (hit) {
            Material &firstMaterial = materials[intersection.materialId];
            if (firstMaterial.emittance > 0)
                gbuffer[idx].emission = firstMaterial.color * firstMaterial.emittance * lightGain;
            gbuffer[idx].albedo = firstMaterial.texid == -1 ? firstMaterial.color :
                                 texts[firstMaterial.texid].getColor(intersection.uv);
            separated = denoise && sepcolor && firstMaterial.emittance <= 0;
        }

        for (int depth = 1; depth <= max_depth; depth++) {
            if (!hit) {
                // Cosine-weighted diffuse continuation samples the constant sky.
                accumulatedColor += segment.color * skyRadiance;
                break;
            }

            unsigned int seed = initRand(idx, frame + depth, 16);
            Material &material = materials[intersection.materialId];

            if (material.emittance > 0.0f) {  // Hit light (terminate ray)
                if (!trace_shadowray || !reduce_var || !segment.diffuse) {
                    accumulatedColor += segment.color * material.color * material.emittance * lightGain;
                }
                break;
            }
            else {                            // Hit material (scatter ray)
                glm::vec3 intersectionPos = segment.ray.origin + intersection.t * segment.ray.direction;
                glm::vec3 &intersectionNormal = intersection.surfaceNormal;
                bool materialIsDiffuse = material.hasReflective < 1e-6 && material.hasRefractive < 1e-6;

                // Apply color
                if (denoise && sepcolor) {
                    if (depth > 1) {
                        segment.color *= material.texid != -1 ? 
                                         texts[material.texid].getColor(intersection.uv) : 
                                         material.color;
                    }
                }
                else {
                    segment.color *= material.texid != -1 ? 
                                     texts[material.texid].getColor(intersection.uv) : 
                                     material.color;
                }
                glm::clamp(segment.color, glm::vec3(0.0f), glm::vec3(1.0f));

                // Trace shadow ray
                if (trace_shadowray && materialIsDiffuse) {
                    // TODO: pick random light
                    int lightIdx = 0;
                    Geom& light = geoms[lightIdx];

                    // generate shadow ray
                    Ray shadowRay;
                    float shadowRayExpectDist = 0.0f;
                    computeShadowRay(shadowRay, intersectionPos + 1e-4f * intersectionNormal, light, lightSampleRadius, shadowRayExpectDist, seed);

                    // Shadow rays need visibility only, not the nearest shaded
                    // surface. Stop at the first opaque blocker before the light.
                    glm::vec3 lightPoint, lightNormal;
                    bool outsideLight;
                    float lightDistance = light.type == CUBE ? boxIntersectionTest(light, shadowRay, lightPoint, lightNormal, outsideLight) :
                                                             sphereIntersectionTest(light, shadowRay, lightPoint, lightNormal, outsideLight);
                    if (lightDistance > 0 && !shadowOccluded(shadowRay, lightDistance, geoms, geoms_size,
                                                           triangles, bvhnodes, materials, texts, accel)) {
                        Material shadowRayMaterial = materials[light.materialid];
                        if (shadowRayMaterial.emittance > 0.0f) {
                            float diffuse = glm::max(0.0f, glm::dot(shadowRay.direction, intersectionNormal));
                            float shadowIntensity = sintensity / pow(shadowRayExpectDist, 2.0f);
                            accumulatedColor += segment.color
                                                * shadowRayMaterial.emittance * shadowRayMaterial.color
                                                * shadowIntensity * diffuse * lightGain;
                        }
                    }
                }

                // Bounce ray and compute intersection
                if (depth < max_depth) {
                    scatterRay(segment, intersectionPos, intersectionNormal, material, seed);
                    hit = computeIntersection(segment.ray, intersection, geoms, geoms_size, triangles,
                                              /* BVH */ boudings, bvhnodes, materials, texts, accel);
                }
            }
        }
        // Preserve the original denoiser's input convention while providing a
        // separate radiance sample to the raw panel and the new denoiser.
        raw[segment.pixelIndex] = separated ? accumulatedColor * gbuffer[idx].albedo : accumulatedColor;
        if (denoise) {
            image[segment.pixelIndex] = accumulatedColor;
        } else {
            image[segment.pixelIndex] = image[segment.pixelIndex] * (float)frame / (float)(frame + 1) + accumulatedColor / (float)(frame + 1);
        }
    }
}

// Path trace one frame
void saveComparison(const std::string &filename) {
    const glm::ivec2 res = hst_scene->state.camera.resolution;
    const int count = res.x * res.y;
    std::vector<glm::vec3> pixels(count);
    image comparison(res.x * 3, res.y);
    const glm::vec3 *buffers[] = {dev_raw_image, dev_denoised_image, dev_optimized_image};
    for (int panel = 0; panel < 3; ++panel) {
        cudaMemcpy(pixels.data(), buffers[panel], count * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        for (int y = 0; y < res.y; ++y) for (int x = 0; x < res.x; ++x)
            comparison.setPixel(panel * res.x + res.x - 1 - x, y, pixels[x + y * res.x]);
    }
    comparison.savePNG(filename);
}

__global__ void accumulateReference(glm::vec3 *mean, const glm::vec3 *sample, int count, int frame) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < count) mean[p] = frame == 0 ? sample[p] : mean[p] + (sample[p] - mean[p]) / (float)(frame + 1);
}

void saveLinearCapture(const std::string &prefix) {
    const auto res = hst_scene->state.camera.resolution;
    capturePFM(prefix + "-raw.pfm", dev_raw_image, res);
    if (referenceRender) capturePFM(prefix + "-reference.pfm", dev_reference_image, res);
    else {
        capturePFM(prefix + "-original.pfm", dev_denoised_image, res);
        capturePFM(prefix + "-optimized.pfm", dev_optimized_image, res);
    }
    std::vector<GBufferTexel> geometry(res.x * res.y);
    checkCudaInterop(cudaMemcpy(geometry.data(), dev_gbuffer, geometry.size() * sizeof(GBufferTexel), cudaMemcpyDeviceToHost), "Capturing geometry");
    std::vector<glm::vec3> field(geometry.size());
    for (int p = 0; p < field.size(); ++p) field[p] = geometry[p].albedo;
    writePFM(prefix + "-albedo.pfm", field, res);
    for (int p = 0; p < field.size(); ++p) field[p] = geometry[p].normal;
    writePFM(prefix + "-normal.pfm", field, res);
    for (int p = 0; p < field.size(); ++p) field[p] = geometry[p].emission;
    writePFM(prefix + "-emission.pfm", field, res);
    const Camera &c = hst_scene->state.camera;
    std::ofstream metadata(prefix + ".json");
    metadata << std::setprecision(9) << "{\"width\":" << res.x << ",\"height\":" << res.y << ",\"frames\":" << frame
             << ",\"seed_offset\":" << sampleSeedOffset << ",\"reference\":" << (referenceRender ? "true" : "false")
             << ",\"eye\":[" << c.position.x << "," << c.position.y << "," << c.position.z
             << "],\"look_at\":[" << c.lookAt.x << "," << c.lookAt.y << "," << c.lookAt.z
             << "],\"pixel_length\":[" << c.pixelLength.x << "," << c.pixelLength.y
             << "],\"falcor_aligned\":" << (optimizedSettings.falcorAligned ? "true" : "false")
             << ",\"alpha\":" << optimizedSettings.alpha << ",\"moments_alpha\":" << optimizedSettings.momentsAlpha
             << ",\"levels\":" << optimizedSettings.levels << ",\"feedback_tap\":" << optimizedSettings.feedbackTap
             << ",\"anti_lag\":" << (optimizedSettings.antiLag ? "true" : "false") << "}\n";
    printf("Saved linear float captures: %s\n", prefix.c_str());
}

void pathtrace(uchar4 *pbo, int frame) {
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;
    dim3 blocksPerGrid1d = (pixelcount + blockSize1d - 1) / blockSize1d;

    ////////////////////////// Ray Tracing ////////////////////////////////////

    // Generate camera rays
    cudaEventRecord(stageEvents[0]);
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d >>>(cam, ui_tracedepth, dev_paths);
    checkCUDAError("generate camera ray");
    
    // Do actual ray tracing
    rt<<<blocksPerGrid1d, blockSize1d>>>(frame + sampleSeedOffset, pixelcount, ui_tracedepth,
        dev_paths, dev_intersections,
        dev_geoms, hst_scene->geoms.size(),
        dev_triangles, dev_materials, dev_gbuffer, dev_image, dev_raw_image,
        ui_shadowray, ui_reducevar, ui_sintensity, ui_lightradius,
        ui_pulse_light && (frame % 180 >= 90) ? 0.05f : 1.0f, hst_scene->state.skyRadiance, ui_denoise_enable, ui_sepcolor,
        /* BVH & Texture */ dev_bounding, dev_bvh_nodes, dev_texts, dev_accel_triangles);
        checkCUDAError("ray tracing");
    cudaEventRecord(stageEvents[1]);
    if (!recordSequence.empty()) {
        static std::ofstream sequence;
        static Camera sequenceCamera;
        std::vector<glm::vec3> samples(pixelcount);
        if (frame == 0) {
            auto parent = std::filesystem::path(recordSequence).parent_path();
            if (!parent.empty()) std::filesystem::create_directories(parent);
            sequence.open(recordSequence, std::ios::binary);
            DenoiseSequenceHeader header;
            header.width = cam.resolution.x; header.height = cam.resolution.y; header.frames = renderFrameLimit;
            sequence.write(reinterpret_cast<const char*>(&header), sizeof(header));
            sequence.write(reinterpret_cast<const char*>(&cam), sizeof(cam));
            sequenceCamera = cam;
            std::vector<GBufferTexel> geometry(pixelcount);
            checkCudaInterop(cudaMemcpy(geometry.data(), dev_gbuffer, geometry.size() * sizeof(GBufferTexel), cudaMemcpyDeviceToHost), "Recording geometry");
            sequence.write(reinterpret_cast<const char*>(geometry.data()), geometry.size() * sizeof(GBufferTexel));
        }
        if (memcmp(&cam, &sequenceCamera, sizeof(cam)) != 0) throw std::runtime_error("Camera changed during stationary recording");
        checkCudaInterop(cudaMemcpy(samples.data(), dev_raw_image, samples.size() * sizeof(glm::vec3), cudaMemcpyDeviceToHost), "Recording samples");
        sequence.write(reinterpret_cast<const char*>(samples.data()), samples.size() * sizeof(glm::vec3));
        if (!sequence) throw std::runtime_error("Cannot write sample sequence");
        if (frame + 1 == renderFrameLimit) sequence.close();
    }

    ////////////////////////// Denosing ///////////////////////////////////////

    // Run denoiser!
    if (referenceRender) {
        accumulateReference<<<blocksPerGrid1d, blockSize1d>>>(dev_reference_image, dev_raw_image, pixelcount, frame);
        cudaMemcpyAsync(dev_denoised_image, dev_reference_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    } else if (ui_denoise_enable) {
        denoise(dev_denoised_image, dev_image, dev_gbuffer);
    }
    else {
        cudaMemcpy(dev_denoised_image, dev_image, sizeof(glm::vec3) * pixelcount, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(stageEvents[2]);
    if (referenceRender) cudaMemcpyAsync(dev_optimized_image, dev_reference_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    else {
        if (!capturePrefix.empty() && frame + 1 == renderFrameLimit) optimizedDenoiseCapture(capturePrefix);
        optimizedDenoise(dev_optimized_image, dev_raw_image, dev_gbuffer, cam);
    }
    checkCUDAError("optimized denoising");
    cudaEventRecord(stageEvents[3]);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendThreeImagesToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_raw_image, dev_denoised_image, dev_optimized_image, displayToneMap, displayExposureEV);
    checkCUDAError("send images to PBO");
    cudaEventRecord(stageEvents[4]);

    // Full-precision CPU readback is only needed when the user saves an image.
}
