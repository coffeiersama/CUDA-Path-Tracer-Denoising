#include "denoise_optimized.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include "capture.h"

// Falcor-aligned stages adapted from NVIDIA Falcor SVGFPass (tag 9.0,
// 759aad033ff610fb0d82c74f7e0a508d0096d5f2). NVIDIA's BSD notice is
// retained in third_party/Falcor-SVGF-LICENSE.txt. This is a CUDA adaptation,
// not the Falcor renderer or a claim to reproduce the paper's complete output.

OptimizedDenoiseSettings optimizedSettings;

namespace {
glm::vec3 *historyColor, *temporalColor, *color[2];
glm::vec3 *historyFast, *fastColor;
glm::vec2 *historyMoments, *moments;
float *historyLength, *lengths, *variance[2];
float *reaction;
GBufferTexel *historyGeometry;
Camera previousCamera;
bool hasHistory = false;
std::string nextCapture;
using GeometryGuide = OptimizedGeometryGuide;
GeometryGuide *guides;

void check(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "Optimized SVGF: %s\n", cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

__device__ float luminance(glm::vec3 c) {
    return glm::dot(c, glm::vec3(0.2126f, 0.7152f, 0.0722f));
}

__device__ glm::vec3 albedo(const GBufferTexel &g, bool demodulate, bool aligned = false) {
    return demodulate && g.geomId >= 0 ? glm::max(g.albedo, glm::vec3(aligned ? 0.001f : 0.01f)) : glm::vec3(1.0f);
}

// Camera rays use possibly unnormalized right/up axes. Invert that exact basis
// rather than assuming a square image, fixed FOV, or orthonormal camera frame.
__device__ glm::vec3 cameraCoordinates(glm::vec3 world, const Camera &cam) {
    return glm::inverse(glm::mat3(cam.right, cam.up, cam.view)) * (world - cam.position);
}

__global__ void prepareGuides(GeometryGuide *out, const GBufferTexel *g, Camera cam) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cam.resolution.x * cam.resolution.y) return;
    int x = p % cam.resolution.x, y = p / cam.resolution.x;
    int qx = glm::min(x ^ 1, cam.resolution.x - 1) + y * cam.resolution.x;
    int qy = x + glm::min(y ^ 1, cam.resolution.y - 1) * cam.resolution.x;
    float z = cameraCoordinates(g[p].position, cam).z;
    float dx = g[qx].geomId >= 0 ? fabsf(cameraCoordinates(g[qx].position, cam).z - z) : 0;
    float dy = g[qy].geomId >= 0 ? fabsf(cameraCoordinates(g[qy].position, cam).z - z) : 0;
    out[p] = {g[p].geomId >= 0 ? z : -1.0f, dx + dy,
              glm::length(g[p].normal - g[qx].normal) + glm::length(g[p].normal - g[qy].normal)};
}

__device__ bool consistent(int p, int x, int y, glm::ivec2 res,
                           const GBufferTexel *g, const GBufferTexel *previous,
                           const float *previousLength, const Camera &cam,
                           const GeometryGuide *guide, bool aligned) {
    if (x < 0 || y < 0 || x >= res.x || y >= res.y) return false;
    int q = x + y * res.x;
    if (previousLength[q] <= 0 || g[p].geomId < 0 || g[p].geomId != previous[q].geomId) return false;
    if (aligned) {
        if (glm::length(g[p].normal - previous[q].normal) > 16.0f * (guide[p].normalWidth + 0.01f)) return false;
    } else if (glm::dot(g[p].normal, previous[q].normal) < 0.95f) return false;
    glm::vec3 delta = g[p].position - previous[q].position;
    float z = fabsf(cameraCoordinates(g[p].position, cam).z);
    float footprint = fmaxf(z * fmaxf(cam.pixelLength.x, cam.pixelLength.y), 1e-4f);
    // Reject other depth layers while allowing neighboring samples on a slope.
    float residual = aligned ? fabsf(cameraCoordinates(g[p].position, cam).z - cameraCoordinates(previous[q].position, cam).z) : fabsf(glm::dot(delta, g[p].normal));
    float tolerance = aligned ? 10.0f * (guide[p].width + 0.01f) : fmaxf(0.02f * z, 1e-3f);
    return residual < tolerance &&
           glm::length(delta) < 3.0f * footprint;
}

__global__ void temporalPass(glm::vec3 *outColor, glm::vec3 *outFast, glm::vec2 *outMoments, float *outLength,
                             const glm::vec3 *input, const GBufferTexel *g,
                             const glm::vec3 *prevColor, const glm::vec3 *prevFast, const glm::vec2 *prevMoments,
                             const float *prevLength, const GBufferTexel *prevG,
                             Camera cam, Camera prevCam, bool reuse, OptimizedDenoiseSettings settings,
                             const GeometryGuide *guide) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int count = cam.resolution.x * cam.resolution.y;
    if (p >= count) return;
    glm::vec3 sample = (input[p] - (settings.falcorAligned ? g[p].emission : glm::vec3(0))) /
                      albedo(g[p], settings.demodulate, settings.falcorAligned);
    float l = luminance(sample);
    glm::vec3 oldColor(0);
    glm::vec3 oldFast(0);
    glm::vec2 oldMoments(0);
    float oldLength = 0, sum = 0;
    if (reuse && settings.temporal && g[p].geomId >= 0) {
        glm::vec3 pos = cameraCoordinates(g[p].position, prevCam);
        if (pos.z > 1e-5f) {
            float px = -pos.x / (pos.z * prevCam.pixelLength.x) + (cam.resolution.x * 0.5f - 0.5f);
            float py = -pos.y / (pos.z * prevCam.pixelLength.y) + (cam.resolution.y * 0.5f - 0.5f);
            // Bounds also prevent converting huge projections to integer indices.
            if (isfinite(px) && isfinite(py) && px >= -1 && py >= -1 &&
                px < cam.resolution.x && py < cam.resolution.y) {
                int bx = (int)floorf(px), by = (int)floorf(py);
                float fx = px - bx, fy = py - by;
                for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
                    int x = bx + dx, y = by + dy;
                    if (!consistent(p, x, y, cam.resolution, g, prevG, prevLength, prevCam, guide, settings.falcorAligned)) continue;
                    int q = x + y * cam.resolution.x;
                    float w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy);
                    oldColor += w * prevColor[q];
                    oldFast += w * prevFast[q];
                    oldMoments += w * prevMoments[q];
                    oldLength += w * prevLength[q];
                    sum += w;
                }
                if (sum < 1e-4f) {
                    oldColor = oldFast = glm::vec3(0); oldMoments = glm::vec2(0); oldLength = sum = 0;
                    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                        int x = bx + dx, y = by + dy;
                        if (!consistent(p, x, y, cam.resolution, g, prevG, prevLength, prevCam, guide, settings.falcorAligned)) continue;
                        int q = x + y * cam.resolution.x;
                        oldColor += prevColor[q]; oldMoments += prevMoments[q];
                        oldFast += prevFast[q];
                        oldLength += prevLength[q]; sum += 1;
                    }
                }
            }
        }
    }
    if (sum > 1e-4f) {
        oldColor /= sum; oldFast /= sum; oldMoments /= sum; oldLength /= sum;
        float a = fmaxf(settings.alpha, 1.0f / (oldLength + 1));
        float fastAlpha = fmaxf(a, 1.0f / fmaxf((float)settings.fastHistoryFrames, 1.0f));
        outFast[p] = fastAlpha * sample + (1 - fastAlpha) * oldFast;
        outColor[p] = a * sample + (1 - a) * oldColor;
        float ma = settings.falcorAligned ? fmaxf(settings.momentsAlpha, 1.0f / (oldLength + 1)) : a;
        outMoments[p] = ma * glm::vec2(l, l * l) + (1 - ma) * oldMoments;
        outLength[p] = fminf(oldLength + 1, 32.0f);
    } else {
        outColor[p] = sample;
        outFast[p] = sample;
        outMoments[p] = glm::vec2(l, l * l);
        outLength[p] = g[p].geomId >= 0 ? 1.0f : 0.0f;
    }
}

__device__ float alignedGeometryWeight(int p, int q, const GBufferTexel *g,
                                      const GeometryGuide *guide, float distance, float normalPower) {
    if (g[p].geomId < 0 || g[q].geomId != g[p].geomId) return 0;
    float phiDepth = fmaxf(guide[p].width, 1e-8f) * distance;
    float depth = fabsf(guide[p].depth - guide[q].depth) / fmaxf(phiDepth, 1e-8f);
    return expf(-depth) * powf(fmaxf(glm::dot(g[p].normal, g[q].normal), 0.0f), normalPower);
}

// Falcor SVGFFilterMoments: filter illumination and accumulated moments
// together for short histories, then boost variance by 4 / history length.
__global__ void filterShortHistory(glm::vec3 *dst, float *outVariance, const glm::vec3 *src,
                                   const glm::vec2 *m, const float *length,
                                   const GBufferTexel *g, const GeometryGuide *guide,
                                   Camera cam, OptimizedDenoiseSettings settings) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cam.resolution.x * cam.resolution.y) return;
    dst[p] = src[p];
    outVariance[p] = fmaxf(m[p].y - m[p].x * m[p].x, 0.0f);
    if (g[p].geomId < 0 || length[p] >= 4) return;
    int x = p % cam.resolution.x, y = p / cam.resolution.x;
    glm::vec3 c(0); glm::vec2 mm(0); float sum = 0;
    for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx) {
        int xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= cam.resolution.x || yy >= cam.resolution.y) continue;
        int q = xx + yy * cam.resolution.x;
        float w = p == q ? 1.0f : alignedGeometryWeight(p, q, g, guide,
            3.0f * sqrtf((float)(dx * dx + dy * dy)), settings.sigmaNormal) *
            expf(-fabsf(luminance(src[p]) - luminance(src[q])) / fmaxf(settings.sigmaLuminance, 1e-6f));
        c += w * src[q]; mm += w * m[q]; sum += w;
    }
    dst[p] = c / fmaxf(sum, 1e-6f);
    mm /= fmaxf(sum, 1e-6f);
    outVariance[p] = fmaxf(mm.y - mm.x * mm.x, 0.0f) * 4.0f / fmaxf(length[p], 1.0f);
}

__device__ float depthWeight(int p, int q, const GBufferTexel *g, const Camera &cam) {
    // Use a depth-relative tangent-plane residual instead of a fixed world-space
    // distance cutoff. This is a geometric variant of SVGF's depth edge stopper.
    glm::vec3 delta = g[q].position - g[p].position;
    float zp = cameraCoordinates(g[p].position, cam).z;
    float plane = fabsf(glm::dot(delta, g[p].normal));
    return expf(-plane / (0.01f * fabsf(zp) + 1e-4f));
}

__device__ float geometryWeight(int p, int q, const GBufferTexel *g, const Camera &cam,
                               float normalPower) {
    if (g[p].geomId < 0 || g[q].geomId != g[p].geomId) return 0;
    float n = fmaxf(glm::dot(g[p].normal, g[q].normal), 0.0f);
    return powf(n, normalPower) * depthWeight(p, q, g, cam);
}

__device__ glm::vec3 toYCoCg(glm::vec3 rgb) {
    return glm::vec3((rgb.r + 2 * rgb.g + rgb.b) * 0.25f,
                     (rgb.r - rgb.b) * 0.5f, (-rgb.r + 2 * rgb.g - rgb.b) * 0.25f);
}

__device__ glm::vec3 fromYCoCg(glm::vec3 c) {
    return glm::vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// A small statistical neighborhood of responsive history constrains the slower
// history. Independent CUDA implementation of the fast/slow-history concept;
// no NRD shaders or A-SVGF random-sequence replay are used here.
__global__ void responsivePass(glm::vec3 *out, glm::vec3 *nextFast, glm::vec2 *m,
                               float *length, float *response, const glm::vec3 *slow,
                               const glm::vec3 *fast, const glm::vec3 *input,
                               const GBufferTexel *g, Camera cam, OptimizedDenoiseSettings settings) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cam.resolution.x * cam.resolution.y) return;
    response[p] = 0;
    if (g[p].geomId < 0 || length[p] < 2) { out[p] = slow[p]; nextFast[p] = fast[p]; return; }
    int x = p % cam.resolution.x, y = p / cam.resolution.x;
    glm::vec3 meanFast(0), secondFast(0), meanRaw(0), secondRaw(0);
    float sum = 0;
    for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx) {
        int xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= cam.resolution.x || yy >= cam.resolution.y) continue;
        int q = xx + yy * cam.resolution.x;
        float w = geometryWeight(p, q, g, cam, settings.sigmaNormal);
        glm::vec3 f = toYCoCg(fast[q]);
        glm::vec3 r = toYCoCg((input[q] - (settings.falcorAligned ? g[q].emission : glm::vec3(0))) /
                              albedo(g[q], settings.demodulate, settings.falcorAligned));
        meanFast += w * f; secondFast += w * f * f;
        meanRaw += w * r; secondRaw += w * r * r; sum += w;
    }
    sum = fmaxf(sum, 1e-6f);
    meanFast /= sum; secondFast /= sum; meanRaw /= sum; secondRaw /= sum;
    glm::vec3 sigmaFast = glm::sqrt(glm::max(secondFast - meanFast * meanFast, glm::vec3(0)));
    glm::vec3 sigmaRaw = glm::sqrt(glm::max(secondRaw - meanRaw * meanRaw, glm::vec3(0)));
    glm::vec3 centerFast = toYCoCg(fast[p]);
    glm::vec3 old = toYCoCg(slow[p]);
    // Include the center sample so that a legitimate isolated highlight is not
    // forcibly replaced with the neighborhood mean.
    glm::vec3 lower = glm::min(meanFast - settings.clampSigma * sigmaFast, centerFast);
    glm::vec3 upper = glm::max(meanFast + settings.clampSigma * sigmaFast, centerFast);
    glm::vec3 clamped = settings.historyClamping ? glm::clamp(old, lower, upper) : old;

    // Confidence is a noise-aware heuristic, not an unbiased temporal gradient.
    // Test chroma as well as luminance to detect equal-brightness color changes.
    float delta = glm::length(old - meanRaw);
    float noise = glm::length(sigmaRaw) * 0.5f + glm::length(sigmaFast) * 2.0f;
    float scale = fmaxf(glm::length(old), glm::length(meanRaw)) + noise + 1e-4f;
    float react = settings.antiLag ? glm::clamp(settings.antiLagStrength * (delta - noise) / scale, 0.0f, 1.0f) : 0;
    glm::vec3 corrected = glm::mix(clamped, meanRaw, react);
    out[p] = glm::max(fromYCoCg(corrected), glm::vec3(0));
    nextFast[p] = glm::max(fromYCoCg(glm::mix(centerFast, meanRaw, react)), glm::vec3(0));
    response[p] = react;
    length[p] = glm::mix(length[p], fminf(length[p], (float)settings.fastHistoryFrames), react);
    // A color adjustment must update both moments. Preserve the existing variance
    // under a mean shift, then blend toward measured current variance on a reset.
    float v = fmaxf(m[p].y - m[p].x * m[p].x, 0.0f);
    float currentVariance = fmaxf(secondRaw.x - meanRaw.x * meanRaw.x, 0.0f);
    float correctedL = luminance(out[p]);
    m[p] = glm::vec2(correctedL, correctedL * correctedL + glm::mix(v, currentVariance, react));
}

__global__ void estimateVariance(float *out, const glm::vec3 *input, const glm::vec2 *m,
                                 const float *length, const GBufferTexel *g, Camera cam,
                                 float normalPower) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cam.resolution.x * cam.resolution.y) return;
    if (g[p].geomId < 0) { out[p] = 0; return; }
    if (length[p] >= 4) { out[p] = fmaxf(m[p].y - m[p].x * m[p].x, 0.0f); return; }
    int x = p % cam.resolution.x, y = p / cam.resolution.x;
    float sum = 0, first = 0, second = 0;
    for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx) {
        int xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= cam.resolution.x || yy >= cam.resolution.y) continue;
        int q = xx + yy * cam.resolution.x;
        float w = geometryWeight(p, q, g, cam, normalPower) * expf(-(dx * dx + dy * dy) / 18.0f);
        float l = luminance(input[q]);
        first += w * l; second += w * l * l; sum += w;
    }
    float mean = first / fmaxf(sum, 1e-6f);
    out[p] = fmaxf(second / fmaxf(sum, 1e-6f) - mean * mean, 0.0f);
}

__global__ void atrousPass(glm::vec3 *dst, float *varianceOut, const glm::vec3 *src,
                          const float *varianceIn, const GBufferTexel *g, Camera cam,
                          int step, OptimizedDenoiseSettings settings, const GeometryGuide *guide) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cam.resolution.x * cam.resolution.y) return;
    if (g[p].geomId < 0) { dst[p] = src[p]; varianceOut[p] = varianceIn[p]; return; }
    int x = p % cam.resolution.x, y = p / cam.resolution.x;
    float smoothVariance = 0, gaussianSum = 0;
    const float gaussian[3] = {1, 2, 1};
    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
        int xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= cam.resolution.x || yy >= cam.resolution.y) continue;
        float w = gaussian[dx + 1] * gaussian[dy + 1];
        smoothVariance += w * varianceIn[xx + yy * cam.resolution.x]; gaussianSum += w;
    }
    float sigma = settings.sigmaLuminance * sqrtf(fmaxf(smoothVariance / gaussianSum, 0.0f)) + 1e-6f;
    const float h[5] = {1, 4, 6, 4, 1};
    float centerLuminance = luminance(src[p]);
    glm::vec3 result(0);
    float sum = 0, var = 0;
    for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx) {
        int xx = x + step * dx, yy = y + step * dy;
        if (xx < 0 || yy < 0 || xx >= cam.resolution.x || yy >= cam.resolution.y) continue;
        int q = xx + yy * cam.resolution.x;
        float geometry = settings.falcorAligned && settings.depthDerivatives ? alignedGeometryWeight(p, q, g, guide,
            step * sqrtf((float)(dx * dx + dy * dy)), settings.sigmaNormal) : geometryWeight(p, q, g, cam, settings.sigmaNormal);
        float w = h[dx + 2] * h[dy + 2] * (p == q ? 1.0f : geometry) *
                  expf(-fabsf(centerLuminance - luminance(src[q])) / sigma);
        result += w * src[q]; sum += w; var += w * w * varianceIn[q];
    }
    dst[p] = sum > 1e-6f ? result / sum : src[p];
    varianceOut[p] = sum > 1e-6f ? var / (sum * sum) : varianceIn[p];
}

__global__ void compose(glm::vec3 *out, const glm::vec3 *filtered, const GBufferTexel *g,
                        int count, bool demodulate, const float *length, const float *response, int debugView, bool aligned) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < count) {
        if (debugView == 1) out[p] = glm::vec3(length[p] / 32.0f);
        else if (debugView == 2) out[p] = glm::vec3(response[p], 0.0f, 0.0f);
        else out[p] = filtered[p] * albedo(g[p], demodulate, aligned) + (aligned ? g[p].emission : glm::vec3(0));
    }
}
}

void optimizedDenoiseReset() { hasHistory = false; }
void optimizedDenoiseCapture(const std::string &prefix) { nextCapture = prefix; }

void optimizedDenoiseInit(int count) {
    check(cudaMalloc(&guides, count * sizeof(GeometryGuide)));
    check(cudaMalloc(&historyColor, count * sizeof(glm::vec3)));
    check(cudaMalloc(&temporalColor, count * sizeof(glm::vec3)));
    check(cudaMalloc(&historyFast, count * sizeof(glm::vec3)));
    check(cudaMalloc(&fastColor, count * sizeof(glm::vec3)));
    check(cudaMalloc(&reaction, count * sizeof(float)));
    check(cudaMalloc(&historyMoments, count * sizeof(glm::vec2)));
    check(cudaMalloc(&moments, count * sizeof(glm::vec2)));
    check(cudaMalloc(&historyLength, count * sizeof(float)));
    check(cudaMalloc(&lengths, count * sizeof(float)));
    check(cudaMalloc(&historyGeometry, count * sizeof(GBufferTexel)));
    for (int i = 0; i < 2; ++i) {
        check(cudaMalloc(&color[i], count * sizeof(glm::vec3)));
        check(cudaMalloc(&variance[i], count * sizeof(float)));
    }
    hasHistory = false;
}

void optimizedDenoiseFree() {
    check(cudaFree(guides)); guides = nullptr;
    check(cudaFree(historyColor)); historyColor = nullptr;
    check(cudaFree(temporalColor)); temporalColor = nullptr;
    check(cudaFree(historyFast)); historyFast = nullptr;
    check(cudaFree(fastColor)); fastColor = nullptr;
    check(cudaFree(reaction)); reaction = nullptr;
    check(cudaFree(historyMoments)); historyMoments = nullptr;
    check(cudaFree(moments)); moments = nullptr;
    check(cudaFree(historyLength)); historyLength = nullptr;
    check(cudaFree(lengths)); lengths = nullptr;
    check(cudaFree(historyGeometry)); historyGeometry = nullptr;
    for (int i = 0; i < 2; ++i) {
        check(cudaFree(color[i])); color[i] = nullptr;
        check(cudaFree(variance[i])); variance[i] = nullptr;
    }
    hasHistory = false;
}

void optimizedDenoise(glm::vec3 *output, const glm::vec3 *input,
                     const GBufferTexel *g, const Camera &cam,
                     const OptimizedGeometryGuide *externalGuides) {
    int count = cam.resolution.x * cam.resolution.y;
    int blocks = (count + 127) / 128;
    if (externalGuides) check(cudaMemcpyAsync(guides, externalGuides, count * sizeof(GeometryGuide), cudaMemcpyDeviceToDevice));
    else if (optimizedSettings.falcorAligned) prepareGuides<<<blocks, 128>>>(guides, g, cam);
    temporalPass<<<blocks, 128>>>(temporalColor, fastColor, moments, lengths, input, g, historyColor,
        historyFast, historyMoments, historyLength, historyGeometry, cam, hasHistory ? previousCamera : cam,
        hasHistory, optimizedSettings, guides);
    if (!nextCapture.empty()) {
        capturePFM(nextCapture + "-temporal.pfm", temporalColor, cam.resolution);
        capturePFM(nextCapture + "-history.pfm", lengths, cam.resolution);
    }
    glm::vec3 *src = temporalColor;
    if (optimizedSettings.historyClamping || optimizedSettings.antiLag) {
        responsivePass<<<blocks, 128>>>(color[0], historyFast, moments, lengths, reaction,
            temporalColor, fastColor, input, g, cam, optimizedSettings);
        src = color[0];
    } else {
        check(cudaMemsetAsync(reaction, 0, count * sizeof(float)));
        check(cudaMemcpyAsync(historyFast, fastColor, count * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));
    }
    if (optimizedSettings.falcorAligned && optimizedSettings.spatial) {
        glm::vec3 *dst = src == color[0] ? color[1] : color[0];
        filterShortHistory<<<blocks, 128>>>(dst, variance[0], src, moments, lengths, g, guides, cam, optimizedSettings);
        src = dst;
    } else estimateVariance<<<blocks, 128>>>(variance[0], src, moments, lengths, g, cam, optimizedSettings.sigmaNormal);
    if (!nextCapture.empty()) {
        capturePFM(nextCapture + "-prefilter.pfm", src, cam.resolution);
        capturePFM(nextCapture + "-variance.pfm", variance[0], cam.resolution);
    }
    int levels = optimizedSettings.spatial ? std::max(0, std::min(optimizedSettings.levels, 7)) : 0;
    for (int i = 0; i < levels; ++i) {
        glm::vec3 *dst = src == color[0] ? color[1] : color[0];
        atrousPass<<<blocks, 128>>>(dst, variance[(i + 1) % 2], src, variance[i % 2],
                                   g, cam, 1 << i, optimizedSettings, guides);
        src = dst;
        int feedback = optimizedSettings.falcorAligned ? std::min(optimizedSettings.feedbackTap, levels - 1) : 0;
        if (i == feedback) check(cudaMemcpyAsync(historyColor, src, count * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));
        if (!nextCapture.empty()) capturePFM(nextCapture + "-atrous-" + std::to_string(i) + ".pfm", src, cam.resolution);
    }
    if (levels == 0) check(cudaMemcpyAsync(historyColor, src, count * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));
    compose<<<blocks, 128>>>(output, src, g, count, optimizedSettings.demodulate,
                             lengths, reaction, optimizedSettings.debugView, optimizedSettings.falcorAligned);
    check(cudaMemcpyAsync(historyGeometry, g, count * sizeof(GBufferTexel), cudaMemcpyDeviceToDevice));
    std::swap(historyMoments, moments);
    std::swap(historyLength, lengths);
    previousCamera = cam;
    hasHistory = true;
    nextCapture.clear();
    check(cudaGetLastError());
}
