#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <cmath>

extern bool displayToneMap;
extern float displayExposureEV;

// Shared by CUDA preview packing and CPU PNG export. Never used on filter data.
__host__ __device__ inline glm::vec3 displayColor(glm::vec3 linear, bool toneMap, float exposureEV) {
    if (!toneMap) return glm::clamp(linear, glm::vec3(0), glm::vec3(1));
    glm::vec3 c = glm::max(linear, glm::vec3(0)) * exp2f(exposureEV);
    c /= glm::vec3(1) + c;
    for (int i = 0; i < 3; ++i)
        c[i] = c[i] <= 0.0031308f ? 12.92f * c[i] : 1.055f * powf(c[i], 1.0f / 2.4f) - 0.055f;
    return glm::clamp(c, glm::vec3(0), glm::vec3(1));
}
