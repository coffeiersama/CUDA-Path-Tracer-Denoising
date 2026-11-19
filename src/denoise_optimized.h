#pragma once
#include "sceneStructs.h"

struct OptimizedDenoiseSettings {
    bool falcorAligned = true;
    bool depthDerivatives = true;
    bool temporal = true;
    bool spatial = true;
    bool demodulate = true;
    // Independently implemented, RELAX-inspired responsive history (not NRD).
    bool historyClamping = false;
    bool antiLag = false;
    int fastHistoryFrames = 3;
    float clampSigma = 2.0f;
    float antiLagStrength = 1.0f;
    int debugView = 0; // 0: filtered, 1: history length, 2: anti-lag reaction
    int levels = 4;
    int feedbackTap = 1;
    float alpha = 0.05f;
    float momentsAlpha = 0.2f;
    float sigmaLuminance = 10.0f;
    float sigmaNormal = 128.0f;
};

extern OptimizedDenoiseSettings optimizedSettings;
// Optional externally supplied raster guides for shared-input Falcor benchmarks.
struct OptimizedGeometryGuide { float depth, width, normalWidth; };
void optimizedDenoiseInit(int pixelCount);
void optimizedDenoiseFree();
void optimizedDenoiseReset();
void optimizedDenoiseCapture(const std::string &prefix);
void optimizedDenoise(glm::vec3 *output, const glm::vec3 *radiance,
                     const GBufferTexel *gbuffer, const Camera &camera,
                     const OptimizedGeometryGuide *externalGuides = nullptr);
