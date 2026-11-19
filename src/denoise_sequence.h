#pragma once
#include <cstdint>
#include "sceneStructs.h"
// Local benchmark format, versioned and checked against this build's layouts.
// Header, Camera, stationary GBufferTexel[], then frames of linear RGB samples.
struct DenoiseSequenceHeader {
    char magic[8] = {'S','V','G','F','S','E','Q','1'};
    uint32_t width, height, frames;
    uint32_t cameraBytes = sizeof(Camera), geometryBytes = sizeof(GBufferTexel);
};
