#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtraceReset();
void retrieveOriginalImage();
void pathtrace(uchar4 *pbo, int frame);
void saveComparison(const std::string &filename);
void saveLinearCapture(const std::string &prefix);
struct FramePerformance {
    float tracing = 0, originalDenoise = 0, optimizedDenoise = 0, packing = 0, transfer = 0;
};
extern FramePerformance framePerformance;
void finishFrameProfiling(float transferMilliseconds);
void waitForFrame();
void printFrameProfiling();
