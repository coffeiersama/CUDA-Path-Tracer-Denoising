#pragma once

#include "scene.h"
#include "pathtrace.h"

void denoiseInit(Scene *scene);
void denoiseFree();
void denoiseReset();
void denoise(glm::vec3 * output, glm::vec3 * input, GBufferTexel * gbuffer);
