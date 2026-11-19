#pragma once

extern GLuint pbo;
void checkCudaInterop(cudaError_t result, const char *operation);
uchar4 *mapPreviewBuffer();
void unmapPreviewBuffer();

std::string currentTimeString();
bool init();
void mainLoop();
