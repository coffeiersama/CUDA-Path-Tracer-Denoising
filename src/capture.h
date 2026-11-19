#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <vector>

// PFM: unbounded float32 linear data, little endian, bottom row first.
// Match the horizontal orientation used by the OpenGL preview and PNG export.
template<class T> inline void writePFM(const std::string &path, const std::vector<T> &pixels, glm::ivec2 res) {
    static_assert(sizeof(T) == sizeof(float) || sizeof(T) == 3 * sizeof(float), "PFM channel count");
    std::vector<T> row(res.x);
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot write capture: " + path);
    stream << (sizeof(T) == sizeof(float) ? "Pf\n" : "PF\n") << res.x << " " << res.y << "\n-1.0\n";
    for (int y = res.y - 1; y >= 0; --y) {
        for (int x = 0; x < res.x; ++x) row[x] = pixels[y * res.x + res.x - 1 - x];
        stream.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(T));
    }
    if (!stream) throw std::runtime_error("Capture write failed: " + path);
}

template<class T> inline void capturePFM(const std::string &path, const T *device, glm::ivec2 res) {
    std::vector<T> pixels(res.x * res.y);
    cudaError_t result = cudaMemcpy(pixels.data(), device, pixels.size() * sizeof(T), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    writePFM(path, pixels, res);
}
