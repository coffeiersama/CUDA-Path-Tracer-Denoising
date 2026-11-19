#include "denoise_optimized.h"
#include "denoise_sequence.h"
#include "capture.h"
#include <iostream>
#include <cstring>
#include <cmath>

static void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

int main(int argc, char** argv) try {
    require(argc == 3, "Usage: falcor_replay INPUT_DIRECTORY OUTPUT_PREFIX");
    const std::filesystem::path dir(argv[1]);
    std::ifstream f(dir / "stationary.seq", std::ios::binary);
    DenoiseSequenceHeader h;
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    require(f && !memcmp(h.magic, "SVGFSEQ1", 8) && h.cameraBytes == sizeof(Camera) &&
            h.geometryBytes == sizeof(GBufferTexel) && h.width && h.width <= 8192 && h.height && h.height <= 8192 &&
            h.frames && h.frames <= 4096, "Invalid Falcor sequence header/layout");
    Camera cam; f.read(reinterpret_cast<char*>(&cam), sizeof(cam));
    require(cam.resolution == glm::ivec2(h.width, h.height), "Camera dimensions mismatch");
    const size_t n = size_t(h.width) * h.height;
    std::vector<GBufferTexel> geo(n);
    f.read(reinterpret_cast<char*>(geo.data()), n * sizeof(GBufferTexel));
    require(!!f, "Truncated G-buffer");
    const auto framesStart = f.tellg();
    std::vector<OptimizedGeometryGuide> guide(n);
    std::ifstream gf(dir / "guides.bin", std::ios::binary);
    gf.read(reinterpret_cast<char*>(guide.data()), n * sizeof(guide[0]));
    require(!!gf, "Truncated exact raster guide data");
    glm::vec3 *di, *doo; GBufferTexel *dg; OptimizedGeometryGuide *dguide;
    check(cudaMalloc(&di, n * sizeof(glm::vec3))); check(cudaMalloc(&doo, n * sizeof(glm::vec3)));
    check(cudaMalloc(&dg, n * sizeof(GBufferTexel))); check(cudaMalloc(&dguide, n * sizeof(guide[0])));
    check(cudaMemcpy(dg, geo.data(), n * sizeof(GBufferTexel), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dguide, guide.data(), n * sizeof(guide[0]), cudaMemcpyHostToDevice));
    optimizedDenoiseInit(int(n));
    std::vector<glm::vec3> input(n), output(n);
    const std::string prefix(argv[2]);
    const auto parent = std::filesystem::path(prefix).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream timing(prefix + "-timing.json"); timing << "{";
    std::ofstream presets(prefix + "-settings.json"); presets << "{";
    for (int variant = 0; variant < 2; ++variant) {
        optimizedSettings = OptimizedDenoiseSettings{};
        const std::string name = variant == 0 ? "current" : "matched";
        if (variant == 1) { optimizedSettings.alpha = .05f; optimizedSettings.sigmaLuminance = 10; optimizedSettings.feedbackTap = 1; }
        if (variant) presets << ",";
        presets << "\"" << name << "\":{\"alpha\":" << optimizedSettings.alpha
                << ",\"moments_alpha\":" << optimizedSettings.momentsAlpha
                << ",\"phi_color\":" << optimizedSettings.sigmaLuminance
                << ",\"phi_normal\":" << optimizedSettings.sigmaNormal
                << ",\"iterations\":" << optimizedSettings.levels
                << ",\"feedback_tap\":" << optimizedSettings.feedbackTap << "}";
        optimizedDenoiseReset(); f.clear(); f.seekg(framesStart);
        double elapsed = 0;
        cudaEvent_t start, end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
        for (uint32_t frame = 0; frame < h.frames; ++frame) {
            f.read(reinterpret_cast<char*>(input.data()), n * sizeof(glm::vec3));
            require(!!f, "Truncated raw frame");
            check(cudaMemcpy(di, input.data(), n * sizeof(glm::vec3), cudaMemcpyHostToDevice));
            check(cudaEventRecord(start));
            optimizedDenoise(doo, di, dg, cam, dguide);
            check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
            float ms; check(cudaEventElapsedTime(&ms, start, end)); if (frame >= 8) elapsed += ms;
            if (frame == 0 || frame + 1 == h.frames || frame + 1 == 32 || frame + 1 == 64) {
                check(cudaMemcpy(output.data(), doo, n * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
                for (const auto& p : output) require(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z), "Nonfinite CUDA output");
                writePFM(prefix + "-" + name + "-" + std::to_string(frame + 1) + ".pfm", output, cam.resolution);
            }
        }
        if (variant) timing << ",";
        timing << "\"" << name << "_gpu_ms\":" << elapsed / std::max(int(h.frames)-8, 1);
        check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
        std::cout << "REPLAY_COMPLETE " << name << " " << h.frames << " frames\n";
    }
    timing << "}\n";
    presets << "}\n";
    optimizedDenoiseFree(); check(cudaFree(di)); check(cudaFree(doo)); check(cudaFree(dg)); check(cudaFree(dguide));
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
