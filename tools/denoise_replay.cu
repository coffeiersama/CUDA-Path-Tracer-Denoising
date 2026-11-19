#include "denoise_optimized.h"
#include "denoise_sequence.h"
#include "capture.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

static void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void cudaCheck(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }

static std::vector<glm::vec3> loadPFM(const char *path, glm::ivec2 res) {
    std::ifstream f(path, std::ios::binary);
    std::string kind; int w, h; float scale;
    f >> kind >> w >> h >> scale; f.get();
    require(f && kind == "PF" && w == res.x && h == res.y && scale == -1.0f, "Reference PFM layout mismatch");
    std::vector<glm::vec3> image(w * h), disk(w * h);
    f.read(reinterpret_cast<char*>(disk.data()), disk.size() * sizeof(glm::vec3));
    require(!!f, "Truncated reference PFM");
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
        image[y * w + x] = disk[(h - 1 - y) * w + w - 1 - x];
    return image;
}

int main(int argc, char **argv) try {
    require(argc == 5 || (argc == 6 && std::string(argv[5]) == "--ablation-only"), "Usage: denoise_replay SEQUENCE REF_A.pfm REF_B.pfm OUTPUT_PREFIX [--ablation-only]");
    std::ifstream f(argv[1], std::ios::binary);
    DenoiseSequenceHeader header;
    f.read(reinterpret_cast<char*>(&header), sizeof(header));
    require(f && memcmp(header.magic, "SVGFSEQ1", 8) == 0 && header.cameraBytes == sizeof(Camera) &&
            header.geometryBytes == sizeof(GBufferTexel) && header.width > 0 && header.width <= 8192 &&
            header.height > 0 && header.height <= 8192 && header.frames > 0 && header.frames <= 4096, "Invalid sequence header");
    Camera cam;
    f.read(reinterpret_cast<char*>(&cam), sizeof(cam));
    require(cam.resolution.x == header.width && cam.resolution.y == header.height, "Sequence camera mismatch");
    const size_t n = (size_t)header.width * header.height;
    std::vector<GBufferTexel> g(n);
    std::vector<glm::vec3> samples(n * header.frames), output(n);
    f.read(reinterpret_cast<char*>(g.data()), g.size() * sizeof(GBufferTexel));
    f.read(reinterpret_cast<char*>(samples.data()), samples.size() * sizeof(glm::vec3));
    require(!!f, "Truncated sequence");
    auto ref = loadPFM(argv[2], cam.resolution), refB = loadPFM(argv[3], cam.resolution);
    for (size_t p = 0; p < n; ++p) ref[p] = (ref[p] + refB[p]) * 0.5f;
    glm::vec3 *di, *doo; GBufferTexel *dg;
    cudaCheck(cudaMalloc(&di, n * sizeof(glm::vec3))); cudaCheck(cudaMalloc(&doo, n * sizeof(glm::vec3)));
    cudaCheck(cudaMalloc(&dg, n * sizeof(GBufferTexel)));
    cudaCheck(cudaMemcpy(dg, g.data(), n * sizeof(GBufferTexel), cudaMemcpyHostToDevice));
    optimizedDenoiseInit((int)n);
    auto run = [&](OptimizedDenoiseSettings settings, bool dump, const std::string &prefix) {
        optimizedSettings = settings; optimizedDenoiseReset();
        for (uint32_t frame = 0; frame < header.frames; ++frame) {
            cudaCheck(cudaMemcpyAsync(di, samples.data() + frame * n, n * sizeof(glm::vec3), cudaMemcpyHostToDevice));
            if (dump && frame + 1 == header.frames) optimizedDenoiseCapture(prefix);
            optimizedDenoise(doo, di, dg, cam);
        }
        cudaCheck(cudaMemcpy(output.data(), doo, n * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
        double mse = 0;
        for (size_t p = 0; p < n; ++p) for (int c = 0; c < 3; ++c) {
            require(std::isfinite(output[p][c]), "Nonfinite filter result");
            double delta = (double)output[p][c] - ref[p][c]; mse += delta * delta;
        }
        if (dump) writePFM(prefix + "-optimized.pfm", output, cam.resolution);
        return mse / (3 * n);
    };
    std::string prefix(argv[4]);
    auto parent = std::filesystem::path(prefix).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream csv(prefix + "-sweep.csv");
    csv << "aligned,depth_derivatives,alpha,sigma_l,feedback,levels,linear_mse\n";
    OptimizedDenoiseSettings previous;
    previous.falcorAligned = false; previous.antiLag = previous.historyClamping = true;
    previous.alpha = 0.2f; previous.sigmaLuminance = 4; previous.levels = 5;
    previous.feedbackTap = 0;
    double oldError = run(previous, true, prefix + "-previous");
    OptimizedDenoiseSettings official;
    official.alpha = 0.05f; official.sigmaLuminance = 10; official.feedbackTap = 1;
    double officialError = run(official, true, prefix + "-official");
    if (argc == 6) {
        auto noResponsive = previous;
        noResponsive.antiLag = noResponsive.historyClamping = false;
        double noResponsiveError = run(noResponsive, false, "");
        auto matchedLegacy = OptimizedDenoiseSettings{};
        matchedLegacy.alpha = .2f; matchedLegacy.sigmaLuminance = 2; matchedLegacy.feedbackTap = 0;
        matchedLegacy.falcorAligned = false;
        double matchedError = run(matchedLegacy, true, prefix + "-matched-legacy");
        auto selected = matchedLegacy; selected.falcorAligned = true;
        double alignedError = run(selected, true, prefix + "-aligned");
        std::ofstream ablation(prefix + "-ablation.json");
        ablation << "{\"previous\":" << oldError << ",\"previous_without_responsive_stage\":" << noResponsiveError
                 << ",\"legacy_core_with_selected_parameters\":" << matchedError
                 << ",\"aligned_core_with_selected_parameters\":" << alignedError
                 << ",\"falcor_defaults\":" << officialError << "}\n";
        std::cout << "Ablation: previous " << oldError << " | no responsive " << noResponsiveError
                  << " | matched legacy " << matchedError << " | aligned " << alignedError << std::endl;
        optimizedDenoiseFree(); cudaCheck(cudaFree(di)); cudaCheck(cudaFree(doo)); cudaCheck(cudaFree(dg));
        return 0;
    }
    double bestError = std::numeric_limits<double>::infinity();
    OptimizedDenoiseSettings best;
    for (bool derivative : {true, false}) for (float alpha : {0.05f, 0.1f, 0.2f})
    for (float sigma : {2.0f, 4.0f, 10.0f}) for (int feedback : {0, 1}) for (int levels : {4, 5}) {
        OptimizedDenoiseSettings candidate;
        candidate.depthDerivatives = derivative; candidate.alpha = alpha; candidate.sigmaLuminance = sigma;
        candidate.feedbackTap = feedback; candidate.levels = levels;
        double error = run(candidate, false, "");
        csv << "1," << derivative << "," << alpha << "," << sigma << "," << feedback << "," << levels << "," << error << "\n";
        if (error < bestError) { bestError = error; best = candidate; }
    }
    run(best, true, prefix + "-best");
    std::ofstream result(prefix + "-sweep.json");
    result << "{\"previous_mse\":" << oldError << ",\"official_defaults_mse\":" << officialError
           << ",\"best_mse\":" << bestError << ",\"best\":{\"depth_derivatives\":" << best.depthDerivatives
           << ",\"alpha\":" << best.alpha << ",\"sigma_l\":" << best.sigmaLuminance
           << ",\"feedback\":" << best.feedbackTap << ",\"levels\":" << best.levels << "}}\n";
    std::cout << "Previous MSE " << oldError << " | Falcor defaults " << officialError << " | selected " << bestError
              << " | alpha " << best.alpha << " sigma " << best.sigmaLuminance << " feedback " << best.feedbackTap
              << " levels " << best.levels << " derivatives " << best.depthDerivatives << std::endl;
    optimizedDenoiseFree(); cudaCheck(cudaFree(di)); cudaCheck(cudaFree(doo)); cudaCheck(cudaFree(dg));
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << std::endl; return 1;
}
