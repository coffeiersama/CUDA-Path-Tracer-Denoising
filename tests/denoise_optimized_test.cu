#include "denoise_optimized.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void require(bool ok, const char *message) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}
static void cudaCheck(cudaError_t result) { require(result == cudaSuccess, cudaGetErrorString(result)); }

int main() {
    // Existing tests retain the exact baseline contract.
    optimizedSettings.antiLag = optimizedSettings.historyClamping = false;
    optimizedSettings.falcorAligned = false;
    optimizedSettings.alpha = 0.2f;
    optimizedSettings.levels = 5;
    optimizedSettings.sigmaLuminance = 4;
    Camera cam{};
    cam.resolution = glm::ivec2(48, 24);
    cam.position = glm::vec3(0);
    cam.right = glm::vec3(1, 0, 0);
    cam.up = glm::vec3(0, 1, 0);
    cam.view = glm::vec3(0, 0, 1);
    cam.pixelLength = glm::vec2(0.07f, 0.11f);
    const int n = cam.resolution.x * cam.resolution.y;
    std::vector<GBufferTexel> g(n);
    std::vector<glm::vec3> input(n), output(n);
    glm::vec3 *devInput, *devOutput;
    GBufferTexel *devG;
    cudaCheck(cudaMalloc(&devInput, n * sizeof(glm::vec3)));
    cudaCheck(cudaMalloc(&devOutput, n * sizeof(glm::vec3)));
    cudaCheck(cudaMalloc(&devG, n * sizeof(GBufferTexel)));
    optimizedDenoiseInit(n);
    auto geometry = [&]() {
        for (int p = 0; p < n; ++p) {
            int x = p % cam.resolution.x, y = p / cam.resolution.x;
            g[p].position = cam.position + 5.0f * (cam.view - cam.right * cam.pixelLength.x *
                (x - cam.resolution.x * 0.5f + 0.5f) - cam.up * cam.pixelLength.y *
                (y - cam.resolution.y * 0.5f + 0.5f));
            g[p].normal = glm::vec3(0, 0, -1);
            g[p].geomId = 1; g[p].albedo = glm::vec3(1); g[p].ialbedo = glm::vec3(1);
        }
    };
    auto run = [&]() {
        cudaCheck(cudaMemcpy(devInput, input.data(), n * sizeof(glm::vec3), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(devG, g.data(), n * sizeof(GBufferTexel), cudaMemcpyHostToDevice));
        optimizedDenoise(devOutput, devInput, devG, cam);
        cudaCheck(cudaMemcpy(output.data(), devOutput, n * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
        for (const auto &c : output) require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z), "finite output");
    };
    geometry();
    std::fill(input.begin(), input.end(), glm::vec3(0.3f, 0.6f, 0.9f));
    for (int i = 0; i < 8; ++i) run();
    for (const auto &c : output) require(glm::length(c - input[0]) < 1e-4f, "constant color preserved, including borders");
    puts("PASS: constant signal and borders");

    optimizedDenoiseReset();
    for (int p = 0; p < n; ++p) {
        g[p].albedo = glm::vec3((p % 2) ? 0.1f : 0.9f);
        input[p] = g[p].albedo * 0.5f;
    }
    for (int i = 0; i < 5; ++i) run();
    for (int p = 0; p < n; ++p) require(glm::length(output[p] - input[p]) < 1e-4f, "albedo texture preserved");
    puts("PASS: sharp albedo texture survives filtering");

    geometry(); optimizedDenoiseReset(); optimizedSettings.spatial = false;
    std::fill(input.begin(), input.end(), glm::vec3(1)); run();
    std::fill(input.begin(), input.end(), glm::vec3(0)); run();
    for (const auto &c : output) require(fabsf(c.x - 0.5f) < 1e-4f, "two-frame temporal average");
    for (auto &v : g) v.position.z += 2;
    std::fill(input.begin(), input.end(), glm::vec3(0.2f)); run();
    for (const auto &c : output) require(fabsf(c.x - 0.2f) < 1e-4f, "reject stale depth layer with identical normals and ID");
    puts("PASS: temporal accumulation and depth disocclusion rejection");

    geometry(); optimizedDenoiseReset();
    for (int p = 0; p < n; ++p) input[p] = glm::vec3(0.1f + 0.01f * (p % cam.resolution.x));
    run();
    cam.position.x += 5 * cam.pixelLength.x; geometry();
    std::fill(input.begin(), input.end(), glm::vec3(0)); run();
    int center = 24 + 12 * cam.resolution.x;
    require(fabsf(output[center].x - 0.5f * (0.1f + 0.01f * 23)) < 1e-4f,
            "camera motion reprojects correct history on non-square, non-default FOV input");
    puts("PASS: camera-motion reprojection with non-default projection");

    optimizedDenoiseReset();
    for (auto &v : g) { v.geomId = -1; v.normal = glm::vec3(0); }
    std::fill(input.begin(), input.end(), glm::vec3(0)); run();
    for (const auto &c : output) require(glm::length(c) < 1e-6f, "background remains black without history");
    puts("PASS: background and history reset");

    geometry(); optimizedDenoiseReset(); optimizedSettings.spatial = true;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise(-0.4f, 0.4f);
    double rawError = 0, filteredError = 0;
    for (int frame = 0; frame < 12; ++frame) {
        for (auto &c : input) c = glm::vec3(0.5f + noise(rng));
        run();
        if (frame >= 4) for (int p = 0; p < n; ++p) {
            rawError += (input[p].x - 0.5) * (input[p].x - 0.5);
            filteredError += (output[p].x - 0.5) * (output[p].x - 0.5);
        }
    }
    printf("Synthetic flat-field MSE ratio (optimized/raw): %.6f\n", filteredError / rawError);
    require(filteredError < rawError * 0.25, "noise reduction against known clean signal");
    puts("PASS: noise reduction against analytic reference");

    // Compare identical noisy sequences with and without responsive history.
    // No resets are allowed when the illumination changes.
    struct Metrics { double stable = 0, transient = 0; };
    auto measure = [&](bool adaptive, bool colorChange) {
        optimizedDenoiseReset();
        optimizedSettings.antiLag = optimizedSettings.historyClamping = adaptive;
        optimizedSettings.spatial = true;
        optimizedSettings.alpha = 0.1f;
        std::mt19937 sequence(1234);
        std::uniform_real_distribution<float> jitter(-0.15f, 0.15f);
        Metrics metrics;
        for (int f = 0; f < 44; ++f) {
            glm::vec3 target = f < 32 ? glm::vec3(0.8f) : glm::vec3(0.1f);
            if (colorChange) target = f < 32 ? glm::vec3(0.8f, 0.1f, 0.1f) : glm::vec3(0.1f, 0.3081f, 0.1f);
            for (auto &c : input) c = glm::max(target + glm::vec3(jitter(sequence)), glm::vec3(0));
            run();
            if ((f >= 24 && f < 32) || (f >= 32 && f < 38)) for (const auto &c : output) {
                glm::vec3 e = c - target;
                if (f < 32) metrics.stable += glm::dot(e, e);
                else metrics.transient += glm::dot(e, e);
            }
        }
        return metrics;
    };
    Metrics baseline = measure(false, false), adaptive = measure(true, false);
    printf("Light-switch transient MSE ratio (adaptive/baseline): %.6f\n", adaptive.transient / baseline.transient);
    printf("Static noisy-field MSE ratio (adaptive/baseline): %.6f\n", adaptive.stable / baseline.stable);
    require(adaptive.transient < baseline.transient * 0.5, "faster response to lighting change without resets");
    require(adaptive.stable < baseline.stable * 2.0 + 1e-5, "bounded noise cost in stationary lighting");
    baseline = measure(false, true); adaptive = measure(true, true);
    printf("Equal-luminance color-switch MSE ratio (adaptive/baseline): %.6f\n", adaptive.transient / baseline.transient);
    require(adaptive.transient < baseline.transient * 0.75, "respond to chromatic changes, not only luminance");
    puts("PASS: adaptive response to lighting and chromatic changes, with stationary-noise control");

    optimizedSettings.debugView = 1; run();
    for (const auto &c : output) require(c.x >= 0 && c.x <= 1, "history debug in range");
    optimizedSettings.debugView = 2; run();
    for (const auto &c : output) require(c.x >= 0 && c.x <= 1 && c.y == 0 && c.z == 0, "reaction debug in range");
    optimizedSettings.debugView = 0;
    puts("PASS: adaptive debug views");
    // Check the aligned path independently of the optional anti-lag heuristics.
    optimizedSettings = OptimizedDenoiseSettings{};
    geometry(); optimizedDenoiseReset();
    for (int p = 0; p < n; ++p) {
        g[p].albedo = glm::vec3(p % 2 ? 0.05f : 0.9f);
        g[p].emission = glm::vec3(2.0f, 0.1f, 0.3f);
        input[p] = g[p].emission + g[p].albedo * 0.5f;
    }
    for (int i = 0; i < 8; ++i) run();
    for (int p = 0; p < n; ++p) require(glm::length(output[p] - input[p]) < 1e-4f, "emission bypass and textured constant illumination");
    puts("PASS: Falcor-aligned emission bypass and albedo remodulation");
    geometry(); for (auto &v : g) v.emission = glm::vec3(0);
    optimizedDenoiseReset(); optimizedSettings.levels = 0;
    for (int p = 0; p < n; ++p) input[p] = glm::vec3(0.5f + (p % 2 ? 0.3f : -0.3f));
    run();
    double shortError = 0;
    for (const auto &c : output) shortError += (c.x - 0.5f) * (c.x - 0.5f);
    require(shortError / n < 0.02, "short-history illumination is prefiltered before any atrous pass");
    puts("PASS: Falcor-aligned short-history illumination prefilter");
    optimizedDenoiseReset(); optimizedSettings.spatial = false;
    optimizedSettings.alpha = 0.05f;
    std::fill(input.begin(), input.end(), glm::vec3(1));
    for (int i = 0; i < 40; ++i) run();
    std::fill(input.begin(), input.end(), glm::vec3(0)); run();
    for (const auto &c : output) require(fabsf(c.x - 0.95f) < 1e-4f, "independent color alpha after mature history");
    puts("PASS: Falcor-aligned temporal-only control and color alpha");
    optimizedDenoiseReset(); geometry();
    for (int p = 0; p < n; ++p) input[p] = glm::vec3(0.1f + 0.01f * (p % cam.resolution.x));
    run();
    cam.position.x += 5 * cam.pixelLength.x; geometry();
    std::fill(input.begin(), input.end(), glm::vec3(0)); run();
    require(fabsf(output[center].x - 0.5f * (0.1f + 0.01f * 23)) < 1e-4f,
            "aligned reprojection samples the expected previous world point");
    for (auto &v : g) v.normal = glm::vec3(1, 0, 0);
    std::fill(input.begin(), input.end(), glm::vec3(0.7f)); run();
    for (const auto &c : output) require(fabsf(c.x - 0.7f) < 1e-4f, "aligned normal consistency rejects stale history");
    puts("PASS: Falcor-aligned camera reprojection and normal rejection");
    // External raster depth guides must actually drive spatial edge stopping.
    optimizedSettings = OptimizedDenoiseSettings{};
    optimizedSettings.temporal = false; optimizedSettings.levels = 1;
    geometry(); optimizedDenoiseReset();
    for (int p = 0; p < n; ++p) input[p] = glm::vec3(p % 2 ? .8f : .2f);
    run();
    double computedDifference = 0;
    for (int p = 0; p < n; ++p) computedDifference += fabsf(output[p].x - input[p].x);
    require(computedDifference / n > .05, "flat computed depth permits spatial noise reduction");
    std::vector<OptimizedGeometryGuide> external(n);
    for (int p = 0; p < n; ++p) external[p] = {p % 2 ? 100.f : 1.f, 0.f, 0.f};
    OptimizedGeometryGuide* devExternal;
    cudaCheck(cudaMalloc(&devExternal, n * sizeof(external[0])));
    cudaCheck(cudaMemcpy(devExternal, external.data(), n * sizeof(external[0]), cudaMemcpyHostToDevice));
    optimizedDenoiseReset();
    optimizedDenoise(devOutput, devInput, devG, cam, devExternal);
    cudaCheck(cudaMemcpy(output.data(), devOutput, n * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
    for (int p = 0; p < n; ++p) require(glm::length(output[p] - input[p]) < 1e-4f, "external depth discontinuity blocks cross-layer filtering");
    cudaCheck(cudaFree(devExternal));
    puts("PASS: externally supplied raster guides control edge stopping");
    optimizedDenoiseFree();
    cudaCheck(cudaFree(devInput)); cudaCheck(cudaFree(devOutput)); cudaCheck(cudaFree(devG));
    return 0;
}
