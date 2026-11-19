#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
using namespace Falcor;

// Format adapter only: raster G-buffer -> NRD's oct-normal/roughness format.
// World motion is zero because this benchmark explicitly freezes scene/camera.
class SharedInputGuides : public RenderPass {
public:
    FALCOR_PLUGIN_CLASS(SharedInputGuides, "SharedInputGuides", "Stationary shared raster guide format adapter.");
    static ref<SharedInputGuides> create(ref<Device> d, const Properties& p) { return make_ref<SharedInputGuides>(d, p); }
    SharedInputGuides(ref<Device> d, const Properties&) : RenderPass(d) {
        pass = ComputePass::create(d, "RenderPasses/SharedInputGuides/SharedInputGuides.cs.slang", "main");
    }
    RenderPassReflection reflect(const CompileData&) override {
        RenderPassReflection r;
        r.addInput("normal", "Raster world guide normal");
        r.addInput("roughness", "Raster specular reflectance and linear roughness");
        r.addInput("diffuse", "Raster diffuse reflection albedo");
        r.addInput("diffuseRadiance", "Raw diffuse path contribution");
        r.addInput("specularRadiance", "Raw specular path contribution");
        r.addOutput("packed", "NRD oct normal, roughness, material ID zero").format(ResourceFormat::RGB10A2Unorm).texture2D();
        r.addOutput("motion", "Stationary world-space motion").format(ResourceFormat::RGBA16Float).texture2D();
        for (const auto* name : {"diffuseFactor", "specularFactor", "diffuseSignal", "specularSignal"})
            r.addOutput(name, "Energy-preserving material normalization").format(ResourceFormat::RGBA32Float).texture2D();
        return r;
    }
    void execute(RenderContext* ctx, const RenderData& data) override {
        auto v = pass->getRootVar();
        v["normal"] = data.getTexture("normal"); v["roughness"] = data.getTexture("roughness");
        v["packed"] = data.getTexture("packed"); v["motion"] = data.getTexture("motion");
        for (const auto* name : {"diffuse", "diffuseRadiance", "specularRadiance", "diffuseFactor", "specularFactor", "diffuseSignal", "specularSignal"})
            v[name] = data.getTexture(name);
        pass->execute(ctx, uint3(data.getTexture("packed")->getWidth(), data.getTexture("packed")->getHeight(), 1));
    }
private:
    ref<ComputePass> pass;
};
extern "C" FALCOR_API_EXPORT void registerPlugin(PluginRegistry& registry) {
    registry.registerClass<RenderPass, SharedInputGuides>();
}
