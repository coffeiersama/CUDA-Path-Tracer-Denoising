#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Core/API/NativeHandleTraits.h"
#include "Core/API/Shared/D3D12DescriptorSet.h"
#include "Core/API/Shared/D3D12RootSignature.h"
#include "Core/API/Shared/D3D12ConstantBufferView.h"
#include <NRD.h>
#include <windows.h>
#include <wrl/client.h>
#include <unordered_map>
using namespace Falcor;
using Microsoft::WRL::ComPtr;

// Native DXIL dispatch adapter for NRD 4.18. The official kernels are loaded
// directly from the matching freshly built SDK, without shader translation.
class NRD4Pass : public RenderPass {
public:
    FALCOR_PLUGIN_CLASS(NRD4Pass, "NRD4", "NRD 4.18 RELAX, direct official DXIL kernels.");
    static ref<NRD4Pass> create(ref<Device> d, const Properties& p) { return make_ref<NRD4Pass>(d,p); }
    NRD4Pass(ref<Device> d, const Properties& p) : RenderPass(d) {
        mpDevice->requireD3D12();
        delta = p.get<bool>("delta",false);
        libraryPath = p.get<std::string>("library","");
        module = LoadLibraryW(std::filesystem::path(libraryPath).c_str());
        FALCOR_CHECK(module, "Could not load NRD4 library: {}", libraryPath);
#define LOAD(name) name = reinterpret_cast<decltype(name)>(GetProcAddress(module, #name)); FALCOR_CHECK(name, "Missing NRD4 export " #name)
        LOAD(CreateInstance); LOAD(DestroyInstance); LOAD(GetLibraryDesc); LOAD(GetInstanceDesc);
        LOAD(SetCommonSettings); LOAD(SetDenoiserSettings); LOAD(GetComputeDispatches);
#undef LOAD
        const auto* lib = GetLibraryDesc();
        FALCOR_CHECK(lib->versionMajor == 4 && lib->versionMinor == 18 && lib->versionBuild == 0, "Wrong NRD runtime version");
        FALCOR_CHECK(lib->normalEncoding == nrd::NormalEncoding::R10_G10_B10_A2_UNORM && lib->roughnessEncoding == nrd::RoughnessEncoding::LINEAR, "NRD4 shader encoding mismatch");
        DefineList defines; defines.add("DELTA_LAYER", delta ? "1" : "0");
        prepare = ComputePass::create(d, "RenderPasses/NRD4Pass/NRD4Prepare.cs.slang", "prepare", defines);
        compose = ComputePass::create(d, "RenderPasses/NRD4Pass/NRD4Prepare.cs.slang", "compose", defines);
        logInfo("NRD4_RUNTIME {}.{}.{} / official embedded DXIL / delta={}", lib->versionMajor,lib->versionMinor,lib->versionBuild,delta);
    }
    ~NRD4Pass() {
        // Make native PSO/SDK teardown safe for any still queued GPU work.
        if (instance) { mpDevice->getRenderContext()->submit(true); DestroyInstance(*instance); }
        pipelines.clear(); if (module) FreeLibrary(module);
    }
    RenderPassReflection reflect(const CompileData&) override {
        RenderPassReflection r;
        for (const char* s : {"diffuse", "specular", "normal", "roughness", "albedo", "view", "depth", "motion"}) r.addInput(s, "Shared Falcor input");
        r.addOutput("output", "Remodulated RELAX radiance").format(ResourceFormat::RGBA32Float).texture2D();
        return r;
    }
    void setScene(RenderContext*, const ref<Scene>& s) override { scene = s; frame = 0; }
    void execute(RenderContext* ctx, const RenderData& data) override {
        FALCOR_CHECK(scene, "NRD4 requires a camera");
        auto output = data.getTexture("output");
        if (!instance || width != output->getWidth() || height != output->getHeight()) init(output->getWidth(), output->getHeight());
        auto v = prepare->getRootVar();
        v["cameraForward"] = normalize(scene->getCamera()->getTarget()-scene->getCamera()->getPosition());
        for (const char* s : {"diffuse", "specular", "normal", "roughness", "albedo", "view", "depth", "motion"}) v[s] = data.getTexture(s);
        for (auto& [name,tex] : textures) v[name] = tex;
        prepare->execute(ctx, uint3(width,height,1));
        nrd::CommonSettings common{};
        copyMatrix(common.worldToViewMatrix, scene->getCamera()->getViewMatrix());
        copyMatrix(common.worldToViewMatrixPrev, scene->getCamera()->getViewMatrix());
        copyMatrix(common.viewToClipMatrix, scene->getCamera()->getData().projMatNoJitter);
        copyMatrix(common.viewToClipMatrixPrev, scene->getCamera()->getData().projMatNoJitter);
        common.resourceSize[0] = common.resourceSizePrev[0] = common.rectSize[0] = common.rectSizePrev[0] = uint16_t(width);
        common.resourceSize[1] = common.resourceSizePrev[1] = common.rectSize[1] = common.rectSizePrev[1] = uint16_t(height);
        common.timeDeltaBetweenFrames = 1000.f / 60.f;
        common.isMotionVectorInWorldSpace = false; common.motionVectorScale[2] = 0;
        common.frameIndex = frame;
        common.accumulationMode = frame ? nrd::AccumulationMode::CONTINUE : nrd::AccumulationMode::CLEAR_AND_RESTART;
        check(SetCommonSettings(*instance, common));
        const nrd::DispatchDesc* dispatches = nullptr; uint32_t count = 0;
        const nrd::Identifier id = 0;
        check(GetComputeDispatches(*instance,&id,1,dispatches,count));
        for (uint32_t i=0; i<count; ++i) dispatch(ctx,dispatches[i]);
        auto c = compose->getRootVar();
        for (auto& [name,tex] : textures) c[name] = tex;
        c["output"] = output;
        compose->execute(ctx,uint3(width,height,1));
        ++frame;
    }
private:
    template<class T> static void check(T r) { FALCOR_CHECK(r == nrd::Result::SUCCESS, "NRD4 API failed: {}", int(r)); }
    static void copyMatrix(float* dst, const float4x4& m) { auto col=transpose(m); memcpy(dst,col.data(),64); }
    static ResourceFormat format(nrd::Format f) {
        switch(f) {
#define F(a,b) case nrd::Format::a: return ResourceFormat::b
        F(R8_UNORM,R8Unorm); F(R8_UINT,R8Uint); F(R8_SNORM,R8Snorm); F(RG8_UNORM,RG8Unorm); F(RG8_UINT,RG8Uint);
        F(RGBA8_UNORM,RGBA8Unorm); F(RGBA8_SNORM,RGBA8Snorm); F(RGBA8_UINT,RGBA8Uint);
        F(R16_SFLOAT,R16Float); F(R16_UNORM,R16Unorm); F(R16_UINT,R16Uint);
        F(RG16_SFLOAT,RG16Float); F(RG16_UNORM,RG16Unorm); F(RG16_UINT,RG16Uint); F(RG16_SNORM,RG16Snorm);
        F(RGBA16_SFLOAT,RGBA16Float); F(RGBA16_UNORM,RGBA16Unorm); F(RGBA16_UINT,RGBA16Uint); F(RGBA16_SNORM,RGBA16Snorm);
        F(R32_SFLOAT,R32Float); F(R32_UINT,R32Uint); F(RG32_SFLOAT,RG32Float); F(RG32_UINT,RG32Uint);
        F(RGBA32_SFLOAT,RGBA32Float); F(RGBA32_UINT,RGBA32Uint); F(R10_G10_B10_A2_UNORM,RGB10A2Unorm); F(R11_G11_B10_UFLOAT,R11G11B10Float);
#undef F
        default: FALCOR_THROW("Unhandled NRD4 pool format {}", int(f));
        }
    }
    ref<Texture> makeTexture(ResourceFormat f,uint32_t w,uint32_t h) {
        return mpDevice->createTexture2D(w,h,f,1,1,nullptr,ResourceBindFlags::ShaderResource|ResourceBindFlags::UnorderedAccess);
    }
    void init(uint32_t w,uint32_t h) {
        if (instance) { mpDevice->getRenderContext()->submit(true); DestroyInstance(*instance); instance=nullptr; }
        pipelines.clear(); roots.clear(); layouts.clear(); permanent.clear(); transient.clear(); textures.clear();
        width=w; height=h; frame=0;
        const nrd::DenoiserDesc denoiser{0, delta ? nrd::Denoiser::RELAX_DIFFUSE : nrd::Denoiser::RELAX_DIFFUSE_SPECULAR};
        nrd::InstanceCreationDesc creation{}; creation.denoisers=&denoiser; creation.denoisersNum=1;
        check(CreateInstance(creation,instance));
        nrd::RelaxSettings settings{};
        // Falcor selects primary lobes probabilistically. Preserve its exact
        // RGB stream; reconstruct missing distances without changing sampling.
        settings.hitDistanceReconstructionMode=nrd::HitDistanceReconstructionMode::AREA_5X5;
        settings.minMaterialForDiffuse=0;
        check(SetDenoiserSettings(*instance,0,&settings));
        const auto& desc=*GetInstanceDesc(*instance);
        for (auto name : {"diffuseSignal","specularSignal","diffuseFactor","specularFactor","filteredDiffuse","filteredSpecular","motionGuide"})
            textures[name]=makeTexture(ResourceFormat::RGBA32Float,w,h);
        textures["normalGuide"]=makeTexture(ResourceFormat::RGB10A2Unorm,w,h);
        textures["depthGuide"]=makeTexture(ResourceFormat::R32Float,w,h);
        for (uint32_t i=0;i<desc.permanentPoolSize;++i) { const auto& t=desc.permanentPool[i]; permanent.push_back(makeTexture(format(t.format),(w+t.downsampleFactor-1)/t.downsampleFactor,(h+t.downsampleFactor-1)/t.downsampleFactor)); }
        for (uint32_t i=0;i<desc.transientPoolSize;++i) { const auto& t=desc.transientPool[i]; transient.push_back(makeTexture(format(t.format),(w+t.downsampleFactor-1)/t.downsampleFactor,(h+t.downsampleFactor-1)/t.downsampleFactor)); }
        D3D12DescriptorSetLayout sl;
        sl.addRange(ShaderResourceType::Sampler,desc.samplersBaseRegisterIndex,desc.samplersNum,desc.constantBufferAndSamplersSpaceIndex);
        samplers=D3D12DescriptorSet::create(mpDevice,sl);
        for (uint32_t i=0;i<desc.samplersNum;++i) {
            Sampler::Desc sd; auto filter=desc.samplers[i]==nrd::Sampler::NEAREST_CLAMP?TextureFilteringMode::Point:TextureFilteringMode::Linear;
            sd.setFilterMode(filter,filter,TextureFilteringMode::Point); sd.setAddressingMode(TextureAddressingMode::Clamp,TextureAddressingMode::Clamp,TextureAddressingMode::Clamp);
            auto sampler=mpDevice->createSampler(sd); samplers->setSampler(0,i,sampler.get()); heldSamplers.push_back(sampler);
        }
        auto* device=mpDevice->getNativeHandle().as<ID3D12Device*>();
        for (uint32_t i=0;i<desc.pipelinesNum;++i) {
            const auto& p=desc.pipelines[i];
            FALCOR_CHECK(p.computeShaderDXIL.bytecode && p.computeShaderDXIL.size,"NRD4 missing embedded DXIL");
            D3D12DescriptorSetLayout layout;
            layout.addRange(ShaderResourceType::Cbv,desc.constantBufferRegisterIndex,1,desc.constantBufferAndSamplersSpaceIndex);
            for (uint32_t j=0;j<p.resourceRangesNum;++j) layout.addRange(p.resourceRanges[j].descriptorType==nrd::DescriptorType::TEXTURE?ShaderResourceType::TextureSrv:ShaderResourceType::TextureUav,desc.resourcesBaseRegisterIndex,p.resourceRanges[j].descriptorsNum,desc.resourcesSpaceIndex);
            D3D12RootSignature::Desc rd; rd.addDescriptorSet(sl); rd.addDescriptorSet(layout);
            auto root=D3D12RootSignature::create(mpDevice,rd);
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature=root->getApiHandle().GetInterfacePtr();
            pd.CS={p.computeShaderDXIL.bytecode,size_t(p.computeShaderDXIL.size)};
            ComPtr<ID3D12PipelineState> pipeline;
            FALCOR_CHECK(SUCCEEDED(device->CreateComputePipelineState(&pd,IID_PPV_ARGS(pipeline.GetAddressOf()))),"NRD4 DXIL PSO creation failed: {}",p.shaderIdentifier);
            pipelines.push_back(pipeline); roots.push_back(root); layouts.push_back(layout);
        }
        logInfo("NRD4_INIT {}x{}: {} pipelines, {} permanent, {} transient textures",w,h,desc.pipelinesNum,desc.permanentPoolSize,desc.transientPoolSize);
    }
    ref<Texture> resource(const nrd::ResourceDesc& r) {
        switch(r.type) {
        case nrd::ResourceType::IN_MV:return textures.at("motionGuide");
        case nrd::ResourceType::IN_NORMAL_ROUGHNESS:return textures.at("normalGuide");
        case nrd::ResourceType::IN_VIEWZ:return textures.at("depthGuide");
        case nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST:return textures.at("diffuseSignal");
        case nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST:return textures.at("specularSignal");
        case nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST:return textures.at("filteredDiffuse");
        case nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST:return textures.at("filteredSpecular");
        case nrd::ResourceType::PERMANENT_POOL:return permanent.at(r.indexInPool);
        case nrd::ResourceType::TRANSIENT_POOL:return transient.at(r.indexInPool);
        default:FALCOR_THROW("Unexpected NRD4 resource {}",int(r.type));
        }
    }
    void dispatch(RenderContext* ctx,const nrd::DispatchDesc& d) {
        const auto& desc=*GetInstanceDesc(*instance); const auto& p=desc.pipelines[d.pipelineIndex];
        auto cb=mpDevice->getUploadHeap()->allocate(std::max(d.constantBufferDataSize,16u),ResourceBindFlags::Constant);
        if (d.constantBufferDataSize) memcpy(cb.pData,d.constantBufferData,d.constantBufferDataSize);
        auto view=D3D12ConstantBufferView::create(mpDevice,cb.getGpuAddress(),cb.size);
        auto set=D3D12DescriptorSet::create(mpDevice,layouts[d.pipelineIndex]);
        set->setCbv(0,0,view.get());
        uint32_t index=0;
        for (uint32_t j=0;j<p.resourceRangesNum;++j) for (uint32_t k=0;k<p.resourceRanges[j].descriptorsNum;++k) {
            const auto& r=d.resources[index++]; auto t=resource(r);
            bool read=r.descriptorType==nrd::DescriptorType::TEXTURE;
            if (!ctx->resourceBarrier(t.get(),read?Resource::State::ShaderResource:Resource::State::UnorderedAccess) && !read) ctx->uavBarrier(t.get());
            if (read) set->setSrv(j+1,k,t->getSRV().get()); else set->setUav(j+1,k,t->getUAV().get());
        }
        FALCOR_CHECK(index==d.resourcesNum,"NRD4 descriptor/resource mismatch");
        roots[d.pipelineIndex]->bindForCompute(ctx);
        samplers->bindForCompute(ctx,roots[d.pipelineIndex].get(),0);
        set->bindForCompute(ctx,roots[d.pipelineIndex].get(),1);
        auto* cmd=ctx->getLowLevelData()->getCommandBufferNativeHandle().as<ID3D12GraphicsCommandList*>();
        cmd->SetPipelineState(pipelines[d.pipelineIndex].Get()); cmd->Dispatch(d.gridWidth,d.gridHeight,1);
        mpDevice->getUploadHeap()->release(cb);
    }
    HMODULE module=nullptr; std::string libraryPath;
#define API(name) decltype(&nrd::name) name=nullptr
    API(CreateInstance); API(DestroyInstance); API(GetLibraryDesc); API(GetInstanceDesc); API(SetCommonSettings); API(SetDenoiserSettings); API(GetComputeDispatches);
#undef API
    bool delta=false; uint32_t frame=0,width=0,height=0; nrd::Instance* instance=nullptr;
    ref<Scene> scene; ref<ComputePass> prepare,compose;
    std::unordered_map<std::string,ref<Texture>> textures;
    std::vector<ref<Texture>> permanent,transient;
    ref<D3D12DescriptorSet> samplers; std::vector<ref<Sampler>> heldSamplers;
    std::vector<ref<D3D12RootSignature>> roots; std::vector<D3D12DescriptorSetLayout> layouts;
    std::vector<ComPtr<ID3D12PipelineState>> pipelines;
};
extern "C" FALCOR_API_EXPORT void registerPlugin(PluginRegistry& r) { r.registerClass<RenderPass,NRD4Pass>(); }
