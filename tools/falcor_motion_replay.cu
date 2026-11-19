#include "denoise_optimized.h"
#include "capture.h"
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <cstring>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

struct MotionHeader {
    char magic[8];
    uint32_t width,height,frames,warmup,stride,cameraBytes,geometryBytes,guideBytes;
};
static_assert(sizeof(MotionHeader)==40,"Motion sequence header layout");
static void require(bool ok,const char* text) { if(!ok) throw std::runtime_error(text); }
static void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void crypto(NTSTATUS r) { require(r>=0,"SHA256 API failed"); }

int main(int argc,char** argv) try {
    require(argc==3,"Usage: falcor_motion_replay MOTION.seq OUTPUT_DIRECTORY");
    std::ifstream f(argv[1],std::ios::binary); MotionHeader h{};
    f.read(reinterpret_cast<char*>(&h),sizeof(h));
    require(f && !memcmp(h.magic,"SVGFMOT2",8) && h.cameraBytes==sizeof(Camera) &&
        h.geometryBytes==sizeof(GBufferTexel) && h.guideBytes==sizeof(OptimizedGeometryGuide) &&
        h.width>1 && h.width<=8192 && h.height>1 && h.height<=8192 && h.frames>h.warmup &&
        h.frames<=4096 && h.stride>0,"Invalid motion sequence/layout");
    std::filesystem::path out(argv[2]); std::filesystem::create_directories(out);
    size_t n=size_t(h.width)*h.height;
    std::vector<GBufferTexel> geo(n); std::vector<OptimizedGeometryGuide> guides(n);
    std::vector<glm::vec3> raw(n),image(n);
    glm::vec3 *di,*doo; GBufferTexel* dg; OptimizedGeometryGuide* dguide;
    check(cudaMalloc(&di,n*sizeof(raw[0]))); check(cudaMalloc(&doo,n*sizeof(raw[0])));
    check(cudaMalloc(&dg,n*sizeof(geo[0]))); check(cudaMalloc(&dguide,n*sizeof(guides[0])));
    BCRYPT_ALG_HANDLE alg=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
    crypto(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0));
    crypto(BCryptCreateHash(alg,&hash,nullptr,0,nullptr,0,0));
    optimizedSettings=OptimizedDenoiseSettings{}; optimizedDenoiseInit(int(n));
    // One reset at startup; camera motion must preserve/reproject history.
    optimizedDenoiseReset();
    for(uint32_t frame=0;frame<h.frames;++frame) {
        Camera cam{};
        f.read(reinterpret_cast<char*>(&cam),sizeof(cam));
        f.read(reinterpret_cast<char*>(geo.data()),n*sizeof(geo[0]));
        f.read(reinterpret_cast<char*>(guides.data()),n*sizeof(guides[0]));
        f.read(reinterpret_cast<char*>(raw.data()),n*sizeof(raw[0]));
        require(f && cam.resolution==glm::ivec2(h.width,h.height),"Truncated motion frame or mismatched camera");
        crypto(BCryptHashData(hash,reinterpret_cast<PUCHAR>(raw.data()),ULONG(n*sizeof(raw[0])),0));
        check(cudaMemcpy(di,raw.data(),n*sizeof(raw[0]),cudaMemcpyHostToDevice));
        check(cudaMemcpy(dg,geo.data(),n*sizeof(geo[0]),cudaMemcpyHostToDevice));
        check(cudaMemcpy(dguide,guides.data(),n*sizeof(guides[0]),cudaMemcpyHostToDevice));
        bool capture=frame>=h.warmup && (frame-h.warmup)%h.stride==0;
        if(frame==h.warmup || frame+1==h.frames)
            optimizedDenoiseCapture((out/("diagnostic-"+std::to_string(frame))).string());
        optimizedDenoise(doo,di,dg,cam,dguide);
        if(capture) {
            check(cudaMemcpy(image.data(),doo,n*sizeof(image[0]),cudaMemcpyDeviceToHost));
            for(const auto& p:image) require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z),"Nonfinite motion output");
            std::ostringstream name; name<<"cuda-"<<std::setw(4)<<std::setfill('0')<<(frame-h.warmup)/h.stride<<".pfm";
            writePFM((out/name.str()).string(),image,cam.resolution);
        }
        if((frame+1)%48==0) std::cout<<"MOTION_REPLAY "<<(frame+1)<<" / "<<h.frames<<std::endl;
    }
    require(f.peek()==std::char_traits<char>::eof(),"Trailing motion sequence data");
    unsigned char digest[32]; crypto(BCryptFinishHash(hash,digest,sizeof(digest),0));
    std::ostringstream hex; for(unsigned char b:digest) hex<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(b);
    std::ofstream meta(out/"replay.json");
    meta<<"{\"raw_sha256\":\""<<hex.str()<<"\",\"frames\":"<<h.frames
        <<",\"warmup\":"<<h.warmup<<",\"stride\":"<<h.stride
        <<",\"history_resets\":1,\"camera_and_geometry_per_frame\":true,\"settings\":{\"alpha\":"<<optimizedSettings.alpha
        <<",\"moments_alpha\":"<<optimizedSettings.momentsAlpha<<",\"phi_color\":"<<optimizedSettings.sigmaLuminance
        <<",\"phi_normal\":"<<optimizedSettings.sigmaNormal<<",\"iterations\":"<<optimizedSettings.levels
        <<",\"feedback_tap\":"<<optimizedSettings.feedbackTap<<"}}\n";
    optimizedDenoiseFree(); check(cudaFree(di));check(cudaFree(doo));check(cudaFree(dg));check(cudaFree(dguide));
    BCryptDestroyHash(hash); BCryptCloseAlgorithmProvider(alg,0);
    std::cout<<"MOTION_REPLAY_COMPLETE "<<hex.str()<<std::endl;
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
