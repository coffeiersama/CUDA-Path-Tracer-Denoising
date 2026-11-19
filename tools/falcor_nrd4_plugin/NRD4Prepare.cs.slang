#include "nrd4/NRD.hlsli"
import Utils.Math.MathHelpers;
Texture2D<float4> diffuse;
Texture2D<float4> specular;
Texture2D<float4> normal;
Texture2D<float4> roughness;
Texture2D<float4> albedo;
Texture2D<float4> view;
Texture2D<float> depth;
Texture2D<float2> motion;
float3 cameraForward;
RWTexture2D<float4> diffuseSignal, specularSignal;
RWTexture2D<float4> diffuseFactor, specularFactor;
RWTexture2D<float4> filteredDiffuse, filteredSpecular;
RWTexture2D<float4> motionGuide;
RWTexture2D<float4> normalGuide;
RWTexture2D<float> depthGuide;
RWTexture2D<float4> output;

[numthreads(16,16,1)]
void prepare(uint3 tid:SV_DispatchThreadID) {
    uint w,h; diffuseSignal.GetDimensions(w,h);
    if (tid.x>=w || tid.y>=h) return;
    uint2 p=tid.xy;
    float3 N=normal[p].xyz, V=view[p].xyz;
    float r=roughness[p].w;
    float z=depth[p];
#if DELTA_LAYER
    // Old Falcor secondary guides use oct.xy / roughness.z. Convert their
    // representation, keeping the archived optical-path guide model explicit.
    float4 nr=normal[p];
    N=oct_to_ndir_snorm(nr.xy*2-1); r=nr.z;
    z*=abs(dot(cameraForward,V));
    if (depth[p]>=60000 || depth[p]<=0) z=1e6;
#endif
    N=length(N)>0 ? normalize(N) : float3(0,1,0);
    V=length(V)>0 ? normalize(V) : float3(0,0,1);
    float materialID=0;
#if !DELTA_LAYER
    materialID=max(albedo[p].r,max(albedo[p].g,albedo[p].b))<1e-5 ? 1 : 0;
#endif
    normalGuide[p]=NRD_FrontEnd_PackNormalAndRoughness(N,r,materialID);
    depthGuide[p]=z;
    motionGuide[p]=float4(motion[p],0,0);
    float3 df,sf;
#if DELTA_LAYER
    // Transparent optical paths are an explicit extension of the standard
    // opaque material model: do not apply a second primary-surface material.
    df=1; sf=1;
#else
    NRD_MaterialFactors(N,V,albedo[p].rgb,roughness[p].rgb,r,df,sf);
#endif
    diffuseFactor[p]=float4(df,0); specularFactor[p]=float4(sf,0);
    float4 d=diffuse[p], s=specular[p];
#if DELTA_LAYER
    s=0;
#endif
    diffuseSignal[p]=RELAX_FrontEnd_PackRadianceAndHitDist(d.rgb/df,max(d.w,0),true);
    specularSignal[p]=RELAX_FrontEnd_PackRadianceAndHitDist(s.rgb/sf,max(s.w,0),true);
    filteredDiffuse[p]=0; filteredSpecular[p]=0;
}
[numthreads(16,16,1)]
void compose(uint3 tid:SV_DispatchThreadID) {
    uint w,h; output.GetDimensions(w,h);
    if(tid.x>=w || tid.y>=h) return;
    uint2 p=tid.xy;
    float3 d=RELAX_BackEnd_UnpackRadiance(filteredDiffuse[p]).rgb*diffuseFactor[p].rgb;
    float3 s=RELAX_BackEnd_UnpackRadiance(filteredSpecular[p]).rgb*specularFactor[p].rgb;
    // NRD does not write out-of-range pixels. Preserve those raw paths.
    if(depthGuide[p]>=500000) { d=diffuseSignal[p].rgb*diffuseFactor[p].rgb; s=specularSignal[p].rgb*specularFactor[p].rgb; }
    output[p]=float4(d+s,1);
}
