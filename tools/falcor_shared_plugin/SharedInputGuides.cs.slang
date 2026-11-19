import Utils.Math.MathHelpers;
Texture2D<float4> normal;
Texture2D<float4> roughness;
Texture2D<float4> diffuse;
Texture2D<float4> diffuseRadiance;
Texture2D<float4> specularRadiance;
RWTexture2D<float4> packed;
RWTexture2D<float4> motion;
RWTexture2D<float4> diffuseFactor;
RWTexture2D<float4> specularFactor;
RWTexture2D<float4> diffuseSignal;
RWTexture2D<float4> specularSignal;
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint w, h; packed.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;
    float3 n = normal[tid.xy].xyz;
    float2 oct = length(n) > 0 ? ndir_to_oct_unorm(normalize(n)) : float2(.5, .5);
    packed[tid.xy] = float4(oct, roughness[tid.xy].w, 0);
    motion[tid.xy] = 0;
    // Use the SAME deterministic factor before and after denoising. Falcor 8's
    // per-sample delta demodulation can disagree with its guide-path factors on
    // this imported transmissive scene. Raw path partitioning is left intact.
    float3 df = max(diffuse[tid.xy].rgb, .01);
    float3 sf = max(roughness[tid.xy].rgb, .01);
    diffuseFactor[tid.xy] = float4(df, 0);
    specularFactor[tid.xy] = float4(sf, 0);
    float4 dr = diffuseRadiance[tid.xy], sr = specularRadiance[tid.xy];
    diffuseSignal[tid.xy] = float4(dr.rgb / df, dr.a);
    specularSignal[tid.xy] = float4(sr.rgb / sf, sr.a);
}
