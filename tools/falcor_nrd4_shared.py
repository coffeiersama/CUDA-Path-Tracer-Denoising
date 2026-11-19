"""NRD 4.18 on the exact frozen RGB sequence of the previous comparison."""
from falcor import *
import hashlib
import json
import os
from pathlib import Path
import struct
import numpy as np

root=Path(os.environ['FALCOR_COURTYARD_ROOT'])
base=Path(os.environ.get('FALCOR_NRD4_BASE',str(root/'build/falcor-shared-final')))
out=Path(os.environ.get('FALCOR_NRD4_OUTPUT',str(root/'build/falcor-nrd4')))
out.mkdir(parents=True,exist_ok=True)
old=json.loads((base/'run.json').read_text())
w,h=old['resolution']; frames=int(os.environ.get('FALCOR_NRD4_FRAMES',str(old['frames'])))
assert 0<frames<=old['frames']
g=RenderGraph('Same Falcor samples: CUDA / official SVGF / NRD 4.18 RELAX')
g.addPass(createPass('GBufferRaster',{'cull':'Back','samplePattern':'Center'}),'GBuffer')
g.addPass(createPass('PathTracer',{'samplesPerPixel':1,'maxSurfaceBounces':6,'maxDiffuseBounces':3,
    'maxSpecularBounces':6,'maxTransmissionBounces':6,'useNEE':True,'useMIS':True,'useNRDDemodulation':False}),'PathTracer')
g.addEdge('GBuffer.vbuffer','PathTracer.vbuffer'); g.addEdge('GBuffer.viewW','PathTracer.viewW')
svgf=old['official_svgf']
g.addPass(createPass('SVGFPass',svgf),'SVGF')
g.addEdge('PathTracer.color','SVGF.Color'); g.addEdge('PathTracer.albedo','SVGF.Albedo')
for s,t in [('emissive','Emission'),('posW','WorldPosition'),('guideNormalW','WorldNormal'),
            ('pnFwidth','PositionNormalFwidth'),('linearZ','LinearZ'),('mvec','MotionVec')]: g.addEdge('GBuffer.'+s,'SVGF.'+t)
library=str(root/'build/Falcor8/build/windows-vs2022/bin/Release/NRD4.dll')
for name,delta,prefix in [('NRD4Primary',False,''),('NRD4Reflection',True,'DeltaReflection'),('NRD4Transmission',True,'DeltaTransmission')]:
    g.addPass(createPass('NRD4',{'library':library,'delta':delta}),name)
    if not delta:
        sources={'diffuse':'PathTracer.nrdDiffuseRadianceHitDist','specular':'PathTracer.nrdSpecularRadianceHitDist',
                 'normal':'GBuffer.guideNormalW','depth':'GBuffer.linearZ'}
    else:
        sources={'diffuse':f'PathTracer.nrd{prefix}RadianceHitDist','specular':f'PathTracer.nrd{prefix}RadianceHitDist',
                 'normal':f'PathTracer.nrd{prefix}NormWRoughMaterialID','depth':f'PathTracer.nrd{prefix}PathLength'}
    sources.update({'roughness':'GBuffer.specRough','albedo':'GBuffer.diffuseOpacity','view':'GBuffer.viewW','motion':'GBuffer.mvec'})
    for target,source in sources.items(): g.addEdge(source,name+'.'+target)
g.addPass(createPass('ModulateIllumination',{'useResidualRadiance':True}),'ComposeNRD4')
for s,t in [('NRD4Primary.output','diffuseRadiance'),('NRD4Reflection.output','deltaReflectionRadiance'),
            ('NRD4Transmission.output','deltaTransmissionRadiance'),('PathTracer.nrdEmission','emission'),
            ('PathTracer.nrdResidualRadianceHitDist','residualRadiance')]: g.addEdge(s,'ComposeNRD4.'+t)
outputs=['PathTracer.color','SVGF.Filtered image','ComposeNRD4.output','NRD4Primary.output',
         'NRD4Reflection.output','NRD4Transmission.output','GBuffer.posW','GBuffer.guideNormalW',
         'GBuffer.emissive','GBuffer.linearZ','GBuffer.pnFwidth','PathTracer.albedo',
         'PathTracer.nrdDiffuseRadianceHitDist','PathTracer.nrdSpecularRadianceHitDist',
         'PathTracer.nrdDeltaReflectionPathLength','PathTracer.nrdDeltaTransmissionPathLength',
         'PathTracer.nrdDeltaReflectionRadianceHitDist','PathTracer.nrdDeltaTransmissionRadianceHitDist']
for s in outputs: g.markOutput(s)
m.addGraph(g); m.setActiveGraph(g); m.resizeFrameBuffer(w,h)
m.loadScene(str(root/'scenes/falcor_san_miguel.pyscene'))
m.ui=False; m.clock.framerate=60; m.clock.pause()

def tex(name):
    t=g.getOutput(name); a=np.asarray(t.to_numpy())
    if a.ndim==1 and str(t.format).endswith('RGBA8Unorm'): a=a.reshape(h,w,4).astype(np.float32)/255
    if a.ndim==2: a=a[...,None]
    a=a.astype(np.float32)
    assert a.shape[:2]==(h,w) and np.isfinite(a).all(),(name,a.shape)
    return a

sha=hashlib.sha256(); checkpoints=[]
with (base/'stationary.seq').open('rb') as seq:
    header=struct.unpack('8s5I',seq.read(28)); assert header==(b'SVGFSEQ1',w,h,old['frames'],84,64)
    seq.seek(84+w*h*64,1)
    for f in range(1,frames+1):
        m.clock.frame=f; m.renderFrame()
        raw=tex('PathTracer.color')[...,:3]
        buf=raw[:,::-1].astype('<f4').tobytes()
        assert buf==seq.read(w*h*12),f'RGB changed from original input at frame {f}'
        sha.update(buf)
        if f==1:
            for s,filename in [('GBuffer.posW','GBufferRT-posW.npy'),('GBuffer.guideNormalW','GBufferRT-guideNormalW.npy'),
                               ('GBuffer.emissive','GBufferRT-emissive.npy'),('GBuffer.linearZ','GBufferRT-linearZ.npy'),
                               ('GBuffer.pnFwidth','GBufferRT-pnFwidth.npy'),('PathTracer.albedo','PathTracer-albedo.npy')]:
                assert np.array_equal(tex(s),np.load(base/filename)),f'Guide changed: {s}'
            assert not tex('GBuffer.emissive').any(),'This shared input experiment assumes no emissive surfaces'
        if f in {1,32,64,frames}:
            for s in outputs:
                if not s.startswith('GBuffer.') and s!='PathTracer.albedo': np.save(out/(s.replace('.','-').replace(' ','_')+f'-{f}.npy'),tex(s))
            if (base/f'SVGF-Filtered_image-{f}.npy').exists():
                old_svgf=np.load(base/f'SVGF-Filtered_image-{f}.npy')
                assert np.array_equal(tex('SVGF.Filtered image')[...,:3],old_svgf),'Official SVGF changed'
            checkpoints.append(f)
            print('NRD4_CAPTURE',f,'RAW_AND_OFFICIAL_SVGF_BITWISE_IDENTICAL',flush=True)
if frames==old['frames']: assert sha.hexdigest()==old['raw_sha256']
(out/'run.json').write_text(json.dumps({'nrd_version':'4.18.0','nrd_commit':'90dea8bd91ec6b58d0f90cb6adf53c7da613b82c',
    'nrd_method':'RELAX_DIFFUSE_SPECULAR primary + RELAX_DIFFUSE secondary reflection/transmission, three independent instances','base_directory':str(base),
    'resolution':[w,h],'frames':frames,'raw_sha256':sha.hexdigest(),'rgb_bitwise_equal':True,'official_svgf_bitwise_equal':True,
    'checkpoints':checkpoints,'normal_encoding':2,'roughness_encoding':1,'backend':'D3D12, official embedded DXIL',
    'nrd_binary_sha256':hashlib.sha256(Path(library).read_bytes()).hexdigest(),
    'guide_arrays_bitwise_equal_to_baseline':True,
    'primary_material_factors':'NRD_MaterialFactors, same factors before/after',
    'hit_distance_reconstruction':'AREA_5X5','prepass_radius_pixels':[30,50],'history_frames':30,
    'fast_history_frames':6,'history_fix_frames':3,'min_material_for_diffuse':0,'anti_firefly':False,
    'path_sampling_unchanged':True,'limitations':['Original Falcor primary lobe sampling is retained; Bayer coverage / probability bounds are not guaranteed.',
    'Secondary transparent layers reuse Falcor optical-path guides; complete NRD-Sample PSR/curvature/virtual-normal reconstruction is not available.',
    'Secondary paths outside their guide coverage are kept unfiltered. Static camera only; no TAA/DLSS.']},indent=2))
print('NRD4_COMPLETE',str(out),flush=True)
exit()
