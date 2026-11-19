"""Inside Mogwai: moving-camera shared RGB/G-buffer sequence, official SVGF."""
from falcor import *
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import numpy as np

root=Path(os.environ['FALCOR_COURTYARD_ROOT'])
out=Path(os.environ.get('FALCOR_MOTION_OUTPUT',str(root/'build/falcor-motion')))
out.mkdir(parents=True,exist_ok=True)
w=int(os.environ.get('FALCOR_MOTION_WIDTH','640')); h=int(os.environ.get('FALCOR_MOTION_HEIGHT','360'))
warmup=int(os.environ.get('FALCOR_MOTION_WARMUP','48'))
moving=int(os.environ.get('FALCOR_MOTION_FRAMES','240'))
stride=int(os.environ.get('FALCOR_MOTION_STRIDE','3'))
g=RenderGraph('Camera-motion shared input: official SVGF / CUDA replay')
g.addPass(createPass('GBufferRaster',{'cull':'Back','samplePattern':'Center'}),'GBuffer')
g.addPass(createPass('PathTracer',{'samplesPerPixel':1,'maxSurfaceBounces':6,'maxDiffuseBounces':3,
    'maxSpecularBounces':6,'maxTransmissionBounces':6,'useNEE':True,'useMIS':True,'useNRDDemodulation':False}),'PathTracer')
g.addEdge('GBuffer.vbuffer','PathTracer.vbuffer'); g.addEdge('GBuffer.viewW','PathTracer.viewW')
settings={'Enabled':True,'Iterations':4,'FeedbackTap':1,'VarianceEpsilon':1e-4,
          'PhiColor':10.,'PhiNormal':128.,'Alpha':.05,'MomentsAlpha':.2}
g.addPass(createPass('SVGFPass',settings),'SVGF')
g.addEdge('PathTracer.color','SVGF.Color'); g.addEdge('PathTracer.albedo','SVGF.Albedo')
for s,t in [('emissive','Emission'),('posW','WorldPosition'),('guideNormalW','WorldNormal'),
            ('pnFwidth','PositionNormalFwidth'),('linearZ','LinearZ'),('mvec','MotionVec')]:
    g.addEdge('GBuffer.'+s,'SVGF.'+t)
outputs=['PathTracer.color','PathTracer.albedo','SVGF.Filtered image']+[
    'GBuffer.'+n for n in ['emissive','posW','guideNormalW','pnFwidth','linearZ','mask','mvec']]
for name in outputs: g.markOutput(name)
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

def vec(v): return np.array([v.x,v.y,v.z],np.float32)

dtype=np.dtype([('emission','<f4',3),('normal','<f4',3),('position','<f4',3),
                ('albedo','<f4',3),('ialbedo','<f4',3),('geomId','<i4')])
assert dtype.itemsize==64
eye0=np.array([10.5,1.5,-2.9]); target=np.array([15.,1.05,2.])
offset=eye0-target
raw_sha=hashlib.sha256(); geometry_sha=hashlib.sha256()
trajectory=[]; captures=[]; max_projection=0; max_motion=0
with (out/'motion.seq').open('wb') as seq:
    seq.write(struct.pack('<8s8I',b'SVGFMOT2',w,h,warmup+moving,warmup,stride,84,64,12))
    for frame in range(warmup+moving):
        k=max(frame-warmup,0); phase=2*math.pi*k/moving
        angle=math.radians(6)*math.sin(phase)
        ca,sa=math.cos(angle),math.sin(angle)
        eye=target+np.array([ca*offset[0]+sa*offset[2],offset[1]+.08*(1-math.cos(phase)),-sa*offset[0]+ca*offset[2]])
        cam=m.scene.camera
        cam.position=float3(*map(float,eye)); cam.target=float3(*map(float,target)); cam.up=float3(0,1,0)
        m.clock.frame=frame+1; m.renderFrame()
        eye=vec(cam.position); at=vec(cam.target); up=vec(cam.up)
        view=at-eye; view/=np.linalg.norm(view)
        right=np.cross(view,up); right/=np.linalg.norm(right); up=np.cross(right,view)
        fy=float(cam.frameHeight)/float(cam.focalLength); fx=fy*w/h
        camera=np.concatenate([eye,at,view,up,right,[fx,fy],[fx/w,fy/h]]).astype('<f4')
        seq.write(struct.pack('<2i',w,h)); seq.write(camera.tobytes())
        geo=np.zeros((h,w),dtype=dtype)
        geo['emission']=tex('GBuffer.emissive')[...,:3]
        geo['normal']=tex('GBuffer.guideNormalW')[...,:3]
        geo['position']=tex('GBuffer.posW')[...,:3]
        geo['albedo']=tex('PathTracer.albedo')[...,:3]
        geo['ialbedo']=1/np.maximum(geo['albedo'],.001)
        valid=tex('GBuffer.mask')[...,0]>0; geo['geomId']=np.where(valid,0,-1)
        z=tex('GBuffer.linearZ'); pn=tex('GBuffer.pnFwidth')
        guide=np.stack([np.where(valid,z[...,0],-1),z[...,1],pn[...,1]],axis=-1)
        raw=tex('PathTracer.color')[...,:3]
        gb=geo[:,::-1].tobytes(); rb=raw[:,::-1].astype('<f4').tobytes()
        seq.write(gb); seq.write(guide[:,::-1].astype('<f4').tobytes()); seq.write(rb)
        geometry_sha.update(gb); raw_sha.update(rb)
        delta=geo['position']-eye; zz=delta@view
        px=-(delta@right)/(np.maximum(zz,1e-8)*fx/w)+w/2-.5
        py=-(delta@up)/(np.maximum(zz,1e-8)*fy/h)+h/2-.5
        yy,xx=np.indices((h,w))
        error=float(np.max(np.maximum(abs(px-(w-1-xx)),abs(py-yy))[valid]))
        # Rasterized world positions are float32; near-clip surfaces amplify
        # their world-coordinate rounding. Keep error below one eighth pixel.
        assert error<.125,('Camera/texture orientation mismatch',frame,error)
        max_projection=max(max_projection,error)
        if frame>0: max_motion=max(max_motion,float(abs(tex('GBuffer.mvec')).max()))
        trajectory.append({'frame':frame,'eye':eye.tolist(),'target':at.tolist(),'phase':phase})
        if frame>=warmup and (frame-warmup)%stride==0:
            idx=(frame-warmup)//stride
            np.save(out/f'raw-{idx:04}.npy',raw)
            np.save(out/f'official-{idx:04}.npy',tex('SVGF.Filtered image')[...,:3])
            captures.append(frame)
        if (frame+1)%48==0: print('MOTION_CAPTURE',frame+1,'/',warmup+moving,flush=True)
assert max_motion>1e-4,'Camera did not move'
(out/'trajectory.json').write_text(json.dumps(trajectory))
(out/'run.json').write_text(json.dumps({'framework':'Falcor 8.0','falcor_commit':'9dc819c162b2070335c65060436041690b7937f8',
    'resolution':[w,h],'frames':warmup+moving,'warmup':warmup,'moving_frames':moving,'stride':stride,
    'simulation_fps':60,'playback_fps':60/stride,'duration_seconds':moving/60,'capture_frames':captures,
    'path_samples_per_pixel_per_frame':1,'svgf':settings,'display':'Shared Reinhard+sRGB, EV +2',
    'raw_sha256':raw_sha.hexdigest(),'geometry_sha256':geometry_sha.hexdigest(),
    'camera_projection_max_error_pixels':max_projection,'max_normalized_motion_vector':max_motion,
    'motion':'6-degree sinusoidal orbit plus 0.16m vertical travel; static scene and lights',
    'jitter':False,'taa':False,'dlss':False,'history_reset_policy':'once at startup; never reset for camera motion',
    'notes':'Actual per-frame path tracing and raster guides, not animated still images. Offline capture; playback fps is not a render-performance measurement.'},indent=2))
print('MOTION_CAPTURE_COMPLETE',str(out),flush=True)
exit()
