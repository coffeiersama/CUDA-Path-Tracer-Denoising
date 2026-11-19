"""Encode actual camera-motion frames with identical display conversion."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from skimage.metrics import structural_similarity
import imageio_ffmpeg
from compare_falcor_shared import pfm,display

def font(size,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/'+('segoeuib.ttf' if bold else 'segoeui.ttf'),size)

def history(path):
    with path.open('rb') as f:
        assert f.readline().strip()==b'Pf'
        w,h=map(int,f.readline().split()); assert float(f.readline())==-1
        x=np.fromfile(f,'<f4').reshape(h,w)[::-1]
    assert np.isfinite(x).all()
    return {'mean':float(x.mean()),'max':float(x.max()),'fraction_at_least_4':float((x>=4).mean())}

def panel_pair(left,right,left_title,right_title,t):
    canvas=Image.new('RGB',(960,336),(17,24,34)); draw=ImageDraw.Draw(canvas)
    draw.text((12,7),left_title,font=font(20,True),fill=(241,245,249))
    draw.text((492,7),right_title,font=font(20,True),fill=(241,245,249))
    for x,a in [(0,left),(480,right)]:
        im=Image.fromarray(np.rint(a*255).astype(np.uint8)).resize((480,270),Image.Resampling.LANCZOS)
        canvas.paste(im,(x,42))
    draw.line((480,0,480,312),fill=(96,110,127),width=1)
    draw.text((12,315),f'Same 1 spp/frame | shared exposure | camera orbit | t = {t:.2f} s',font=font(13),fill=(197,207,219))
    return canvas

def encode(ffmpeg,frames,out,fps):
    # No interpolation or synthetic motion; one decoded frame per capture.
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-y','-framerate',str(fps),'-i',str(frames),
         '-filter_complex','[0:v]split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none',
         '-loop','0',str(out.with_suffix('.gif'))]
    subprocess.run(cmd,check=True)
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-framerate',str(fps),'-i',str(frames),
        '-c:v','libx264','-preset','medium','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(out.with_suffix('.mp4'))],check=True)

def validate_gif(path,count,fps):
    im=Image.open(path); assert im.is_animated and im.n_frames==count,(path,im.n_frames,count)
    frames=[]; duration=0
    for i in range(im.n_frames):
        im.seek(i); duration+=im.info.get('duration',0)
        if i in [0,count//4,count//2,3*count//4,count-1]: frames.append(im.convert('RGB').copy())
    assert duration==round(count/fps*1000),(path,duration)
    assert len({hashlib.sha256(f.tobytes()).hexdigest() for f in frames})==len(frames),'Frozen animation'
    return {'frames':count,'duration_ms':duration,'size_bytes':path.stat().st_size}

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output-directory',type=Path,default=Path('img'))
    args=p.parse_args();d=args.directory;out=args.output_directory;out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((d/'run.json').read_text()); replay=json.loads((d/'replay.json').read_text())
    assert meta['raw_sha256']==replay['raw_sha256'],'Falcor/CUDA raw samples differ'
    assert replay['history_resets']==1 and replay['camera_and_geometry_per_frame']
    assert replay['settings']=={'alpha':.05,'moments_alpha':.2,'phi_color':10,'phi_normal':128,'iterations':4,'feedback_tap':1}
    poses=json.loads((d/'trajectory.json').read_text()); eyes=np.array([x['eye'] for x in poses])
    assert float(np.linalg.norm(eyes-eyes[0],axis=1).max())>.1,'Camera is stationary'
    count=len(meta['capture_frames']);fps=meta['playback_fps'];frame_dir=d/'animation-frames';frame_dir.mkdir(exist_ok=True)
    ssims=[]; psnrs=[]
    for i in range(count):
        raw=np.load(d/f'raw-{i:04}.npy'); official=np.load(d/f'official-{i:04}.npy'); cuda=pfm(d/f'cuda-{i:04}.pfm')
        assert raw.shape==official.shape==cuda.shape and np.isfinite(cuda).all()
        r,c,o=[display(a,2) for a in [raw,cuda,official]]
        t=i*meta['stride']/meta['simulation_fps']
        panel_pair(r,c,'Raw / 1 spp','CUDA SVGF / Falcor parameters',t).save(frame_dir/f'demo-{i:04}.png')
        panel_pair(c,o,'CUDA SVGF / Falcor parameters','Official Falcor SVGF',t).save(frame_dir/f'compare-{i:04}.png')
        ssims.append(float(structural_similarity(c,o,channel_axis=2,data_range=1)))
        psnrs.append(float(-10*np.log10(max(np.mean((c-o)**2),1e-20))))
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    encode(ffmpeg,frame_dir/'demo-%04d.png',out/'courtyard-denoising',fps)
    encode(ffmpeg,frame_dir/'compare-%04d.png',out/'svgf-falcor-motion',fps)
    verification={name:validate_gif(out/(name+'.gif'),count,fps) for name in ['courtyard-denoising','svgf-falcor-motion']}
    # Contact sheet makes frame order and real camera parallax easy to inspect.
    contact=Image.new('RGB',(960*2,336*2))
    for j,i in enumerate([0,count//4,count//2,3*count//4]):
        contact.paste(Image.open(frame_dir/f'compare-{i:04}.png'),((j%2)*960,(j//2)*336))
    contact.save(d/'animation-contact-sheet.png')
    h=history(d/f'diagnostic-{meta["frames"]-1}-history.pfm')
    assert h['max']>4,'Motion replay did not reuse temporal history'
    report={'raw_sha256':meta['raw_sha256'],'same_input_verified':True,'history_reset_count':1,
        'per_frame_camera_and_geometry':True,'history_at_last_frame':h,'output_fps':fps,'duration_seconds':count/fps,
        'native_resolution':meta['resolution'],'gif_dimensions':[960,336],'gif_palette':'256 colors, shared global palette, no dithering',
        'camera_displacement_max_m':float(np.linalg.norm(eyes-eyes[0],axis=1).max()),
        'cuda_vs_official_display_ssim':{'mean':float(np.mean(ssims)),'min':float(min(ssims)),'max':float(max(ssims))},
        'cuda_vs_official_display_psnr_mean_db':float(np.mean(psnrs)),'gif_verification':verification,
        'notes':['Actual 60 Hz camera-motion rendering; each third frame encoded at 20 fps.','48 warmup frames omitted; history stays active.',
                 'No generated/interpolated frames. MP4 copies avoid GIF palette limits.','Playback fps is not rendering throughput; no moving high-spp reference is claimed.']}
    (d/'animation-verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
