"""Compare actual NRD 4.18 output to the previously frozen CUDA/SVGF outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from compare_falcor_shared import pfm, display

def main():
    p=argparse.ArgumentParser(); p.add_argument('directory',type=Path); args=p.parse_args()
    out=args.directory; run=json.loads((out/'run.json').read_text()); base=Path(run['base_directory'])
    old=json.loads((base/'run.json').read_text()); frame=run['frames']; w,h=run['resolution']
    assert frame==old['frames'] and run['raw_sha256']==old['raw_sha256']
    assert run['rgb_bitwise_equal'] and run['official_svgf_bitwise_equal']
    ref=np.load(base/'reference.npy'); rd=display(ref,2)
    images={
        'Your CUDA / current settings':pfm(base/f'cuda-current-{frame}.pfm'),
        'Official Falcor SVGF':np.load(base/f'SVGF-Filtered_image-{frame}.npy'),
        'NRD 4.18 RELAX':np.load(out/f'ComposeNRD4-output-{frame}.npy')[...,:3],
        f'Reference / {old["reference_spp"]} spp':ref,
        'NRD 3.1 / previous integration':np.load(base/f'ModulateIllumination-output-{frame}.npy'),
        'Raw / 1 spp':np.load(base/f'PathTracer-color-{frame}.npy')}
    scores={}; displays={}
    filenames=['cuda-current','official-svgf','nrd4-relax','reference','nrd3-previous','raw']
    for filename,(name,a) in zip(filenames,images.items()):
        assert a.shape==(h,w,3) and np.isfinite(a).all(),name
        d=display(a,2); displays[name]=d
        scores[name]={'linear_mse':float(np.mean((a.astype(np.float64)-ref)**2)),
            'display_psnr':float(-10*np.log10(max(np.mean((d-rd)**2),1e-20))),
            'display_ssim':float(structural_similarity(d,rd,channel_axis=2,data_range=1))}
        Image.fromarray(np.rint(d*255).astype(np.uint8)).save(out/(filename+'.png'))
    metrics={'nrd_version':run['nrd_version'],'nrd_commit':run['nrd_commit'],'resolution':[w,h],
        'same_input_raw_sha256':run['raw_sha256'],'same_official_svgf':True,'frames':frame,
        'reference_spp':old['reference_spp'],'display':'Shared Reinhard+sRGB / EV +2','scores':scores,
        'limitations':run['limitations']}
    metrics['reference_noise_mse_estimate']=json.loads((base/'metrics.json').read_text())['reference_noise_mse_estimate']
    layer_data={}
    for layer in ['Reflection','Transmission']:
        zfile=out/f'PathTracer-nrdDelta{layer}PathLength-{frame}.npy'
        if zfile.exists():
            z=np.load(zfile)[...,0]; invalid=(z<=0)|(z>=60000)
            a=np.load(out/f'PathTracer-nrdDelta{layer}RadianceHitDist-{frame}.npy')[...,:3]
            layer_data[layer]={'pixels_without_guide_fraction':float(invalid.mean()),
                'unfiltered_fallback_rgb_energy_fraction_of_layer':float(a[invalid].sum()/max(a.sum(),1e-20))}
    metrics['secondary_guide_coverage']=layer_data
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    names=list(images)[:4]
    fig,axes=plt.subplots(1,4,figsize=(20,3.45),dpi=160)
    for ax,name in zip(axes,names):
        ax.imshow(displays[name]); ax.axis('off')
        ax.set_title(name+(f"\nPSNR {scores[name]['display_psnr']:.2f} dB / SSIM {scores[name]['display_ssim']:.3f}" if 'Reference' not in name else '\nUnfiltered path-sample average'),fontsize=10)
    fig.suptitle(f'Same frozen Falcor samples + G-buffer | {w}x{h}, 1 spp x {frame} frames | shared display transform',fontsize=12)
    fig.tight_layout(); fig.savefig(out/'comparison.png'); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,6.8),dpi=160)
    for ax,(name,d) in zip(axes.flat,displays.items()): ax.imshow(d); ax.axis('off'); ax.set_title(name,fontsize=11)
    fig.tight_layout(); fig.savefig(out/'all-variants.png'); plt.close(fig)
    regions=[(int(w*.32),int(h*.37),int(w*.70),int(h*.90)),(int(w*.57),int(h*.04),int(w*.97),int(h*.42))]
    fig,axes=plt.subplots(2,4,figsize=(16,8),dpi=160)
    for row,(x0,y0,x1,y1) in enumerate(regions):
        for col,name in enumerate(names):
            axes[row,col].imshow(displays[name][y0:y1,x0:x1],interpolation='nearest'); axes[row,col].axis('off'); axes[row,col].set_title(name,fontsize=10)
    fig.tight_layout(); fig.savefig(out/'crops.png'); plt.close(fig)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
