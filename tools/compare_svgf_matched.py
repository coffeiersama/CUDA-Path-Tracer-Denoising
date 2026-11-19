"""Fresh default-CUDA vs fresh official SVGF on the same stationary sequence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from compare_falcor_shared import pfm,display

def main():
    p=argparse.ArgumentParser(); p.add_argument('directory',type=Path)
    p.add_argument('--reference-directory',type=Path,default=Path('build/falcor-shared-final'))
    args=p.parse_args(); out=args.directory; base=args.reference_directory
    run=json.loads((out/'run.json').read_text()); original=json.loads((base/'run.json').read_text())
    settings=json.loads((out/'cuda-settings.json').read_text())
    expected={'alpha':.05,'moments_alpha':.2,'phi_color':10,'phi_normal':128,'iterations':4,'feedback_tap':1}
    assert settings['current']==settings['matched']==expected
    assert run['resolution']==original['resolution'] and run['frames']==original['frames']
    assert run['raw_sha256']==original['raw_sha256'] and run['geometry_sha256']==original['geometry_sha256']
    assert hashlib.sha256((out/'stationary.seq').read_bytes()).digest()==hashlib.sha256((base/'stationary.seq').read_bytes()).digest()
    assert (out/'guides.bin').read_bytes()==(base/'guides.bin').read_bytes()
    frames=run['frames']; w,h=run['resolution']
    current=pfm(out/f'cuda-current-{frames}.pfm')
    assert np.array_equal(current,pfm(out/f'cuda-matched-{frames}.pfm'))
    official=np.load(out/f'SVGF-Filtered_image-{frames}.npy')
    for f in [1,32,64,frames]:
        assert np.array_equal(np.load(out/f'SVGF-Filtered_image-{f}.npy'),np.load(base/f'SVGF-Filtered_image-{f}.npy'))
    reference=np.load(base/'reference.npy'); assert original['reference_spp']>0
    images={'Your CUDA / Falcor parameters':current,'Official Falcor SVGF':official,
            f'Reference / {original["reference_spp"]} spp':reference}
    rd=display(reference,2); ds={}; scores={}
    for filename,(name,a) in zip(['cuda-falcor-defaults','official-svgf','reference'],images.items()):
        assert a.shape==(h,w,3) and np.isfinite(a).all()
        b=display(a,2); ds[name]=b
        scores[name]={'linear_mse_to_reference':float(np.mean((a.astype(np.float64)-reference)**2)),
            'display_psnr_to_reference':float(-10*np.log10(max(np.mean((b-rd)**2),1e-20))),
            'display_ssim_to_reference':float(structural_similarity(b,rd,channel_axis=2,data_range=1))}
        Image.fromarray(np.rint(b*255).astype(np.uint8)).save(out/(filename+'.png'))
    x,y=list(ds.values())[:2]
    pair={'display_psnr':float(-10*np.log10(np.mean((x-y)**2))),
          'display_ssim':float(structural_similarity(x,y,channel_axis=2,data_range=1)),
          'linear_mse':float(np.mean((current.astype(np.float64)-official)**2))}
    report={'resolution':[w,h],'frames':frames,'reference_spp':original['reference_spp'],
        'reference_directory':str(base.resolve()),'shared_settings':expected,
        'raw_sha256':run['raw_sha256'],'entire_sequence_and_guides_bitwise_equal_to_original':True,
        'new_default_equals_explicit_falcor_preset_bitwise':True,'official_checkpoints_unchanged_bitwise':True,
        'scores':scores,'cuda_vs_official':pair,'display':'Shared Reinhard+sRGB, EV +2',
        'notes':['Parameters changed; CUDA filter algorithms unchanged.','Static camera, 1 spp per frame, 128 frames. No TAA.',
                 'The reference is the existing independent unfiltered 4096-sample average, not the zero-reference placeholder in this fresh capture.']}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(16,3.8),dpi=160)
    for ax,(name,b) in zip(axes,ds.items()): ax.imshow(b); ax.axis('off'); ax.set_title(name,fontsize=12)
    fig.suptitle(f'Identical Falcor input + matching parameters | 1 spp x {frames} frames | CUDA / official SSIM {pair["display_ssim"]:.5f}',fontsize=13)
    fig.tight_layout(); fig.savefig(out/'comparison.png'); plt.close(fig)
    regions=[(int(w*.32),int(h*.37),int(w*.70),int(h*.90)),(int(w*.57),int(h*.04),int(w*.97),int(h*.42))]
    fig,axes=plt.subplots(2,3,figsize=(13,8),dpi=160)
    for row,(x0,y0,x1,y1) in enumerate(regions):
        for col,(name,b) in enumerate(ds.items()):
            axes[row,col].imshow(b[y0:y1,x0:x1],interpolation='nearest'); axes[row,col].axis('off'); axes[row,col].set_title(name,fontsize=11)
    fig.tight_layout(); fig.savefig(out/'crops.png'); plt.close(fig)
    previous=display(pfm(base/f'cuda-current-{frames}.pfm'),2)
    fig,axes=plt.subplots(1,4,figsize=(20,3.5),dpi=160)
    for ax,name,b in zip(axes,['Your previous parameters']+list(ds),[previous]+list(ds.values())):
        ax.imshow(b); ax.axis('off'); ax.set_title(name,fontsize=11)
    fig.tight_layout(); fig.savefig(out/'before-after.png'); plt.close(fig)
    Image.fromarray(np.rint(np.clip(abs(x-y)*16,0,1)*255).astype(np.uint8)).save(out/'difference-x16.png')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
