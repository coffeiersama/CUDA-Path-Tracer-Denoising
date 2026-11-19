"""Validate same-input replay and render all algorithms through one display transform."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

def pfm(path):
    with path.open("rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = map(int, f.readline().split())
        assert float(f.readline()) == -1
        data = np.fromfile(f, dtype="<f4")
    return data.reshape(h, w, 3)[::-1].copy()

def display(a, ev):
    x = np.maximum(a.astype(np.float64), 0) * 2**ev
    x = x / (1 + x)
    return np.clip(np.where(x <= .0031308, x * 12.92, 1.055 * x**(1/2.4) - .055), 0, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ev", type=float, default=2)
    args = parser.parse_args()
    d = args.directory
    meta = json.loads((d / "run.json").read_text())
    w, h = meta["resolution"]; frame = meta["frames"]
    n = w * h
    with (d / "stationary.seq").open("rb") as f:
        header = struct.unpack("8s5I", f.read(28))
        assert header == (b"SVGFSEQ1", w, h, frame, 84, 64)
        f.seek(84 + n * 64, 1)
        sha = hashlib.sha256()
        last = None
        for i in range(frame):
            data = f.read(n * 12)
            assert len(data) == n * 12
            sha.update(data)
            last = np.frombuffer(data, "<f4").reshape(h, w, 3)[:, ::-1]
        assert not f.read(1), "Trailing sequence bytes"
    assert sha.hexdigest() == meta["raw_sha256"], "Raw sequence hash mismatch"
    assert max(e["relative_rmse"] for e in meta["raw_lobe_reconstruction_errors"]) < 1e-4, "Invalid NRD input partition"
    images = {
        "Raw 1 spp": np.load(d / f"PathTracer-color-{frame}.npy"),
        "Your CUDA / current settings": pfm(d / f"cuda-current-{frame}.pfm"),
        "Official Falcor SVGF": np.load(d / f"SVGF-Filtered_image-{frame}.npy"),
        "NRD 3.1 RELAX": np.load(d / f"ModulateIllumination-output-{frame}.npy"),
        "Your CUDA / Falcor settings": pfm(d / f"cuda-matched-{frame}.pfm"),
        f"Reference / {meta['reference_spp']} spp": np.load(d / "reference.npy"),
    }
    assert np.array_equal(last, images["Raw 1 spp"]), "Falcor/CUDA raw pixels differ"
    reference = list(images.values())[-1]
    assert meta["reference_spp"] > 0
    reference_display = display(reference, args.ev)
    scores = {}
    displays = {}
    for name, a in images.items():
        assert a.shape == (h, w, 3) and np.isfinite(a).all(), name
        b = display(a, args.ev)
        displays[name] = b
        mse = float(np.mean((a.astype(np.float64) - reference)**2))
        mse_d = float(np.mean((b - reference_display)**2))
        scores[name] = {"linear_mse": mse, "display_psnr": float(-10 * np.log10(max(mse_d, 1e-20))),
                        "display_ssim": float(structural_similarity(b, reference_display, channel_axis=2, data_range=1))}
        filename = {"Raw 1 spp": "raw", "Your CUDA / current settings": "cuda-current",
                    "Your CUDA / Falcor settings": "cuda-matched", "Official Falcor SVGF": "official-svgf",
                    "NRD 3.1 RELAX": "nrd-relax"}.get(name, "reference")
        Image.fromarray(np.rint(b * 255).astype(np.uint8)).save(d / f"{filename}.png")
    report = {"resolution": [w, h], "input_frames": frame, "reference_spp": meta["reference_spp"],
              "same_raw_bitwise_verified": True, "raw_sha256": sha.hexdigest(), "display_ev": args.ev,
              "scores": scores, "notes": ["Stationary camera and center-pixel sampling; no TAA/DLSS.",
                "NRD uses additional path-lobe/hit-distance/roughness guides from the same path stream.",
                "CUDA receives the exact exported raster depth/derivative/normal-width guides.",
                "NRD version is bundled 3.1.0; this is not a test of NRD 4.18 master.",
                meta["nrd_normalization"],
                "NRD uses Falcor adapter intensity limit 250 on demodulated inputs (algorithm-specific preprocessing).",
                "Current settings and matched Falcor settings are both shown; no tuning against this reference."]}
    matched = images["Your CUDA / Falcor settings"]
    official = images["Official Falcor SVGF"]
    report["matched_cuda_vs_official"] = {
        "linear_mse_between_outputs": float(np.mean((matched.astype(np.float64) - official)**2)),
        "display_psnr_between_outputs": float(-10 * np.log10(np.mean((display(matched,args.ev)-display(official,args.ev))**2))),
        "display_ssim_between_outputs": float(structural_similarity(display(matched,args.ev), display(official,args.ev), channel_axis=2, data_range=1))}
    report["cuda_presets"] = {
        "current": {"alpha": .2, "moments_alpha": .2, "phi_color": 2, "phi_normal": 128, "iterations": 4, "feedback_tap": 0},
        "matched": {"alpha": .05, "moments_alpha": .2, "phi_color": 10, "phi_normal": 128, "iterations": 4, "feedback_tap": 1}}
    if (d / "cuda-settings.json").exists():
        report["cuda_presets"] = json.loads((d / "cuda-settings.json").read_text())
    if (d / "reference-a.npy").exists():
        a, b = np.load(d / "reference-a.npy"), np.load(d / "reference-b.npy")
        report["reference_noise_mse_estimate"] = float(np.mean((a.astype(np.float64)-b)**2) / 4)
    (d / "metrics.json").write_text(json.dumps(report, indent=2))
    # Main comparison keeps the three actual denoisers together at equal scale.
    main_names = ["Your CUDA / current settings", "Official Falcor SVGF", "NRD 3.1 RELAX", list(images)[-1]]
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.45), dpi=160)
    for ax, name in zip(axes, main_names):
        ax.imshow(displays[name]); ax.axis("off")
        s = scores[name]
        ax.set_title(name + (f"\nPSNR {s['display_psnr']:.2f} dB / SSIM {s['display_ssim']:.3f}" if "Reference" not in name else "\nUnfiltered path-sample average"), fontsize=10)
    fig.suptitle(f"Same Falcor samples + G-buffer | {w}x{h}, 1 spp x {frame} frames | shared Reinhard+sRGB, EV {args.ev:+g}", fontsize=12)
    fig.tight_layout(); fig.savefig(d / "comparison.png"); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.8), dpi=160)
    for ax, (name, b) in zip(axes.flat, displays.items()):
        ax.imshow(b); ax.set_title(name, fontsize=11); ax.axis("off")
    fig.tight_layout(); fig.savefig(d / "all-variants.png"); plt.close(fig)
    # Fixed scene-relative crops: glossy cups/table and background foliage.
    regions = [(int(w*.32), int(h*.37), int(w*.70), int(h*.90)),
               (int(w*.57), int(h*.04), int(w*.97), int(h*.42))]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=160)
    for row, (x0,y0,x1,y1) in enumerate(regions):
        for col, name in enumerate(main_names):
            axes[row,col].imshow(displays[name][y0:y1,x0:x1], interpolation="nearest")
            axes[row,col].set_title(name, fontsize=10); axes[row,col].axis("off")
    fig.tight_layout(); fig.savefig(d / "crops.png"); plt.close(fig)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
