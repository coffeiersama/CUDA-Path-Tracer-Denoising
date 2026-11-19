"""Evaluate linear PFM captures against independent high-spp reference captures.

All display comparisons use the same Reinhard + sRGB transform. Linear errors
use unclipped radiance; SSIM/PSNR use the shared display transform. Requires
NumPy, Pillow and matplotlib (scikit-image optionally enables SSIM).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_pfm(path):
    with open(path, "rb") as f:
        kind = f.readline().strip()
        if kind not in (b"PF", b"Pf"):
            raise ValueError(f"Not a PFM: {path}")
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        channels = 3 if kind == b"PF" else 1
        data = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4")
        data = data.reshape(h, w, channels)[::-1].astype(np.float64) * abs(scale)
        if not np.all(np.isfinite(data)):
            raise ValueError(f"Nonfinite values: {path}")
        return data


def display(rgb):
    c = np.maximum(rgb, 0)
    c = c / (1 + c)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def errors(image, reference):
    mse = float(np.mean((image - reference) ** 2))
    d = display(image)
    r = display(reference)
    dmse = float(np.mean((d - r) ** 2))
    result = {"linear_mse": mse, "linear_relative_rmse": float(np.sqrt(mse / max(np.mean(reference ** 2), 1e-12))),
              "display_psnr_db": float(-10 * np.log10(max(dmse, 1e-16)))}
    try:
        from skimage.metrics import structural_similarity
        result["display_ssim"] = float(structural_similarity(r, d, channel_axis=2, data_range=1.0))
    except ImportError:
        pass
    return result


def evaluate(directory, before, after, refs, title):
    d = Path(directory)
    reference_halves = [read_pfm(p) for p in refs]
    reference_metadata = [json.loads(Path(str(p).replace("-reference.pfm", ".json")).read_text()) for p in refs]
    before_metadata = json.loads((d / f"{before}.json").read_text())
    after_metadata = json.loads((d / f"{after}.json").read_text())
    for metadata in [before_metadata, *reference_metadata]:
        for key in ("width", "height", "eye", "look_at", "pixel_length"):
            if not np.allclose(metadata[key], after_metadata[key], rtol=0, atol=1e-4):
                raise ValueError(f"Camera mismatch in {key}")
    ranges = [(after_metadata["seed_offset"], after_metadata["seed_offset"] + after_metadata["frames"])]
    for metadata in reference_metadata:
        if not metadata["reference"]:
            raise ValueError("Reference must be an unfiltered reference-mode capture")
        interval = (metadata["seed_offset"], metadata["seed_offset"] + metadata["frames"])
        if any(interval[0] < high and interval[1] > low for low, high in ranges):
            raise ValueError("Reference sample ranges must be independent of the input and one another")
        ranges.append(interval)
    sample_counts = [m["frames"] for m in reference_metadata]
    if len(set(sample_counts)) != 1:
        raise ValueError("Reference groups must have equal sample counts")
    reference = np.mean(reference_halves, axis=0)
    images = {"Raw 1 spp": read_pfm(d / f"{after}-raw.pfm"),
              "Original center": read_pfm(d / f"{after}-original.pfm"),
              "Previous right": read_pfm(d / f"{before}-optimized.pfm"),
              "Aligned SVGF": read_pfm(d / f"{after}-optimized.pfm")}
    previous_raw = read_pfm(d / f"{before}-raw.pfm")
    raw_delta = float(np.max(np.abs(previous_raw - images["Raw 1 spp"])))
    if raw_delta > 1e-6:
        raise ValueError(f"Before/after inputs differ by {raw_delta}; this comparison is not controlled")
    report = {"reference": {"paths": [str(p) for p in refs], "renderer": "this project's renderer, fixed primary rays",
              "raw_max_abs_before_after": raw_delta, "samples": sum(sample_counts)}, "metrics": {k: errors(v, reference) for k, v in images.items()}}
    if len(reference_halves) == 2:
        # Equal-size independent halves: estimated variance of their average.
        report["reference"]["estimated_mean_noise_mse"] = float(np.mean((reference_halves[0] - reference_halves[1]) ** 2) / 4)
    report["after_before_linear_mse_ratio"] = report["metrics"]["Aligned SVGF"]["linear_mse"] / report["metrics"]["Previous right"]["linear_mse"]
    albedo_path = d / f"{after}-albedo.pfm"
    if albedo_path.exists():
        a = np.maximum(read_pfm(albedo_path), 0.001)
        e = read_pfm(d / f"{after}-emission.pfm")
        report["stages"] = {}
        stage_paths = [d / f"{after}-temporal.pfm", d / f"{after}-prefilter.pfm"] + sorted(d.glob(f"{after}-atrous-*.pfm"))
        for p in stage_paths:
            report["stages"][p.stem] = errors(read_pfm(p) * a + e, reference)
    history = d / f"{after}-history.pfm"
    if history.exists():
        h = read_pfm(history)
        report["history"] = {"mean": float(h.mean()), "shorter_than_4_fraction": float(np.mean(h < 4))}
    (d / f"{after}-metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    chosen = [("Previous right", images["Previous right"]), ("Aligned SVGF", images["Aligned SVGF"]), (f"{sum(sample_counts)} spp reference", reference)]
    fig, ax = plt.subplots(2, 3, figsize=(16, 6.8), constrained_layout=True)
    fig.suptitle(title + " | Same renderer, camera and input samples | Reinhard + sRGB", fontsize=11)
    max_error = max(float(np.percentile(np.mean(np.abs(display(im) - display(reference)), axis=2), 99)) for _, im in chosen[:2])
    for col, (label, im) in enumerate(chosen):
        ax[0, col].imshow(display(im)); ax[0, col].set_title(label); ax[0, col].axis("off")
        if col < 2:
            heat = np.mean(np.abs(display(im) - display(reference)), axis=2)
            ax[1, col].imshow(heat, cmap="magma", vmin=0, vmax=max_error)
            m = report["metrics"][label]
            ax[1, col].set_title(f"Display error | linear rRMSE {m['linear_relative_rmse']:.4f} | PSNR {m['display_psnr_db']:.2f} dB")
        else:
            ax[1, col].imshow(display(images["Raw 1 spp"])); ax[1, col].set_title("Shared 1 spp input")
        ax[1, col].axis("off")
    fig.savefig(d / f"{after}-comparison.png", dpi=150)
    plt.close(fig)
    h, w = reference.shape[:2]
    crops = [("Central detail", (0.32, 0.20, 0.70, 0.88)),
             ("Lower-right detail / shadows", (0.58, 0.62, 0.98, 0.98))]
    fig, ax = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, (region, (x0, y0, x1, y1)) in enumerate(crops):
        for col, (label, im) in enumerate(chosen):
            ax[row, col].imshow(display(im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]), interpolation="nearest")
            ax[row, col].set_title(label + "\n" + region, fontsize=10)
            ax[row, col].axis("off")
    fig.savefig(d / f"{after}-crops.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--before", default="before")
    parser.add_argument("--after", default="after")
    parser.add_argument("--references", type=Path, nargs="+", required=True)
    parser.add_argument("--title", default="San Miguel stationary camera")
    args = parser.parse_args()
    evaluate(args.directory, args.before, args.after, args.references, args.title)
