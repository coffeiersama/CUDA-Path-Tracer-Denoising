"""Label native Falcor captures for comparison; do not alter either image's colors."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
d = args.directory
metadata = json.loads((d / "run.json").read_text())
frame = max(metadata["capture_frames"])
paths = [d / f"falcor-sanmiguel.Noisy.dst.{frame}.png", d / f"falcor-sanmiguel.Denoised.dst.{frame}.png"]
images = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
assert images[0].shape == images[1].shape
assert np.ptp(images[0]) > 0 and np.ptp(images[1]) > 0
first = min(metadata["capture_frames"])
if first != frame:
    first_raw = np.asarray(Image.open(d / f"falcor-sanmiguel.Noisy.dst.{first}.png").convert("RGB"))
    assert not np.array_equal(first_raw, images[0]), "Raw path-tracing samples did not change between captured frames"
shutil.copyfile(paths[0], d / "falcor-raw-1spp.png")
shutil.copyfile(paths[1], d / "falcor-official-svgf.png")
fig, axes = plt.subplots(1, 2, figsize=(18, 5.6), constrained_layout=True)
labels = ["Raw path tracing | 1 spp / frame | no accumulation",
          f"Official Falcor SVGF | 1 spp / frame | {frame}-frame warm-up"]
for ax, im, label in zip(axes, images, labels):
    ax.imshow(im, interpolation="nearest")
    ax.set_title(label, fontsize=11)
    ax.axis("off")
fig.suptitle(f"{metadata['framework']} / D3D12 / San Miguel | Shared input, camera and exposure", fontsize=12)
fig.savefig(d / "falcor-comparison.png", dpi=150)
plt.close(fig)
print(json.dumps({"frame": frame, "resolution": metadata["resolution"],
                  "raw_mean_byte": float(images[0].mean()), "svgf_mean_byte": float(images[1].mean()),
                  "raw_changes_between_frames": first != frame,
                  "comparison": str(d / "falcor-comparison.png")}))
