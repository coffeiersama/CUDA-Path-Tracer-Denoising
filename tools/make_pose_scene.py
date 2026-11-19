"""Freeze a captured camera pose while retaining the source scene and asset paths."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("scene", type=Path)
parser.add_argument("capture_metadata", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
metadata = json.loads(args.capture_metadata.read_text())
lines = args.scene.read_text().splitlines()
for i, line in enumerate(lines):
    tokens = line.split()
    if not tokens:
        continue
    if tokens[0] == "EYE":
        lines[i] = "EYE " + " ".join(format(v, ".9g") for v in metadata["eye"])
    elif tokens[0] == "LOOKAT":
        lines[i] = "LOOKAT " + " ".join(format(v, ".9g") for v in metadata["look_at"])
    elif line.strip() == "mesh":
        asset = Path(lines[i + 5].strip())
        if not asset.is_absolute():
            asset = args.scene.resolve().parent / "Models" / asset
        lines[i + 5] = asset.resolve().as_posix()
    elif tokens[0] == "TEXTURE":
        # The audit scenes import textures through MTL. Fail rather than silently
        # change meaning for the separate legacy texture-path convention.
        raise ValueError("Pose wrapper currently supports MTL-textured audit scenes only")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
