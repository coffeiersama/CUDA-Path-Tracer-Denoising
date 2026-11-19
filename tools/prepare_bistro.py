"""Convert exported Bistro DDS base-color textures to bounded-size RGBA PNGs.

Run with ordinary Python after export_bistro.py. Requires Pillow.
"""
import argparse
import json
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("obj", type=Path)
parser.add_argument("--texture-size", type=int, default=1024)
args = parser.parse_args()
mtl = args.obj.with_suffix(".mtl")
texture_dir = mtl.parent / "textures"
texture_dir.mkdir(exist_ok=True)
converted = {}
lines = []
for line in mtl.read_text(encoding="utf-8").splitlines():
    if line.startswith("map_Kd "):
        source = Path(line[7:].strip())
        if not source.is_absolute():
            source = mtl.parent / source
        if not source.is_file():
            raise FileNotFoundError(source)
        target = texture_dir / (source.stem + ".png")
        if source not in converted:
            with Image.open(source) as im:
                im = im.convert("RGBA")
                im.thumbnail((args.texture_size, args.texture_size), Image.Resampling.LANCZOS)
                im.save(target)
            converted[source] = target
        line = "map_Kd " + target.relative_to(mtl.parent).as_posix()
    # Unsupported maps remain available in the source archive, but are not
    # referenced as if the current renderer supported them.
    elif line.startswith(("map_", "bump ", "disp ")):
        continue
    lines.append(line)
mtl.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"base_color_textures": len(converted), "max_texture_size": args.texture_size}))
