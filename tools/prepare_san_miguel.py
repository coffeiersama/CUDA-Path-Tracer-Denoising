"""Prepare McGuire's San Miguel OBJ/MTL and bounded RGBA textures. Requires Pillow."""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


def prepare(source: Path, destination: Path, texture_size: int, full: bool):
    if texture_size < 1:
        raise ValueError("Texture size must be positive")
    name = "san-miguel" if full else "san-miguel-low-poly"
    destination.mkdir(parents=True, exist_ok=True)
    texture_dir = destination / "textures"
    texture_dir.mkdir(exist_ok=True)
    converted = {}
    material_lines = []
    for line in (source / (name + ".mtl")).read_text(encoding="utf-8").splitlines():
        if line.startswith("map_Kd "):
            texture = (source / line[7:].strip().replace("\\", "/")).resolve()
            target = texture_dir / texture.name
            if texture not in converted:
                with Image.open(texture) as im:
                    im = im.convert("RGBA")
                    im.thumbnail((texture_size, texture_size), Image.Resampling.LANCZOS)
                    im.save(target)
                converted[texture] = target
            line = "map_Kd " + target.relative_to(destination).as_posix()
        elif line.startswith(("map_", "bump ", "disp ")):
            continue
        material_lines.append(line)
    (destination / "san-miguel.mtl").write_text("\n".join(material_lines) + "\n", encoding="utf-8")
    # Stream the large mesh without holding the source text in memory.
    triangles = 0
    wrote_library = False
    with (source / (name + ".obj")).open(encoding="utf-8") as src, \
            (destination / "san-miguel.obj").open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            if line.startswith("mtllib "):
                if wrote_library:
                    continue
                line = "mtllib san-miguel.mtl\n"
                wrote_library = True
            if line.startswith("f "):
                vertices = line.partition("#")[0].split()[1:]
                for i in range(1, len(vertices) - 1):
                    dst.write(f"f {vertices[0]} {vertices[i]} {vertices[i + 1]}\n")
                    triangles += 1
                continue
            dst.write(line)
    shutil.copyfile(source / "license.txt", destination / "license.txt")
    metadata = {"source": "https://casual-effects.com/g3d/data10/research/model/San_Miguel/San_Miguel.zip",
                "variant": name, "triangles": triangles, "base_color_textures": len(converted),
                "max_texture_size": texture_size,
                "limitations": "Base color and cutout alpha only; simplified renderer materials and test lighting."}
    (destination / "san-miguel.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--full", action="store_true", help="Use the full-resolution mesh instead of the real-time variant")
    args = parser.parse_args()
    prepare(args.source, args.destination, args.texture_size, args.full)
