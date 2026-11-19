"""Run with Blender --background --factory-startup --python tools/export_bistro.py -- source.fbx output.obj."""
import json
import sys
from pathlib import Path

import bpy
from bpy_extras.io_utils import axis_conversion

source, destination = map(Path, sys.argv[sys.argv.index("--") + 1:])
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.fbx(filepath=str(source.resolve()))
destination.parent.mkdir(parents=True, exist_ok=True)
transform = axis_conversion(to_forward="-Z", to_up="Y").to_4x4()
meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
corners = [transform @ obj.matrix_world @ __import__('mathutils').Vector(v)
           for obj in meshes for v in obj.bound_box]
bounds = [[min(p[i] for p in corners) for i in range(3)],
          [max(p[i] for p in corners) for i in range(3)]]
cameras = [{"name": obj.name, "position": list(transform @ obj.matrix_world.translation),
            "direction": list(transform.to_3x3() @ (obj.matrix_world.to_3x3() @ __import__('mathutils').Vector((0, 0, -1))))}
           for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
bpy.ops.wm.obj_export(filepath=str(destination.resolve()), export_triangulated_mesh=True,
                      export_materials=True, export_uv=True, export_normals=True,
                      forward_axis="NEGATIVE_Z", up_axis="Y", path_mode="ABSOLUTE")
metadata = {"source": source.name, "bounds": bounds, "mesh_objects": len(meshes),
            "materials": len(bpy.data.materials), "cameras": cameras}
destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("BISTRO_EXPORT", json.dumps(metadata))
