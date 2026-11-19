"""Inside official Falcor 8 Mogwai: one path stream -> SVGF, NRD and CUDA export.

NRD wiring is taken from Falcor 8 scripts/PathTracerNRD.py, without DLSS/TAA.
Stationary pixel centers make a fixed G-buffer sequence valid for CUDA replay.
"""
from falcor import *
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import numpy as np

root = Path(os.environ["FALCOR_COURTYARD_ROOT"])
out = Path(os.environ.get("FALCOR_SHARED_OUTPUT", str(root / "build/falcor-shared")))
out.mkdir(parents=True, exist_ok=True)
w = int(os.environ.get("FALCOR_SHARED_WIDTH", "640"))
h = int(os.environ.get("FALCOR_SHARED_HEIGHT", "360"))
frames = int(os.environ.get("FALCOR_SHARED_FRAMES", "128"))
ref_frames = int(os.environ.get("FALCOR_SHARED_REFERENCE", "4096"))
g = RenderGraph("Same Falcor path stream: SVGF / NRD RELAX / CUDA replay")
g.addPass(createPass("GBufferRaster", {"cull": "Back", "samplePattern": "Center"}), "GBufferRT")
g.addPass(createPass("PathTracer", {
    "samplesPerPixel": 1, "maxSurfaceBounces": 6, "maxDiffuseBounces": 3,
    "maxSpecularBounces": 6, "maxTransmissionBounces": 6,
    "useNEE": True, "useMIS": True,
    "useNRDDemodulation": False,
}), "PathTracer")
g.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer")
g.addEdge("GBufferRT.viewW", "PathTracer.viewW")
g.addPass(createPass("SharedInputGuides", {}), "NRDGuides")
g.addEdge("GBufferRT.guideNormalW", "NRDGuides.normal")
g.addEdge("GBufferRT.specRough", "NRDGuides.roughness")
g.addEdge("GBufferRT.diffuseOpacity", "NRDGuides.diffuse")
g.addEdge("PathTracer.nrdDiffuseRadianceHitDist", "NRDGuides.diffuseRadiance")
g.addEdge("PathTracer.nrdSpecularRadianceHitDist", "NRDGuides.specularRadiance")

settings = {"Enabled": True, "Iterations": 4, "FeedbackTap": 1, "VarianceEpsilon": 1e-4,
            "PhiColor": 10.0, "PhiNormal": 128.0, "Alpha": .05, "MomentsAlpha": .2}
g.addPass(createPass("SVGFPass", settings), "SVGF")
g.addEdge("PathTracer.color", "SVGF.Color")
g.addEdge("PathTracer.albedo", "SVGF.Albedo")
for source, target in [("emissive", "Emission"), ("posW", "WorldPosition"),
                       ("guideNormalW", "WorldNormal"), ("pnFwidth", "PositionNormalFwidth"),
                       ("linearZ", "LinearZ"), ("mvec", "MotionVec")]:
    g.addEdge("GBufferRT." + source, "SVGF." + target)

# Use the official full decomposition, including delta reflection/transmission.
for name, props in {
    "NRDDiffuseSpecular": {"maxIntensity": 250.0},
    "NRDDeltaReflection": {"method": "RelaxDiffuse", "maxIntensity": 250.0, "worldSpaceMotion": False,
                           "enableReprojectionTestSkippingWithoutMotion": True,
                           "spatialVarianceEstimationHistoryThreshold": 1},
    "NRDDeltaTransmission": {"method": "RelaxDiffuse", "maxIntensity": 250.0, "worldSpaceMotion": False,
                             "enableReprojectionTestSkippingWithoutMotion": True},
    "NRDReflectionMotionVectors": {"method": "SpecularReflectionMv", "worldSpaceMotion": False},
    "NRDTransmissionMotionVectors": {"method": "SpecularDeltaMv", "worldSpaceMotion": False},
}.items():
    g.addPass(createPass("NRD", props), name)
g.addPass(createPass("ModulateIllumination", {"useResidualRadiance": True}), "ModulateIllumination")
g.addPass(createPass("ModulateIllumination", {"useResidualRadiance": True}), "ReconstructRaw")

official_script = root / "build/Falcor8/scripts/PathTracerNRD.py"
copy_passes = {}
for node in ast.walk(ast.parse(official_script.read_text())):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "addEdge":
        source, target = [ast.literal_eval(arg) for arg in node.args]
        if target.split(".")[0] in {"NRDDiffuseSpecular", "NRDDeltaReflection", "NRDDeltaTransmission",
                                   "NRDReflectionMotionVectors", "NRDTransmissionMotionVectors", "ModulateIllumination"}:
            if source == "GBufferRT.mvecW":
                source = "NRDGuides.motion"
            elif source == "GBufferRT.normWRoughnessMaterialID":
                source = "NRDGuides.packed"
            if source.startswith("PathTracer.nrd") and target.split(".")[0] in {
                    "NRDDiffuseSpecular", "NRDDeltaReflection", "NRDDeltaTransmission"} and target.endswith("RadianceHitDist"):
                # Falcor NRD pack pass clamps inputs IN PLACE. Give it private
                # copies so exported unfiltered lobes remain independently auditable.
                if source == "PathTracer.nrdDiffuseRadianceHitDist":
                    copy_passes[source] = "NRDGuides.diffuseSignal"
                elif source == "PathTracer.nrdSpecularRadianceHitDist":
                    copy_passes[source] = "NRDGuides.specularSignal"
                elif source not in copy_passes:
                    copy_name = "NRDInputCopy" + str(len(copy_passes))
                    g.addPass(createPass("BlitPass", {"filter": "Point", "outputFormat": "RGBA32Float"}), copy_name)
                    g.addEdge(source, copy_name + ".src")
                    copy_passes[source] = copy_name + ".dst"
                g.addEdge(copy_passes[source], target)
            elif target == "ModulateIllumination.diffuseReflectance":
                g.addEdge("NRDGuides.diffuseFactor", target)
            elif target == "ModulateIllumination.specularReflectance":
                g.addEdge("NRDGuides.specularFactor", target)
            else:
                g.addEdge(source, target)
            if target.startswith("ModulateIllumination."):
                # Also reconstruct the same raw sample from unfiltered lobe buffers.
                raw_source = source
                if source.startswith("NRDDiffuseSpecular.filteredDiffuse"):
                    raw_source = "PathTracer.nrdDiffuseRadianceHitDist"
                elif source.startswith("NRDDiffuseSpecular.filteredSpecular"):
                    raw_source = "PathTracer.nrdSpecularRadianceHitDist"
                elif source.startswith("NRDDeltaReflection.filtered"):
                    raw_source = "PathTracer.nrdDeltaReflectionRadianceHitDist"
                elif source.startswith("NRDDeltaTransmission.filtered"):
                    raw_source = "PathTracer.nrdDeltaTransmissionRadianceHitDist"
                g.addEdge(raw_source, target.replace("ModulateIllumination.", "ReconstructRaw."))

outputs = ["PathTracer.color", "PathTracer.albedo", "SVGF.Filtered image", "ModulateIllumination.output",
           "ReconstructRaw.output"] + ["GBufferRT." + s for s in
           ["posW", "guideNormalW", "emissive", "pnFwidth", "linearZ", "mvec", "mask"]]
debug_outputs = []
if os.environ.get("FALCOR_SHARED_DEBUG", "0") == "1":
    debug_outputs = ["PathTracer." + name for name in ["nrdDiffuseRadianceHitDist", "nrdSpecularRadianceHitDist",
        "nrdDiffuseReflectance", "nrdSpecularReflectance", "nrdEmission", "nrdDeltaReflectionRadianceHitDist",
        "nrdDeltaReflectionReflectance", "nrdDeltaReflectionEmission", "nrdDeltaTransmissionRadianceHitDist",
        "nrdDeltaTransmissionReflectance", "nrdDeltaTransmissionEmission", "nrdResidualRadianceHitDist"]] + list(copy_passes.values())
    outputs += debug_outputs
for name in outputs:
    g.markOutput(name)
m.addGraph(g)
m.setActiveGraph(g)
m.resizeFrameBuffer(w, h)
m.loadScene(str(root / "scenes/falcor_san_miguel.pyscene"))
m.ui = False
m.clock.framerate = 60
m.clock.pause()

def tex(name):
    texture = g.getOutput(name)
    a = np.asarray(texture.to_numpy())
    if a.ndim == 1:
        # Normalized formats are returned as their raw bytes by Falcor 8.
        fmt = str(texture.format)
        if fmt.endswith("RGBA8Unorm"):
            a = a.reshape(h, w, 4).astype(np.float32) / 255
        elif fmt.endswith("RGBA16Float"):
            a = a.view(np.float16).reshape(h, w, 4)
        else:
            raise ValueError((name, fmt, a.shape, a.dtype))
    a = a.astype(np.float32)
    if a.ndim == 2:
        a = a[..., None]
    assert a.shape[:2] == (h, w), (name, a.shape)
    assert np.isfinite(a).all(), name
    return a

def vec(v):
    return np.array([v.x, v.y, v.z], dtype=np.float32)

sha = hashlib.sha256()
geometry_hash = None
reconstruction_errors = []
with (out / "stationary.seq").open("wb") as seq:
    seq.write(struct.pack("8s5I", b"SVGFSEQ1", w, h, frames, 84, 64))
    for f in range(1, frames + 1):
        m.clock.frame = f
        m.renderFrame()
        raw = tex("PathTracer.color")[..., :3]
        if f == 1:
            cam = m.scene.camera
            eye, target, up = vec(cam.position), vec(cam.target), vec(cam.up)
            view = target - eye; view /= np.linalg.norm(view)
            right = np.cross(view, up); right /= np.linalg.norm(right)
            up = np.cross(right, view)
            fy = float(cam.frameHeight) / float(cam.focalLength)
            fx = fy * w / h
            camera = np.concatenate([eye, target, view, up, right, [fx, fy], [fx / w, fy / h]]).astype("<f4")
            seq.write(struct.pack("2i", w, h)); seq.write(camera.tobytes())
            geo = np.zeros((h, w), dtype=[("emission", "<f4", 3), ("normal", "<f4", 3),
                           ("position", "<f4", 3), ("albedo", "<f4", 3), ("ialbedo", "<f4", 3), ("geomId", "<i4")])
            geo["emission"] = tex("GBufferRT.emissive")[..., :3]
            geo["normal"] = tex("GBufferRT.guideNormalW")[..., :3]
            geo["position"] = tex("GBufferRT.posW")[..., :3]
            geo["albedo"] = tex("PathTracer.albedo")[..., :3]
            geo["ialbedo"] = 1 / np.maximum(geo["albedo"], .001)
            valid = tex("GBufferRT.mask")[..., 0] > 0
            geo["geomId"] = np.where(valid, 0, -1)
            seq.write(geo[:, ::-1].tobytes())
            z = tex("GBufferRT.linearZ")
            pn = tex("GBufferRT.pnFwidth")
            guide = np.stack([np.where(valid, z[..., 0], -1), z[..., 1], pn[..., 1]], axis=-1)
            guide[:, ::-1].astype("<f4").tofile(str(out / "guides.bin"))
            for name in outputs:
                if name.startswith("GBufferRT.") or name == "PathTracer.albedo":
                    np.save(str(out / (name.replace(".", "-") + ".npy")), tex(name))
            geometry_hash = hashlib.sha256(geo.tobytes()).hexdigest()
            np.save(str(out / "gbuffer.npy"), geo)
            # Verify the exact projection convention used by the CUDA reprojector.
            delta = geo["position"] - eye
            zz = delta @ view
            px = -(delta @ right) / (np.maximum(zz, 1e-8) * fx / w) + w / 2 - .5
            py = -(delta @ up) / (np.maximum(zz, 1e-8) * fy / h) + h / 2 - .5
            yy, xx = np.indices((h, w))
            projection_error = float(np.max(np.maximum(abs(px - (w - 1 - xx)), abs(py - yy))[valid]))
            assert projection_error < .05, ("camera projection mismatch", projection_error)
            (out / "camera.json").write_text(json.dumps({"eye": eye.tolist(), "target": target.tolist(),
                "up": up.tolist(), "frameHeight": float(cam.frameHeight), "focalLength": float(cam.focalLength),
                "projection_max_error_pixels": projection_error}, indent=2))
        else:
            assert np.array_equal(tex("GBufferRT.posW")[..., :3], geo["position"]), "Geometry changed during static export"
            assert np.array_equal(tex("PathTracer.albedo")[..., :3], geo["albedo"]), "Albedo changed during static export"
            assert np.array_equal(tex("GBufferRT.guideNormalW")[..., :3], geo["normal"]), "Normal changed during static export"
            assert np.array_equal(tex("GBufferRT.emissive")[..., :3], geo["emission"]), "Emission changed during static export"
            assert float(abs(tex("GBufferRT.mvec")).max()) * max(w, h) < .05, "Unexpected motion after first frame"
        sample_bytes = raw[:, ::-1].astype("<f4").tobytes()
        sha.update(sample_bytes); seq.write(sample_bytes)
        if f in {1, 32, 64, frames}:
            if f == frames:
                for name in debug_outputs:
                    np.save(str(out / (name.replace(".", "-") + ".npy")), tex(name))
            for name in ["PathTracer.color", "SVGF.Filtered image", "ModulateIllumination.output", "ReconstructRaw.output"]:
                np.save(str(out / (name.replace(".", "-").replace(" ", "_") + "-" + str(f) + ".npy")), tex(name)[..., :3])
            rec = tex("ReconstructRaw.output")[..., :3]
            reconstruction_errors.append({"frame": f, "mse": float(np.mean((raw.astype(np.float64) - rec)**2)),
                                          "max_abs": float(abs(raw - rec).max()),
                                          "relative_rmse": float(np.sqrt(np.mean((raw.astype(np.float64)-rec)**2) / max(np.mean(raw.astype(np.float64)**2), 1e-20)))})
            assert reconstruction_errors[-1]["relative_rmse"] < 1e-4, "NRD path partition does not reconstruct the original RGB"
            print("SHARED_CAPTURE", f, "raw/lobe reconstruction", reconstruction_errors[-1], flush=True)

# Independent subsequent path samples at the same fixed pixel centers.
# Average in float64 on the CPU; this reference does not go through a denoiser.
ref = np.zeros((h, w, 3), dtype=np.float64)
ref_a = np.zeros_like(ref)
for i in range(ref_frames):
    m.clock.frame = frames + i + 1
    m.renderFrame()
    ref += tex("PathTracer.color")[..., :3]
    if i + 1 == ref_frames // 2:
        ref_a = ref.copy() / (i + 1)
    if (i + 1) % 64 == 0:
        print("SHARED_REFERENCE", i + 1, flush=True)
ref /= max(ref_frames, 1)
np.save(str(out / "reference.npy"), ref.astype(np.float32))
if ref_frames >= 2 and ref_frames % 2 == 0:
    ref_b = 2 * ref - ref_a
    np.save(str(out / "reference-a.npy"), ref_a.astype(np.float32))
    np.save(str(out / "reference-b.npy"), ref_b.astype(np.float32))
metadata = {"framework": "Falcor 8.0", "falcor_commit": "9dc819c162b2070335c65060436041690b7937f8",
    "nrd_version": "3.1.0 (Falcor 8 bundled SDK; not current NRD master)",
    "nrd_method": "RELAX diffuse/specular plus delta reflection/transmission", "nrd_max_intensity": 250,
    "resolution": [w, h], "frames": frames, "reference_spp": ref_frames,
    "reference_sample_range": [frames + 1, frames + ref_frames], "input_sample_range": [1, frames],
    "raw_sha256": sha.hexdigest(), "geometry_sha256": geometry_hash,
    "official_svgf": settings, "pixel_jitter": False, "motion": "stationary",
    "cuda_current": {"Alpha": .05, "MomentsAlpha": .2, "PhiColor": 10, "PhiNormal": 128, "Iterations": 4,
                     "FeedbackTap": 1, "antiLag": False, "historyClamping": False},
    "cuda_matched": settings,
    "nrd_has_additional_lobe_guides": True, "dlss": False, "taa": False,
    "nrd_input_copies": True, "nrd_residual_radiance": "included unfiltered in reconstruction",
    "nrd_normalization": "Falcor per-sample demodulation disabled; diffuse/specular divided and remultiplied by the same deterministic raster factors (floor .01). Delta channels remain unmodulated. RELAX denoiser is unmodified.",
    "raw_lobe_reconstruction_errors": reconstruction_errors,
    "cuda_guides": "Exact exported Falcor linearZ + depth derivative + normal fwidth; shared albedo, emission, world position and normal",
    "display": "all variants use the same offline Reinhard+sRGB, EV +2"}
(out / "run.json").write_text(json.dumps(metadata, indent=2))
print("SHARED_EXPORT_COMPLETE", str(out), flush=True)
exit()
