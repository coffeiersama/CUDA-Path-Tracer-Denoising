"""Run inside official Falcor Mogwai: one PathTracer output, raw and SVGF views.

Graph wiring follows Falcor 8.0's tests/image_tests/renderpasses/graphs/SVGF.py.
The official PathTracer and SVGFPass implementations are unmodified.
"""
from falcor import *
import json
import os
from pathlib import Path

root = Path(os.environ["FALCOR_COURTYARD_ROOT"])
output = Path(os.environ.get("FALCOR_COURTYARD_OUTPUT", str(root / "build/falcor-demo")))
output.mkdir(parents=True, exist_ok=True)
width = int(os.environ.get("FALCOR_COURTYARD_WIDTH", "1280"))
height = int(os.environ.get("FALCOR_COURTYARD_HEIGHT", "720"))
exposure_ev = float(os.environ.get("FALCOR_COURTYARD_EXPOSURE_EV", "2.0"))
capture_frames = sorted(set(map(int, os.environ.get("FALCOR_COURTYARD_FRAMES", "1,64,128").split(","))))

g = RenderGraph("Official Falcor - San Miguel raw vs SVGF")
g.addPass(createPass("GBufferRaster", {"cull": "Back"}), "GBuffer")
g.addPass(createPass("PathTracer", {
    "samplesPerPixel": 1,
    "maxSurfaceBounces": 6,
    "maxDiffuseBounces": 3,
    "maxSpecularBounces": 6,
    "maxTransmissionBounces": 6,
    "useNEE": True,
    "useMIS": True,
}), "PathTracer")
svgf_settings = {"Enabled": True, "Iterations": 4, "FeedbackTap": 1,
                 "VarianceEpsilon": 1e-4, "PhiColor": 10.0, "PhiNormal": 128.0,
                 "Alpha": 0.05, "MomentsAlpha": 0.2}
g.addPass(createPass("SVGFPass", svgf_settings), "SVGF")
g.addEdge("GBuffer.vbuffer", "PathTracer.vbuffer")
g.addEdge("PathTracer.color", "SVGF.Color")
g.addEdge("PathTracer.albedo", "SVGF.Albedo")
for source, target in [("emissive", "Emission"), ("posW", "WorldPosition"),
                       ("guideNormalW", "WorldNormal"), ("pnFwidth", "PositionNormalFwidth"),
                       ("linearZ", "LinearZ"), ("mvec", "MotionVec")]:
    g.addEdge("GBuffer." + source, "SVGF." + target)

tone_settings = {"autoExposure": False, "useSceneMetadata": False,
                 "exposureCompensation": exposure_ev, "operator": "Aces",
                 "outputFormat": "RGBA8UnormSrgb"}
g.addPass(createPass("ToneMapper", tone_settings), "Noisy")
g.addPass(createPass("ToneMapper", tone_settings), "Denoised")
g.addEdge("PathTracer.color", "Noisy.src")
g.addEdge("SVGF.Filtered image", "Denoised.src")
g.addPass(createPass("SplitScreenPass", {"splitLocation": 0.5, "showTextLabels": True,
          "leftLabel": "Raw / 1 spp", "rightLabel": "Official Falcor SVGF / 1 spp"}), "Compare")
g.addEdge("Noisy.dst", "Compare.leftInput")
g.addEdge("Denoised.dst", "Compare.rightInput")
for name in ["Compare.output", "Noisy.dst", "Denoised.dst", "PathTracer.color", "SVGF.Filtered image"]:
    g.markOutput(name)

m.addGraph(g)
m.setActiveGraph(g)
m.resizeFrameBuffer(width, height)
m.loadScene(str(root / "scenes/falcor_san_miguel.pyscene"))
m.ui = False
m.clock.framerate = 60
m.clock.time = 0
m.clock.pause()
m.frameCapture.outputDir = str(output)
m.frameCapture.baseFilename = "falcor-sanmiguel"
for f in range(1, max(capture_frames) + 1):
    m.clock.frame = f
    m.renderFrame()
    if f in capture_frames:
        m.frameCapture.capture()
        print("FALCOR_CAPTURE", f, flush=True)

(output / "run.json").write_text(json.dumps({
    "framework": "NVIDIA Falcor 8.0",
    "commit": "9dc819c162b2070335c65060436041690b7937f8",
    "scene": "San Miguel native source OBJ/MTL",
    "resolution": [width, height], "path_samples_per_pixel_per_frame": 1,
    "capture_frames": capture_frames, "svgf": svgf_settings,
    "shared_exposure_compensation_ev": exposure_ev,
    "environment_intensity": float(os.environ.get("FALCOR_COURTYARD_ENVIRONMENT_INTENSITY", "8.0")),
    "raw_accumulation": False, "same_path_tracer_output": True,
    "lighting": "demo directional sun plus Falcor Arcade HDR environment",
    "notes": "Official Falcor PathTracer/SVGFPass, not an exact recreation of the 2017 paper setup."
}, indent=2), encoding="utf-8")

if os.environ.get("FALCOR_COURTYARD_LIVE", "0") != "1":
    exit()
else:
    m.ui = True
    m.clock.play()
