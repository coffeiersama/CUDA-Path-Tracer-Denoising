"""Patch only Falcor's runtime NRD auxiliary alpha outputs; preserve RGB/RNG.

Keep skipped-lobe distances zero, as required by NRD 4.18. This writes the
runtime shader copy, not the official checkout's source shader.
"""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
source = root / 'build/Falcor8/Source/RenderPasses/PathTracer/ResolvePass.cs.slang'
dest = root / 'build/Falcor8/build/windows-vs2022/bin/Release/shaders/RenderPasses/PathTracer/ResolvePass.cs.slang'
s = source.read_text()
assert s.count('float hitDist = 0.f;') == 1
s = s.replace('float hitDist = 0.f;', 'float hitDist = 0.f;\n            float4 nrd4LobeHitDist = 0.f;')
for i, name in enumerate(['Diffuse', 'Specular', 'DeltaReflection', 'DeltaTransmission']):
    key = f'case NRDPathType::{name}:'
    assert s.count(key) == 1
    s = s.replace(key, key + f'\n                    nrd4LobeHitDist[{i}] += sampleNRDHitDist[idx];')
for i, (name, value) in enumerate([('Diffuse','diffuseRadiance'),('Specular','specularRadiance'),
                                 ('DeltaReflection','deltaReflectionRadiance'),('DeltaTransmission','deltaTransmissionRadiance')]):
    old_alpha = 'invSpp * hitDist' if i < 2 else '0.f'
    old = f'outputNRD{name}RadianceHitDist[pixel] = float4(invSpp * {value}, {old_alpha});'
    assert s.count(old) == 1
    s = s.replace(old, f'outputNRD{name}RadianceHitDist[pixel] = float4(invSpp * {value}, nrd4LobeHitDist[{i}]);')
dest.write_text(s)
(root/'build/nrd4-metadata-patch.json').write_text(json.dumps({
    'scope':'Runtime shader auxiliary NRD alpha only; RGB and path RNG unchanged',
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'runtime_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
    'spp_contract':1},indent=2))
print('NRD4 auxiliary lobe distances prepared')
