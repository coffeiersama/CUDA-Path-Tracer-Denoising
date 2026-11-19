param(
    [string]$Scene = 'san-miguel-benchmark',
    [string]$OutputDirectory = 'build/svgf-audit-reproduced',
    [int]$Frames = 128,
    [int]$ReferenceHalfSamples = 1024,
    [switch]$Motion,
    [switch]$Replay,
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if ($Frames -lt 1 -or $ReferenceHalfSamples -lt 1) { throw 'Frame/sample counts must be positive.' }
$exe = Join-Path $repo 'build/bin/Release/cis565_path_tracer.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build Release first.' }
if (-not $Python) { $Python = Join-Path $repo 'build/audit-env/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Create build/audit-env and install tools/audit-requirements.txt, or pass -Python.' }
if (Test-Path -LiteralPath $Scene -PathType Leaf) { $scenePath = (Resolve-Path -LiteralPath $Scene).Path }
else { $scenePath = Join-Path $repo "scenes/$Scene.txt" }
if (-not (Test-Path -LiteralPath $scenePath)) { throw "Scene not found: $scenePath" }
if (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory = Join-Path $repo $OutputDirectory }
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
function Render-Audit([string]$InputScene, [string]$Name, [string[]]$Options) {
    Write-Output "Rendering $Name"
    & $exe $InputScene --hidden --no-vsync @Options --capture (Join-Path $OutputDirectory $Name) > (Join-Path $OutputDirectory "$Name.log") 2>&1
    if ($LASTEXITCODE -ne 0) { Get-Content (Join-Path $OutputDirectory "$Name.log") -Tail 20; throw "Rendering $Name failed." }
}
$beforeOptions = @('--frames', "$Frames", '--denoiser', 'previous')
if ($Replay) { $beforeOptions += @('--record', (Join-Path $OutputDirectory 'stationary.seq')) }
Render-Audit $scenePath 'before' $beforeOptions
Render-Audit $scenePath 'after' @('--frames', "$Frames", '--denoiser', 'svgf')
Render-Audit $scenePath 'ref-a' @('--reference', "$ReferenceHalfSamples", '--seed-offset', '100000')
Render-Audit $scenePath 'ref-b' @('--reference', "$ReferenceHalfSamples", '--seed-offset', '200000')
& $Python (Join-Path $PSScriptRoot 'evaluate_svgf.py') $OutputDirectory --references (Join-Path $OutputDirectory 'ref-a-reference.pfm') (Join-Path $OutputDirectory 'ref-b-reference.pfm') > (Join-Path $OutputDirectory 'metrics.log')
if ($LASTEXITCODE -ne 0) { throw 'Static evaluation failed.' }
if ($Replay) {
    & (Join-Path $repo 'build/bin/Release/denoise_replay.exe') (Join-Path $OutputDirectory 'stationary.seq') (Join-Path $OutputDirectory 'ref-a-reference.pfm') (Join-Path $OutputDirectory 'ref-b-reference.pfm') (Join-Path $OutputDirectory 'replay')
    if ($LASTEXITCODE -ne 0) { throw 'Replay evaluation failed.' }
}
if ($Motion) {
    $motionOptions = @('--frames', "$Frames", '--auto-camera', '--fixed-dt', '0.016666667')
    Render-Audit $scenePath 'motion-before' ($motionOptions + @('--denoiser', 'previous'))
    Render-Audit $scenePath 'motion-after' ($motionOptions + @('--denoiser', 'svgf'))
    $pose = Join-Path $OutputDirectory 'motion-pose.txt'
    & $Python (Join-Path $PSScriptRoot 'make_pose_scene.py') $scenePath (Join-Path $OutputDirectory 'motion-after.json') $pose
    if ($LASTEXITCODE -ne 0) { throw 'Motion pose construction failed.' }
    Render-Audit $pose 'motion-ref-a' @('--reference', "$ReferenceHalfSamples", '--seed-offset', '100000')
    Render-Audit $pose 'motion-ref-b' @('--reference', "$ReferenceHalfSamples", '--seed-offset', '200000')
    & $Python (Join-Path $PSScriptRoot 'evaluate_svgf.py') $OutputDirectory --before motion-before --after motion-after --references (Join-Path $OutputDirectory 'motion-ref-a-reference.pfm') (Join-Path $OutputDirectory 'motion-ref-b-reference.pfm') --title 'San Miguel moving camera, final frame' > (Join-Path $OutputDirectory 'motion-metrics.log')
    if ($LASTEXITCODE -ne 0) { throw 'Motion evaluation failed.' }
}
Write-Output "Audit finished: $OutputDirectory"
