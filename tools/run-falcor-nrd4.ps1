param([int]$Frames = 128,[string]$OutputDirectory='build/falcor-nrd4',[switch]$DebugLayer)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$exe=Join-Path $repo 'build/Falcor8/build/windows-vs2022/bin/Release/Mogwai.exe'
if (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory=Join-Path $repo $OutputDirectory }
$values=@{ FALCOR_COURTYARD_ROOT=$repo; FALCOR_COURTYARD_ENVIRONMENT_INTENSITY='8.0'; FALCOR_NRD4_OUTPUT=$OutputDirectory; FALCOR_NRD4_FRAMES="$Frames" }
$previous=@{}
foreach($key in $values.Keys) { $previous[$key]=[Environment]::GetEnvironmentVariable($key,'Process'); [Environment]::SetEnvironmentVariable($key,$values[$key],'Process') }
try {
    & (Join-Path $repo 'build/audit-env/Scripts/python.exe') (Join-Path $PSScriptRoot 'prepare_nrd4_metadata.py')
    if ($LASTEXITCODE -ne 0) { throw 'NRD4 auxiliary distance preparation failed.' }
    $rendererArgs=@('--gpu','0','--headless','--script',(Join-Path $PSScriptRoot 'falcor_nrd4_shared.py'))
    if ($DebugLayer) { $rendererArgs += '--enable-debug-layer' }
    & $exe @rendererArgs
    if ($LASTEXITCODE -ne 0) { throw "NRD4 run failed: $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath (Join-Path $OutputDirectory 'run.json'))) { throw 'NRD4 run did not complete.' }
    if ($Frames -eq 128) {
        & (Join-Path $repo 'build/audit-env/Scripts/python.exe') (Join-Path $PSScriptRoot 'compare_nrd4.py') $OutputDirectory
        if ($LASTEXITCODE -ne 0) { throw 'NRD4 comparison validation failed.' }
    }
} finally {
    # Isolate the auxiliary-alpha adaptation to this run. Legacy Falcor/NRD3
    # launchers must keep their original runtime shader convention.
    Copy-Item -LiteralPath (Join-Path $repo 'build/Falcor8/Source/RenderPasses/PathTracer/ResolvePass.cs.slang') -Destination (Join-Path $repo 'build/Falcor8/build/windows-vs2022/bin/Release/shaders/RenderPasses/PathTracer/ResolvePass.cs.slang') -Force
    foreach($key in $values.Keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
}
