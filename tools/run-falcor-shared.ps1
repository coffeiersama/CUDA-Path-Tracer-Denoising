param(
    [int]$Width = 640,
    [int]$Height = 360,
    [int]$Frames = 128,
    [int]$ReferenceSamples = 4096,
    [string]$OutputDirectory = 'build/falcor-shared-final'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $repo 'build/Falcor8/build/windows-vs2022/bin/Release/Mogwai.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build official Falcor first with tools/run-falcor-courtyard.ps1 -Build.' }
if ($Width -lt 2 -or $Height -lt 2 -or $Frames -lt 1 -or $ReferenceSamples -lt 0) { throw 'Invalid capture dimensions or sample counts.' }
if (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory = Join-Path $repo $OutputDirectory }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$values = @{
    FALCOR_COURTYARD_ROOT = $repo
    FALCOR_COURTYARD_ENVIRONMENT_INTENSITY = '8.0'
    FALCOR_SHARED_OUTPUT = $OutputDirectory
    FALCOR_SHARED_WIDTH = "$Width"
    FALCOR_SHARED_HEIGHT = "$Height"
    FALCOR_SHARED_FRAMES = "$Frames"
    FALCOR_SHARED_REFERENCE = "$ReferenceSamples"
}
$previous = @{}
foreach ($key in $values.Keys) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
    [Environment]::SetEnvironmentVariable($key, $values[$key], 'Process')
}
try {
    & $exe --gpu 0 --headless --script (Join-Path $PSScriptRoot 'falcor_shared_inputs.py')
    if ($LASTEXITCODE -ne 0) { throw "Falcor shared capture failed: $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath (Join-Path $OutputDirectory 'run.json'))) { throw 'Falcor did not complete the export.' }
    & (Join-Path $repo 'build/bin/Release/falcor_replay.exe') $OutputDirectory (Join-Path $OutputDirectory 'cuda')
    if ($LASTEXITCODE -ne 0) { throw 'CUDA shared-input replay failed.' }
    if ($ReferenceSamples -gt 0) {
        & (Join-Path $repo 'build/audit-env/Scripts/python.exe') (Join-Path $PSScriptRoot 'compare_falcor_shared.py') $OutputDirectory
        if ($LASTEXITCODE -ne 0) { throw 'Shared-input validation or comparison failed.' }
    }
} finally {
    foreach ($key in $values.Keys) { [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process') }
}
