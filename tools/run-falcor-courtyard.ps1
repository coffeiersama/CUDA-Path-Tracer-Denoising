param(
    [switch]$Build,
    [switch]$Headless,
    [int]$Width = 1280,
    [int]$Height = 720,
    [int]$Gpu = 0,
    [double]$ExposureEV = 2.0,
    [double]$EnvironmentIntensity = 8.0,
    [int[]]$CaptureFrames = @(1, 64, 128),
    [string]$OutputDirectory = 'build/falcor-demo'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$falcorRoot = Join-Path $repo 'build/Falcor8'
$falcorBuild = Join-Path $falcorRoot 'build/windows-vs2022'
$exe = Join-Path $falcorBuild 'bin/Release/Mogwai.exe'
if ($Width -lt 1 -or $Height -lt 1 -or $CaptureFrames.Count -lt 1 -or ($CaptureFrames | Where-Object { $_ -lt 1 })) { throw 'Invalid image dimensions or capture frame list.' }
if (-not (Test-Path -LiteralPath (Join-Path $repo 'scenes/Models/san-miguel/source/san-miguel-low-poly.obj'))) {
    throw 'Download San Miguel first: .\tools\download-san-miguel.ps1'
}
if ($Build -or -not (Test-Path -LiteralPath $exe)) {
    if (-not (Test-Path -LiteralPath (Join-Path $falcorRoot 'CMakeLists.txt'))) {
        & git -c core.longpaths=true clone --depth 1 --branch 8.0 --recurse-submodules 'https://github.com/NVIDIAGameWorks/Falcor.git' $falcorRoot
        if ($LASTEXITCODE -ne 0) { throw 'Falcor clone failed.' }
    }
    $commit = & git -C $falcorRoot rev-parse HEAD
    if ($commit -ne '9dc819c162b2070335c65060436041690b7937f8') { throw "Expected official Falcor 8.0, found $commit" }
    & cmake -S $falcorRoot -B $falcorBuild -G 'Visual Studio 17 2022' -A x64 -T host=x64 '-DFALCOR_ENABLE_USD=OFF' '-DCMAKE_CUDA_COMPILER=NOTFOUND'
    if ($LASTEXITCODE -ne 0) { throw 'Falcor configuration failed.' }
    & cmake --build $falcorBuild --config Release --target Mogwai --parallel 8
    if ($LASTEXITCODE -ne 0) { throw 'Falcor build failed.' }
}
if (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory = Join-Path $repo $OutputDirectory }
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$values = @{
    FALCOR_COURTYARD_ROOT = $repo
    FALCOR_COURTYARD_OUTPUT = $OutputDirectory
    FALCOR_COURTYARD_WIDTH = "$Width"
    FALCOR_COURTYARD_HEIGHT = "$Height"
    FALCOR_COURTYARD_EXPOSURE_EV = $ExposureEV.ToString([Globalization.CultureInfo]::InvariantCulture)
    FALCOR_COURTYARD_ENVIRONMENT_INTENSITY = $EnvironmentIntensity.ToString([Globalization.CultureInfo]::InvariantCulture)
    FALCOR_COURTYARD_FRAMES = ($CaptureFrames -join ',')
    FALCOR_COURTYARD_LIVE = $(if ($Headless) { '0' } else { '1' })
}
$previous = @{}
foreach ($key in $values.Keys) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
    [Environment]::SetEnvironmentVariable($key, $values[$key], 'Process')
}
try {
    $rendererArgs = @('--gpu', "$Gpu", '--script', (Join-Path $PSScriptRoot 'falcor_courtyard.py'))
    if ($Headless) { $rendererArgs += '--headless' }
    & $exe @rendererArgs
    if ($LASTEXITCODE -ne 0) { throw "Mogwai exited with code $LASTEXITCODE" }
} finally {
    foreach ($key in $values.Keys) { [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process') }
}
