param([switch]$BuildSDK)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$nrd = Join-Path $repo 'build/NRD4'
$falcor = Join-Path $repo 'build/Falcor8'
if (-not (Test-Path -LiteralPath (Join-Path $nrd 'Include/NRD.h'))) {
    & git clone --depth 1 https://github.com/NVIDIA-RTX/NRD.git $nrd
    if ($LASTEXITCODE -ne 0) { throw 'NRD4 clone failed.' }
}
$commit = & git -C $nrd rev-parse HEAD
if ($commit -ne '90dea8bd91ec6b58d0f90cb6adf53c7da613b82c') { throw "Expected tested NRD 4.18 commit, found $commit" }
$sdkDll = Join-Path $nrd '_Bin/Release/NRD.dll'
if ($BuildSDK -or -not (Test-Path -LiteralPath $sdkDll)) {
    & cmake -S $nrd -B (Join-Path $nrd 'build') -G 'Visual Studio 17 2022' -A x64 '-DNRD_EMBEDS_DXBC_SHADERS=OFF' '-DNRD_EMBEDS_SPIRV_SHADERS=OFF' '-DNRD_EMBEDS_DXIL_SHADERS=ON' '-DNRD_NORMAL_ENCODING=2' '-DNRD_ROUGHNESS_ENCODING=1'
    if ($LASTEXITCODE -ne 0) { throw 'NRD4 SDK configuration failed.' }
    & cmake --build (Join-Path $nrd 'build') --config Release --target NRD --parallel 8
    if ($LASTEXITCODE -ne 0) { throw 'NRD4 SDK build failed.' }
}
$dest = Join-Path $falcor 'Source/RenderPasses/NRD4Pass'
New-Item -ItemType Directory -Path $dest -Force | Out-Null
Copy-Item -Path (Join-Path $PSScriptRoot 'falcor_nrd4_plugin/*') -Destination $dest
$cmake = Join-Path $falcor 'Source/RenderPasses/CMakeLists.txt'
if (-not ([IO.File]::ReadAllText($cmake)).Contains('add_subdirectory(NRD4Pass)')) { [IO.File]::AppendAllText($cmake,"`nadd_subdirectory(NRD4Pass)`n") }
$bin = Join-Path $falcor 'build/windows-vs2022/bin/Release'
Copy-Item -LiteralPath $sdkDll -Destination (Join-Path $bin 'NRD4.dll') -Force
$shader = Join-Path $bin 'shaders/nrd4'
New-Item -ItemType Directory -Path $shader -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $nrd 'Shaders/NRD.hlsli'), (Join-Path $nrd 'Shaders/NRDConfig.hlsli') -Destination $shader -Force
& cmake --build (Join-Path $falcor 'build/windows-vs2022') --config Release --target NRD4Pass --parallel 8
if ($LASTEXITCODE -ne 0) { throw 'NRD4 Falcor plugin build failed.' }
