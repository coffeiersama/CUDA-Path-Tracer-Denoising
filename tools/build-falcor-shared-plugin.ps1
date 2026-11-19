$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$falcor = Join-Path $repo 'build/Falcor8'
$dest = Join-Path $falcor 'Source/RenderPasses/SharedInputGuides'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'falcor_shared_plugin/CMakeLists.txt'), (Join-Path $PSScriptRoot 'falcor_shared_plugin/SharedInputGuides.cpp'), (Join-Path $PSScriptRoot 'falcor_shared_plugin/SharedInputGuides.cs.slang') -Destination $dest
$cmake = Join-Path $falcor 'Source/RenderPasses/CMakeLists.txt'
$content = [IO.File]::ReadAllText($cmake)
if (-not $content.Contains('add_subdirectory(SharedInputGuides)')) {
    [IO.File]::AppendAllText($cmake, "`nadd_subdirectory(SharedInputGuides)`n")
}
& cmake --build (Join-Path $falcor 'build/windows-vs2022') --config Release --target SharedInputGuides NRDPass ModulateIllumination --parallel 8
if ($LASTEXITCODE -ne 0) { throw 'Falcor shared-input adapter build failed.' }
