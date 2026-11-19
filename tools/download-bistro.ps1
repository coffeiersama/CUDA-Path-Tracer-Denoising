param(
    [string]$Blender = 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe',
    [int]$TextureSize = 1024
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$archive = Join-Path $repo 'build/Bistro_v5_2.zip'
$source = Join-Path $repo 'scenes/Models/bistro/source'
$fbx = Join-Path $source 'Bistro_v5_2/BistroInterior.fbx'
$obj = Join-Path $repo 'scenes/Models/bistro/converted/bistro.obj'
if (-not (Test-Path -LiteralPath $Blender)) { throw "Blender executable not found: $Blender" }
New-Item -ItemType Directory -Force -Path (Split-Path $archive), $source | Out-Null
if (-not (Test-Path -LiteralPath $archive)) {
    & curl.exe -L --fail --retry 2 --max-time 600 -o $archive 'https://developer.nvidia.com/bistro'
    if ($LASTEXITCODE -ne 0) { throw 'Bistro download failed.' }
}
$expectedHash = '0D50E3C724C6C5DA19F8EB99AD3F53E36FEC37FFA2DF9621F9CCF0603F3934E1'
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedHash) {
    throw 'Archive differs from the verified Bistro v5.2 package. Check the download before continuing.'
}
if (-not (Test-Path -LiteralPath $fbx)) { Expand-Archive -LiteralPath $archive -DestinationPath $source -Force }
& $Blender --background --factory-startup --python (Join-Path $PSScriptRoot 'export_bistro.py') -- $fbx $obj
if ($LASTEXITCODE -ne 0) { throw 'FBX conversion failed.' }
& python (Join-Path $PSScriptRoot 'prepare_bistro.py') $obj --texture-size $TextureSize
if ($LASTEXITCODE -ne 0) { throw 'Texture conversion failed. Python requires Pillow.' }
Write-Output 'Bistro is ready. From the repository root, run: .\run.ps1 -Scene bistro'
