param([int]$TextureSize = 1024)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$archive = Join-Path $repo 'build/San_Miguel.zip'
$source = Join-Path $repo 'scenes/Models/san-miguel/source'
$destination = Join-Path $repo 'scenes/Models/san-miguel/converted'
New-Item -ItemType Directory -Force -Path (Split-Path $archive), $source | Out-Null
if (-not (Test-Path -LiteralPath $archive)) {
    & curl.exe -L --fail --retry 2 --max-time 600 -o $archive 'https://casual-effects.com/g3d/data10/research/model/San_Miguel/San_Miguel.zip'
    if ($LASTEXITCODE -ne 0) { throw 'San Miguel download failed.' }
}
$expectedHash = '85874077735808150E679B3C71D70A37A270CB8833F4911325AA1099DA3F7D4A'
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedHash) {
    throw 'Archive differs from the verified San Miguel package. Check the download before continuing.'
}
if (-not (Test-Path -LiteralPath (Join-Path $source 'san-miguel-low-poly.obj'))) {
    & python -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' $archive $source
    if ($LASTEXITCODE -ne 0) { throw 'San Miguel extraction failed.' }
}
& python (Join-Path $PSScriptRoot 'prepare_san_miguel.py') $source $destination --texture-size $TextureSize
if ($LASTEXITCODE -ne 0) { throw 'Scene preparation failed. Python requires Pillow.' }
Write-Output 'San Miguel is ready. Run: .\run.ps1 -Scene san-miguel'
