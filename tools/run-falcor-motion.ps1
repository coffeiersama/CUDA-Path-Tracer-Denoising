param([int]$Width=640,[int]$Height=360,[int]$Warmup=48,[int]$Frames=240,[int]$Stride=3,[string]$OutputDirectory='build/falcor-motion')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if ($Width -lt 2 -or $Height -lt 2 -or $Warmup -lt 1 -or $Frames -lt 1 -or $Stride -lt 1) { throw 'Invalid animation capture configuration.' }
if (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory=Join-Path $repo $OutputDirectory }
$values=@{FALCOR_COURTYARD_ROOT=$repo;FALCOR_COURTYARD_ENVIRONMENT_INTENSITY='8.0';FALCOR_MOTION_OUTPUT=$OutputDirectory;
    FALCOR_MOTION_WIDTH="$Width";FALCOR_MOTION_HEIGHT="$Height";FALCOR_MOTION_WARMUP="$Warmup";FALCOR_MOTION_FRAMES="$Frames";FALCOR_MOTION_STRIDE="$Stride"}
$previous=@{}
foreach($key in $values.Keys) { $previous[$key]=[Environment]::GetEnvironmentVariable($key,'Process');[Environment]::SetEnvironmentVariable($key,$values[$key],'Process') }
try {
    & (Join-Path $repo 'build/Falcor8/build/windows-vs2022/bin/Release/Mogwai.exe') --gpu 0 --headless --script (Join-Path $PSScriptRoot 'falcor_motion_capture.py')
    if ($LASTEXITCODE -ne 0) { throw 'Falcor motion capture failed.' }
    if (-not (Test-Path -LiteralPath (Join-Path $OutputDirectory 'run.json'))) { throw 'Motion capture did not complete.' }
    & (Join-Path $repo 'build/bin/Release/falcor_motion_replay.exe') (Join-Path $OutputDirectory 'motion.seq') $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw 'CUDA motion replay failed.' }
} finally { foreach($key in $values.Keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') } }
