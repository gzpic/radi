param(
    [Parameter(Mandatory = $true)]
    [string]$Model,

    [int]$Runs = 5,

    [string]$BuildDir = "",

    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $RepoRoot "build"
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutDir = Join-Path $RepoRoot ("artifacts\agent-runtime-comparison\" + $ts)
}

if (-not (Test-Path $Model)) {
    throw "Model not found: $Model"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

# Ensure MSYS2 runtime DLLs are discoverable for test executables.
$env:PATH = "D:\SoftwareFilePlace\MSYS2\ucrt64\bin;$env:PATH"

Write-Host "== Build target: test-agent-runtime-planner-bench =="
cmake --build $BuildDir --target test-agent-runtime-planner-bench -j4

$Exe = Join-Path $BuildDir "bin\test-agent-runtime-planner-bench.exe"
if (-not (Test-Path $Exe)) {
    throw "Benchmark executable not found: $Exe"
}

Write-Host "== Run benchmark =="
Write-Host "Model: $Model"
Write-Host "Runs : $Runs"
Write-Host "Out  : $OutDir"

& $Exe $Model --runs $Runs --out $OutDir

Write-Host ""
Write-Host "Done."
Write-Host "Summary JSON: $(Join-Path $OutDir 'planner-comparison-summary.json')"
Write-Host "Summary MD  : $(Join-Path $OutDir 'planner-comparison-summary.md')"
