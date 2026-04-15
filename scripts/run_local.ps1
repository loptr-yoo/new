param(
  [Parameter(Mandatory = $true)]
  [Alias("p")]
  [string]$Prompt,

  [Parameter(Mandatory = $true)]
  [Alias("m")]
  [string]$Model,

  [ValidateSet("north", "center", "south")]
  [Alias("c")]
  [string]$Core = "north",

  [string]$OutDir = "out",
  [string]$ProjectName = $(Get-Date -Format "yyyyMMdd_HHmmss")
)

$ErrorActionPreference = "Stop"

$TargetDir = Join-Path $OutDir $ProjectName
New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

$layoutJsonBase = Join-Path $TargetDir "layout.json"
$baseName = [System.IO.Path]::GetFileNameWithoutExtension($layoutJsonBase)

Write-Host "Starting headless generation..." -ForegroundColor Cyan
Write-Host "Output directory: $TargetDir" -ForegroundColor Cyan

python scripts/cli_runner.py -p "$Prompt" -m "$Model" -c "$Core" -o "$layoutJsonBase"

$jsonFiles = Get-ChildItem -Path $TargetDir -Filter "$($baseName)_*.json" -ErrorAction SilentlyContinue

if ($null -eq $jsonFiles -or $jsonFiles.Count -eq 0) {
  Write-Host "No per-floor JSON files found ($($baseName)_*.json). Rendering layout.json directly." -ForegroundColor Yellow

  $pngOut = Join-Path $TargetDir "${baseName}.png"
  $svgOut = Join-Path $TargetDir "${baseName}.svg"

  python scripts/local_renderer.py -i "$layoutJsonBase" -o "$pngOut"
  python scripts/local_renderer.py -i "$layoutJsonBase" -o "$svgOut"

  Write-Host "Done. See: $TargetDir" -ForegroundColor Green
  exit 0
}

Write-Host "Found $($jsonFiles.Count) floor files. Rendering..." -ForegroundColor Cyan

foreach ($file in $jsonFiles) {
  $floorSuffix = $file.BaseName.Substring($baseName.Length)

  $pngOut = Join-Path $TargetDir "${baseName}${floorSuffix}.png"
  $svgOut = Join-Path $TargetDir "${baseName}${floorSuffix}.svg"

  Write-Host "Rendering: ${baseName}${floorSuffix} ..."
  python scripts/local_renderer.py -i "$($file.FullName)" -o "$pngOut"
  python scripts/local_renderer.py -i "$($file.FullName)" -o "$svgOut"
}

Write-Host "Done. See: $TargetDir" -ForegroundColor Green
