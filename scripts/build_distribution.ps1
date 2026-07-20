param(
    [string]$OutputDirectory = "dist\AviationVisionApp"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destination = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
$distRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "dist"))

if (-not $destination.StartsWith($distRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must be located under $distRoot."
}

if (Test-Path -LiteralPath $destination) {
    Remove-Item -LiteralPath $destination -Recurse -Force
}
New-Item -ItemType Directory -Path $destination -Force | Out-Null

$files = @(
    "streamlit_app.py",
    "requirements.txt",
    "packages.txt",
    "README.md",
    "setup_windows.bat",
    "run_app.bat",
    "start_folder_sync.bat",
    "yolov8n.pt"
)
$directories = @(
    ".streamlit",
    "assets",
    "configs",
    "src",
    "models",
    "local_uploader",
    "supabase",
    "docs",
    "scripts",
    "tests"
)

foreach ($file in $files) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $file) -Destination $destination
}
foreach ($directory in $directories) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $directory) -Destination $destination -Recurse
}

Get-ChildItem -Path $destination -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
}
Get-ChildItem -Path $destination -Recurse -File -Include "*.pyc", "secrets.toml", "uploader.toml", ".uploader_state.json" | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Force
}

$zipPath = "$destination.zip"
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path (Join-Path $destination "*") -DestinationPath $zipPath -CompressionLevel Optimal
Write-Host "Distribution package created: $zipPath"
