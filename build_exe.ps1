$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Python venv not found at $pythonExe"
}

$distDir = Join-Path $projectRoot "dist"
if (Test-Path $distDir) {
    Remove-Item $distDir -Recurse -Force
}

& $pythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --name "Sesli" `
    --windowed `
    --add-data "templates;templates" `
    --add-data "static;static" `
    --add-data "tr_TR-dfki-medium.onnx;." `
    --add-data "tr_TR-dfki-medium.onnx.json;." `
    --hidden-import "flask" `
    --hidden-import "yt_dlp" `
    --hidden-import "faster_whisper" `
    --hidden-import "transformers" `
    --hidden-import "torch" `
    --hidden-import "pydub" `
    --hidden-import "piper" `
    app.py

Write-Host ""
Write-Host "Build complete. EXE output is in: $projectRoot\dist\Sesli\"
