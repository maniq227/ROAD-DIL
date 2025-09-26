param(
    [string]$CondaEnv = "",
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu121"
)

$ErrorActionPreference = "Stop"

Write-Host "[SETUP] Installing packages for Ultralytics YOLO + Avalanche" -ForegroundColor Cyan

if ($CondaEnv -ne "") {
    Write-Host "[SETUP] Using conda env: $CondaEnv" -ForegroundColor Yellow
}

function RunPip {
    param([string]$Args)
    if ($CondaEnv -ne "") {
        conda run -n $CondaEnv python -m pip $Args
    } else {
        python -m pip $Args
    }
}

RunPip "install --upgrade pip setuptools wheel"

# Install torch first to ensure correct CUDA
RunPip "install --index-url $CudaIndex torch torchvision torchaudio"

# Ultralytics + optional tools
RunPip "install ultralytics opencv-python tqdm tensorboard"

# Avalanche and COCO tools
RunPip "install avalanche-lib pycocotools"

Write-Host "[SETUP] Done." -ForegroundColor Green


