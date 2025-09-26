param(
    [string]$DatasetRoot = "YOLO-ER/datasets",
    [string]$OutputsRoot = "YOLO-GDUMB/optimal_output",
    [string]$Model = "yolo12s.pt",
    [int]$Batch = 64,
    [int]$Epochs = 20,
    [int]$ImgSz = 640,
    [int]$MemSize = 10000
)

$ErrorActionPreference = 'Stop'

Write-Host "Preparing YOLO-GDUMB dry setup..." -ForegroundColor Green

if (-not (Test-Path -LiteralPath $OutputsRoot)) { New-Item -ItemType Directory -Path $OutputsRoot | Out-Null }
if (-not (Test-Path -LiteralPath (Join-Path $OutputsRoot 'models'))) { New-Item -ItemType Directory -Path (Join-Path $OutputsRoot 'models') | Out-Null }
if (-not (Test-Path -LiteralPath (Join-Path $OutputsRoot 'tensorboard_logs'))) { New-Item -ItemType Directory -Path (Join-Path $OutputsRoot 'tensorboard_logs') | Out-Null }

python YOLO-GDUMB\scripts\gdumb_yolo_v12.py `
  --dataset_root $DatasetRoot `
  --outputs_root $OutputsRoot `
  --model $Model `
  --batch $Batch `
  --epochs $Epochs `
  --imgsz $ImgSz `
  --mem_size $MemSize `
  --dry_run

Write-Host "Dry setup done. Lists/YAMLs prepared under $OutputsRoot" -ForegroundColor Cyan


