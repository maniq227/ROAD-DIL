param(
    [string]$DatasetRoot = "YOLO-ER/datasets",
    [string]$OutputsRoot = "YOLO-NAIVE/optimal_output",
    [string[]]$Domains = @("sunny","overcast","snowy","night"),
    [string]$Model = "yolo12s.pt",
    [int]$ImgSz = 640
)

Write-Host "[SETUP] Preparing YOLO-NAIVE dry run layout..."

New-Item -ItemType Directory -Force -Path $OutputsRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutputsRoot 'models') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutputsRoot 'tensorboard_logs') | Out-Null

$namesPath = Join-Path (Split-Path -Parent $DatasetRoot) 'configs/names.json'
if (-not (Test-Path $namesPath)) {
    Write-Error "names.json not found at $namesPath. Run YOLO-ER conversion first."
    exit 1
}

$classNames = Get-Content $namesPath -Raw | ConvertFrom-Json

foreach ($i in 1..$Domains.Count) {
    $domain = $Domains[$i-1]
    $expDir = Join-Path $OutputsRoot ("exp{0}_{1}" -f $i, $domain)
    New-Item -ItemType Directory -Force -Path $expDir | Out-Null

    $trainList = Join-Path (Join-Path $DatasetRoot $domain) 'train.txt'
    $valList = Join-Path (Join-Path $DatasetRoot $domain) 'val.txt'
    if (-not (Test-Path $trainList) -or -not (Test-Path $valList)) {
        Write-Error "Missing train/val lists for domain '$domain' under $DatasetRoot"
        exit 1
    }

    Copy-Item $trainList (Join-Path $expDir 'train.txt') -Force
    Copy-Item $valList (Join-Path $expDir 'val.txt') -Force

    $yamlPath = Join-Path $expDir 'data.yaml'
    $trainAbs = (Resolve-Path (Join-Path $expDir 'train.txt')).Path
    $valAbs = (Resolve-Path (Join-Path $expDir 'val.txt')).Path
    $namesJson = ($classNames | ConvertTo-Json -Compress)
    @(
        "train: $trainAbs",
        "val: $valAbs",
        "names: $namesJson"
    ) | Set-Content -Encoding UTF8 $yamlPath

    # Per-seen-domain val-only yamls will be created by the Python script at runtime.
}

"" | Set-Content -Encoding UTF8 (Join-Path $OutputsRoot 'continual_eval_matrix.csv')
"[]" | Set-Content -Encoding UTF8 (Join-Path $OutputsRoot 'continual_eval_matrix.json')

Write-Host "[OK] YOLO-NAIVE dry run files prepared in $OutputsRoot"


