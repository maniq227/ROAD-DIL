### YOLO-ER: Setup and Experience Replay Strategy

#### Overview
- All YOLO work is isolated in `YOLO-ER/` (scripts, configs, datasets, outputs).
- Uses real ROAD annotations under `road-real-annotations/data` (10 Active Agent classes), and writes outputs only to `YOLO-ER/optimal_output/`.

#### Environment
- Quick install (PowerShell):
```powershell
pwsh YOLO-ER/scripts/00_setup_env.ps1 -CondaEnv avalanche-env -CudaIndex https://download.pytorch.org/whl/cu121
```
- Verify:
```powershell
yolo version
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
```

#### Data conversion (COCO → YOLO)
- Convert per-domain COCO (real) to YOLO:
```powershell
python YOLO-ER/scripts/01_convert_coco_to_yolo.py --coco_root road-real-annotations/data --out_root YOLO-ER/datasets
```
- Produces:
  - Labels: `road-real-annotations/data/<domain>/labels/*.txt`
  - Splits: `YOLO-ER/datasets/<domain>/{train,val}.txt` (absolute paths)
  - Eval YAML: `YOLO-ER/datasets/<domain>/data_val_only.yaml`
  - Class names: `YOLO-ER/configs/names.json`

#### Model/variants
- Default: YOLOv12 small (`--model .\yolo12s.pt`).
- Alternatives: `yolo12n.pt` (≤8GB), `yolo12m.pt` (≥12GB), `yolo12l/x.pt` (≥24GB).

#### Experience Replay (ER) strategy
- Scenario: domain-incremental `sunny → overcast → night → snowy`.
- Per-experience flow:
  1) Merge current-domain train with replay-buffer images.
  2) Write per-experience `data.yaml` with absolute `train/val` (no `path:`).
  3) Train for configured epochs; save best to `YOLO-ER/optimal_output/models/exp{idx}_{domain}_best.pt`.
  4) Update replay buffer (image-level reservoir sampling).
  5) Evaluate on all seen domains; append mAP50-95 to `YOLO-ER/optimal_output/continual_eval_matrix.csv/json`.
- Replay buffer:
  - `--mem_size` (default 5000 images).
  - Reservoir sampling (uses Avalanche `ReservoirSamplingBuffer` if available; else built-in equivalent).
- Defaults: `epochs=30`, `batch=16`, `imgsz=640`, `patience=5`, `device=0`.

#### How to run

- Smoke test (sunny only, quick):
```powershell
python YOLO-ER/scripts/er_yolo_v12.py `
  --domains sunny `
  --model .\yolo12s.pt `
  --epochs 3 --batch 12 --imgsz 640 --patience 2 --mem_size 0 `
  --dataset_root YOLO-ER/datasets `
  --outputs_root YOLO-ER/optimal_output `
  --device 0 2>&1 | Tee-Object -FilePath YOLO-ER\optimal_output\training_output.log
```
  - Tail live:
```powershell
Get-Content YOLO-ER\optimal_output\training_output.log -Wait -Tail 20
```
  - Expect mAP50-95 > 0 (typical for smoke: ~0.6 on sunny with 3 epochs).

- Full ER run (4 domains):
```powershell
python YOLO-ER/scripts/er_yolo_v12.py `
  --domains sunny overcast night snowy `
  --model .\yolo12s.pt `
  --epochs 30 --batch 16 --imgsz 640 --patience 5 --mem_size 5000 `
  --dataset_root YOLO-ER/datasets `
  --outputs_root YOLO-ER/optimal_output `
  --device 0 2>&1 | Tee-Object -FilePath YOLO-ER\optimal_output\training_output_full.log
```

#### Monitoring and artifacts
- Live tail (epoch/iter/GPU details if using Tee-Object):
```powershell
Get-Content YOLO-ER\outputs\training_output_full.log -Wait -Tail 20
```
- Epoch summaries:
```powershell
Get-Content YOLO-ER\optimal_output\eta_progress.log -Wait -Tail 20
```
- Artifacts:
  - Models: `YOLO-ER/optimal_output/models/exp{N}_{domain}_best.pt`
  - Matrix: `YOLO-ER/optimal_output/continual_eval_matrix.csv` and `.json`
  - Summary: `YOLO-ER/optimal_output/continual_learning_metrics.json`
  - Ultralytics run assets: `YOLO-ER/optimal_output/runs/*`

#### Post-run summary
- Aggregate metrics:
```powershell
python YOLO-ER/scripts/eval_matrix.py --inputs_root YOLO-ER/optimal_output --domains sunny overcast night snowy
Get-Content YOLO-ER\optimal_output\continual_learning_metrics.json
```

#### Notes
- Always use real annotations under `road-real-annotations/data`.
- `data.yaml` files use absolute `train/val` paths and no `path:` to avoid path-join issues on Windows.