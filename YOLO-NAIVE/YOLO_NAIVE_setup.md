## YOLO-NAIVE: Naive Fine-tuning with Ultralytics YOLOv12 (Domain-Incremental)

### Overview
- Scripts live in `YOLO-NAIVE/scripts/`, outputs under `YOLO-NAIVE/optimal_output/`.
- Consumes real ROAD annotations converted to YOLO lists under `YOLO-ER/datasets/`.
- Strategy: Naive sequential fine-tuning across domains (no replay or regularization).
- Evaluates mAP50–95 on all seen domains after each experience and writes a matrix.

### Prerequisites
1) Install environment (same as YOLO-ER). Verify:
```powershell
yolo version
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
```
2) Convert real ROAD COCO → YOLO lists (absolute paths; no `path:` key):
```powershell
python YOLO-ER\scripts\01_convert_coco_to_yolo.py --coco_root road-real-annotations\data --out_root YOLO-ER\datasets
```

### Run (dry-run prep only; does not start training)
```powershell
pwsh YOLO-NAIVE\scripts\setup_dry_run.ps1
```

### Train + Evaluate (when ready)
```powershell
python YOLO-NAIVE\scripts\naive_yolo_v12.py `
  --dataset_root YOLO-ER\datasets `
  --outputs_root YOLO-NAIVE\optimal_output `
  --domains sunny overcast snowy night `
  --model yolo12s.pt `
  --epochs 30 --batch 48 --imgsz 640 --patience 5 `
  --workers 0 --device 0
```
Notes:
- If VRAM is tight, reduce `--batch` instead of using `--accumulate` (your Ultralytics install rejects it in overrides).
- Caching: `--cache disk` by default. Use `--cache ram` only if you have sufficient host memory.

### Metrics captured
- Per-experience row includes both `mAP50-95` and `mAP50` for each seen domain.
- Aggregates: `avg_seen_mAP50_95` and `avg_seen_mAP50`.

### Live monitoring
- Terminal emits per-epoch and per-iteration ETA with GPU memory:
  - EPOCH_START/EPOCH_END lines
  - ITER lines: `iter i/total, epoch e/E, imgs/s, GPUmem, remaining seconds`
- TensorBoard:
```powershell
tensorboard --logdir YOLO-NAIVE\optimal_output\tensorboard_logs --port 6006
```
- Tail ETA log (PowerShell):
```powershell
Get-Content YOLO-NAIVE\outputs\eta_progress.log -Wait
```

### Artifacts
- `YOLO-NAIVE/optimal_output/continual_eval_matrix.{json,csv}`
- `YOLO-NAIVE/optimal_output/continual_learning_metrics.json`
- `YOLO-NAIVE/optimal_output/models/exp{i}_{domain}_best.pt`
- `YOLO-NAIVE/optimal_output/tensorboard_logs/`, `YOLO-NAIVE/optimal_output/eta_progress.log`

### Domains
- Default order: `sunny → overcast → snowy → night`


