<<<<<<< HEAD
## ROAD-DIL: Domain-Incremental Learning for Road Scenes

Domain-incremental learning (DIL) experiments for road-scene object detection across sequential domains: sunny → overcast → night → snowy. This repo organizes baselines and metrics for three strategies:
- ER (Experience Replay)
- GDumb (Greedy Dumb)
- Naive (sequential fine-tuning)

Missing heavy artifacts (trained weights, full logs, large run folders) and real annotations are available via Google Drive:
[ADD GOOGLE DRIVE LINK HERE]

### Tech stack
- Python 3.10+
- PyTorch + Ultralytics YOLO (v12 family)
- Optional Avalanche (for ER variant)
- NumPy, Pandas, Matplotlib/Seaborn, TQDM
- TensorBoard (optional; training works even if TB is not installed)

### Conventions used in this project
- Dataset YAMLs use absolute `train`/`val` paths, and omit the `path` key to avoid Ultralytics root prefixing.
- We do not pass `accumulate` in Ultralytics `.train` overrides in this environment; tune batch size instead if needed.
- Each strategy writes curated artifacts under its own `optimal_output/` to avoid mixing results.

---

## Repository structure

Top-level tools
- `compare_standard_metrics.py`, `compare_extra_metrics.py`: Merge/compare metrics across strategies and export tables.
- `convert_road_to_coco.py`, `visualize_coco_annotations.py`: Data conversion/visualization utilities.
- `domain-shifts/`: Domain-shift analysis
  - `compute_domain_shifts.py`
  - `domain_shift_summary.json`, `domain_stats.csv` (small summaries, versioned)
  - `run.log` (optional to keep)
- `standard_metrics_comparison.{csv,json}`, `extra_metrics_comparison.{csv,json}`: Cross-strategy summary tables.
- `yolo12s.pt`: Base weights (large; excluded by default – see Drive link above).

Per-strategy layout
- `YOLO-ER/`
  - `configs/` (e.g., `names.json`)
  - `datasets/<domain>/*.yaml`: Absolute paths, no `path` key
  - `scripts/`: training + export utilities
    - `er_yolo_v12.py`, `er_yolo_avalanche.py`
    - `compute_cl_metrics_and_figs.py`, `eval_matrix.py`, `export_standard_metrics.py`
  - `optimal_output/`
    - Light artifacts (versioned): `continual_eval_matrix.{csv,json}`, `continual_learning_metrics.json`, `standard_metrics.json`, `analysis_extended_ap/*.json`
    - Report images in `runs/` (selected files versioned): `results.png`, `BoxPR_curve.png`, `confusion_matrix_normalized.png`, `labels.jpg`, `val_batch0_pred.jpg`, `train_batch0.jpg`, `args.yaml`, `results.csv`
    - Heavy artifacts (external via Drive): `runs/weights/*.pt`, full `runs/` dump, `tensorboard_logs/`, `models/*.pt`
- `YOLO-GDUMB/` and `YOLO-NAIVE/`: same pattern as `YOLO-ER/` with their own scripts and `optimal_output/`

Archives
- `Archives/` (historic/large bundles; excluded by default; see Drive link)

---

## Getting started

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install ultralytics numpy pandas matplotlib seaborn tqdm
# Install a matching torch/torchvision for your CUDA/CPU setup
# Optional ER variant:
# pip install avalanche-lib
```

### 2) Data and annotations
- Real annotations and any missing large folders are provided via Google Drive:
  [ADD GOOGLE DRIVE LINK HERE]
- Place datasets as referenced by the YAMLs in `YOLO-ER/datasets/<domain>/*.yaml`.
- Use absolute paths for `train`/`val` and omit the `path` key in YAMLs.

### 3) Run baselines (order: sunny → overcast → night → snowy)

ER
```bash
python YOLO-ER/scripts/er_yolo_v12.py
```

GDumb
```bash
python YOLO-GDUMB/scripts/gdumb_yolo_v12.py
```

Naive
```bash
python YOLO-NAIVE/scripts/naive_yolo_v12.py
```

### 4) Export metrics and figures

Per strategy
```bash
# Standard metrics (mAP, P, R, etc.)
python YOLO-ER/scripts/export_standard_metrics.py

# Continual-learning metrics & figures (FWT/BWT, matrices, PR/CM curves)
python YOLO-ER/scripts/compute_cl_metrics_and_figs.py
```

Cross-strategy comparisons at repo root
```bash
python compare_standard_metrics.py
python compare_extra_metrics.py
```

---

## What is versioned vs external

Included (in Git)
- Source code and configs (`*.py`, `*.yaml`, `configs/`)
- Lightweight metrics under `*/optimal_output/`: `continual_eval_matrix.{csv,json}`, `continual_learning_metrics.json`, `standard_metrics.json`, `analysis_extended_ap/*.json`
- Selected report images from `*/optimal_output/runs/`: `results.png`, `BoxPR_curve.png`, `confusion_matrix_normalized.png`, `labels.jpg`, `val_batch0_pred.jpg`, `train_batch0.jpg`, plus `args.yaml` and `results.csv`
- Domain-shift summaries: `domain-shifts/domain_shift_summary.json`, `domain-shifts/domain_stats.csv`
- Cross-strategy tables: `standard_metrics_comparison.{csv,json}`, `extra_metrics_comparison.{csv,json}`

External (Google Drive; not versioned)
- Full `*/optimal_output/runs/` dumps and all `train_batch*.jpg`
- `*/optimal_output/runs/weights/*.pt`, `*/optimal_output/models/*.pt` (trained weights)
- `*/optimal_output/tensorboard_logs/` (event files)
- `Archives/` bundles, large raw assets, base weights like `yolo12s.pt`
- Real annotations datasets

If you must keep weights in the repo, use Git LFS or publish them as GitHub Releases.

---

## Report assets (where to look)
- Curves and summaries: `*/optimal_output/runs/results.png`
- PR curve: `*/optimal_output/runs/BoxPR_curve.png`
- Confusion matrix (normalized): `*/optimal_output/runs/confusion_matrix_normalized.png`
- Qualitative predictions: `*/optimal_output/runs/val_batch*_pred.jpg`
- Label distribution: `*/optimal_output/runs/labels.jpg`

---

## Acknowledgements
- Ultralytics YOLO
- Avalanche (if using ER with Avalanche)

## License
Add your chosen license (e.g., MIT, Apache-2.0).


=======
# ROAD-DIL
Domain incremental learning benchmark for autonmous driving
>>>>>>> ab5df300dcf7f7ac2768cbb197240883c84d8ce3
