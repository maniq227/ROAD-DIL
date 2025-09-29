#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Optional


def safe_mean(xs: List[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval-only baseline + recompute FWT (no fallback)")
    ap.add_argument("--outputs_root", type=Path, required=True)
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--domains", nargs="+", default=["sunny", "overcast", "night", "snowy"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument(
        "--metrics_json",
        type=Path,
        default=None,
        help="Target metrics JSON to update (defaults to continual_learning_metrics_adjusted.json if present else continual_learning_metrics.json)",
    )
    args = ap.parse_args()

    eval_json = args.outputs_root / "continual_eval_matrix.json"
    if not eval_json.exists():
        raise FileNotFoundError(f"Missing continual_eval_matrix.json under {args.outputs_root}")
    rows = json.loads(eval_json.read_text(encoding="utf-8"))

    baseline_path = args.outputs_root / "baseline.json"
    try:
        baseline_raw = json.loads(baseline_path.read_text(encoding="utf-8"))
        if isinstance(baseline_raw, list):
            baseline_data = {"mAP50_95": baseline_raw, "mAP50": []}
        else:
            baseline_data = baseline_raw
        b95_list = baseline_data.get("mAP50_95")
        b50_list = baseline_data.get("mAP50")
        if not isinstance(b95_list, list):
            raise ValueError
        # b50_list may be missing; that's fine, we'll compute only 95
        if not isinstance(b50_list, list):
            b50_list = []
            baseline_data["mAP50"] = b50_list
    except Exception:
        # Try fallback baseline from ER or GDUMB outputs (mAP50-95 list)
        fallback_paths = [
            Path("YOLO-ER/optimal_output/baseline.json"),
            Path("YOLO-GDUMB/optimal_output/baseline.json"),
        ]
        baseline_data = None
        for fp in fallback_paths:
            if fp.exists():
                try:
                    bl = json.loads(fp.read_text(encoding="utf-8"))
                    if isinstance(bl, list):
                        baseline_data = {"mAP50_95": bl, "mAP50": []}
                        break
                except Exception:
                    pass
        if baseline_data is None:
            # Compute baselines via eval-only using fresh minimal YAMLs with absolute paths
            try:
                from ultralytics import YOLO  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("Ultralytics is required to compute baselines.") from e
            import tempfile
            model = YOLO(args.model)
            # Load class names
            names_json = args.dataset_root.parent / "configs" / "names.json"
            names = json.loads(names_json.read_text(encoding="utf-8")) if names_json.exists() else []
            b95_list = []
            b50_list = []
            for d in args.domains:
                val_txt = (args.dataset_root / d / "val.txt").resolve()
                # Create a minimal YAML pointing val to the list and an empty train
                with tempfile.TemporaryDirectory() as td:
                    ypath = Path(td) / f"val_only_{d}.yaml"
                    ytext = f"train: {val_txt}\nval: {val_txt}\nnames: {json.dumps(names)}\n"
                    ypath.write_text(ytext, encoding="utf-8")
                    res = model.val(data=str(ypass := ypath), split="val", workers=int(args.workers))
                mp95 = res.results_dict.get("metrics/mAP50-95(B)") if hasattr(res, "results_dict") else None
                mp50 = res.results_dict.get("metrics/mAP50(B)") if hasattr(res, "results_dict") else None
                b95_list.append(float(mp95) if mp95 is not None else None)
                b50_list.append(float(mp50) if mp50 is not None else None)
            baseline_data = {"mAP50_95": b95_list, "mAP50": b50_list}
            baseline_path.write_text(json.dumps(baseline_data, indent=2), encoding="utf-8")

    # Recompute FWT using only baseline-subtracted terms (no fallback)
    fwt95_vals: List[float] = []
    fwt50_vals: List[float] = []
    T = min(len(args.domains), len(rows))
    for j in range(1, T):
        dom = args.domains[j]
        r_prev = rows[j - 1]
        v95 = r_prev.get(f"{dom}_mAP50_95")
        v50 = r_prev.get(f"{dom}_mAP50")
        b95 = baseline_data["mAP50_95"][j] if j < len(baseline_data["mAP50_95"]) else None
        b50 = baseline_data["mAP50"][j] if j < len(baseline_data["mAP50"]) else None
        if isinstance(v95, (int, float)) and isinstance(b95, (int, float)):
            fwt95_vals.append(float(v95) - float(b95))
        if isinstance(v50, (int, float)) and isinstance(b50, (int, float)):
            fwt50_vals.append(float(v50) - float(b50))

    fwt95 = safe_mean(fwt95_vals)
    fwt50 = safe_mean(fwt50_vals)

    # Update metrics JSON
    target = args.metrics_json
    if target is None:
        candidate1 = args.outputs_root / "continual_learning_metrics_adjusted.json"
        candidate2 = args.outputs_root / "continual_learning_metrics.json"
        target = candidate1 if candidate1.exists() else candidate2
    metrics = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {"final": {}, "rows": rows}
    metrics.setdefault("final", {})
    metrics["final"]["forward_transfer_mAP50_95"] = fwt95
    metrics["final"]["forward_transfer_mAP50"] = fwt50
    # Track how many domains contributed for transparency
    metrics["final"]["FWT_terms_used_mAP50_95"] = len(fwt95_vals)
    metrics["final"]["FWT_terms_used_mAP50"] = len(fwt50_vals)
    target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


