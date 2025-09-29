#!/usr/bin/env python3
import json, argparse, csv
from pathlib import Path
from typing import Any, Dict

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def main():
    ap = argparse.ArgumentParser("Compare extra metrics (FWT, Extended AP) across strategies")
    ap.add_argument("--er", type=Path, default=Path("YOLO-ER/optimal_output/continual_learning_metrics.json"))
    ap.add_argument("--gdumb", type=Path, default=Path("YOLO-GDUMB/optimal_output/continual_learning_metrics.json"))
    ap.add_argument("--naive", type=Path, default=Path("YOLO-NAIVE/optimal_output/continual_learning_metrics.json"))
    ap.add_argument("--out_json", type=Path, default=Path("extra_metrics_comparison.json"))
    ap.add_argument("--out_csv", type=Path, default=Path("extra_metrics_comparison.csv"))
    args = ap.parse_args()

    dj = {
        "YOLO-ER": load_json(args.er),
        "YOLO-GDUMB": load_json(args.gdumb),
        "YOLO-NAIVE": load_json(args.naive),
    }
    strategies = list(dj.keys())

    comp: Dict[str, Any] = {"strategies": strategies}

    # FWT mean/per-domain
    comp["FWT_mean"] = {s: dj[s].get("FWT_mean") or dj[s].get("final", {}).get("FWT_mean") for s in strategies}
    comp["FWT_per_domain"] = {}
    # Unify per-domain keys
    dom_union = []
    for s in strategies:
        ds = dj[s].get("domains")
        if isinstance(ds, list):
            for d in ds:
                if d not in dom_union:
                    dom_union.append(d)
    for d in dom_union:
        row = {}
        for s in strategies:
            src = dj[s].get("FWT_per_domain") or dj[s].get("final", {}).get("FWT_per_domain") or {}
            row[s] = src.get(d)
        comp["FWT_per_domain"][d] = row

    # Extended AP final (per-domain and avg) if present
    comp["ExtendedAP_final_avg"] = {}
    comp["ExtendedAP_final_per_domain"] = {}
    for s in strategies:
        final = dj[s].get("final", {})
        comp["ExtendedAP_final_avg"][s] = final.get("ExtendedAP_final_avg")
    # Per-domain keys per AP metric
    ap_keys = ["AP@0.20","AP@0.50","AP@0.75","AP@[0.50:0.90]","AP@[0.50:0.95]"]
    # Build nested: ExtendedAP_final_per_domain[domain][AP_key][strategy]
    for d in dom_union:
        comp["ExtendedAP_final_per_domain"][d] = {}
        for k in ap_keys:
            row = {}
            for s in strategies:
                per = (dj[s].get("final", {}).get("ExtendedAP_final_per_domain") or {}).get(d)
                row[s] = (per or {}).get(k) if isinstance(per, dict) else None
            comp["ExtendedAP_final_per_domain"][d][k] = row

    # Write JSON
    args.out_json.write_text(json.dumps(comp, indent=2), encoding="utf-8")

    # Write CSV with a few useful slices
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # FWT mean
        w.writerow(["Metric","Domain","YOLO-ER","YOLO-GDUMB","YOLO-NAIVE"])
        w.writerow(["FWT_mean","", comp["FWT_mean"]["YOLO-ER"], comp["FWT_mean"]["YOLO-GDUMB"], comp["FWT_mean"]["YOLO-NAIVE"]])
        # FWT per-domain
        for d in dom_union:
            row = comp["FWT_per_domain"][d]
            w.writerow(["FWT_per_domain", d, row.get("YOLO-ER"), row.get("YOLO-GDUMB"), row.get("YOLO-NAIVE")])
        # Extended AP (final avg) as a JSON blob per strategy
        w.writerow(["ExtendedAP_final_avg","", json.dumps(comp["ExtendedAP_final_avg"].get("YOLO-ER")), json.dumps(comp["ExtendedAP_final_avg"].get("YOLO-GDUMB")), json.dumps(comp["ExtendedAP_final_avg"].get("YOLO-NAIVE"))])
        # Extended AP per-domain per key
        for d in dom_union:
            for k in ap_keys:
                row = comp["ExtendedAP_final_per_domain"][d][k]
                w.writerow([f"ExtendedAP_final_per_domain:{k}", d, row.get("YOLO-ER"), row.get("YOLO-GDUMB"), row.get("YOLO-NAIVE")])

    print("[OK] Wrote", args.out_json)
    print("[OK] Wrote", args.out_csv)

if __name__ == "__main__":
    main()


