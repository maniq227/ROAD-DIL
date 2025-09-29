#!/usr/bin/env python3
import json, argparse, csv
from pathlib import Path
from typing import Dict, Any

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def main():
    ap = argparse.ArgumentParser("Compare standard metrics across strategies")
    ap.add_argument("--er_json", type=Path, default=Path("YOLO-ER/optimal_output/standard_metrics.json"))
    ap.add_argument("--gdumb_json", type=Path, default=Path("YOLO-GDUMB/optimal_output/standard_metrics.json"))
    ap.add_argument("--naive_json", type=Path, default=Path("YOLO-NAIVE/optimal_output/standard_metrics.json"))
    ap.add_argument("--out_json", type=Path, default=Path("standard_metrics_comparison.json"))
    ap.add_argument("--out_csv", type=Path, default=Path("standard_metrics_comparison.csv"))
    args = ap.parse_args()

    data = {
        "YOLO-ER": load_json(args.er_json),
        "YOLO-GDUMB": load_json(args.gdumb_json),
        "YOLO-NAIVE": load_json(args.naive_json),
    }

    strategies = list(data.keys())
    comp = {"strategies": strategies}

    for key in ["ACC_final", "ACC_mean_over_time", "BWT_mean", "Forgetting_mean"]:
        comp[key] = {s: data[s].get(key) for s in strategies}

    # Union of domains across inputs
    dom_union = []
    for s in strategies:
        for d in (data[s].get("domains") or []):
            if d not in dom_union:
                dom_union.append(d)

    per_bwt = {}
    per_forget = {}
    for d in dom_union:
        per_bwt[d] = {s: (data[s].get("BWT_per_domain") or {}).get(d) for s in strategies}
        per_forget[d] = {s: (data[s].get("Forgetting_per_domain") or {}).get(d) for s in strategies}
    comp["BWT_per_domain"] = per_bwt
    comp["Forgetting_per_domain"] = per_forget

    args.out_json.write_text(json.dumps(comp, indent=2), encoding="utf-8")

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Domain", "YOLO-ER", "YOLO-GDUMB", "YOLO-NAIVE"])
        for key in ["ACC_final", "ACC_mean_over_time", "BWT_mean", "Forgetting_mean"]:
            w.writerow([key, "", comp[key].get("YOLO-ER"), comp[key].get("YOLO-GDUMB"), comp[key].get("YOLO-NAIVE")])
        for d in dom_union:
            bb = comp["BWT_per_domain"][d]
            w.writerow(["BWT_per_domain", d, bb.get("YOLO-ER"), bb.get("YOLO-GDUMB"), bb.get("YOLO-NAIVE")])
        for d in dom_union:
            ff = comp["Forgetting_per_domain"][d]
            w.writerow(["Forgetting_per_domain", d, ff.get("YOLO-ER"), ff.get("YOLO-GDUMB"), ff.get("YOLO-NAIVE")])

    print("[OK] Wrote", args.out_json)
    print("[OK] Wrote", args.out_csv)

if __name__ == "__main__":
    main()


