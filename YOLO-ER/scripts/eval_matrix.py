#!/usr/bin/env python3
import argparse, json, csv
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Summarize continual eval matrix into CSV + basic metrics")
    ap.add_argument("--inputs_root", type=Path, default=Path("YOLO-ER/optimal_output"))
    ap.add_argument("--domains", nargs="+", default=["sunny", "overcast", "night", "snowy"])
    args = ap.parse_args()

    eval_json = args.inputs_root / "continual_eval_matrix.json"
    eval_csv = args.inputs_root / "continual_eval_matrix.csv"
    metrics_json = args.inputs_root / "continual_learning_metrics.json"

    if not eval_json.exists():
        raise FileNotFoundError(f"Missing {eval_json}")

    rows = json.loads(eval_json.read_text(encoding="utf-8"))
    # Re-write CSV to match domains order
    with eval_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experience"] + [f"{d}_mAP50_95" for d in args.domains])
        for r in rows:
            w.writerow([r.get("experience")] + [r.get(f"{d}_mAP50_95") for d in args.domains])

    # Basic metrics: average over seen domains at last exp, and across matrix
    last = rows[-1] if rows else {}
    last_seen = [last.get(f"{d}_mAP50_95") for d in args.domains if last.get(f"{d}_mAP50_95") is not None]
    final_avg = sum(last_seen) / len(last_seen) if last_seen else None

    # Per-domain final perf and Chaudhry forgetting: max over steps up to T-1 minus final; only first T-1 domains
    E = len(rows); T = len(args.domains)
    per_domain = {}
    if E > 1:
        K_max = min(T - 1, E - 1)
        for j, d in enumerate(args.domains):
            if j >= K_max:
                continue
            pre_vals = []
            for i in range(j, E - 1):
                v = rows[i].get(f"{d}_mAP50_95")
                if isinstance(v, (int, float)):
                    pre_vals.append(float(v))
            last_v = rows[-1].get(f"{d}_mAP50_95")
            if pre_vals and isinstance(last_v, (int, float)):
                per_domain[d] = {
                    "final": float(last_v),
                    "best": max(pre_vals),
                    "forgetting": float(max(pre_vals) - float(last_v))
                }

    metrics = {
        "final_avg_mAP50_95": final_avg,
        "per_domain": per_domain,
        "num_experiences": len(rows),
        "domains": args.domains,
    }
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {eval_csv} and {metrics_json}")

if __name__ == "__main__":
    main()


