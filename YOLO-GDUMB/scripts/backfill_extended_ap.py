#!/usr/bin/env python3
import json, csv
from pathlib import Path

ROOT = Path("YOLO-GDUMB/optimal_output")
DOMAINS = ["sunny","overcast","night","snowy"]

def latest_extap(p: Path):
    if not p.exists():
        return None
    files = sorted(p.glob("extended_ap_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return None

def main():
    jpath = ROOT / "continual_eval_matrix.json"
    if not jpath.exists():
        raise FileNotFoundError(jpath)
    rows = json.loads(jpath.read_text(encoding="utf-8"))

    for i, row in enumerate(rows, start=1):
        cur = DOMAINS[i-1]
        for d in DOMAINS[:i]:
            out_dir = ROOT / "analysis_extended_ap" / f"exp{i}_{cur}" / d
            info = latest_extap(out_dir)
            if not info:
                continue
            mets = info.get("metrics", {})
            row[f"{d}_AP20"]    = float(mets.get("AP@0.20")) if mets.get("AP@0.20") is not None else None
            row[f"{d}_AP50"]    = float(mets.get("AP@0.50")) if mets.get("AP@0.50") is not None else None
            row[f"{d}_AP75"]    = float(mets.get("AP@0.75")) if mets.get("AP@0.75") is not None else None
            row[f"{d}_AP50_90"] = float(mets.get("AP@[0.50:0.90]")) if mets.get("AP@[0.50:0.90]") is not None else None
            row[f"{d}_AP50_95"] = float(mets.get("AP@[0.50:0.95]")) if mets.get("AP@[0.50:0.95]") is not None else None

    # Write JSON
    jpath.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Rebuild CSV with extended AP columns
    csv_path = ROOT / "continual_eval_matrix.csv"
    header = (["experience"]
              + [f"{d}_mAP50_95" for d in DOMAINS]
              + [f"{d}_mAP50"    for d in DOMAINS]
              + [f"{d}_AP20"     for d in DOMAINS]
              + [f"{d}_AP50"     for d in DOMAINS]
              + [f"{d}_AP75"     for d in DOMAINS]
              + [f"{d}_AP50_90"  for d in DOMAINS]
              + [f"{d}_AP50_95"  for d in DOMAINS]
              + ["avg_seen_mAP50_95"]) 
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            line = [r.get("experience")]
            def g(keys):
                out = []
                for k in keys:
                    v = r.get(k)
                    out.append("" if v is None else f"{float(v):.6f}")
                return out
            line += g([f"{d}_mAP50_95" for d in DOMAINS])
            line += g([f"{d}_mAP50"    for d in DOMAINS])
            line += g([f"{d}_AP20"     for d in DOMAINS])
            line += g([f"{d}_AP50"     for d in DOMAINS])
            line += g([f"{d}_AP75"     for d in DOMAINS])
            line += g([f"{d}_AP50_90"  for d in DOMAINS])
            line += g([f"{d}_AP50_95"  for d in DOMAINS])
            avg = r.get("avg_seen_mAP50_95")
            line += ["" if avg is None else f"{float(avg):.6f}"]
            w.writerow(line)
    print("Backfill complete.")

    # Also update continual_learning_metrics.json with final extended APs
    metrics_path = ROOT / "continual_learning_metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
        # Collect final (last experience) extended APs per domain from expT
        T = len(DOMAINS)
        exp_final_dir = ROOT / "analysis_extended_ap" / f"exp{T}_{DOMAINS[T-1]}"
        per_domain: dict[str, dict[str, float]] = {}
        keys = ["AP@0.20","AP@0.50","AP@0.75","AP@[0.50:0.90]","AP@[0.50:0.95]"]
        for d in DOMAINS:
            info = latest_extap(exp_final_dir / d)
            if not info:
                continue
            mets = info.get("metrics", {})
            per_domain[d] = {k: float(mets.get(k)) for k in keys if mets.get(k) is not None}
        # Averages across domains for any key present in all collected domains
        averages: dict[str, float] = {}
        for k in keys:
            vals = [pd[k] for pd in per_domain.values() if k in pd]
            if vals:
                try:
                    averages[k] = float(sum(vals) / len(vals))
                except Exception:
                    pass
        if "final" not in metrics or not isinstance(metrics["final"], dict):
            metrics["final"] = {}
        metrics["final"]["ExtendedAP_final_per_domain"] = per_domain
        metrics["final"]["ExtendedAP_final_avg"] = averages
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()


