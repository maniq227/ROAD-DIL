#!/usr/bin/env python3
import json, csv
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path("YOLO-ER/optimal_output")
DEFAULT_DOMAINS = ["sunny","overcast","night","snowy"]

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

def load_domains() -> List[str]:
    # Try to read domains from continual_learning_metrics.json; fallback to default
    mpath = ROOT / "continual_learning_metrics.json"
    try:
        if mpath.exists():
            mj = json.loads(mpath.read_text(encoding="utf-8"))
            ds = mj.get("domains")
            if isinstance(ds, list) and ds:
                return [str(d) for d in ds]
    except Exception:
        pass
    return list(DEFAULT_DOMAINS)

def main():
    domains = load_domains()
    jpath = ROOT / "continual_eval_matrix.json"
    if not jpath.exists():
        raise FileNotFoundError(jpath)
    rows: List[Dict[str, Any]] = json.loads(jpath.read_text(encoding="utf-8"))

    # Backfill per-experience Extended AP columns into rows (if available)
    for i, row in enumerate(rows, start=1):
        if i - 1 >= len(domains):
            break
        cur = domains[i-1]
        for d in domains[:i]:
            out_dir = ROOT / "analysis_extended_ap" / f"exp{i}_{cur}" / d
            info = latest_extap(out_dir)
            if not info:
                continue
            mets = (info or {}).get("metrics", {})
            def getf(k: str):
                v = mets.get(k)
                return float(v) if isinstance(v, (int, float)) else None
            row[f"{d}_AP20"]    = getf("AP@0.20")
            row[f"{d}_AP50"]    = getf("AP@0.50")
            row[f"{d}_AP75"]    = getf("AP@0.75")
            row[f"{d}_AP50_90"] = getf("AP@[0.50:0.90]")
            row[f"{d}_AP50_95"] = getf("AP@[0.50:0.95]")

    # Write JSON back
    jpath.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Rebuild CSV with extended AP columns
    csv_path = ROOT / "continual_eval_matrix.csv"
    header = ( ["experience"]
               + [f"{d}_mAP50_95" for d in domains]
               + [f"{d}_mAP50"    for d in domains]
               + [f"{d}_AP20"     for d in domains]
               + [f"{d}_AP50"     for d in domains]
               + [f"{d}_AP75"     for d in domains]
               + [f"{d}_AP50_90"  for d in domains]
               + [f"{d}_AP50_95"  for d in domains]
               + ["avg_seen_mAP50_95"] )
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
            line += g([f"{d}_mAP50_95" for d in domains])
            line += g([f"{d}_mAP50"    for d in domains])
            line += g([f"{d}_AP20"     for d in domains])
            line += g([f"{d}_AP50"     for d in domains])
            line += g([f"{d}_AP75"     for d in domains])
            line += g([f"{d}_AP50_90"  for d in domains])
            line += g([f"{d}_AP50_95"  for d in domains])
            avg = r.get("avg_seen_mAP50_95")
            line += ["" if avg is None else f"{float(avg):.6f}"]
            w.writerow(line)

    # Update continual_learning_metrics.json with final extended APs (from last experience dir)
    metrics_path = ROOT / "continual_learning_metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
        T = len(domains)
        if T > 0:
            exp_final_dir = ROOT / "analysis_extended_ap" / f"exp{T}_{domains[T-1]}"
            per_domain: Dict[str, Dict[str, float]] = {}
            keys = ["AP@0.20","AP@0.50","AP@0.75","AP@[0.50:0.90]","AP@[0.50:0.95]"]
            for d in domains:
                info = latest_extap(exp_final_dir / d)
                if not info:
                    continue
                mets = (info or {}).get("metrics", {})
                pd = {}
                for k in keys:
                    v = mets.get(k)
                    if isinstance(v, (int, float)):
                        pd[k] = float(v)
                if pd:
                    per_domain[d] = pd
            averages: Dict[str, float] = {}
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
    print("[OK] Backfill complete for YOLO-ER.")

if __name__ == "__main__":
    main()


