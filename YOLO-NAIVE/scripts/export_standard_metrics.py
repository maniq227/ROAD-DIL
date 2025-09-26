#!/usr/bin/env python3
import json, argparse, math
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_DOMAINS = ["sunny", "overcast", "night", "snowy"]

def load_rows(inputs_root: Path) -> List[Dict[str, Any]]:
    p = inputs_root / "continual_eval_matrix.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def to_matrix(rows: List[Dict[str, Any]], domains: List[str]) -> List[List[float]]:
    E, T = len(rows), len(domains)
    A = [[math.nan for _ in range(T)] for _ in range(E)]
    for i, r in enumerate(rows):
        for j, d in enumerate(domains):
            v = r.get(f"{d}_mAP50_95")
            if isinstance(v, (int, float)):
                A[i][j] = float(v)
    return A

def mean(vals: List[float]) -> Optional[float]:
    xs = [x for x in vals if not math.isnan(x)]
    return float(sum(xs) / len(xs)) if xs else None

def acc_final(A: List[List[float]]) -> Optional[float]:
    return mean(A[-1]) if A else None

def acc_mean_over_time(A: List[List[float]]) -> Optional[float]:
    if not A: return None
    E, T = len(A), len(A[0]) if A else 0
    per: List[float] = []
    for i in range(E):
        seen = [A[i][j] for j in range(min(i+1, T))]
        m = mean(seen)
        if m is not None:
            per.append(m)
    return float(sum(per) / len(per)) if per else None

def bwt(A: List[List[float]], domains: List[str]) -> (Optional[float], Dict[str, float]):
    if not A: return None, {}
    E, T = len(A), len(A[0])
    last = A[-1]
    K_max = min(T - 1, E - 1)
    per: Dict[str, float] = {}
    vals: List[float] = []
    for j in range(max(0, K_max)):
        a_jj, a_tj = A[j][j], last[j]
        if not math.isnan(a_jj) and not math.isnan(a_tj):
            diff = float(a_tj - a_jj)
            per[domains[j]] = diff
            vals.append(diff)
    return (float(sum(vals) / len(vals)) if vals else None), per

def forgetting(A: List[List[float]], domains: List[str]) -> (Optional[float], Dict[str, float]):
    if not A: return None, {}
    E, T = len(A), len(A[0])
    last = A[-1]
    K_max = min(T - 1, E - 1)
    per: Dict[str, float] = {}
    vals: List[float] = []
    for j in range(max(0, K_max)):
        pre = [A[i][j] for i in range(j, E - 1)]
        pre = [x for x in pre if not math.isnan(x)]
        if pre and not math.isnan(last[j]):
            f = float(max(pre) - last[j])
            per[domains[j]] = f
            vals.append(f)
    return (float(sum(vals) / len(vals)) if vals else None), per

def main():
    ap = argparse.ArgumentParser("Export standard metrics (mAP@0.5:0.95) to standard_metrics.json")
    ap.add_argument("--inputs_root", type=Path, default=Path("YOLO-NAIVE/optimal_output"))
    ap.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    args = ap.parse_args()

    rows = load_rows(args.inputs_root)
    A = to_matrix(rows, args.domains)
    out = {
        "domains": args.domains,
        "num_experiences": len(rows),
        "ACC_final": acc_final(A),
        "ACC_mean_over_time": acc_mean_over_time(A),
    }
    bwt_mean, bwt_per = bwt(A, args.domains)
    f_mean, f_per = forgetting(A, args.domains)
    out["BWT_mean"] = bwt_mean
    out["BWT_per_domain"] = bwt_per
    out["Forgetting_mean"] = f_mean
    out["Forgetting_per_domain"] = f_per

    target = args.inputs_root / "standard_metrics.json"
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[OK] Wrote", target)

if __name__ == "__main__":
    main()


