#!/usr/bin/env python3
import argparse, json, csv
from pathlib import Path
from typing import List, Dict, Any
import math

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

def load_matrix(inputs_root: Path, domains: List[str]) -> List[Dict[str, Any]]:
    j = inputs_root / "continual_eval_matrix.json"
    c = inputs_root / "continual_eval_matrix.csv"
    if j.exists():
        return json.loads(j.read_text(encoding="utf-8"))
    rows = []
    if c.exists():
        for r in csv.DictReader(c.open("r", encoding="utf-8")):
            # cast numeric cols
            for d in domains:
                k = f"{d}_mAP50_95"
                if k in r and r[k] not in ("", None):
                    try: r[k] = float(r[k])
                    except: r[k] = None
                else:
                    r[k] = None
            r["experience"] = int(r.get("experience", len(rows)+1))
            rows.append(r)
    return rows

def to_matrix(rows: List[Dict[str, Any]], domains: List[str]) -> np.ndarray:
    # A[i,j] = mAP of test domain j after experience i (1-indexed in file → 0-index here)
    T = len(domains)
    E = len(rows)
    A = np.full((E, T), np.nan, dtype=float)
    for i, r in enumerate(rows):
        for j, d in enumerate(domains):
            v = r.get(f"{d}_mAP50_95", None)
            if v is not None:
                try: A[i, j] = float(v)
                except: pass
    return A

def acc_final(A: np.ndarray) -> float:
    last = A[-1, :]
    vals = last[~np.isnan(last)]
    return float(np.mean(vals)) if vals.size else float("nan")

def acc_mean_over_time(A: np.ndarray) -> float:
    # mean over time of mean over seen tasks
    accs = []
    for i in range(A.shape[0]):
        seen = A[i, :i+1]
        seen = seen[~np.isnan(seen)]
        if seen.size: accs.append(np.mean(seen))
    return float(np.mean(accs)) if accs else float("nan")

def bwt(A: np.ndarray) -> (float, Dict[str, float]):
    # Chaudhry BWT: mean over j=0..T-2 of (A[T-1,j] - A[j,j])
    import math
    E, T = A.shape
    last = A[-1, :]
    per = {}
    diffs = []
    K_max = min(T - 1, E - 1)
    for j in range(max(0, K_max)):
        a_jj = A[j, j]
        a_tj = last[j]
        if not (math.isnan(a_jj) or math.isnan(a_tj)):
            d = float(a_tj - a_jj)
            per[j] = d
            diffs.append(d)
    return (float(np.mean(diffs)) if diffs else float("nan")), per

def forgetting(A: np.ndarray) -> (float, Dict[str, float]):
    # Chaudhry (2018): f_k(T) = max_{t=k..T-1} a_{k,t} - a_{k,T}; mean over k=1..T-1
    import math
    E, T = A.shape
    if E == 0 or T == 0:
        return float("nan"), {}
    last = A[-1, :]
    per: Dict[int, float] = {}
    vals: List[float] = []
    # Only domains k=0..min(T-2, E-2) are valid (exclude final column index T-1 and require at least 2 rows)
    K_max = min(T - 1, E - 1)
    for j in range(K_max):
        # consider rows j..E-2 (exclude final row E-1)
        pre = A[j:E-1, j]
        pre = pre[~np.isnan(pre)]
        if pre.size and not math.isnan(last[j]):
            f = float(np.nanmax(pre) - last[j])
            per[j] = f
            vals.append(f)
    return (float(np.mean(vals)) if vals else float("nan")), per

def fwt(A: np.ndarray, baseline: Dict[int, float] | None = None) -> (float, Dict[str, float]):
    # Approx FWT (Chaudhry’18 style): mean over j>0 of (A[j-1, j] - b_j)
    # If baseline not provided, use A[0,j] if exists; else 0.0
    per = {}
    diffs = []
    for j in range(1, A.shape[1]):
        a_prev = A[min(j-1, A.shape[0]-1), j]
        if math.isnan(a_prev): continue
        b = baseline.get(j, np.nan) if baseline else (A[0, j] if not math.isnan(A[0, j]) else 0.0)
        if math.isnan(b): b = 0.0
        d = float(a_prev - b)
        per[j] = d
        diffs.append(d)
    return (float(np.mean(diffs)) if diffs else float("nan")), per

def save_figs(A: np.ndarray, domains: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    x = list(range(1, A.shape[0]+1))
    # 1) Heatmap
    plt.figure(figsize=(1.6*len(domains)+2, 1.2*A.shape[0]+2))
    data = np.copy(A)
    vmax = 1.0 if np.nanmax(data) <= 1.0 else np.nanmax(data)
    if HAS_SNS:
        import seaborn as sns
        ax = sns.heatmap(data, annot=True, fmt=".3f", cmap="viridis", vmin=0.0, vmax=vmax)
    else:
        plt.imshow(np.nan_to_num(data, nan=-1), cmap="viridis", vmin=0.0, vmax=vmax)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v): plt.text(j, i, f"{v:.3f}", ha="center", va="center", color="w")
    plt.xticks(ticks=np.arange(len(domains))+0.5, labels=domains, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(A.shape[0])+0.5, labels=[f"exp{i}" for i in x])
    plt.title("Continual Eval Matrix (mAP50-95)")
    plt.tight_layout()
    plt.savefig(out_dir / "cl_matrix_heatmap.png", dpi=150)
    plt.close()

    # 2) Trends per domain
    plt.figure(figsize=(10, 5))
    for j, d in enumerate(domains):
        plt.plot(x, A[:, j], marker="o", label=d)
    plt.xlabel("Experience")
    plt.ylabel("mAP50-95")
    plt.title("Per-domain mAP50-95 over time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cl_trends.png", dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs_root", type=Path, default=Path("YOLO-ER/optimal_output"))
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("YOLO-ER/optimal_output/figures"))
    ap.add_argument("--baseline_json", type=Path, default=None, help="Optional per-domain baseline mAP50-95 JSON list aligned to domains")
    args = ap.parse_args()

    rows = load_matrix(args.inputs_root, args.domains)
    if not rows:
        raise FileNotFoundError("No continual_eval_matrix found.")
    A = to_matrix(rows, args.domains)
    # Metrics
    accF = acc_final(A)
    accMean = acc_mean_over_time(A)
    bwt_mean, bwt_per = bwt(A)
    forget_mean, forget_per = forgetting(A)
    baseline = None
    if args.baseline_json and args.baseline_json.exists():
        bl = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        if isinstance(bl, list) and len(bl) == len(args.domains):
            baseline = {i: float(bl[i]) for i in range(len(bl))}
    fwt_mean, fwt_per = fwt(A, baseline=baseline)

    metrics = {
        "domains": args.domains,
        "num_experiences": int(A.shape[0]),
        "ACC_final": accF,
        "ACC_mean_over_time": accMean,
        "BWT_mean": bwt_mean,
        "BWT_per_domain": {args.domains[k]: v for k, v in bwt_per.items()},
        "Forgetting_mean": forget_mean,
        "Forgetting_per_domain": {args.domains[k]: v for k, v in forget_per.items()},
        "FWT_mean": fwt_mean,
        "FWT_per_domain": {args.domains[k]: v for k, v in fwt_per.items()},
    }
    # Save metrics (extend existing file if present)
    out_json = args.inputs_root / "continual_learning_metrics.json"
    if out_json.exists():
        try:
            old = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        old.update(metrics)
        out_json.write_text(json.dumps(old, indent=2), encoding="utf-8")
    else:
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Figures
    save_figs(A, args.domains, args.out_dir)
    print("[OK] Wrote metrics to", out_json)
    print("[OK] Figures in", args.out_dir)

if __name__ == "__main__":
    main()
