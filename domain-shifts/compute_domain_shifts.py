#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


DOMAINS_DEFAULT = ["sunny", "overcast", "night", "snowy"]


def read_list(p: Path) -> List[str]:
    if not p.exists():
        return []
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def sample_paths(paths: List[str], k: int, seed: int) -> List[str]:
    if k <= 0 or len(paths) <= k:
        return list(paths)
    rnd = random.Random(seed)
    return rnd.sample(paths, k)


def entropy_from_hist(hist: np.ndarray, eps: float = 1e-12) -> float:
    p = hist.astype(np.float64)
    p = p / (p.sum() + eps)
    return float(-(p * np.log2(p + eps)).sum())


def compute_image_stats(img_path: str) -> Tuple[Dict[str, float], np.ndarray]:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return {}, np.array([])
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Photometric stats
    lum_mean = float(gray.mean())
    lum_std = float(gray.std())
    sat_mean = float(hsv[:, :, 1].mean())
    sat_std = float(hsv[:, :, 1].std())

    # Entropy (grayscale)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    ent = entropy_from_hist(hist_gray)

    # Edge density (Canny)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).mean())

    # Gradient magnitude (Sobel)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_mean = float(np.sqrt(gx * gx + gy * gy).mean())

    # Keypoint density (ORB, per megapixel)
    try:
        orb = cv2.ORB_create(nfeatures=500)
        kps = orb.detect(gray, None)
        kp_per_mp = float(len(kps)) / max(1.0, (h * w) / 1_000_000.0)
    except Exception:
        kp_per_mp = float("nan")

    # Compact 64-bin grayscale histogram (normalized)
    hist64 = cv2.calcHist([gray], [0], None, [64], [0, 256]).ravel().astype(np.float64)
    s = hist64.sum()
    if s > 0:
        hist64 /= s

    stats = {
        "lum_mean": lum_mean,
        "lum_std": lum_std,
        "sat_mean": sat_mean,
        "sat_std": sat_std,
        "entropy_gray": ent,
        "edge_density": edge_density,
        "grad_mag_mean": grad_mag_mean,
        "orb_kp_per_mp": kp_per_mp,
        "height_mean": float(h),
        "width_mean": float(w),
    }
    return stats, hist64


def aggregate_domain(paths: List[str]) -> Tuple[Dict[str, float], np.ndarray, int]:
    agg: Dict[str, List[float]] = {}
    hists: List[np.ndarray] = []
    n_used = 0

    for p in paths:
        st, h = compute_image_stats(p)
        if not st:
            continue
        for k, v in st.items():
            if np.isnan(v) if isinstance(v, float) else False:
                continue
            agg.setdefault(k, []).append(float(v))
        if h.size:
            hists.append(h)
        n_used += 1

    summary = {k: float(np.mean(v)) for k, v in agg.items() if len(v) > 0}
    if hists:
        H = np.stack(hists, axis=0)
        hist_mean = H.mean(axis=0)
        s = hist_mean.sum()
        if s > 0:
            hist_mean = hist_mean / s
    else:
        hist_mean = np.zeros(64, dtype=np.float64)
    return summary, hist_mean, n_used


def pairwise_l1(hist_means: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    doms = list(hist_means.keys())
    out: Dict[str, Dict[str, float]] = {d: {} for d in doms}
    for i, di in enumerate(doms):
        for dj in doms:
            d = float(np.abs(hist_means[di] - hist_means[dj]).sum())
            out[di][dj] = d
    return out


def main():
    ap = argparse.ArgumentParser("Compute CPU-only domain shift metrics (photometric/structural)")
    ap.add_argument("--dataset_root", type=Path, default=Path("YOLO-ER/datasets"))
    ap.add_argument("--domains", nargs="+", default=DOMAINS_DEFAULT)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--max_images", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/analysis/domain_shifts"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_domain_stats: Dict[str, Dict[str, float]] = {}
    per_domain_hist: Dict[str, np.ndarray] = {}
    per_domain_counts: Dict[str, int] = {}

    for d in args.domains:
        txt = args.dataset_root / d / f"{args.split}.txt"
        if not txt.exists():
            print(f"[warn] missing list: {txt}")
            continue
        paths = read_list(txt)
        sel = sample_paths(paths, args.max_images, args.seed)
        stats, hist_mean, n_used = aggregate_domain(sel)
        stats["num_images_used"] = n_used
        per_domain_stats[d] = stats
        per_domain_hist[d] = hist_mean
        per_domain_counts[d] = n_used
        print(f"[OK] {d}: used={n_used} images")

    # Pairwise distances (64-bin grayscale hist L1)
    dist_gray_hist_L1 = pairwise_l1(per_domain_hist)

    # Save JSON
    out_json = {
        "domains": args.domains,
        "split": args.split,
        "max_images": args.max_images,
        "per_domain_stats": {k: v for k, v in per_domain_stats.items()},
        "pairwise": {
            "gray_hist_L1": dist_gray_hist_L1,
        },
    }
    (args.out_dir / "domain_shift_summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # Save CSV summary
    headers = [
        "domain",
        "num_images_used",
        "lum_mean",
        "lum_std",
        "sat_mean",
        "sat_std",
        "entropy_gray",
        "edge_density",
        "grad_mag_mean",
        "orb_kp_per_mp",
        "height_mean",
        "width_mean",
    ]
    lines = [",".join(headers)]
    for d in args.domains:
        s = per_domain_stats.get(d, {})
        row = [
            d,
            str(s.get("num_images_used", 0)),
            f"{s.get('lum_mean', np.nan):.6f}",
            f"{s.get('lum_std', np.nan):.6f}",
            f"{s.get('sat_mean', np.nan):.6f}",
            f"{s.get('sat_std', np.nan):.6f}",
            f"{s.get('entropy_gray', np.nan):.6f}",
            f"{s.get('edge_density', np.nan):.6f}",
            f"{s.get('grad_mag_mean', np.nan):.6f}",
            f"{s.get('orb_kp_per_mp', np.nan):.6f}",
            f"{s.get('height_mean', np.nan):.6f}",
            f"{s.get('width_mean', np.nan):.6f}",
        ]
        lines.append(",".join(row))
    (args.out_dir / "domain_stats.csv").write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] wrote {args.out_dir / 'domain_shift_summary.json'}")
    print(f"[OK] wrote {args.out_dir / 'domain_stats.csv'}")


if __name__ == "__main__":
    main() 