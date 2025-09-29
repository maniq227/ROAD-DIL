#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Optional

from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
import gc
import subprocess
import math

DOMAINS = ["sunny", "overcast", "night", "snowy"]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_lines(p: Path) -> List[str]:
    if not p.exists():
        return []
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_lines(p: Path, lines: List[str]) -> None:
    p.write_text("\n".join(lines), encoding="utf-8")


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def build_yaml(train_txt: Path, val_txt: Path, class_names: List[str], out_dir: Path) -> Path:
    # Absolute train/val, omit "path" to avoid Ultralytics prefixing dataset root
    yaml_text = f"train: {train_txt.resolve()}\nval: {val_txt.resolve()}\nnames: {class_names}\n"
    y = out_dir / "data.yaml"
    y.write_text(yaml_text, encoding="utf-8")
    return y


class DomainReservoir:
    def __init__(self, domains: List[str], quota_per_domain: int, seed: int = 42) -> None:
        self.domains = list(domains)
        self.quota = int(quota_per_domain)
        self.buffers: Dict[str, List[str]] = {d: [] for d in self.domains}
        self.seen: Dict[str, int] = {d: 0 for d in self.domains}
        random.seed(seed)

    def update_with_list(self, domain: str, img_paths: List[str]) -> None:
        for p in img_paths:
            self.seen[domain] += 1
            buf = self.buffers[domain]
            if len(buf) < self.quota:
                buf.append(p)
            else:
                j = random.randint(0, self.seen[domain])
                if j < self.quota:
                    buf[j] = p

    def memory_union(self, upto_domains: List[str]) -> List[str]:
        out: List[str] = []
        for d in upto_domains:
            out.extend(self.buffers[d])
        seen = set()
        uniq: List[str] = []
        for p in out:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        return uniq


def main() -> None:
    ap = argparse.ArgumentParser("GDumb with YOLOv12 Small on ROAD (Domain-Incremental)")
    ap.add_argument("--dataset_root", type=Path, default=Path("YOLO-ER/datasets"))
    ap.add_argument("--outputs_root", type=Path, default=Path("YOLO-GDUMB/optimal_output"))
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--model", type=str, default=os.getenv("YOLO_MODEL", "yolo12s.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--mem_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--accumulate", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--dry_run", action="store_true", help="Do not start training; set up files only")
    ap.add_argument("--cache", type=str, default="disk", help="Ultralytics caching: 'ram', 'disk', or 'none'")
    ap.add_argument("--multi_scale", action="store_true", help="Enable multi-scale training (disabled by default)")
    ap.add_argument("--mosaic", type=float, default=0.5, help="Mosaic probability (0.0-1.0)")
    args = ap.parse_args()

    random.seed(args.seed)

    ensure_dir(args.outputs_root)
    ensure_dir(args.outputs_root / "models")
    ensure_dir(args.outputs_root / "tensorboard_logs")

    writer = SummaryWriter(log_dir=str(args.outputs_root / "tensorboard_logs"))
    eta_log = args.outputs_root / "eta_progress.log"
    eval_json = args.outputs_root / "continual_eval_matrix.json"
    eval_csv = args.outputs_root / "continual_eval_matrix.csv"

    # Class names
    names_path = args.dataset_root.parent / "configs" / "names.json"
    if not names_path.exists():
        raise FileNotFoundError("names.json not found under YOLO-ER/configs; convert data to YOLO first")
    class_names: List[str] = json.loads(names_path.read_text(encoding="utf-8"))

    # Compute zero-shot baseline per domain (initial model) for authentic FWT
    baseline_path = args.outputs_root / "baseline.json"
    try:
        base_model = YOLO(args.model)
        baselines: List[Optional[float]] = []
        for d in args.domains:
            dyaml = args.dataset_root / d / "data_val_only.yaml"
            res = base_model.val(data=str(dyaml), split="val", workers=int(args.workers))
            mp95 = res.results_dict.get("metrics/mAP50-95(B)") if hasattr(res, "results_dict") else None
            baselines.append(float(mp95) if mp95 is not None else None)
        baseline_path.write_text(json.dumps(baselines, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Init matrix files
    rows: List[Dict[str, Optional[float]]] = []
    if eval_json.exists():
        try:
            rows = json.loads(eval_json.read_text(encoding="utf-8"))
        except Exception:
            rows = []
    if not eval_csv.exists() or not eval_csv.read_text(encoding="utf-8").strip():
        header = ["experience"]
        header += [f"{d}_mAP50_95" for d in args.domains]
        header += [f"{d}_mAP50" for d in args.domains]
        header += [f"{d}_AP20" for d in args.domains]
        header += [f"{d}_AP50" for d in args.domains]
        header += [f"{d}_AP75" for d in args.domains]
        header += [f"{d}_AP50_90" for d in args.domains]
        header += [f"{d}_AP50_95" for d in args.domains]
        header += ["avg_seen_mAP50_95"]
        eval_csv.write_text(
            ",".join(header) + "\n",
            encoding="utf-8",
        )

    # Memory reservoir
    quota = int(args.mem_size // max(1, len(args.domains)))
    reservoir = DomainReservoir(args.domains, quota, seed=args.seed)

    # Loop experiences without starting training if --dry_run
    for i, domain in enumerate(args.domains, start=1):
        seen_domains = args.domains[:i]
        exp_dir = args.outputs_root / f"exp{i}_{domain}"
        ensure_dir(exp_dir)

        # Load lists
        cur_train = load_lines(args.dataset_root / domain / "train.txt")
        cur_val = load_lines(args.dataset_root / domain / "val.txt")

        # Update reservoir
        reservoir.update_with_list(domain, cur_train)

        # Build memory and seen-val union
        mem_train = reservoir.memory_union(seen_domains)
        val_seen: List[str] = []
        for d in seen_domains:
            val_seen.extend(load_lines(args.dataset_root / d / "val.txt"))

        # Write lists
        mem_train_txt = exp_dir / "train_memory.txt"
        val_seen_txt = exp_dir / "val_seen.txt"
        write_lines(mem_train_txt, mem_train)
        write_lines(val_seen_txt, val_seen)

        data_yaml = build_yaml(mem_train_txt, val_seen_txt, class_names, exp_dir)

        # Prepare model and callbacks (training gated by --dry_run)
        model = YOLO(args.model)

        # ETA + TB callbacks
        import torch  # local import to avoid hard dependency if unused

        t_epoch0 = None
        t_last = None

        def on_train_epoch_start(trainer):
            nonlocal t_epoch0, t_last
            t_epoch0 = time.perf_counter()
            t_last = t_epoch0
            # Try to read total iterations for this epoch
            try:
                total_iters = int(getattr(trainer, "nb", len(getattr(trainer, "dataloader", []))) or 0)
            except Exception:
                total_iters = 0
            msg = f"[EPOCH_START] total_iters={total_iters} epoch={trainer.epoch+1}/{trainer.epochs} time={now_str()}"
            print(msg, flush=True)
            with eta_log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

        def on_train_batch_end(trainer):
            nonlocal t_last, t_epoch0
            now = time.perf_counter()
            dt = max(1e-6, now - (t_last or now))
            elapsed = now - (t_epoch0 or now)
            try:
                i_it = int(getattr(trainer, "batch_i", 0)) + 1
            except Exception:
                i_it = 0
            try:
                total = int(getattr(trainer, "nb", len(getattr(trainer, "dataloader", []))) or 0)
            except Exception:
                total = 0
            mb = int(getattr(trainer, "batch_size", int(args.batch)))
            imgsps = (mb / dt) if dt > 0 else 0.0
            gpumem = 0.0
            try:
                if torch.cuda.is_available():
                    gpumem = torch.cuda.memory_reserved() / (1024 ** 3)
            except Exception:
                pass
            rem_iter = max(0, total - i_it)
            eta_s = (elapsed / max(1, i_it)) * rem_iter if i_it > 0 else 0.0
            msg = (
                f"[ITER] iter={i_it}/{total} epoch={trainer.epoch+1}/{trainer.epochs} "
                f"t={elapsed:7.2f}s dt={dt:6.3f}s imgs/s={imgsps:7.2f} GPUmem={gpumem:5.2f}GB "
                f"started={now_str()} remaining={eta_s:7.1f}s"
            )
            print(msg, flush=True)
            with eta_log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            t_last = now

        def on_train_epoch_end(trainer):
            msg = f"[EPOCH_END] epoch={trainer.epoch+1}/{trainer.epochs} time={now_str()}"
            print(msg, flush=True)
            with eta_log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            try:
                metrics = getattr(trainer, "metrics", None) or getattr(trainer.validator, "metrics", {}) or {}
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(k, v, getattr(trainer, "epoch", 0))
                writer.flush()
            except Exception:
                pass

        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_train_batch_end", on_train_batch_end)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        # Training block (skipped in dry run)
        if not args.dry_run:
            with open(eta_log, "a", encoding="utf-8") as f:
                f.write(f"[{now_str()}] [exp{i}:{domain}] train start: {args.model}, epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, mem={len(mem_train)}\n")
            # Resolve cache mode to avoid host RAM spikes
            cache_mode = str(args.cache).lower().strip()
            cache_val = False
            if cache_mode == "ram":
                cache_val = "ram"
            elif cache_mode == "disk":
                cache_val = "disk"

            model.train(
                data=str(data_yaml),
                epochs=int(args.epochs),
                imgsz=int(args.imgsz),
                batch=int(args.batch),
                patience=int(args.patience),
                device=args.device or None,
                project=str(args.outputs_root),
                name="runs",
                exist_ok=True,
                workers=int(args.workers),
                optimizer="AdamW",
                cos_lr=True,
                lr0=0.002,
                lrf=0.01,
                weight_decay=0.0005,
                warmup_epochs=3,
                mosaic=float(args.mosaic),
                mixup=0.1,
                close_mosaic=5,
                multi_scale=bool(args.multi_scale),
                cache=cache_val,
            )

            # Save best checkpoint for this experience
            best = getattr(getattr(model, "trainer", None), "best", None)
            if best:
                bp = Path(best)
                if bp.exists():
                    dst = args.outputs_root / "models" / f"gdumb_model_after_exp_{i}.pt"
                    try:
                        dst.write_bytes(bp.read_bytes())
                    except Exception:
                        pass

            # Evaluate on all seen domains and pre-task for next domain (for FWT)
            row: Dict[str, Optional[float]] = {"experience": i}
            for d in args.domains:
                if d in seen_domains:
                    dyaml = exp_dir / f"val_only_{d}.yaml"
                    dyaml.write_text(
                        f"train: {mem_train_txt.resolve()}\nval: {(args.dataset_root / d / 'val.txt').resolve()}\nnames: {class_names}\n",
                        encoding="utf-8",
                    )
                    res = model.val(data=str(dyaml), split="val", workers=int(args.workers))
                    mp95 = res.results_dict.get("metrics/mAP50-95(B)") if hasattr(res, "results_dict") else None
                    mp50 = res.results_dict.get("metrics/mAP50(B)") if hasattr(res, "results_dict") else None
                    row[f"{d}_mAP50_95"] = float(mp95) if mp95 is not None else None
                    row[f"{d}_mAP50"] = float(mp50) if mp50 is not None else None
                    # Extended AP via helper script
                    try:
                        out_dir = args.outputs_root / "analysis_extended_ap" / f"exp{i}_{domain}" / d
                        out_dir.mkdir(parents=True, exist_ok=True)
                        cmd = [
                            "python",
                            str(Path.cwd() / "compute_extended_ap.py"),
                            "--weights",
                            str(getattr(getattr(model, "trainer", None), "best", "")),
                            "--data_yaml",
                            str(dyaml),
                            "--names_json",
                            str(args.dataset_root.parent / "configs" / "names.json"),
                            "--out_dir",
                            str(out_dir),
                        ]
                        subprocess.run(cmd, check=False, capture_output=True)
                        # Read the latest JSON from out_dir
                        jfiles = sorted(out_dir.glob("extended_ap_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                        if jfiles:
                            info = json.loads(jfiles[0].read_text(encoding="utf-8"))
                            mets = info.get("metrics", {})
                            row[f"{d}_AP20"] = float(mets.get("AP@0.20")) if mets.get("AP@0.20") is not None else None
                            row[f"{d}_AP50"] = float(mets.get("AP@0.50")) if mets.get("AP@0.50") is not None else None
                            row[f"{d}_AP75"] = float(mets.get("AP@0.75")) if mets.get("AP@0.75") is not None else None
                            row[f"{d}_AP50_90"] = float(mets.get("AP@[0.50:0.90]")) if mets.get("AP@[0.50:0.90]") is not None else None
                            row[f"{d}_AP50_95"] = float(mets.get("AP@[0.50:0.95]")) if mets.get("AP@[0.50:0.95]") is not None else None
                    except Exception:
                        pass
                else:
                    row[f"{d}_mAP50_95"] = None
                    row[f"{d}_mAP50"] = None
                    row[f"{d}_AP20"] = None
                    row[f"{d}_AP50"] = None
                    row[f"{d}_AP75"] = None
                    row[f"{d}_AP50_90"] = None
                    row[f"{d}_AP50_95"] = None

            # Pre-task off-diagonal for next domain (A[i-1, i])
            if i < len(args.domains):
                next_d = args.domains[i]
                dyaml_next = exp_dir / f"val_only_{next_d}_pre.yaml"
                dyaml_next.write_text(
                    f"train: {mem_train_txt.resolve()}\nval: {(args.dataset_root / next_d / 'val.txt').resolve()}\nnames: {class_names}\n",
                    encoding="utf-8",
                )
                res_next = model.val(data=str(dyaml_next), split="val", workers=int(args.workers))
                mp_next = res_next.results_dict.get("metrics/mAP50-95(B)") if hasattr(res_next, "results_dict") else None
                if mp_next is not None:
                    row[f"{next_d}_mAP50_95"] = float(mp_next)

            # Average across seen domains
            seen_vals = [row.get(f"{d}_mAP50_95") for d in seen_domains]
            seen_vals = [float(v) for v in seen_vals if v is not None]
            row["avg_seen_mAP50_95"] = float(sum(seen_vals) / max(1, len(seen_vals))) if seen_vals else None

            rows.append(row)
            eval_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            with eval_csv.open("a", encoding="utf-8") as f:
                line = [str(i)]
                line += ["" if row.get(f"{d}_mAP50_95") is None else f"{row[f'{d}_mAP50_95']:.6f}" for d in args.domains]
                line += ["" if row.get(f"{d}_mAP50") is None else f"{row[f'{d}_mAP50']:.6f}" for d in args.domains]
                line += ["" if row.get(f"{d}_AP20") is None else f"{row[f'{d}_AP20']:.6f}" for d in args.domains]
                line += ["" if row.get(f"{d}_AP50") is None else f"{row[f'{d}_AP50']:.6f}" for d in args.domains]
                line += ["" if row.get(f"{d}_AP75") is None else f"{row[f'{d}_AP75']:.6f}" for d in args.domains]
                line += ["" if row.get(f"{d}_AP50_90") is None else f"{row[f'{d}_AP50_90']:.6f}" for d in args.domains]
                line += ["" if row.get(f"{d}_AP50_95") is None else f"{row[f'{d}_AP50_95']:.6f}" for d in args.domains]
                avg_str = "" if row.get("avg_seen_mAP50_95") is None else f"{row['avg_seen_mAP50_95']:.6f}"
                line.append(avg_str)
                f.write(",".join(line) + "\n")

            with open(eta_log, "a", encoding="utf-8") as f:
                f.write(f"[{now_str()}] [exp{i}:{domain}] train+eval done\n")

            # Cleanup to reduce VRAM/RAM growth between experiences
            try:
                del res
            except Exception:
                pass
            try:
                del res_next
            except Exception:
                pass
            try:
                del model
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    try:
        writer.flush(); writer.close()
    except Exception:
        pass

    # Write final summary after full run (ACC/BWT/Forgetting + baseline-adjusted FWT if baseline.json present)
    if rows:
        # Build A (E x T) matrix of mAP50-95
        E = len(rows)
        T = len(args.domains)
        A: list[list[float]] = [[math.nan for _ in range(T)] for _ in range(E)]
        for i, r in enumerate(rows):
            for j, d in enumerate(args.domains):
                v = r.get(f"{d}_mAP50_95")
                if isinstance(v, (int, float)):
                    A[i][j] = float(v)

        def mean_ignore_nan(vals: list[float]) -> float:
            xs = [float(x) for x in vals if not math.isnan(x)]
            return float(sum(xs) / len(xs)) if xs else math.nan

        # ACC_final = mean over domains of last row
        acc_final = mean_ignore_nan(A[-1])

        # ACC_mean_over_time = mean over i of mean over seen domains 0..i
        accs_over_time: list[float] = []
        for i in range(E):
            seen_vals = [A[i][j] for j in range(min(i + 1, T))]
            accs_over_time.append(mean_ignore_nan(seen_vals))
        acc_mean_over_time = mean_ignore_nan(accs_over_time)

        # BWT (Chaudhry): mean over j=0..T-2 of (A[T-1,j] - A[j,j])
        bwt_per: dict[str, float] = {}
        bwt_vals: list[float] = []
        last_row = A[-1]
        if E > 1:
            K_max = min(T - 1, E - 1)
            for j in range(K_max):
                d = args.domains[j]
                a_jj = A[j][j]
                a_tj = last_row[j]
                if not math.isnan(a_jj) and not math.isnan(a_tj):
                    diff = float(a_tj - a_jj)
                    bwt_per[d] = diff
                    bwt_vals.append(diff)
        bwt_mean = float(sum(bwt_vals) / len(bwt_vals)) if bwt_vals else math.nan

        # Forgetting (Chaudhry 2018): max_{t=k..T-1} a_{k,t} - a_{k,T}; mean over k=1..T-1
        forget_per: dict[str, float] = {}
        forget_vals: list[float] = []
        if E > 1:
            K_max = min(T - 1, E - 1)
            for j in range(K_max):
                d = args.domains[j]
                pre = [A[i][j] for i in range(j, E - 1)]  # exclude final row
                pre_no_nan = [x for x in pre if not math.isnan(x)]
                if pre_no_nan and not math.isnan(last_row[j]):
                    f = float(max(pre_no_nan) - last_row[j])
                    forget_per[d] = f
                    forget_vals.append(f)
        forget_mean = float(sum(forget_vals) / len(forget_vals)) if forget_vals else math.nan

        # Baseline-adjusted FWT from pre-task cells
        try:
            baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception:
            baselines = None
        fwt_per: dict[str, float] = {}
        fwt_vals: List[float] = []
        if baselines and isinstance(baselines, list):
            for j in range(1, T):
                d = args.domains[j]
                a_prev = A[j - 1][j] if j - 1 < E else math.nan
                b = baselines[j] if j < len(baselines) else None
                if not math.isnan(a_prev) and isinstance(b, (int, float)):
                    diff = float(a_prev) - float(b)
                    fwt_per[d] = diff
                    fwt_vals.append(diff)
        fwt_mean = float(sum(fwt_vals) / len(fwt_vals)) if fwt_vals else math.nan

        summary = {
            "final": {
                "ACC_final": acc_final,
                "ACC_mean_over_time": acc_mean_over_time,
                "BWT_mean": bwt_mean,
                "BWT_per_domain": bwt_per,
                "Forgetting_mean": forget_mean,
                "Forgetting_per_domain": forget_per,
                "FWT_mean": fwt_mean,
                "FWT_per_domain": fwt_per,
            },
            "rows": rows,
        }
        (args.outputs_root / "continual_learning_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


