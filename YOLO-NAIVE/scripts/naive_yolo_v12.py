#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Optional

from ultralytics import YOLO
try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
except Exception:
    _TBWriter = None
import gc
import subprocess


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
    # Absolute train/val; omit "path" to avoid Ultralytics prefixing dataset root
    y = out_dir / "data.yaml"
    y.write_text(
        f"train: {train_txt.resolve()}\nval: {val_txt.resolve()}\nnames: {class_names}\n",
        encoding="utf-8",
    )
    return y


class _NoOpWriter:
    def add_scalar(self, *args, **kwargs):
        return None
    def flush(self):
        return None
    def close(self):
        return None


def main() -> None:
    ap = argparse.ArgumentParser("Naive Fine-tuning with YOLOv12 on ROAD (Domain-Incremental)")
    ap.add_argument("--dataset_root", type=Path, default=Path("YOLO-ER/datasets"))
    ap.add_argument("--outputs_root", type=Path, default=Path("YOLO-NAIVE/optimal_output"))
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--model", type=str, default=os.getenv("YOLO_MODEL", "yolo12s.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", type=str, default="disk", help="ram|disk|none")
    ap.add_argument("--multi_scale", action="store_true")
    ap.add_argument("--mosaic", type=float, default=0.5)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    ensure_dir(args.outputs_root)
    ensure_dir(args.outputs_root / "models")
    ensure_dir(args.outputs_root / "tensorboard_logs")
    writer = _NoOpWriter() if _TBWriter is None else _TBWriter(log_dir=str(args.outputs_root / "tensorboard_logs"))
    eta_log = args.outputs_root / "eta_progress.log"

    # Class names
    names_path = args.dataset_root.parent / "configs" / "names.json"
    if not names_path.exists():
        raise FileNotFoundError(
            "names.json not found under YOLO-ER/configs; run YOLO-ER/scripts/01_convert_coco_to_yolo.py first"
        )
    class_names: List[str] = json.loads(names_path.read_text(encoding="utf-8"))

    # Compute zero-shot baselines (mAP50-95 and mAP50) for authentic FWT
    baseline_path = args.outputs_root / "baseline.json"
    try:
        base_model = YOLO(args.model)
        baselines_95: List[Optional[float]] = []
        baselines_50: List[Optional[float]] = []
        for d in args.domains:
            dyaml = args.dataset_root / d / "data_val_only.yaml"
            res = base_model.val(data=str(dyaml), split="val", workers=int(args.workers))
            mp95 = res.results_dict.get("metrics/mAP50-95(B)") if hasattr(res, "results_dict") else None
            mp50 = res.results_dict.get("metrics/mAP50(B)") if hasattr(res, "results_dict") else None
            baselines_95.append(float(mp95) if mp95 is not None else None)
            baselines_50.append(float(mp50) if mp50 is not None else None)
        baseline_payload = {"mAP50_95": baselines_95, "mAP50": baselines_50}
        baseline_path.write_text(json.dumps(baseline_payload, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Eval artifacts
    eval_json = args.outputs_root / "continual_eval_matrix.json"
    eval_csv = args.outputs_root / "continual_eval_matrix.csv"
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
        header += ["avg_seen_mAP50_95", "avg_seen_mAP50"]
        eval_csv.write_text(",".join(header) + "\n", encoding="utf-8")

    # ETA callbacks
    t_epoch0 = None
    t_last = None
    t_epochs_total = None

    def on_train_epoch_start(trainer):
        nonlocal t_epoch0, t_last
        t_epoch0 = time.perf_counter()
        t_last = t_epoch0
        try:
            # total epochs known from trainer
            nonlocal t_epochs_total
            t_epochs_total = int(getattr(trainer, "epochs", int(args.epochs)))
        except Exception:
            pass
        try:
            total_iters = int(getattr(trainer, "nb", len(getattr(trainer, "dataloader", []))) or 0)
        except Exception:
            total_iters = 0
        msg = f"[EPOCH_START] total_iters={total_iters} epoch={trainer.epoch+1}/{trainer.epochs} time={now_str()}"
        print(msg, flush=True)
        try:
            with eta_log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

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
        # GPU mem
        gpumem = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpumem = torch.cuda.memory_reserved() / (1024 ** 3)
        except Exception:
            pass
        # ETA remaining for this epoch
        rem_iter = max(0, total - i_it)
        eta_s = (elapsed / max(1, i_it)) * rem_iter if i_it > 0 else 0.0
        msg = (
            f"[ITER] iter={i_it}/{total} epoch={trainer.epoch+1}/{trainer.epochs} "
            f"t={elapsed:7.2f}s dt={dt:6.3f}s imgs/s={imgsps:7.2f} GPUmem={gpumem:5.2f}GB "
            f"remaining={eta_s:7.1f}s"
        )
        print(msg, flush=True)
        try:
            with eta_log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass
        t_last = now

    def on_train_epoch_end(trainer):
        # rough total ETA across epochs
        try:
            e_idx = int(getattr(trainer, "epoch", 0)) + 1
            frac = e_idx / float(t_epochs_total or int(args.epochs) or 1)
        except Exception:
            frac = 0.0
        msg = f"[EPOCH_END] epoch={trainer.epoch+1}/{trainer.epochs} progress={frac:5.2%} time={now_str()}"
        print(msg, flush=True)
        try:
            with eta_log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass
        try:
            metrics = getattr(trainer, "metrics", None) or getattr(trainer.validator, "metrics", {}) or {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(k, v, getattr(trainer, "epoch", 0))
            writer.flush()
        except Exception:
            pass

    # Initialize model once; keep weights across domains (naive FT)
    model = YOLO(args.model)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    for i, domain in enumerate(args.domains, start=1):
        seen_domains = args.domains[:i]
        exp_dir = args.outputs_root / f"exp{i}_{domain}"
        ensure_dir(exp_dir)

        cur_train = load_lines(args.dataset_root / domain / "train.txt")
        cur_val = load_lines(args.dataset_root / domain / "val.txt")
        if not cur_train or not cur_val:
            raise FileNotFoundError(
                f"Missing train/val lists for domain '{domain}' under {args.dataset_root / domain}"
            )

        train_txt = exp_dir / "train.txt"
        val_txt = exp_dir / "val.txt"
        write_lines(train_txt, cur_train)
        # Train-time validation: union of seen domains for domain-incremental consistency
        val_seen: List[str] = []
        for d_seen in seen_domains:
            val_seen.extend(load_lines(args.dataset_root / d_seen / "val.txt"))
        write_lines(val_txt, val_seen)
        data_yaml = build_yaml(train_txt, val_txt, class_names, exp_dir)

        if not args.dry_run:
            cache_mode = str(args.cache).lower().strip()
            cache_val: object = False
            if cache_mode == "ram":
                cache_val = "ram"
            elif cache_mode == "disk":
                cache_val = "disk"

            print(
                f"[{now_str()}] [exp{i}:{domain}] train start: {args.model}, epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}"
            )
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

            # Save best
            best = getattr(getattr(model, "trainer", None), "best", None)
            best_weights_path: Optional[Path] = None
            if best:
                bp = Path(best)
                if bp.exists():
                    (args.outputs_root / "models").mkdir(parents=True, exist_ok=True)
                    dst = args.outputs_root / "models" / f"exp{i}_{domain}_best.pt"
                    dst.write_bytes(bp.read_bytes())
                    best_weights_path = dst

            # Evaluate on seen domains
            row: Dict[str, Optional[float]] = {"experience": i}
            for d in args.domains:
                k95 = f"{d}_mAP50_95"; k50 = f"{d}_mAP50"
                if d in seen_domains:
                    dyaml = exp_dir / f"val_only_{d}.yaml"
                    dyaml.write_text(
                        f"train: {train_txt.resolve()}\nval: {(args.dataset_root / d / 'val.txt').resolve()}\nnames: {class_names}\n",
                        encoding="utf-8",
                    )
                    res = model.val(data=str(dyaml), split="val", workers=int(args.workers))
                    mp95 = res.results_dict.get("metrics/mAP50-95(B)") if hasattr(res, "results_dict") else None
                    mp50 = res.results_dict.get("metrics/mAP50(B)") if hasattr(res, "results_dict") else None
                    row[k95] = float(mp95) if mp95 is not None else None
                    row[k50] = float(mp50) if mp50 is not None else None
                    # Extended AP via helper script (AP@0.20, AP@0.50, AP@0.75, AP@[0.50:0.90], AP@[0.50:0.95])
                    try:
                        if best_weights_path and best_weights_path.exists():
                            out_dir = args.outputs_root / "analysis_extended_ap" / f"exp{i}_{domain}" / d
                            out_dir.mkdir(parents=True, exist_ok=True)
                            cmd = [
                                "python",
                                str(Path.cwd() / "compute_extended_ap.py"),
                                "--weights", str(best_weights_path),
                                "--data_yaml", str(dyaml),
                                "--names_json", str(names_path),
                                "--out_dir", str(out_dir),
                            ]
                            subprocess.run(cmd, check=False, capture_output=True)
                            # Read latest extended AP JSON and map into row
                            jfiles = sorted(out_dir.glob("extended_ap_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                            if jfiles:
                                info = json.loads(jfiles[0].read_text(encoding="utf-8"))
                                mets = (info or {}).get("metrics", {})
                                row[f"{d}_AP20"] = float(mets.get("AP@0.20")) if mets.get("AP@0.20") is not None else None
                                row[f"{d}_AP50"] = float(mets.get("AP@0.50")) if mets.get("AP@0.50") is not None else None
                                row[f"{d}_AP75"] = float(mets.get("AP@0.75")) if mets.get("AP@0.75") is not None else None
                                row[f"{d}_AP50_90"] = float(mets.get("AP@[0.50:0.90]")) if mets.get("AP@[0.50:0.90]") is not None else None
                                row[f"{d}_AP50_95"] = float(mets.get("AP@[0.50:0.95]")) if mets.get("AP@[0.50:0.95]") is not None else None
                    except Exception:
                        pass
                else:
                    row[k95] = None
                    row[k50] = None
                    row[f"{d}_AP20"] = None
                    row[f"{d}_AP50"] = None
                    row[f"{d}_AP75"] = None
                    row[f"{d}_AP50_90"] = None
                    row[f"{d}_AP50_95"] = None

            # Pre-task off-diagonal for FWT (A[i-1, i])
            if i < len(args.domains):
                nxt = args.domains[i]
                dyaml_pre = exp_dir / f"val_only_{nxt}_pre.yaml"
                dyaml_pre.write_text(
                    f"train: {train_txt.resolve()}\nval: {(args.dataset_root / nxt / 'val.txt').resolve()}\nnames: {class_names}\n",
                    encoding="utf-8",
                )
                res_pre = model.val(data=str(dyaml_pre), split="val", workers=int(args.workers))
                mp95_pre = res_pre.results_dict.get("metrics/mAP50-95(B)") if hasattr(res_pre, "results_dict") else None
                mp50_pre = res_pre.results_dict.get("metrics/mAP50(B)") if hasattr(res_pre, "results_dict") else None
                row[f"{nxt}_mAP50_95"] = float(mp95_pre) if mp95_pre is not None else row.get(f"{nxt}_mAP50_95")
                row[f"{nxt}_mAP50"] = float(mp50_pre) if mp50_pre is not None else row.get(f"{nxt}_mAP50")

            seen_vals95 = [row.get(f"{d}_mAP50_95") for d in seen_domains]
            seen_vals95 = [float(v) for v in seen_vals95 if v is not None]
            row["avg_seen_mAP50_95"] = (
                float(sum(seen_vals95) / max(1, len(seen_vals95))) if seen_vals95 else None
            )
            seen_vals50 = [row.get(f"{d}_mAP50") for d in seen_domains]
            seen_vals50 = [float(v) for v in seen_vals50 if v is not None]
            row["avg_seen_mAP50"] = (
                float(sum(seen_vals50) / max(1, len(seen_vals50))) if seen_vals50 else None
            )

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
                avg95 = "" if row.get("avg_seen_mAP50_95") is None else f"{row['avg_seen_mAP50_95']:.6f}"
                avg50 = "" if row.get("avg_seen_mAP50") is None else f"{row['avg_seen_mAP50']:.6f}"
                line += [avg95, avg50]
                f.write(",".join(line) + "\n")

            print(f"[{now_str()}] [exp{i}:{domain}] train+eval done")

    # Final summary
    try:
        writer.flush()
        writer.close()
    except Exception:
        pass

    if rows:
        # Compute Average Accuracy (final row over all domains present), BWT, FWT, and Chaudhry Forgetting
        def safe_mean(vals: List[Optional[float]]) -> Optional[float]:
            xs = [float(v) for v in vals if v is not None]
            return float(sum(xs) / max(1, len(xs))) if xs else None

        final_row = rows[-1]
        avg_acc_95 = safe_mean([final_row.get(f"{d}_mAP50_95") for d in args.domains])
        avg_acc_50 = safe_mean([final_row.get(f"{d}_mAP50") for d in args.domains])

        # BWT (Chaudhry): mean over j=0..T-2 of (A[T-1,j] - A[j,j]) using rows/diagonal
        try:
            bwt_95_vals: List[float] = []
            bwt_50_vals: List[float] = []
            K_max = min(len(args.domains) - 1, len(rows) - 1)
            for j in range(max(0, K_max)):
                d = args.domains[j]
                diag_95 = rows[j].get(f"{d}_mAP50_95")
                last_95 = final_row.get(f"{d}_mAP50_95")
                if isinstance(diag_95, (int, float)) and isinstance(last_95, (int, float)):
                    bwt_95_vals.append(float(last_95) - float(diag_95))
                diag_50 = rows[j].get(f"{d}_mAP50")
                last_50 = final_row.get(f"{d}_mAP50")
                if isinstance(diag_50, (int, float)) and isinstance(last_50, (int, float)):
                    bwt_50_vals.append(float(last_50) - float(diag_50))
            bwt_95 = float(sum(bwt_95_vals) / max(1, len(bwt_95_vals))) if bwt_95_vals else None
            bwt_50 = float(sum(bwt_50_vals) / max(1, len(bwt_50_vals))) if bwt_50_vals else None
        except Exception:
            bwt_95 = None
            bwt_50 = None

        # FWT: baseline-adjusted using zero-shot baseline if available (for both mAP50-95 and mAP50)
        fwt_95_vals: List[float] = []
        fwt_50_vals: List[float] = []
        try:
            baseline_data = json.loads((args.outputs_root / "baseline.json").read_text(encoding="utf-8"))
        except Exception:
            baseline_data = None
        try:
            if isinstance(baseline_data, dict):
                b95_list = baseline_data.get("mAP50_95")
                b50_list = baseline_data.get("mAP50")
            else:
                # Backward compatible: if baseline.json is a list, treat it as mAP50-95 baselines only
                b95_list = baseline_data if isinstance(baseline_data, list) else None
                b50_list = None

            for i_dom in range(1, len(args.domains)):
                d = args.domains[i_dom]
                # Pre-task value resides in row i-1 at column d
                if i_dom <= len(rows) - 1:
                    r_prev = rows[i_dom - 1]
                    v95_prev = r_prev.get(f"{d}_mAP50_95")
                    v50_prev = r_prev.get(f"{d}_mAP50")
                    # mAP50-95
                    if isinstance(v95_prev, (int, float)):
                        b95 = b95_list[i_dom] if (isinstance(b95_list, list) and i_dom < len(b95_list)) else None
                        fwt_95_vals.append(float(v95_prev) - float(b95)) if isinstance(b95, (int, float)) else fwt_95_vals.append(float(v95_prev))
                    # mAP50
                    if isinstance(v50_prev, (int, float)):
                        b50 = b50_list[i_dom] if (isinstance(b50_list, list) and i_dom < len(b50_list)) else None
                        fwt_50_vals.append(float(v50_prev) - float(b50)) if isinstance(b50, (int, float)) else fwt_50_vals.append(float(v50_prev))
        except Exception:
            pass
        fwt_95 = float(sum(fwt_95_vals) / max(1, len(fwt_95_vals))) if fwt_95_vals else None
        fwt_50 = float(sum(fwt_50_vals) / max(1, len(fwt_50_vals))) if fwt_50_vals else None

        # Forgetting (Chaudhry 2018) on mAP50-95
        forget_per_95: Dict[str, float] = {}
        forget_vals_95: List[float] = []
        E = len(rows); T = len(args.domains)
        if E > 1:
            K_max = min(T - 1, E - 1)
            for j in range(K_max):
                d = args.domains[j]
                pre: List[float] = []
                for i in range(j, E - 1):
                    v = rows[i].get(f"{d}_mAP50_95")
                    if isinstance(v, (int, float)):
                        pre.append(float(v))
                last_v = rows[-1].get(f"{d}_mAP50_95")
                if pre and isinstance(last_v, (int, float)):
                    f = float(max(pre) - float(last_v))
                    forget_per_95[d] = f
                    forget_vals_95.append(f)
        forget_mean_95 = float(sum(forget_vals_95) / max(1, len(forget_vals_95))) if forget_vals_95 else None

        # Collect final extended APs from analysis_extended_ap for last experience
        try:
            final_exp = len(args.domains)
            final_dom = args.domains[-1]
            exp_dir_ext = args.outputs_root / "analysis_extended_ap" / f"exp{final_exp}_{final_dom}"
            per_domain_ext: Dict[str, Dict[str, float]] = {}
            keys = ["AP@0.20", "AP@0.50", "AP@0.75", "AP@[0.50:0.90]", "AP@[0.50:0.95]"]
            for d in args.domains:
                jfiles = sorted((exp_dir_ext / d).glob("extended_ap_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not jfiles:
                    continue
                try:
                    info = json.loads(jfiles[0].read_text(encoding="utf-8"))
                except Exception:
                    continue
                mets = (info or {}).get("metrics", {})
                per_domain_ext[d] = {k: float(mets.get(k)) for k in keys if mets.get(k) is not None}
            averages_ext: Dict[str, float] = {}
            for k in keys:
                vals = [pd[k] for pd in per_domain_ext.values() if k in pd]
                if vals:
                    try:
                        averages_ext[k] = float(sum(vals) / len(vals))
                    except Exception:
                        pass
        except Exception:
            per_domain_ext = {}
            averages_ext = {}

        summary = {
            "final": {
                "average_accuracy_mAP50_95": avg_acc_95,
                "average_accuracy_mAP50": avg_acc_50,
                "backward_transfer_mAP50_95": bwt_95,
                "backward_transfer_mAP50": bwt_50,
                "forward_transfer_mAP50_95": fwt_95,
                "forward_transfer_mAP50": fwt_50,
                "Forgetting_mean": forget_mean_95,
                "Forgetting_per_domain": forget_per_95,
                "ExtendedAP_final_per_domain": per_domain_ext,
                "ExtendedAP_final_avg": averages_ext,
            },
            "rows": rows,
        }
        (args.outputs_root / "continual_learning_metrics.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()


