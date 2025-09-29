#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from pathlib import Path

from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
import subprocess
import math

DOMAINS = ["sunny", "overcast", "night", "snowy"]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_lines(p: Path) -> list[str]:
    if not p.exists():
        return []
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def write_lines(p: Path, lines: list[str]) -> None:
    p.write_text("\n".join(lines), encoding="utf-8")

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class DomainReservoir:
    """Domain-aware replay buffer with equal per-domain quotas.

    Mirrors YOLO-GDUMB's reservoir behavior to enable fair comparison.
    """

    def __init__(self, domains: list[str], quota_per_domain: int, seed: int = 42) -> None:
        self.domains = list(domains)
        self.quota = int(max(1, quota_per_domain))
        self.buffers: dict[str, list[str]] = {d: [] for d in self.domains}
        self.seen: dict[str, int] = {d: 0 for d in self.domains}
        random.seed(seed)

    def update_with_list(self, domain: str, img_paths: list[str]) -> None:
        if domain not in self.buffers:
            return
        for p in img_paths:
            self.seen[domain] += 1
            buf = self.buffers[domain]
            if len(buf) < self.quota:
                buf.append(p)
            else:
                j = random.randint(0, self.seen[domain])
                if j < self.quota:
                    buf[j] = p

    def memory_union(self, upto_domains: list[str]) -> list[str]:
        out: list[str] = []
        for d in upto_domains:
            out.extend(self.buffers.get(d, []))
        # Deduplicate preserving order
        seen_paths: set[str] = set()
        uniq: list[str] = []
        for p in out:
            if p not in seen_paths:
                uniq.append(p)
                seen_paths.add(p)
        return uniq

    def load_flat_union(self, paths: list[str]) -> None:
        """Best-effort reconstruction of domain buffers from a flat union list.

        Paths are assigned to domain buffers by inferring the domain from the path.
        """
        for p in paths:
            pd = infer_domain_from_path(p, self.domains)
            if pd is None:
                continue
            self.update_with_list(pd, [p])


def infer_domain_from_path(p: str, domains: list[str]) -> str | None:
    lp = p.replace("\\", "/").lower()
    for d in domains:
        if f"/{d.lower()}/" in lp:
            return d
    return None

def build_exp_yaml(dataset_root: Path, exp_dir: Path, class_names: list[str]) -> Path:
    # Use absolute paths for train/val lists and omit `path` to prevent Ultralytics
    # from incorrectly joining paths.
    train_txt_abs = (exp_dir / "train.txt").resolve()
    val_txt_abs = (exp_dir / "val.txt").resolve()
    yaml_text = f"train: {train_txt_abs}\nval: {val_txt_abs}\nnames: {class_names}\n"
    data_yaml = exp_dir / "data.yaml"
    data_yaml.write_text(yaml_text, encoding="utf-8")
    return data_yaml

def format_eta_line(exp_idx: int, domain: str, msg: str) -> str:
    return f"[{now_str()}] [exp{exp_idx}:{domain}] {msg}"

def main():
    ap = argparse.ArgumentParser(description="YOLO ER with local replay on ROAD (real)")
    ap.add_argument("--dataset_root", type=Path, default=Path("YOLO-ER/datasets"))
    ap.add_argument("--outputs_root", type=Path, default=Path("YOLO-ER/optimal_output"))
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--model", type=str, default=os.getenv("YOLO_MODEL", "yolo12s.pt"), help="Ultralytics weights or checkpoint path")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--replay_ratio", type=float, default=0.5, help="Max replay/current ratio for ER (e.g., 0.5 → at most 1 replay per 2 current)")
    ap.add_argument("--mem_size", type=int, default=5000, help="Replay buffer size in images (0 disables replay)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume_seen", nargs="*", default=[], help="Domains already trained (e.g. sunny)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    random.seed(args.seed)

    ensure_dir(args.outputs_root)
    ensure_dir(args.outputs_root / "models")
    ensure_dir(args.outputs_root / "tensorboard_logs")

    # TensorBoard writer (events)
    tb_dir = args.outputs_root / "tensorboard_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    eta_log = args.outputs_root / "eta_progress.log"
    train_log = args.outputs_root / "training_log.txt"
    eval_json = args.outputs_root / "continual_eval_matrix.json"
    eval_csv = args.outputs_root / "continual_eval_matrix.csv"

    # Load canonical class names if present
    names_path = args.dataset_root.parent / "configs" / "names.json"
    if names_path.exists():
        class_names = json.loads(names_path.read_text(encoding="utf-8"))
    else:
        first_yaml = args.dataset_root / args.domains[0] / "data_val_only.yaml"
        if not first_yaml.exists():
            raise FileNotFoundError("Class names not found: run 01_convert_coco_to_yolo.py first")
        import yaml  # type: ignore
        class_names = yaml.safe_load(first_yaml.read_text(encoding="utf-8")).get("names", [])

    # Initialize YOLO model (can be pretrained or a previous exp best)
    model = YOLO(args.model)

    # ETA + TensorBoard callbacks
    t_epoch0 = None
    t_last = None

    def on_train_epoch_start(trainer):
        nonlocal t_epoch0, t_last
        import time as _time  # local import
        t_epoch0 = _time.perf_counter()
        t_last = t_epoch0
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
        import time as _time
        now = _time.perf_counter()
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
            import torch  # type: ignore
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
        # TensorBoard metrics
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

    # Domain-aware reservoir (image paths only) — consistent with GDUMB
    quota = int(args.mem_size // max(1, len(args.domains)))
    reservoir = DomainReservoir(args.domains, quota, seed=args.seed)

    # Persistent replay buffer load (flat union list)
    rb_path = args.outputs_root / "replay_paths.json"
    if rb_path.exists():
        try:
            rp = json.loads(rb_path.read_text(encoding="utf-8"))
            if isinstance(rp, list) and rp:
                reservoir.load_flat_union(rp[: args.mem_size])
                print(f"[replay] loaded {min(len(rp), args.mem_size)} items into domain-aware reservoir")
        except Exception:
            pass

    # Compute zero-shot baseline per domain (initial model) for authentic FWT
    baseline_path = args.outputs_root / "baseline.json"
    try:
        base_model = YOLO(args.model)
        baselines: list[float] = []
        for d in args.domains:
            dyaml = args.dataset_root / d / "data_val_only.yaml"
            res = base_model.val(data=str(dyaml), split="val", workers=int(args.workers))
            mp95 = res.results_dict.get("metrics/mAP50-95(B)") if hasattr(res, "results_dict") else None
            baselines.append(float(mp95) if mp95 is not None else None)  # type: ignore
        baseline_path.write_text(json.dumps(baselines, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Load existing eval rows if present (to append)
    if eval_json.exists():
        try:
            eval_rows = json.loads(eval_json.read_text(encoding="utf-8"))
        except Exception:
            eval_rows = []
    else:
        eval_rows = []

    # Ensure CSV header exists once
    if not eval_csv.exists() or eval_csv.read_text(encoding="utf-8").strip() == "":
        header = ["experience"]
        header += [f"{d}_mAP50_95" for d in args.domains]
        header += [f"{d}_mAP50" for d in args.domains]
        header += [f"{d}_AP20" for d in args.domains]
        header += [f"{d}_AP50" for d in args.domains]
        header += [f"{d}_AP75" for d in args.domains]
        header += [f"{d}_AP50_90" for d in args.domains]
        header += [f"{d}_AP50_95" for d in args.domains]
        eval_csv.write_text(
            ",".join(header) + "\n",
            encoding="utf-8",
        )

    # If resuming and matrix is empty, backfill the first seen domain from runs/results.csv
    if args.resume_seen and not eval_rows:
        runs_csv = args.outputs_root / "runs" / "results.csv"
        if runs_csv.exists():
            import csv as _csv
            with runs_csv.open("r", encoding="utf-8") as f:
                recs = list(_csv.DictReader(f))
            if recs:
                last = recs[-1]
                backfill = {"experience": 1}
                for d in args.domains:
                    if d == args.resume_seen[0]:
                        try:
                            backfill[f"{d}_mAP50_95"] = float(last.get("metrics/mAP50-95(B)", ""))
                        except Exception:
                            backfill[f"{d}_mAP50_95"] = None
                    else:
                        backfill[f"{d}_mAP50_95"] = None
                eval_rows.append(backfill)
                eval_json.write_text(json.dumps(eval_rows, indent=2), encoding="utf-8")
                with eval_csv.open("a", encoding="utf-8") as f:
                    vals = ["" if backfill[f"{d}_mAP50_95"] is None else f"{backfill[f'{d}_mAP50_95']:.6f}" for d in args.domains]
                    f.write(",".join(["1"] + vals) + "\n")

    # Determine starting experience index and domains to train
    seen_domains: list[str] = list(args.resume_seen)
    base_exp_idx = len(seen_domains)
    domains_to_train = [d for d in args.domains if d not in set(seen_domains)]

    for i, domain in enumerate(domains_to_train, start=1):
        exp_idx = base_exp_idx + i
        # Mark domain as seen before evaluation
        if domain not in seen_domains:
            seen_domains.append(domain)

        exp_dir = args.outputs_root / f"exp{exp_idx}_{domain}"
        ensure_dir(exp_dir)
        (args.outputs_root / "models").mkdir(parents=True, exist_ok=True)

        # Build merged train list with capped replay proportion
        cur_train = load_lines(args.dataset_root / domain / "train.txt")
        # Update reservoir with current domain samples
        reservoir.update_with_list(domain, cur_train)
        replay_union = reservoir.memory_union(seen_domains)
        # Cap replay at replay_ratio * len(current)
        max_replay = int(max(0, min(len(replay_union), int(args.replay_ratio * len(cur_train)))))
        replay_sample = []
        if max_replay > 0:
            try:
                replay_sample = random.sample(replay_union, max_replay)
            except Exception:
                replay_sample = list(replay_union)[:max_replay]
        merged = list(cur_train) + replay_sample
        random.shuffle(merged)
        write_lines(exp_dir / "train.txt", merged)
        # Train-time validation: union of seen domains (consistent with domain-incremental retention)
        val_seen: list[str] = []
        for d in seen_domains:
            val_seen.extend(load_lines(args.dataset_root / d / "val.txt"))
        write_lines(exp_dir / "val.txt", val_seen)

        data_yaml = build_exp_yaml(args.dataset_root, exp_dir, class_names)

        with eta_log.open("a", encoding="utf-8") as f:
            f.write(format_eta_line(exp_idx, domain, f"train start: {args.model}, epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, replay={len(replay_union)}") + "\n")

        # Train
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
            mosaic=0.5,
            mixup=0.1,
            close_mosaic=5,
            cache="disk",
        )

        # Save best checkpoint
        best = getattr(getattr(model, "trainer", None), "best", None)
        best_weights_path = None
        if best:
            best_p = Path(best)
            if best_p.exists():
                dst = args.outputs_root / "models" / f"exp{exp_idx}_{domain}_best.pt"
                dst.write_bytes(best_p.read_bytes())
                best_weights_path = dst

        # Evaluate on all seen domains (build a row) BEFORE buffer update
        row = {"experience": exp_idx}
        for d in args.domains:
            if d in seen_domains:
                dyaml = args.dataset_root / d / "data_val_only.yaml"
                res = model.val(data=str(dyaml), split="val")
                mp95 = res.results_dict.get("metrics/mAP50-95(B)") if hasattr(res, "results_dict") else None
                mp50 = res.results_dict.get("metrics/mAP50(B)") if hasattr(res, "results_dict") else None
                row[f"{d}_mAP50_95"] = mp95
                row[f"{d}_mAP50"] = mp50
                # Extended AP via helper script
                try:
                    if best_weights_path and best_weights_path.exists():
                        out_dir = args.outputs_root / "analysis_extended_ap" / f"exp{exp_idx}_{domain}" / d
                        out_dir.mkdir(parents=True, exist_ok=True)
                        cmd = [
                            "python",
                            str(Path.cwd() / "compute_extended_ap.py"),
                            "--weights",
                            str(best_weights_path),
                            "--data_yaml",
                            str(dyaml),
                            "--names_json",
                            str(args.dataset_root.parent / "configs" / "names.json"),
                            "--out_dir",
                            str(out_dir),
                        ]
                        subprocess.run(cmd, check=False, capture_output=True)
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

        # Pre-task off-diagonal for next domain (A[i-1, i]) to enable authentic FWT
        if exp_idx < len(args.domains):
            next_d = args.domains[exp_idx]
            dyaml_pre = args.dataset_root / next_d / "data_val_only.yaml"
            try:
                res_pre = model.val(data=str(dyaml_pre), split="val")
                mp95_pre = res_pre.results_dict.get("metrics/mAP50-95(B)") if hasattr(res_pre, "results_dict") else None
                mp50_pre = res_pre.results_dict.get("metrics/mAP50(B)") if hasattr(res_pre, "results_dict") else None
                if mp95_pre is not None:
                    row[f"{next_d}_mAP50_95"] = float(mp95_pre)
                if mp50_pre is not None:
                    row[f"{next_d}_mAP50"] = float(mp50_pre)
            except Exception:
                pass

        # Persist
        eval_rows.append(row)
        eval_json.write_text(json.dumps(eval_rows, indent=2), encoding="utf-8")
        with eval_csv.open("a", encoding="utf-8") as f:
            csv_line = [str(exp_idx)]
            csv_line += ["" if row.get(f"{d}_mAP50_95") is None else f"{row[f'{d}_mAP50_95']:.6f}" for d in args.domains]
            csv_line += ["" if row.get(f"{d}_mAP50") is None else f"{row[f'{d}_mAP50']:.6f}" for d in args.domains]
            csv_line += ["" if row.get(f"{d}_AP20") is None else f"{row[f'{d}_AP20']:.6f}" for d in args.domains]
            csv_line += ["" if row.get(f"{d}_AP50") is None else f"{row[f'{d}_AP50']:.6f}" for d in args.domains]
            csv_line += ["" if row.get(f"{d}_AP75") is None else f"{row[f'{d}_AP75']:.6f}" for d in args.domains]
            csv_line += ["" if row.get(f"{d}_AP50_90") is None else f"{row[f'{d}_AP50_90']:.6f}" for d in args.domains]
            csv_line += ["" if row.get(f"{d}_AP50_95") is None else f"{row[f'{d}_AP50_95']:.6f}" for d in args.domains]
            f.write(",".join(csv_line) + "\n")

        with eta_log.open("a", encoding="utf-8") as f:
            f.write(format_eta_line(exp_idx, domain, "train+eval done") + "\n")

        # Persist reservoir as a flat union list (cap at mem_size)
        try:
            all_union = reservoir.memory_union(args.domains)
            rb_path.write_text(json.dumps(all_union[: args.mem_size], indent=2), encoding="utf-8")
        except Exception:
            pass

    # Final training log marker
    with train_log.open("a", encoding="utf-8") as f:
        f.write(f"{now_str()} Completed YOLO ER run across domains: {args.domains}\n")
    try:
        writer.flush(); writer.close()
    except Exception:
        pass

    # Write baseline-adjusted FWT and other CL metrics to continual_learning_metrics.json
    try:
        # Build matrix A (E x T) of mAP50-95
        E = len(eval_rows)
        T = len(args.domains)
        A: list[list[float]] = [[math.nan for _ in range(T)] for _ in range(E)]
        for i, r in enumerate(eval_rows):
            for j, d in enumerate(args.domains):
                v = r.get(f"{d}_mAP50_95")
                if isinstance(v, (int, float)):
                    A[i][j] = float(v)

        def mean_ignore_nan(vals: list[float]) -> float:
            xs = [float(x) for x in vals if not math.isnan(x)]
            return float(sum(xs) / len(xs)) if xs else math.nan

        # ACC_final: mean over domains of last row
        acc_final = mean_ignore_nan(A[-1]) if E > 0 else math.nan

        # ACC_mean_over_time: mean over i of mean over seen domains 0..i
        accs_over_time: list[float] = []
        for i in range(E):
            seen_vals = [A[i][j] for j in range(min(i + 1, T))]
            accs_over_time.append(mean_ignore_nan(seen_vals))
        acc_mean_over_time = mean_ignore_nan(accs_over_time)

        # BWT (Chaudhry): mean over j=0..T-2 of (A[T-1,j] - A[j,j])
        bwt_per: dict[str, float] = {}
        bwt_vals: list[float] = []
        if E > 1:
            last = A[-1]
            K_max = min(T - 1, E - 1)
            for j in range(K_max):
                d = args.domains[j]
                a_jj = A[j][j]
                a_tj = last[j]
                if not math.isnan(a_jj) and not math.isnan(a_tj):
                    diff = float(a_tj - a_jj)
                    bwt_per[d] = diff
                    bwt_vals.append(diff)
        bwt_mean = float(sum(bwt_vals) / len(bwt_vals)) if bwt_vals else math.nan

        # Forgetting (Chaudhry 2018): max over t=k..T-1 minus final; mean over k=1..T-1
        forget_per: dict[str, float] = {}
        forget_vals: list[float] = []
        if E > 1:
            last = A[-1]
            K_max = min(T - 1, E - 1)
            for j in range(K_max):
                d = args.domains[j]
                pre = [A[i][j] for i in range(j, E - 1)]  # exclude final row
                pre_no_nan = [x for x in pre if not math.isnan(x)]
                if pre_no_nan and not math.isnan(last[j]):
                    f = float(max(pre_no_nan) - last[j])
                    forget_per[d] = f
                    forget_vals.append(f)
        forget_mean = float(sum(forget_vals) / len(forget_vals)) if forget_vals else math.nan

        # Baseline-adjusted FWT using baseline.json
        fwt_per: dict[str, float] = {}
        fwt_vals: list[float] = []
        baselines = None
        try:
            baselines = json.loads((args.outputs_root / "baseline.json").read_text(encoding="utf-8"))
        except Exception:
            baselines = None
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

        metrics = {
            "domains": args.domains,
            "num_experiences": E,
            "ACC_final": acc_final,
            "ACC_mean_over_time": acc_mean_over_time,
            "BWT_mean": bwt_mean,
            "BWT_per_domain": bwt_per,
            "Forgetting_mean": forget_mean,
            "Forgetting_per_domain": forget_per,
            "FWT_mean": fwt_mean,
            "FWT_per_domain": fwt_per,
        }

        out_json = args.outputs_root / "continual_learning_metrics.json"
        try:
            old = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
        except Exception:
            old = {}
        old.update(metrics)
        out_json.write_text(json.dumps(old, indent=2), encoding="utf-8")
    except Exception:
        pass

if __name__ == "__main__":
    main()


