#!/usr/bin/env python3
import argparse, json
from pathlib import Path

DOMAINS = ["sunny", "overcast", "snowy", "night"]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_coco(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    cats = sorted(d["categories"], key=lambda c: c["id"])  # ids expected 1..10
    id2name = {c["id"]: c["name"] for c in cats}
    name_order = [c["name"] for c in cats]
    return d, id2name, name_order

def write_split_txt(img_root: Path, images, out_txt: Path):
    lines = []
    for im in images:
        img_path = img_root / im["file_name"]
        lines.append(str(img_path.resolve()))
    out_txt.write_text("\n".join(lines), encoding="utf-8")

def coco_to_yolo_labels(coco, id2name, name_to_idx, img_root: Path, labels_root: Path):
    ensure_dir(labels_root)
    imgid_to_size = {im["id"]: (im["width"], im["height"], im["file_name"]) for im in coco["images"]}
    imgid_to_lines = {im["id"]: [] for im in coco["images"]}

    for a in coco["annotations"]:
        iid = a["image_id"]
        x, y, w, h = a["bbox"]
        W, H, _ = imgid_to_size[iid]
        if W <= 0 or H <= 0 or w <= 0 or h <= 0:
            continue
        xc = (x + w / 2.0) / float(W)
        yc = (y + h / 2.0) / float(H)
        wn = w / float(W)
        hn = h / float(H)
        cid = a["category_id"]
        cname = id2name.get(cid)
        if cname is None:
            continue
        cls = name_to_idx.get(cname)
        if cls is None:
            continue
        imgid_to_lines[iid].append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    for iid, (_, _, fname) in imgid_to_size.items():
        lbl = labels_root / (Path(fname).name.replace(".jpg", ".txt"))
        lines = imgid_to_lines.get(iid, [])
        lbl.write_text("\n".join(lines), encoding="utf-8")

def build_domain_yaml(domain_root: Path, names: list[str]):
    # Use absolute paths for train/val lists and omit `path` to prevent Ultralytics
    # from incorrectly joining paths.
    train_txt_abs = (domain_root / "train.txt").resolve()
    val_txt_abs = (domain_root / "val.txt").resolve()
    yaml_text = (
        f"train: {train_txt_abs}\n"
        f"val: {val_txt_abs}\n"
        f"names: {names}\n"
    )
    (domain_root / "data_val_only.yaml").write_text(yaml_text, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", type=Path, default=Path("road-real-annotations/data"))
    ap.add_argument("--out_root", type=Path, default=Path("YOLO-ER/datasets"))
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    args = ap.parse_args()

    ensure_dir(args.out_root)
    names_global = None

    for domain in args.domains:
        src_domain = args.coco_root / domain
        coco_train_p = src_domain / "annotations" / "train.json"
        coco_val_p   = src_domain / "annotations" / "test.json"
        img_root     = src_domain

        if not coco_train_p.exists() or not coco_val_p.exists():
            print(f"[WARN] Missing COCO JSON in {src_domain}, skipping {domain}")
            continue

        coco_train, id2name_train, names_train = load_coco(coco_train_p)
        coco_val,   id2name_val,   names_val   = load_coco(coco_val_p)
        if names_train != names_val:
            print(f"[WARN] Train/Val class lists differ in {domain}; using train.json order")

        names = names_train
        if names_global is None:
            names_global = names
        name_to_idx = {n: i for i, n in enumerate(names)}

        out_domain = args.out_root / domain
        ensure_dir(out_domain)
        # IMPORTANT: place labels alongside real images so Ultralytics can map
        # absolute image paths .../images/*.jpg -> .../labels/*.txt
        labels_dir = img_root / "labels"
        ensure_dir(labels_dir)

        coco_to_yolo_labels(coco_train, id2name_train, name_to_idx, img_root, labels_dir)
        coco_to_yolo_labels(coco_val,   id2name_val,   name_to_idx, img_root, labels_dir)

        # absolute paths are Windows-friendly; Ultralytics replaces images->labels
        write_split_txt(img_root, coco_train["images"], out_domain / "train.txt")
        write_split_txt(img_root, coco_val["images"],   out_domain / "val.txt")

        build_domain_yaml(out_domain, names)
        print(f"[OK] {domain}: labels={labels_dir}, splits=train/val.txt, yaml={out_domain / 'data_val_only.yaml'}")

    if names_global:
        names_json = args.out_root.parent / "configs" / "names.json"
        ensure_dir(names_json.parent)
        names_json.write_text(json.dumps(names_global, indent=2), encoding="utf-8")
        print(f"[OK] Saved canonical class list to {names_json}")

if __name__ == "__main__":
    main()


