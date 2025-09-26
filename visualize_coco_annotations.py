import argparse
import json
import re
from pathlib import Path
import cv2


DOMAIN_BY_BASENAME = {
    "2015-02-06-13-57-16_stereo_centre_01": "sunny",
    "2014-06-26-09-31-18_stereo_centre_02": "overcast",
    "2015-02-03-08-45-10_stereo_centre_04": "snowy",
    "2014-12-10-18-10-50_stereo_centre_02": "night",
}


def parse_idx_and_base(file_name: str):
    # Ex: images/2015-02-06-13-57-16_stereo_centre_01_frame_000002.jpg
    m = re.search(r"([^/\\]+)_frame_(\d+)\.jpg$", file_name)
    if not m:
        return None, None
    base = m.group(1)
    idx = int(m.group(2))
    return idx, base


def color_for_cat(cid: int):
    # deterministic pseudo-random color per class
    import random
    rnd = random.Random(cid * 9973)
    return (int(rnd.random() * 255), int(rnd.random() * 255), int(rnd.random() * 255))


def draw_boxes(img, anns, id2name):
    for a in anns:
        x, y, w, h = a["bbox"]
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        cid = a["category_id"]
        name = id2name.get(cid, str(cid))
        col = color_for_cat(cid)
        cv2.rectangle(img, p1, p2, col, 2)
        cv2.putText(img, name, (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize COCO annotations and export per-domain videos")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--domains", nargs="+", default=["sunny", "overcast", "snowy", "night"])
    parser.add_argument("--videos_dir", type=Path, default=Path("videos"))
    parser.add_argument("--save_video", type=int, default=1)
    parser.add_argument("--resume", type=int, default=1, help="If 1, skip drawing for frames that already exist in viz/")
    args = parser.parse_args()

    for domain in args.domains:
        ann_path = args.root / domain / "annotations" / "train.json"
        img_root = args.root / domain
        if not ann_path.exists():
            print(f"[WARN] Missing {ann_path}, skipping {domain}")
            continue

        coco = json.loads(ann_path.read_text(encoding="utf-8"))
        id2img = {im["id"]: im for im in coco["images"]}
        id2name = {c["id"]: c["name"] for c in coco["categories"]}

        img2anns = {i: [] for i in id2img}
        for a in coco["annotations"]:
            img2anns[a["image_id"]].append(a)

        items = []
        base_name = None
        for iid, im in id2img.items():
            idx, base = parse_idx_and_base(im["file_name"])
            if idx is None:
                continue
            items.append((idx, iid))
            if base_name is None:
                base_name = base
        items.sort(key=lambda x: x[0])

        writer = None
        if args.save_video and items:
            fps = 10.0
            if base_name:
                src = args.videos_dir / f"{base_name}.mp4"
                cap = cv2.VideoCapture(str(src))
                if cap.isOpened():
                    f = cap.get(cv2.CAP_PROP_FPS)
                    if f and f > 0:
                        fps = float(f)
                cap.release()
            h = id2img[items[0][1]]["height"]
            w = id2img[items[0][1]]["width"]
            out_path = args.root / domain / f"{domain}_annotations.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            print(f"[INFO] Writing video: {out_path} @ {fps:.2f} fps")

        viz_dir = args.root / domain / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        frames_written = 0
        for _, iid in items:
            im = id2img[iid]
            img_path = img_root / im["file_name"]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Cannot read {img_path}")
                continue
            anns = img2anns.get(iid, [])
            out_img = viz_dir / Path(im["file_name"]).name
            if args.resume and out_img.exists():
                # Skip re-draw if already exists
                if writer is not None:
                    # append existing image to writer for contiguous video
                    prior = cv2.imread(str(out_img))
                    if prior is not None:
                        writer.write(prior)
                        frames_written += 1
                        continue
            img = draw_boxes(img, anns, id2name)

            cv2.imwrite(str(out_img), img)

            if writer is not None:
                writer.write(img)
            frames_written += 1

        if writer is not None:
            writer.release()
            print(f"[OK] {domain}: wrote {frames_written} annotated frames to video.")


if __name__ == "__main__":
    main()


