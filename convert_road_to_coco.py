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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def coco_skeleton(categories):
    return {"images": [], "annotations": [], "categories": categories}


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def main():
    parser = argparse.ArgumentParser(description="Convert ROAD annotations to COCO per-domain for RF-DETR")
    parser.add_argument("--json", type=Path, default=Path("road_test_v1.0.json"))
    parser.add_argument("--yolo_json", type=Path, default=Path("YOLO_road_test_v1.0.json"))
    parser.add_argument("--videos_dir", type=Path, default=Path("videos"))
    parser.add_argument("--out_root", type=Path, default=Path("data"))
    parser.add_argument("--domains", nargs="+", default=["sunny", "overcast", "snowy", "night"], help="Domains to convert")
    parser.add_argument("--max_frames", type=int, default=0, help="Limit number of annotated frames per video (0 = all)")
    parser.add_argument("--resume", type=int, default=1, help="If 1, skip decoding frames that already exist and include them in JSON")
    parser.add_argument("--checkpoint_every", type=int, default=500, help="Write JSON every N newly saved frames (0 = only at end)")
    parser.add_argument("--skip_json", type=int, default=0, help="If 1, skip writing COCO JSON files (extract images only)")
    parser.add_argument("--jpg_quality", type=int, default=95, help="JPEG quality for saved frames (1-100)")
    args = parser.parse_args()

    # Read once
    raw = json.loads(args.json.read_text(encoding="utf-8"))
    db = raw["db"]
    # Prefer agent_labels from YOLO JSON if present; fallback to road JSON
    agent_labels = None
    try:
        y = json.loads(args.yolo_json.read_text(encoding="utf-8"))
        agent_labels = y.get("agent_labels")
    except Exception:
        agent_labels = None
    if not agent_labels:
        agent_labels = raw["agent_labels"]  # exact 10 classes from JSON (no ego)

    # Categories: keep exactly as in JSON, contiguous IDs starting at 1
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(agent_labels)]
    name_to_catid = {c["name"]: c["id"] for c in categories}

    for video_key, vinfo in db.items():
        base = video_key  # key matches basename (without .mp4)
        domain = DOMAIN_BY_BASENAME.get(base, "unknown")
        if domain not in args.domains:
            continue
        video_fp = args.videos_dir / f"{base}.mp4"
        if not video_fp.exists():
            print(f"[WARN] Missing video: {video_fp}, skipping.")
            continue

        out_images = args.out_root / domain / "images"
        out_ann = args.out_root / domain / "annotations"
        ensure_dir(out_images)
        ensure_dir(out_ann)

        coco = coco_skeleton(categories)
        ann_id = 1
        img_id = 1

        # Find already saved images for this video (by frame index)
        existing_idxs = set()
        if args.resume and out_images.exists():
            for p in out_images.glob(f"{base}_frame_*.jpg"):
                m = re.search(rf"{re.escape(base)}_frame_(\d+)\.jpg$", p.name)
                if m:
                    try:
                        existing_idxs.add(int(m.group(1)))
                    except Exception:
                        pass

        frames = vinfo.get("frames", {})
        annotated_indices = sorted(
            int(k) for k, finfo in frames.items() if finfo.get("annotated", 0) == 1 and "annos" in finfo
        )
        if args.max_frames and len(annotated_indices) > args.max_frames:
            annotated_indices = annotated_indices[:args.max_frames]
        if not annotated_indices:
            print(f"[INFO] No annotated frames for {video_fp}")
            continue

        # First, if resuming, include existing frames into COCO (no decoding)
        if args.resume and existing_idxs:
            for idx in sorted(existing_idxs):
                finfo = frames.get(str(idx), {})
                if not finfo:
                    continue
                W = int(finfo.get("width", 0))
                H = int(finfo.get("height", 0))
                fname = f"{base}_frame_{idx:06d}.jpg"
                rel_file = f"images/{fname}"
                coco["images"].append({
                    "id": img_id,
                    "file_name": rel_file,
                    "width": W,
                    "height": H,
                })

                for _, anno in finfo.get("annos", {}).items():
                    box = anno.get("box")
                    agent_ids = anno.get("agent_ids", [])
                    if not box or not agent_ids:
                        continue

                    # Pick first valid agent id within range; skip malformed ids
                    valid_name = None
                    for aid in agent_ids:
                        try:
                            if isinstance(aid, int) and 0 <= aid < len(agent_labels):
                                valid_name = agent_labels[aid]
                                break
                        except Exception:
                            continue
                    if valid_name is None:
                        continue

                    x1 = clamp01(float(box[0])); y1 = clamp01(float(box[1]))
                    x2 = clamp01(float(box[2])); y2 = clamp01(float(box[3]))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    x = x1 * W
                    y = y1 * H
                    w = (x2 - x1) * W
                    h = (y2 - y1) * H
                    if w <= 0 or h <= 0:
                        continue

                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": name_to_catid[valid_name],
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "segmentation": [],
                    })
                    ann_id += 1

                img_id += 1

        # Determine which frames still need decoding
        to_decode = [idx for idx in annotated_indices if idx not in existing_idxs]
        if not to_decode:
            if not args.skip_json:
                out_json = out_ann / "train.json"
                out_json.write_text(json.dumps(coco), encoding="utf-8")
                print(f"[OK] {domain}: {len(coco['images'])} images, {len(coco['annotations'])} annos -> {out_json}")
            else:
                print(f"[OK] {domain}: images verified; skipping JSON as requested")
            continue

        cap = cv2.VideoCapture(str(video_fp))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video {video_fp}")
            continue

        # Read sequentially from first to last annotated frame to avoid slow random seeks
        annotated_set = set(to_decode)
        min_idx = to_decode[0]
        max_idx = to_decode[-1]

        # Position to the first annotated frame (OpenCV is 0-based)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(min_idx - 1, 0))
        cur_idx = min_idx
        saved_since_checkpoint = 0
        while cur_idx <= max_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(cur_idx - 1, 0))
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[WARN] Could not read frame {cur_idx} from {video_fp}, stopping early for this video.")
                break

            if cur_idx in annotated_set:
                finfo = frames.get(str(cur_idx), {})
                W = int(finfo.get("width", frame.shape[1]))
                H = int(finfo.get("height", frame.shape[0]))

                fname = f"{base}_frame_{cur_idx:06d}.jpg"
                rel_file = f"images/{fname}"
                out_path = out_images / fname

                # Save RGB frame with configured quality
                try:
                    q = int(max(1, min(100, args.jpg_quality)))
                except Exception:
                    q = 95
                cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])

                coco["images"].append({
                    "id": img_id,
                    "file_name": rel_file,
                    "width": W,
                    "height": H,
                })

                for _, anno in finfo.get("annos", {}).items():
                    box = anno.get("box")
                    agent_ids = anno.get("agent_ids", [])
                    if not box or not agent_ids:
                        continue

                    # Pick first valid agent id within range; skip malformed ids
                    valid_name = None
                    for aid in agent_ids:
                        try:
                            if isinstance(aid, int) and 0 <= aid < len(agent_labels):
                                valid_name = agent_labels[aid]
                                break
                        except Exception:
                            continue
                    if valid_name is None:
                        continue

                    agent_name = valid_name
                    # Convert normalized xyxy -> pixel xywh
                    x1 = clamp01(float(box[0])); y1 = clamp01(float(box[1]))
                    x2 = clamp01(float(box[2])); y2 = clamp01(float(box[3]))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    x = x1 * W
                    y = y1 * H
                    w = (x2 - x1) * W
                    h = (y2 - y1) * H
                    if w <= 0 or h <= 0:
                        continue

                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": name_to_catid[agent_name],
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "segmentation": [],
                    })
                    ann_id += 1

                img_id += 1
                saved_since_checkpoint += 1

                if not args.skip_json and args.checkpoint_every and saved_since_checkpoint >= args.checkpoint_every:
                    out_json = out_ann / "train.json"
                    out_json.write_text(json.dumps(coco), encoding="utf-8")
                    saved_since_checkpoint = 0

            # Move to next needed frame index
            # If there is a next frame in to_decode after cur_idx, jump; else increment
            next_needed = None
            for idx in to_decode:
                if idx > cur_idx:
                    next_needed = idx
                    break
            cur_idx = next_needed if next_needed is not None else (cur_idx + 1)

        cap.release()
        if not args.skip_json:
            out_json = out_ann / "train.json"
            out_json.write_text(json.dumps(coco), encoding="utf-8")
            print(f"[OK] {domain}: {len(coco['images'])} images, {len(coco['annotations'])} annos -> {out_json}")
        else:
            print(f"[OK] {domain}: extracted {len(coco['images'])} frames; skipping JSON as requested")


if __name__ == "__main__":
    main()


