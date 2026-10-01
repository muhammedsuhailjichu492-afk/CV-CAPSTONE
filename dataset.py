from __future__ import annotations

import random
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .utils import banner, out_dir, p, save_json
from .visualize import bar_chart, draw_detections, grid, yolo_labels_to_xyxy

SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _progress(block, bsize, total):
    done = block * bsize
    if total > 0:
        pct = min(100, done * 100 / total)
        print(f"\r  downloading… {pct:5.1f}%  ({done / 1e6:6.1f} / {total / 1e6:.1f} MB)", end="", flush=True)


def download(cfg: dict) -> Path:
    root = p(cfg["data"]["root"])
    if (root / "images" / "train").exists():
        print(f"  dataset already present → {root}")
        return root
    root.mkdir(parents=True, exist_ok=True)
    zpath = root.parent / f"{cfg['data']['name']}.zip"
    if not zpath.exists():
        print(f"  source: {cfg['data']['url']}")
        urllib.request.urlretrieve(cfg["data"]["url"], zpath, _progress)
        print()
    print("  unzipping…")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    # some zips contain a single top folder — flatten it
    inner = [d for d in root.iterdir() if d.is_dir()]
    if not (root / "images").exists() and len(inner) == 1 and (inner[0] / "images").exists():
        for item in inner[0].iterdir():
            item.rename(root / item.name)
        inner[0].rmdir()
    print(f"  dataset ready → {root}")
    return root


def list_images(folder: Path):
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT) if folder.exists() else []


def explore(cfg: dict, root: Path | None = None, tag: str = "raw") -> dict:
    root = root or p(cfg["data"]["root"])
    names = cfg["data"]["class_names"]
    od = out_dir(cfg, "1_dataset")
    stats = {"splits": {}, "instances": {}, "boxes_per_image": {}, "box_area_pct": []}
    inst = Counter()
    per_img = []
    for s in SPLITS:
        imgs = list_images(root / "images" / s)
        labels = list((root / "labels" / s).glob("*.txt")) if (root / "labels" / s).exists() else []
        stats["splits"][s] = {"images": len(imgs), "label_files": len(labels)}
        for lf in labels:
            lines = [l for l in lf.read_text().splitlines() if l.strip()]
            per_img.append(len(lines))
            for l in lines:
                c, *_, bw, bh = l.split()[:5]
                inst[names[int(c)]] += 1
                stats["box_area_pct"].append(float(bw) * float(bh) * 100)
    stats["instances"] = {n: inst.get(n, 0) for n in names}
    stats["boxes_per_image"] = {"mean": round(float(np.mean(per_img)), 2), "max": int(np.max(per_img))}
    areas = np.array(stats["box_area_pct"])
    stats["box_area_pct"] = {"small(<1%)": int((areas < 1).sum()), "medium(1-10%)": int(((areas >= 1) & (areas < 10)).sum()),
                             "large(>=10%)": int((areas >= 10).sum())}

    bar_chart(stats["instances"], f"Objects per class ({tag})", od / f"class_balance_{tag}.png")
    bar_chart({s: v["images"] for s, v in stats["splits"].items()}, f"Images per split ({tag})",
              od / f"split_sizes_{tag}.png", horizontal=False)

    # sample grid with ground-truth boxes
    random.seed(0)
    train_imgs = list_images(root / "images" / "train")
    picks = random.sample(train_imgs, min(8, len(train_imgs)))
    tiles, titles = [], []
    for f in picks:
        im = cv2.imread(str(f))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = yolo_labels_to_xyxy(root / "labels" / "train" / f"{f.stem}.txt", w, h, names)
        tiles.append(draw_detections(im, gt, show_conf=False))
        titles.append(f"{f.name} - {len(gt)} objects")
    cv2.imwrite(str(od / f"samples_{tag}.jpg"), grid(tiles, cols=4, cell=(360, 270), titles=titles))
    save_json(stats, od / f"stats_{tag}.json")

    print(f"  splits: " + ", ".join(f"{s}={v['images']} imgs/{v['label_files']} labels" for s, v in stats["splits"].items()))
    print(f"  objects: {sum(inst.values())} total · {stats['boxes_per_image']['mean']} per image on average")
    print(f"  rarest class: {min(stats['instances'], key=stats['instances'].get)} "
          f"({min(stats['instances'].values())}) · most common: {max(stats['instances'], key=stats['instances'].get)} "
          f"({max(stats['instances'].values())})")
    print(f"  saved charts + sample grid → {od}")
    return stats


def run(cfg: dict) -> dict:
    banner("STEP 1 · DATASET")
    root = download(cfg)
    return explore(cfg, root, "raw")
