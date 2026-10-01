from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from .dataset import SPLITS, explore, list_images
from .utils import banner, out_dir, p, save_json
from .visualize import draw_detections, grid, yolo_labels_to_xyxy


def letterbox(img, size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    k = size / max(h, w)
    nw, nh = int(round(w * k)), int(round(h * k))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = cv2.copyMakeBorder(resized, top, size - nh - top, left, size - nw - left, cv2.BORDER_CONSTANT, value=color)
    return out, k, (left, top)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def clean_labels(lines, n_cls, min_box, report: Counter):
    good, seen = [], set()
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            report["malformed_lines"] += 1
            continue
        try:
            c = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            report["malformed_lines"] += 1
            continue
        if not 0 <= c < n_cls:
            report["bad_class_id"] += 1
            continue
        # clip the box to the image
        x1, y1, x2, y2 = max(0, cx - w / 2), max(0, cy - h / 2), min(1, cx + w / 2), min(1, cy + h / 2)
        if (x1, y1, x2, y2) != (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2):
            report["boxes_clipped"] += 1
        w, h = x2 - x1, y2 - y1
        if w < min_box or h < min_box:
            report["tiny_boxes_dropped"] += 1
            continue
        key = (c, round((x1 + x2) / 2, 4), round((y1 + y2) / 2, 4), round(w, 4), round(h, 4))
        if key in seen:
            report["duplicate_boxes_dropped"] += 1
            continue
        seen.add(key)
        good.append(f"{c} {key[1]:.6f} {key[2]:.6f} {key[3]:.6f} {key[4]:.6f}")
    return good


def build_clean_dataset(cfg: dict) -> Path:
    src, dst = p(cfg["data"]["root"]), p(cfg["data"]["clean_root"])
    names = cfg["data"]["class_names"]
    min_box = cfg["preprocess"]["min_box_size"]
    if dst.exists():
        shutil.rmtree(dst)
    rep = Counter()

    # --- leakage check: identical images in train and val/test
    drop_train = set()
    if cfg["preprocess"].get("check_duplicates", True):
        eval_hashes = {_md5(f) for s in ("val", "test") for f in list_images(src / "images" / s)}
        for f in list_images(src / "images" / "train"):
            if _md5(f) in eval_hashes:
                drop_train.add(f.name)
        rep["train_images_duplicated_in_val_test"] = len(drop_train)

    for s in SPLITS:
        (dst / "images" / s).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / s).mkdir(parents=True, exist_ok=True)
        imgs = list_images(src / "images" / s)
        stems = {f.stem for f in imgs}
        rep[f"orphan_labels_{s}"] = sum(1 for lf in (src / "labels" / s).glob("*.txt") if lf.stem not in stems)
        for f in imgs:
            if s == "train" and f.name in drop_train:
                continue
            im = cv2.imread(str(f))
            if im is None or im.shape[0] < 16 or im.shape[1] < 16:
                rep["corrupt_images_removed"] += 1
                continue
            lf = src / "labels" / s / f"{f.stem}.txt"
            lines = [l for l in lf.read_text().splitlines() if l.strip()] if lf.exists() else []
            if not lines:
                rep["background_images(no objects)"] += 1
            good = clean_labels(lines, len(names), min_box, rep)
            shutil.copy2(f, dst / "images" / s / f.name)
            (dst / "labels" / s / f"{f.stem}.txt").write_text("\n".join(good) + ("\n" if good else ""))
            rep[f"kept_{s}"] += 1

    data_yaml = dst / "data.yaml"
    yaml.safe_dump({"path": str(dst.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
                    "names": {i: n for i, n in enumerate(names)}}, open(data_yaml, "w"), sort_keys=False)
    save_json(dict(rep), out_dir(cfg, "2_preprocess") / "cleaning_report.json")
    print("  cleaning report:")
    for k, v in rep.items():
        print(f"    {k:<38} {v}")
    print(f"  clean dataset + data.yaml → {dst}")
    return data_yaml


def augment_examples(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.35, 0, 255)
    bright = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv2[..., 0] = (hsv2[..., 0] + 12) % 180
    hsv2[..., 1] = np.clip(hsv2[..., 1] * 0.6, 0, 255)
    hue = cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2BGR)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.15)
    rot = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
    return [("horizontal flip", cv2.flip(img, 1)), ("brightness +35%", bright), ("hue / saturation shift", hue),
            ("rotate + scale", rot)]


def visual_demo(cfg: dict, data_root: Path):
    od = out_dir(cfg, "2_preprocess")
    names = cfg["data"]["class_names"]
    size = cfg["preprocess"]["imgsz"]
    imgs = list_images(data_root / "images" / "train")
    # pick a wide image so the padding is visible
    f = max(imgs[:200], key=lambda x: (lambda im: im.shape[1] / im.shape[0] if im is not None else 0)(cv2.imread(str(x))))
    im = cv2.imread(str(f))
    h, w = im.shape[:2]
    gt = yolo_labels_to_xyxy(data_root / "labels" / "train" / f"{f.stem}.txt", w, h, names)
    lb, k, (dx, dy) = letterbox(im, size)
    gt_lb = [dict(d, xyxy=[d["xyxy"][0] * k + dx, d["xyxy"][1] * k + dy, d["xyxy"][2] * k + dx, d["xyxy"][3] * k + dy]) for d in gt]
    norm = lb.astype(np.float32) / 255.0                           # what the network receives (0..1)
    norm_vis = (norm * 255).astype(np.uint8)
    stages = [draw_detections(im, gt, False), draw_detections(lb, gt_lb, False), norm_vis]
    titles = [f"original {w}x{h}", f"letterbox {size}x{size}", f"normalised 0-1 (mean {norm.mean():.2f})"]
    cv2.imwrite(str(od / "pipeline.jpg"), grid(stages, cols=3, cell=(420, 320), titles=titles))
    aug = augment_examples(lb)
    cv2.imwrite(str(od / "augmentations.jpg"), grid([lb] + [a[1] for a in aug], cols=5, cell=(300, 300),
                                                   titles=["input"] + [a[0] for a in aug]))
    print(f"  saved preprocessing + augmentation visuals → {od}")


def run(cfg: dict) -> Path:
    banner("STEP 2 · PREPROCESSING")
    data_yaml = build_clean_dataset(cfg)
    visual_demo(cfg, data_yaml.parent)
    explore(cfg, data_yaml.parent, "clean")
    return data_yaml
