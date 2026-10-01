from __future__ import annotations

import random
from collections import Counter

import cv2
import numpy as np

from .dataset import list_images
from .detect import PPEDetector, classify_compliance
from .utils import banner, out_dir, p
from .visualize import draw_compliance, draw_detections, grid, label_box, yolo_labels_to_xyxy


def _tag(img, text, color=(35, 25, 35)):
    out = img.copy()
    s = max(0.6, out.shape[1] / 900)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, s, 1)
    cv2.rectangle(out, (0, 0), (tw + 16, th + 16), color, -1)
    cv2.putText(out, text, (8, th + 8), cv2.FONT_HERSHEY_DUPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run(cfg: dict):
    banner("STEP 5 · RESULT VISUALIZATION")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    det = PPEDetector(cfg)
    names = cfg["data"]["class_names"]
    root = p(cfg["data"]["clean_root"])
    imgs = list_images(root / "images" / "test")
    od = out_dir(cfg, "5_visualization")
    random.seed(1)
    show = set(random.sample(range(len(imgs)), min(6, len(imgs))))

    gt_counts, pr_counts, status = Counter(), Counter(), Counter()
    pairs, gallery, gtitles = [], [], []
    for i, f in enumerate(imgs):
        im = cv2.imread(str(f))
        h, w = im.shape[:2]
        gt = yolo_labels_to_xyxy(root / "labels" / "test" / f"{f.stem}.txt", w, h, names)
        dets, people, summary, _ = det(im)
        gt_counts.update(d["cls"] for d in gt)
        pr_counts.update(d["cls"] for d in dets)
        status.update(pp["status"] for pp in people)
        if i in show:
            pairs += [_tag(draw_detections(im, gt, False), f"GROUND TRUTH ({len(gt)})"),
                      _tag(draw_detections(im, dets), f"PREDICTION ({len(dets)})")]
        if people and len(gallery) < 8:
            gallery.append(draw_compliance(im, people, summary))
            gtitles.append(f"{f.name}: {summary['compliant']} ok / {summary['violations']} violation")

    cv2.imwrite(str(od / "prediction_vs_ground_truth.jpg"), grid(pairs, cols=2, cell=(560, 400)))
    if gallery:
        cv2.imwrite(str(od / "compliance_gallery.jpg"), grid(gallery, cols=4, cell=(400, 300), titles=gtitles))

    # grouped bar: objects per class, ground truth vs predicted
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - 0.2, [gt_counts[n] for n in names], 0.4, label="ground truth", color="#B9A3B7")
    ax.bar(x + 0.2, [pr_counts[n] for n in names], 0.4, label="predicted", color="#8C3C84")
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_title("Test set: objects per class — ground truth vs model")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(od / "class_counts_gt_vs_pred.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    keys = ["COMPLIANT", "VIOLATION", "CHECK"]
    ax.bar(keys, [status[k] for k in keys], color=["#2E7D32", "#B03030", "#C88220"])
    for i, k in enumerate(keys):
        ax.text(i, status[k], str(status[k]), ha="center", va="bottom")
    ax.set_title("Test set: people by PPE status"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(od / "compliance_status.png", dpi=130); plt.close(fig)
    print(f"  people classified on the test set: {dict(status)}")
    print(f"  prediction-vs-truth grid, compliance gallery and charts → {od}")
