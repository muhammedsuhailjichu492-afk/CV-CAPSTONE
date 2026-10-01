from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .dataset import list_images
from .detect import PPEDetector, classify_compliance
from .model import load
from .utils import banner, out_dir, p, pick_device, save_json
from .visualize import bar_chart, yolo_labels_to_xyxy


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def detection_metrics(cfg: dict) -> dict:
    model = load(cfg)
    data_yaml = p(cfg["data"]["clean_root"]) / "data.yaml"
    od = out_dir(cfg, "6_evaluation")
    m = model.val(data=str(data_yaml), split="test", imgsz=cfg["model"]["imgsz"], batch=8,
                  device=pick_device(cfg["model"]["device"]), project=str(p(cfg["paths"]["runs"])),
                  name="test_eval", exist_ok=True, plots=True, verbose=False)
    names = m.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        per_class[names[int(c)]] = {"precision": round(float(m.box.p[i]), 3), "recall": round(float(m.box.r[i]), 3),
                                    "mAP50": round(float(m.box.ap50[i]), 3), "mAP50_95": round(float(m.box.ap[i]), 3)}
    res = {"precision": round(float(m.box.mp), 3), "recall": round(float(m.box.mr), 3),
           "mAP50": round(float(m.box.map50), 3), "mAP50_95": round(float(m.box.map), 3),
           "speed_ms": {k: round(v, 2) for k, v in m.speed.items()}, "per_class": per_class}
    for f in ("confusion_matrix_normalized.png", "confusion_matrix.png", "BoxPR_curve.png", "PR_curve.png",
              "BoxF1_curve.png", "F1_curve.png"):
        src = Path(m.save_dir) / f
        if src.exists():
            shutil.copy2(src, od / f)
    bar_chart({k: v["mAP50"] for k, v in per_class.items()}, "Test mAP@0.5 per class", od / "map50_per_class.png")
    return res


def compliance_metrics(cfg: dict) -> dict:
    det = PPEDetector(cfg)
    names = cfg["data"]["class_names"]
    req, person, exp = dict(cfg["compliance"]["required"]), cfg["compliance"]["person_class"], cfg["compliance"]["expand"]
    root = p(cfg["data"]["clean_root"])
    labels = ["COMPLIANT", "VIOLATION"]
    cm = np.zeros((2, 2), int)
    missed = extra = 0
    times = []
    for f in list_images(root / "images" / "test"):
        im = cv2.imread(str(f))
        h, w = im.shape[:2]
        gt = yolo_labels_to_xyxy(root / "labels" / "test" / f"{f.stem}.txt", w, h, names)
        gt_people, _ = classify_compliance(gt, req, person, exp)
        dets, pr_people, _, ms = det(im)
        times.append(ms)
        used = set()
        for g in gt_people:
            if g["status"] not in labels:
                continue
            best, bj = 0, -1
            for j, q in enumerate(pr_people):
                if j not in used and (v := _iou(g["xyxy"], q["xyxy"])) > best:
                    best, bj = v, j
            if best >= 0.5 and pr_people[bj]["status"] in labels:
                used.add(bj)
                cm[labels.index(g["status"]), labels.index(pr_people[bj]["status"])] += 1
            else:
                missed += 1
        extra += len(pr_people) - len(used)
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else 0.0
    viol_recall = float(cm[1, 1] / cm[1].sum()) if cm[1].sum() else 0.0
    viol_prec = float(cm[1, 1] / cm[:, 1].sum()) if cm[:, 1].sum() else 0.0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Purples")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
    ax.set_title(f"PPE status per person · accuracy {acc:.1%}")
    fig.tight_layout(); fig.savefig(out_dir(cfg, "6_evaluation") / "compliance_confusion.png", dpi=130); plt.close(fig)
    ms = float(np.median(times))
    return {"people_matched": int(total), "accuracy": round(acc, 3), "violation_recall": round(viol_recall, 3),
            "violation_precision": round(viol_prec, 3), "confusion_matrix": {"rows=truth, cols=pred": labels, "values": cm.tolist()},
            "gt_people_missed": missed, "extra_predicted_people": extra,
            "median_ms_per_image": round(ms, 1), "fps": round(1000 / ms, 1)}


def run(cfg: dict) -> dict:
    banner("STEP 6 · PERFORMANCE EVALUATION")
    detm = detection_metrics(cfg)
    comp = compliance_metrics(cfg)
    report = {"detection_test_split": detm, "compliance_classification": comp}
    od = out_dir(cfg, "6_evaluation")
    save_json(report, od / "metrics.json")

    lines = ["# Evaluation report (test split)", "", "## Detection", "",
             f"| precision | recall | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|",
             f"| {detm['precision']} | {detm['recall']} | {detm['mAP50']} | {detm['mAP50_95']} |", "",
             "| class | precision | recall | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['precision']} | {v['recall']} | {v['mAP50']} | {v['mAP50_95']} |" for k, v in detm["per_class"].items()]
    lines += ["", "## PPE compliance classification (per person)", "",
              f"- people matched: {comp['people_matched']}",
              f"- accuracy: {comp['accuracy']:.1%}",
              f"- violation recall (violations caught): {comp['violation_recall']:.1%}",
              f"- violation precision (alarms that were real): {comp['violation_precision']:.1%}",
              f"- people missed by the detector: {comp['gt_people_missed']}",
              "", "## Speed", "", f"- median {comp['median_ms_per_image']} ms per image ≈ {comp['fps']} FPS"]
    (od / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"  DETECTION   precision {detm['precision']} · recall {detm['recall']} · "
          f"mAP50 {detm['mAP50']} · mAP50-95 {detm['mAP50_95']}")
    print(f"  COMPLIANCE  accuracy {comp['accuracy']:.1%} on {comp['people_matched']} people · "
          f"violation recall {comp['violation_recall']:.1%}")
    print(f"  SPEED       {comp['median_ms_per_image']} ms / image ≈ {comp['fps']} FPS")
    print(f"  report.md, metrics.json, confusion matrices, PR curve → {od}")
    return report
