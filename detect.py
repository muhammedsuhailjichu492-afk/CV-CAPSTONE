from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2

from .dataset import list_images
from .model import load
from .utils import banner, out_dir, p, pick_device, save_json
from .visualize import draw_compliance, draw_detections


class PPEDetector:
    def __init__(self, cfg: dict, weights: str | Path | None = None):
        self.cfg = cfg
        self.model = load(cfg, weights)
        self.device = pick_device(cfg["model"]["device"])
        self.conf = cfg["inference"]["conf"]
        self.iou = cfg["inference"]["iou"]
        self.required = dict(cfg["compliance"]["required"])
        self.person = cfg["compliance"]["person_class"]
        self.expand = cfg["compliance"]["expand"]

    # ---------------------------------------------------------------- detection
    def detect(self, img, conf: float | None = None):
        t = time.perf_counter()
        r = self.model.predict(img, conf=conf or self.conf, iou=self.iou, imgsz=self.cfg["model"]["imgsz"],
                               device=self.device, verbose=False)[0]
        ms = (time.perf_counter() - t) * 1000
        dets = [{"cls": r.names[int(c)], "conf": float(s), "xyxy": [float(v) for v in b]}
                for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        return dets, ms

    # ----------------------------------------------------------- classification
    def classify(self, dets, required: dict | None = None):
        return classify_compliance(dets, required or self.required, self.person, self.expand)

    def __call__(self, img, conf=None, required=None):
        dets, ms = self.detect(img, conf)
        people, summary = self.classify(dets, required)
        return dets, people, summary, ms


def _inside(pt, box, e):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return x1 - e * w <= pt[0] <= x2 + e * w and y1 - e * h <= pt[1] <= y2 + e * h


def classify_compliance(dets, required: dict, person_cls="Person", expand=0.15):
    people = []
    persons = [d for d in dets if d["cls"] == person_cls]
    for pd in persons:
        items, reasons = {}, []
        for item, (pos, neg) in required.items():
            ctr = lambda d: ((d["xyxy"][0] + d["xyxy"][2]) / 2, (d["xyxy"][1] + d["xyxy"][3]) / 2)
            pos_hit = [d for d in dets if d["cls"] == pos and _inside(ctr(d), pd["xyxy"], expand)]
            neg_hit = [d for d in dets if neg and d["cls"] == neg and _inside(ctr(d), pd["xyxy"], expand)]
            if neg_hit and (not pos_hit or max(x["conf"] for x in neg_hit) > max(x["conf"] for x in pos_hit)):
                items[item] = "no"; reasons.append(f"{neg} detected")
            elif pos_hit:
                items[item] = "yes"
            else:
                items[item] = "no"; reasons.append(f"{item} not found")
        x1, y1, x2, y2 = pd["xyxy"]
        tiny = (y2 - y1) < 40
        status = "CHECK" if tiny else ("COMPLIANT" if all(v == "yes" for v in items.values()) else "VIOLATION")
        people.append({"xyxy": pd["xyxy"], "conf": pd["conf"], "items": items, "status": status, "reasons": reasons})
    summary = {"people": len(people), "compliant": sum(p_["status"] == "COMPLIANT" for p_ in people),
               "violations": sum(p_["status"] == "VIOLATION" for p_ in people),
               "check": sum(p_["status"] == "CHECK" for p_ in people)}
    return people, summary


def run(cfg: dict, n: int = 12) -> Path:
    banner("STEP 4 · DETECTION + CLASSIFICATION")
    det = PPEDetector(cfg)
    test_dir = p(cfg["data"]["clean_root"]) / "images" / "test"
    imgs = list_images(test_dir)
    od = out_dir(cfg, "4_detection")
    rows, times = [], []
    for i, f in enumerate(imgs):
        im = cv2.imread(str(f))
        dets, people, summary, ms = det(im)
        times.append(ms)
        rows.append({"image": f.name, "detections": len(dets), **summary,
                     "classes": ";".join(sorted({d["cls"] for d in dets}))})
        if i < n:
            cv2.imwrite(str(od / f"{f.stem}_detections.jpg"), draw_detections(im, dets))
            cv2.imwrite(str(od / f"{f.stem}_compliance.jpg"), draw_compliance(im, people, summary))
    with open(od / "test_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    tot = {k: sum(r[k] for r in rows) for k in ("people", "compliant", "violations", "check")}
    save_json({"images": len(rows), **tot, "median_ms_per_image": round(sorted(times)[len(times) // 2], 1)},
              od / "summary.json")
    print(f"  {len(rows)} test images · {tot['people']} people → {tot['compliant']} compliant, "
          f"{tot['violations']} violations, {tot['check']} to check")
    print(f"  median inference time: {sorted(times)[len(times) // 2]:.1f} ms / image")
    print(f"  annotated images + test_results.csv → {od}")
    return od
