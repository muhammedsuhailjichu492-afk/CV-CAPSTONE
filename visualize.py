from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# BGR colours per class name (anything not listed gets a stable hashed colour)
COLORS = {
    "Person": (200, 120, 40), "helmet": (0, 200, 255), "vest": (0, 140, 255),
    "gloves": (180, 105, 255), "boots": (140, 90, 40), "goggles": (255, 220, 0),
    "none": (160, 160, 160), "no_helmet": (40, 40, 230), "no_goggle": (60, 60, 200),
    "no_gloves": (80, 80, 210), "no_boots": (100, 100, 190),
}
GREEN, RED, AMBER = (60, 180, 60), (40, 40, 220), (0, 160, 240)


def color_for(name: str):
    if name in COLORS:
        return COLORS[name]
    h = abs(hash(name)) % 180
    c = cv2.cvtColor(np.uint8([[[h, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(v) for v in c)


def label_box(img, xyxy, text, color, thick=2, scale=None):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    scale = scale or max(0.4, min(img.shape[:2]) / 1100)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if text:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        yy = max(y1, th + 6)
        cv2.rectangle(img, (x1, yy - th - 6), (x1 + tw + 6, yy), color, -1)
        cv2.putText(img, text, (x1 + 3, yy - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_detections(img, dets, show_conf=True):
    out = img.copy()
    for d in dets:
        t = f"{d['cls']} {d['conf']:.2f}" if show_conf else d["cls"]
        label_box(out, d["xyxy"], t, color_for(d["cls"]))
    return out


def draw_compliance(img, people, summary=None):
    out = img.copy()
    col = {"COMPLIANT": GREEN, "VIOLATION": RED, "CHECK": AMBER}
    for i, pr in enumerate(people):
        c = col[pr["status"]]
        label_box(out, pr["xyxy"], f"P{i + 1} {pr['status']}", c, thick=3)
        x1, _, _, y2 = [int(v) for v in pr["xyxy"]]
        items = "  ".join(f"{k}:{'OK' if v == 'yes' else 'X' if v == 'no' else '?'}" for k, v in pr["items"].items())
        scale = max(0.4, min(out.shape[:2]) / 1300)
        (tw, th), _ = cv2.getTextSize(items, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(out, (x1, y2 - th - 8), (x1 + tw + 8, y2), c, -1)
        cv2.putText(out, items, (x1 + 4, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    if summary is not None:
        txt = f"People {summary['people']} | Compliant {summary['compliant']} | Violations {summary['violations']}"
        scale = max(0.5, out.shape[1] / 1400)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
        H = out.shape[0]   # banner at the bottom-left so it never hides a person label
        cv2.rectangle(out, (0, H - th - 20), (tw + 20, H), (35, 25, 35), -1)
        cv2.putText(out, txt, (10, H - 10), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def yolo_labels_to_xyxy(label_path: Path, w: int, h: int, names: list[str]):
    dets = []
    if not label_path.exists():
        return dets
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
        dets.append({"cls": names[c] if c < len(names) else str(c), "conf": 1.0,
                     "xyxy": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]})
    return dets


def grid(images, cols=4, cell=(320, 240), titles=None):
    cw, ch = cell
    tiles = []
    for i, im in enumerate(images):
        h, w = im.shape[:2]
        k = min(cw / w, ch / h)
        r = cv2.resize(im, (int(w * k), int(h * k)))
        t = np.full((ch + (22 if titles else 0), cw, 3), 255, np.uint8)
        t[:r.shape[0], :r.shape[1]] = r
        if titles:
            cv2.putText(t, titles[i][:40], (4, ch + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def bar_chart(values: dict, title: str, path: Path, color="#8C3C84", horizontal=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys, vals = list(values.keys()), list(values.values())
    fig, ax = plt.subplots(figsize=(8, max(3, len(keys) * 0.38)) if horizontal else (8, 4))
    if horizontal:
        ax.barh(keys[::-1], vals[::-1], color=color)
        for i, v in enumerate(vals[::-1]):
            ax.text(v, i, f" {v}", va="center", fontsize=9)
    else:
        ax.bar(keys, vals, color=color)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
