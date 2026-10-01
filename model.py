from __future__ import annotations

import shutil
from pathlib import Path

from .utils import banner, out_dir, p, pick_device, save_json

BEST = "models/best.pt"


def train(cfg: dict, data_yaml: Path, quick: bool = False) -> Path:
    banner("STEP 3 · CV MODEL (training)")
    from ultralytics import YOLO

    m = cfg["model"]
    device = pick_device(m["device"])
    args = dict(data=str(data_yaml), epochs=m["epochs"], imgsz=m["imgsz"], batch=m["batch"], patience=m["patience"],
                workers=m["workers"], seed=m["seed"], device=device, project=str(p(cfg["paths"]["runs"])),
                name=m["run_name"], exist_ok=True, plots=True, verbose=False)
    if quick:   # smoke test: a few minutes on CPU, just to prove the pipeline runs end to end
        args.update(epochs=1, imgsz=320, batch=8, fraction=0.15, plots=False)
        print("  QUICK MODE: 1 epoch on 15% of the training data (results will be poor — that's expected)")
    print(f"  base model: {m['weights']} · device: {'GPU ' + str(device) if device != 'cpu' else 'CPU'} · "
          f"epochs: {args['epochs']} · imgsz: {args['imgsz']} · batch: {args['batch']}")

    model = YOLO(m["weights"])            # downloads the COCO-pretrained weights on first use
    results = model.train(**args)
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    dst = p(BEST)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst)

    od = out_dir(cfg, "3_model")
    for f in ("results.png", "labels.jpg", "train_batch0.jpg", "val_batch0_pred.jpg"):
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, od / f)
    save_json({"base_weights": m["weights"], "run_dir": str(run_dir), "best_weights": str(dst), "quick": quick,
               **{k: v for k, v in args.items() if k not in ("data", "project")}}, od / "training_summary.json")
    print(f"  best weights → {dst}")
    print(f"  training curves + batches → {od}")
    return dst


def load(cfg: dict | None = None, weights: str | Path | None = None):
    from ultralytics import YOLO
    w = p(weights or BEST)
    if not w.exists():
        raise FileNotFoundError(f"{w} not found — run `python main.py train` first.")
    return YOLO(str(w))
