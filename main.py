import argparse
import sys

# Windows consoles may default to cp1252 — make sure symbols like → and · print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import dataset, detect, evaluate, model, preprocess, results_viz
from src.utils import Timer, load_config, p

STEPS = ["dataset", "preprocess", "train", "detect", "visualize", "evaluate"]


def main():
    ap = argparse.ArgumentParser(description="CV Capstone: PPE Compliance Monitor")
    ap.add_argument("step", choices=STEPS + ["all"], help="which step to run")
    ap.add_argument("--quick", action="store_true", help="tiny training run to test the pipeline")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(p(args.config))
    todo = STEPS if args.step == "all" else [args.step]

    with Timer() as t:
        for step in todo:
            if step == "dataset":
                dataset.run(cfg)
            elif step == "preprocess":
                preprocess.run(cfg)
            elif step == "train":
                data_yaml = p(cfg["data"]["clean_root"]) / "data.yaml"
                if not data_yaml.exists():
                    raise SystemExit("data.yaml missing — run `python main.py preprocess` first.")
                model.train(cfg, data_yaml, quick=args.quick)
            elif step == "detect":
                detect.run(cfg)
            elif step == "visualize":
                results_viz.run(cfg)
            elif step == "evaluate":
                evaluate.run(cfg)
    print(f"\nDone in {t.s / 60:.1f} min. Outputs are in the '{cfg['paths']['outputs']}' folder.")
    if "evaluate" in todo:
        print("Next: launch the app →  python -m streamlit run app.py")


if __name__ == "__main__":      # required on Windows (DataLoader workers use multiprocessing)
    main()
