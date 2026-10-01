# Computer Vision Capstone Project — PPE Compliance Monitor

**Dataset → Preprocessing → CV Model → Detection/Classification → Result Visualization → Performance Evaluation → Final Application**

A complete, end-to-end computer vision project. A YOLOv8 detector is fine-tuned to find people and
their safety equipment on construction sites. Each person is then **classified** as
**COMPLIANT / VIOLATION** according to the PPE you require (helmet, vest, …). The results are visualized
and evaluated on a held-out test set, then served in a **Streamlit web app** that works with images,
videos and a live webcam.

| Step | File | What it does | Output folder |
|---|---|---|---|
| 1 Dataset | `src/dataset.py` | Downloads the Construction-PPE dataset (1,416 images, 11 classes), counts images/objects, and draws class-balance charts and a sample grid | `outputs/1_dataset` |
| 2 Preprocessing | `src/preprocess.py` | Cleans the data (orphan labels, broken or duplicate boxes, corrupt images, train/test leakage), writes `data.yaml`, and visualizes the letterbox, normalization and augmentation steps | `outputs/2_preprocess` |
| 3 CV Model | `src/model.py` | Fine-tunes COCO-pretrained **YOLOv8n** with early stopping and saves `models/best.pt` | `outputs/3_model`, `runs/` |
| 4 Detection + Classification | `src/detect.py` | Detects the PPE items, then classifies each person as COMPLIANT / VIOLATION (with reasons) | `outputs/4_detection` |
| 5 Result Visualization | `src/results_viz.py` | Shows prediction vs ground truth, a compliance gallery, and class-count and status charts | `outputs/5_visualization` |
| 6 Performance Evaluation | `src/evaluate.py` | Detection: precision, recall, mAP@0.5 and mAP@0.5:0.95 per class, confusion matrix, PR curve. Compliance: accuracy and violation recall. Speed in FPS | `outputs/6_evaluation` |
| 7 Final Application | `app.py` | Streamlit app with tabs for image upload, video analysis with a timeline, live webcam, test images and the model report | browser |

Everything is configured in **`config.yaml`**: model size, epochs, batch size, confidence threshold and required PPE.

---

## 1. Setup (Windows · VS Code · PowerShell)

Open the `cv_capstone` folder in VS Code, then open a terminal (**Ctrl + `**):

```powershell
# 1. virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
#   If activation is blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   (then activate again)

# 2. PyTorch WITH CUDA for the RTX GPU — install this BEFORE requirements.txt
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. the rest
python -m pip install -r requirements.txt

# 4. check that the GPU is visible (should print True and the GPU name)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

In VS Code, choose the interpreter: **Ctrl+Shift+P → Python: Select Interpreter → `.venv`**.

> Always use `python -m pip …` and `python -m streamlit …`. That avoids broken pip launchers and PATH problems on Windows.

## 2. Run the pipeline

```powershell
python main.py all --quick     # ~5 min smoke test: proves every step runs (the model will be weak)
python main.py all             # the real run: steps 1-6 (training uses the GPU)
```

Or run one step at a time: `python main.py dataset | preprocess | train | detect | visualize | evaluate`.
In VS Code you can also use **Run and Debug (Ctrl+Shift+D)**, where every step has its own launch entry.

Training time depends on your machine. On an RTX 3050 with YOLOv8n at 640 px, expect roughly
**1 minute per epoch**, and early stopping usually ends the run before 60 epochs. If you get
*CUDA out of memory*, set `batch: 8` in `config.yaml`.

## 3. Launch the final application

```powershell
python -m streamlit run app.py
```

It opens at http://localhost:8501.
- **Image:** upload a site photo to get the per-person status, reasons, and a download of the annotated image.
- **Video:** analyse every Nth frame. You get a live preview, a people/violations timeline chart, the annotated .mp4 and a CSV.
- **Webcam:** live monitoring with OpenCV (the DirectShow backend on Windows). If a phone webcam such as Iriun doesn't show up, set the camera index to `1`. There's also a browser snapshot option.
- **Test images:** held-out images the model never saw during training.
- **Model report:** mAP, precision, recall, compliance accuracy, confusion matrices and the PR curve.
- **Sidebar:** confidence threshold, which PPE is required (helmet, vest, gloves, boots, goggles), and a switch to show every detection.

## 4. How the classification works

The detector finds `Person`, `helmet`, `vest`, `no_helmet`, and so on. For every person:
1. PPE boxes whose centre falls inside the person box (enlarged by 15%) belong to that person.
2. For each required item, it's **yes** if the positive class is found. It's **no** if the negative class is found (e.g. `no_helmet`) or nothing is found.
3. The person is **COMPLIANT** if every required item is *yes* and a **VIOLATION** otherwise. Very small persons (< 40 px tall) are marked **CHECK**.

The evaluation applies the same rule to the ground-truth labels. That produces a real accuracy for the classification step, not just for the detector.

## 5. Project structure

```
cv_capstone/
├── main.py              ← runs pipeline steps 1-6
├── app.py               ← step 7: Streamlit application
├── config.yaml          ← every setting in one place
├── requirements.txt
├── src/
│   ├── utils.py         ← config, paths, device selection
│   ├── dataset.py       ← 1 dataset
│   ├── preprocess.py    ← 2 preprocessing
│   ├── model.py         ← 3 CV model
│   ├── detect.py        ← 4 detection + classification
│   ├── visualize.py     ← drawing helpers
│   ├── results_viz.py   ← 5 result visualization
│   └── evaluate.py      ← 6 performance evaluation
├── .vscode/             ← launch configs for every step + the app
├── datasets/            ← created: raw + cleaned data
├── runs/                ← created: YOLO training runs
├── models/best.pt       ← created: trained model
└── outputs/1_… 6_…      ← created: charts, images, reports
```

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` is False | Reinstall torch with the `--index-url …/cu121` command above. Update the NVIDIA driver |
| CUDA out of memory | Set `batch: 8` (or 4) in `config.yaml` |
| DataLoader / worker error on Windows | Set `workers: 0` in `config.yaml` |
| `streamlit` is not recognized | Run it as `python -m streamlit run app.py` |
| Webcam is black or doesn't open | Try camera index 1 or 2, and close other apps that use the camera |
| Dataset download fails | Download the zip from the URL in `config.yaml` and put it at `datasets/construction-ppe.zip`, then rerun |

## 7. Ideas to extend it

- Use `yolov8s.pt` for higher accuracy (set `batch: 8`).
- Add a violation log with timestamps, plus e-mail or Telegram alerts.
- Use tracking (`model.track`) to count each worker once in a video.
- Export to ONNX or TensorRT for faster inference: `yolo export model=models/best.pt format=onnx`.

**Dataset:** Ultralytics Construction-PPE (https://docs.ultralytics.com/datasets/detect/construction-ppe). **Model:** Ultralytics YOLOv8 (AGPL-3.0).
