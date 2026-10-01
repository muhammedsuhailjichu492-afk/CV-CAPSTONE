import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.dataset import list_images                      # noqa: E402
from src.detect import PPEDetector                        # noqa: E402
from src.utils import load_config, load_json, p           # noqa: E402
from src.visualize import draw_compliance, draw_detections  # noqa: E402

st.set_page_config(page_title="PPE Compliance Monitor", page_icon="🦺", layout="wide")
CFG = load_config()
ALL_ITEMS = {"helmet": ["helmet", "no_helmet"], "vest": ["vest", None], "gloves": ["gloves", "no_gloves"],
             "boots": ["boots", "no_boots"], "goggles": ["goggles", "no_goggle"]}


@st.cache_resource(show_spinner="Loading model…")
def get_detector(weights: str):
    return PPEDetector(CFG, weights)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🦺 PPE Compliance Monitor")
st.sidebar.caption("Computer Vision Capstone Project")
weights = st.sidebar.text_input("Model weights", "models/best.pt")
if not p(weights).exists():
    st.error(f"`{weights}` not found. Train first:  `python main.py all`  (or `--quick` for a test run).")
    st.stop()
det = get_detector(weights)
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, float(CFG["inference"]["conf"]), 0.05)
req_names = st.sidebar.multiselect("Required PPE", list(ALL_ITEMS), default=list(CFG["compliance"]["required"]))
required = {k: ALL_ITEMS[k] for k in req_names} or {"helmet": ALL_ITEMS["helmet"]}
show_raw = st.sidebar.toggle("Show all detections (not just people)", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("**Pipeline**  \nDataset → Preprocessing → CV Model → Detection/Classification → "
                    "Visualization → Evaluation → **Application**")


def analyse(bgr):
    dets, people, summary, ms = det(bgr, conf=conf, required=required)
    out = draw_compliance(draw_detections(bgr, dets) if show_raw else bgr, people, summary)
    return out, dets, people, summary, ms


def people_table(people):
    rows = []
    for i, pp in enumerate(people):
        r = {"person": f"P{i + 1}", "status": pp["status"], "score": round(pp["conf"], 2)}
        r.update({k: {"yes": "✅", "no": "❌"}.get(v, "?") for k, v in pp["items"].items()})
        r["reason"] = ", ".join(pp["reasons"]) or "-"
        rows.append(r)
    return pd.DataFrame(rows)


def show_result(bgr):
    out, dets, people, summary, ms = analyse(bgr)
    c1, c2 = st.columns([3, 2])
    c1.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), width="stretch")
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("People", summary["people"])
        m2.metric("Compliant", summary["compliant"])
        m3.metric("Violations", summary["violations"])
        if summary["violations"]:
            st.error(f"⚠️ {summary['violations']} safety violation(s) detected")
        elif summary["people"]:
            st.success("✅ Everyone is wearing the required PPE")
        else:
            st.info("No people detected")
        if people:
            st.dataframe(people_table(people), hide_index=True, width="stretch")
        st.caption(f"{len(dets)} objects detected · {ms:.0f} ms")
        ok, buf = cv2.imencode(".jpg", out)
        st.download_button("Download annotated image", buf.tobytes(), "ppe_result.jpg", "image/jpeg")


tab_img, tab_vid, tab_cam, tab_test, tab_rep = st.tabs(["🖼️ Image", "🎞️ Video", "📷 Webcam", "🧪 Test images", "📊 Model report"])

# ------------------------------------------------------------------ image
with tab_img:
    up = st.file_uploader("Upload a site photo", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        show_result(img)

# ------------------------------------------------------------------ video
with tab_vid:
    vup = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    step = st.slider("Analyse every Nth frame", 1, 10, 3)
    if vup and st.button("▶ Analyse video", type="primary"):
        tmp_in = Path(tempfile.gettempdir()) / f"ppe_in_{int(time.time())}{Path(vup.name).suffix}"
        tmp_in.write_bytes(vup.read())
        cap = cv2.VideoCapture(str(tmp_in))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tmp_out = Path(tempfile.gettempdir()) / f"ppe_out_{int(time.time())}.mp4"
        writer = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (W, H))
        frame_box, bar, timeline = st.empty(), st.progress(0.0), []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                out, dets, people, summary, ms = analyse(frame)
                writer.write(out)
                timeline.append({"time (s)": round(i / fps, 2), "people": summary["people"],
                                 "compliant": summary["compliant"], "violations": summary["violations"]})
                frame_box.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), width="stretch",
                                caption=f"t = {i / fps:.1f} s · {ms:.0f} ms")
                bar.progress(min(1.0, i / n))
            i += 1
        cap.release(); writer.release(); bar.progress(1.0)
        df = pd.DataFrame(timeline)
        if len(df):
            c1, c2, c3 = st.columns(3)
            c1.metric("Frames analysed", len(df))
            c2.metric("Max people in a frame", int(df["people"].max()))
            c3.metric("Frames with a violation", f"{(df['violations'] > 0).mean():.0%}")
            st.line_chart(df.set_index("time (s)")[["people", "compliant", "violations"]])
            st.download_button("Download annotated video (.mp4)", tmp_out.read_bytes(), "ppe_result.mp4", "video/mp4")
            st.download_button("Download timeline (.csv)", df.to_csv(index=False), "ppe_timeline.csv", "text/csv")

# ------------------------------------------------------------------ webcam
with tab_cam:
    st.write("Live check from a webcam connected to this computer (phone webcams such as Iriun are often index 1).")
    idx = st.number_input("Camera index", 0, 10, int(CFG["app"]["webcam_index"]))
    run = st.toggle("Start live monitoring")
    live = st.empty()
    stats = st.empty()
    if run:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(idx), backend)
        if not cap.isOpened():
            st.error(f"Could not open camera {idx}. Try another index.")
        while run and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                st.warning("Camera returned no frame."); break
            out, dets, people, summary, ms = analyse(frame)
            live.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), width="stretch")
            stats.markdown(f"**People:** {summary['people']} · **Compliant:** {summary['compliant']} · "
                           f"**Violations:** {summary['violations']} · {1000 / max(ms, 1):.1f} FPS")
        cap.release()
    st.markdown("---")
    snap = st.camera_input("…or take a single snapshot in the browser")
    if snap:
        show_result(cv2.imdecode(np.frombuffer(snap.getvalue(), np.uint8), cv2.IMREAD_COLOR))

# ------------------------------------------------------------------ test images
with tab_test:
    test_imgs = list_images(p(CFG["data"]["clean_root"]) / "images" / "test")
    if not test_imgs:
        st.info("Run `python main.py preprocess` to create the test split.")
    else:
        pick = st.selectbox("Held-out test image (never seen in training)", [f.name for f in test_imgs])
        show_result(cv2.imread(str(test_imgs[[f.name for f in test_imgs].index(pick)])))

# ------------------------------------------------------------------ report
with tab_rep:
    ev = p(CFG["paths"]["outputs"]) / "6_evaluation"
    if not (ev / "metrics.json").exists():
        st.info("Run `python main.py evaluate` to create the evaluation report.")
    else:
        rep = load_json(ev / "metrics.json")
        d, c = rep["detection_test_split"], rep["compliance_classification"]
        a, b, cc, dd, e = st.columns(5)
        a.metric("mAP@0.5", d["mAP50"]); b.metric("mAP@0.5:0.95", d["mAP50_95"])
        cc.metric("Precision", d["precision"]); dd.metric("Recall", d["recall"])
        e.metric("Compliance accuracy", f"{c['accuracy']:.0%}")
        st.dataframe(pd.DataFrame(d["per_class"]).T, width="stretch")
        imgs = [f for f in ["compliance_confusion.png", "map50_per_class.png", "confusion_matrix_normalized.png",
                            "BoxPR_curve.png", "PR_curve.png"] if (ev / f).exists()]
        cols = st.columns(2)
        for i, f in enumerate(imgs):
            cols[i % 2].image(str(ev / f), caption=f, width="stretch")
