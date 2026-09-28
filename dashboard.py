import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI-Assisted Illegal Waste Dumping Review", layout="wide")

st.title("AI-Assisted Detection and Prioritisation of Illegal Waste Dumping")
st.caption("College prototype | Dataset: public MIVIA-IWDD-500 (NOT Mumbai CCTV footage) | "
           "Mumbai is only the proposed real-world application context")

here = Path(__file__).parent
res = here

st.warning("**Human review required.** This system only flags *suspected* dumping incidents for an authorised "
           "officer to review. It never accuses a person, issues a fine, or triggers enforcement action by itself.")

st.header("Try it live: upload a video")
st.caption("Upload any short surveillance-style clip (mp4/avi/mov). The trained model below analyses it and returns a prediction.")

MODEL_PATH = res / "iwdd_model.keras"
MODEL_FILE_ID = "PASTE_YOUR_DRIVE_FILE_ID_HERE"

def ensure_model():
    if not MODEL_PATH.exists() and MODEL_FILE_ID and not MODEL_FILE_ID.startswith("PASTE"):
        import gdown
        gdown.download(id=MODEL_FILE_ID, output=str(MODEL_PATH), quiet=True)
    return MODEL_PATH.exists()

NUM_FRAMES, IMG_SIZE, THRESH = 24, 224, 0.5

@st.cache_resource(show_spinner="Loading trained model...")
def load_pipeline():
    import tensorflow as tf
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone.trainable = False
    model = tf.keras.models.load_model(MODEL_PATH)
    return backbone, model

def sample_frames(video_path, n=NUM_FRAMES, size=IMG_SIZE):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        total = 0
        while cap.grab(): total += 1
        cap.release(); cap = cv2.VideoCapture(str(video_path))
    if total <= 0:
        raise ValueError("Could not read this video file.")
    idxs = np.linspace(0, total - 1, n).astype(int)
    frames, last = [], np.zeros((size, size, 3), np.uint8)
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            last = cv2.cvtColor(cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        frames.append(last)
    cap.release()
    return np.stack(frames), idxs / fps

def predict(video_path, backbone, model):
    import tensorflow as tf
    frames, times = sample_frames(video_path)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(frames.astype("float32"))
    feats = backbone(x, training=False).numpy()[None, ...]
    delta = np.diff(feats, axis=1, prepend=feats[:, :1, :])
    feats = np.concatenate([feats, delta], axis=-1)
    p = float(model.predict(feats, verbose=0)[0, 0])
    flagged = p >= THRESH
    return frames, times, p, flagged

uploaded = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
if uploaded is not None:
    with st.spinner("Fetching trained model (first time only)..."):
        model_ok = ensure_model()
    if not model_ok:
        st.error("Could not get the trained model. Check that MODEL_FILE_ID is set and the Drive file is shared as Anyone with the link.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            backbone, model = load_pipeline()
            with st.spinner("Analysing video..."):
                frames, times, p, flagged = predict(tmp_path, backbone, model)
            conf = p if flagged else 1 - p
            label = "DUMPING" if flagged else "NO DUMPING"
            color = "#b91c1c" if flagged else "#15803d"
            st.markdown(f"### Video ID: `{uploaded.name}`")
            st.markdown(f"<h3 style='color:{color}'>Prediction: {label} &nbsp;|&nbsp; Confidence: {conf:.1%}</h3>",
                        unsafe_allow_html=True)
            cols = st.columns(6)
            show_idx = np.linspace(0, len(frames) - 1, 6).astype(int)
            for c, i in zip(cols, show_idx):
                c.image(frames[i], caption=f"t = {times[i]:.1f}s", use_container_width=True)
            if flagged:
                st.error("SUSPECTED incident: sent for HUMAN REVIEW. No automatic accusation, fine or enforcement.")
            else:
                st.success("No suspected dumping detected.")
        except Exception as e:
            st.error(f"Could not process this video: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

st.divider()

st.header("Test-set evaluation")
pred_file = res / "predictions_test.csv"
if not pred_file.exists():
    st.info("predictions_test.csv not found next to this app - test-set results section skipped.")
else:
    df = pd.read_csv(pred_file)
    metrics = json.load(open(res / "metrics.json")) if (res / "metrics.json").exists() else {}

    total = len(df)
    flagged_n = int((df["pred_label"] == 1).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total videos analysed", total)
    c2.metric("Dumping detected (flagged)", flagged_n)
    c3.metric("No dumping detected", total - flagged_n)
    c4.metric("Average prediction confidence", f"{df['confidence'].mean():.1%}")
    st.caption("Videos analysed = the official test set, which the model never saw during training.")

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        m2.metric("Precision", f"{metrics['precision']:.1%}")
        m3.metric("Recall", f"{metrics['recall']:.1%}")
        m4.metric("F1-score", f"{metrics['f1']:.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        if (res / "confusion_matrix.png").exists():
            st.image(str(res / "confusion_matrix.png"))
    with right:
        st.subheader("Training curves")
        for name in ["accuracy_curve.png", "loss_curve.png"]:
            if (res / name).exists():
                st.image(str(res / name))

    st.subheader("Static vs dynamic dumping")
    pos = df[df["true_label"] == 1]
    if len(pos) and "event_type" in pos:
        tab = pos.groupby("event_type").agg(videos=("video_id", "count"), detected=("pred_label", "sum"))
        tab["detection_rate"] = (tab["detected"] / tab["videos"]).round(3)
        a, b = st.columns(2)
        a.dataframe(tab)
        b.bar_chart(tab["detection_rate"])

    st.subheader("Prediction confidence distribution")
    counts = pd.cut(df["confidence"], bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True).value_counts().sort_index()
    counts.index = counts.index.astype(str)
    st.bar_chart(counts)

    st.subheader("Flagged incidents: priority review queue")
    min_conf = st.slider("Minimum confidence to show", 0.5, 1.0, 0.5, 0.01)
    q = df[(df["pred_label"] == 1) & (df["confidence"] >= min_conf)].sort_values("dumping_probability", ascending=False)
    if len(q) == 0:
        st.info("No flagged incidents at this confidence level.")
    else:
        view = pd.DataFrame({
            "Priority": range(1, len(q) + 1),
            "Video ID": q["video_id"].values,
            "Dumping probability": q["dumping_probability"].round(3).values,
            "Confidence": q["confidence"].round(3).values,
            "Annotated type (dataset)": q["event_type"].values,
            "Annotated onset, seconds (dataset)": q["onset_sec"].where(q["onset_sec"] >= 0).values,
            "Time of day (dataset)": q["time_of_day"].values,
            "Reviewer decision": "Pending",
        })
        st.data_editor(
            view, hide_index=True,
            disabled=[c for c in view.columns if c != "Reviewer decision"],
            column_config={"Reviewer decision": st.column_config.SelectboxColumn(
                options=["Pending", "Confirmed by officer", "Dismissed by officer"])},
        )
        st.caption("Ranked by model probability so reviewers see the most likely incidents first. "
                   "The 'annotated' columns are ground truth from the dataset, shown for evaluation only.")

    with st.expander("Error analysis: videos the model got wrong"):
        st.dataframe(df[~df["correct"]][["video_id", "true_label", "pred_label", "dumping_probability", "event_type", "time_of_day"]])
    with st.expander("All test predictions"):
        st.dataframe(df)

st.header("Responsible use")
st.markdown("- Predictions are **suspicions, not evidence**. A trained officer must watch the flagged clip.\n"
            "- False alarms are expected; recall and precision are reported so the trade-off is visible.\n"
            "- No person is identified, accused, or fined by this software.\n"
            "- The model was trained on public MIVIA-IWDD-500 videos, not on Mumbai footage, so it would need local validation before any real deployment.")
