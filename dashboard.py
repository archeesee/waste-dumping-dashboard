import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI-Assisted Illegal Waste Dumping Review", layout="wide")

st.markdown("""<style>
.stApp{background:#e9ecf1 !important}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp li,.stApp label,.stApp span,.stApp div[data-testid="stMarkdownContainer"]{color:#1f2937 !important}
.block-container{padding-top:0 !important;max-width:1200px}
header[data-testid="stHeader"]{background:transparent}
.pb-top{background:#252423;border-bottom:4px solid #f2c811;margin:0 -5rem 14px;padding:18px 5rem}
.pb-top h1{color:#fff !important;font-size:22px !important;margin:0 !important;padding:0 !important}
.pb-top p{color:#c9c9c9 !important;font-size:12.5px;margin:4px 0 0}
.pb-note{background:#fff8d6;border-left:5px solid #f2c811;padding:10px 14px;font-size:13.5px;margin-bottom:14px}
.pb-note,.pb-note *{color:#5a4a00 !important}
.pb-card{background:#fff;border:1px solid #e5e7eb;border-radius:4px;padding:14px 16px;height:100%}
.pb-card h4{margin:0 0 10px;font-size:14px;font-weight:600;color:#1f2937 !important}
.pb-kpi{border-top:4px solid #118dff}.pb-kpi.a{border-top-color:#d64550}.pb-kpi.b{border-top-color:#1aab40}.pb-kpi.c{border-top-color:#f2c811}
.pb-kpi .n{font-size:34px;font-weight:600;line-height:1.1;color:#1f2937 !important}
.pb-kpi .l{font-size:12.5px;color:#6b7280 !important;margin-top:2px}
.pb-row{margin:9px 0}.pb-row .t{display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px}
.pb-tr{background:#e5e7eb;height:12px;border-radius:2px}.pb-tr b{display:block;height:100%;border-radius:2px;background:#118dff}.pb-tr.y b{background:#f2c811}
.pb-cm{display:grid;grid-template-columns:auto 1fr 1fr;gap:4px;font-size:12px;text-align:center}
.pb-cm .h{color:#6b7280 !important;align-self:center;padding:2px 6px}
.pb-cm .c{border-radius:3px;padding:16px 4px;font-size:26px;font-weight:600;color:#fff !important}
.stApp [data-testid="stFileUploader"] section{background:#fff}
</style>""", unsafe_allow_html=True)

def card(title, body, extra=""):
    return f'<div class="pb-card {extra}"><h4>{title}</h4>{body}</div>'

def kpi(n, label, cls=""):
    return f'<div class="pb-card pb-kpi {cls}"><div class="n">{n}</div><div class="l">{label}</div></div>'

def bar(label, pct, yellow=False):
    return (f'<div class="pb-row"><div class="t"><span>{label}</span><b>{pct:.1f}%</b></div>'
            f'<div class="pb-tr {"y" if yellow else ""}"><b style="width:{pct:.1f}%"></b></div></div>')

st.markdown("""<div class="pb-top"><h1>AI-Assisted Detection and Prioritisation of Illegal Waste Dumping</h1>
<p>Design Thinking project | Archee Arolkar, Roll No. 38 | Dataset: public MIVIA-IWDD-500 (not Mumbai footage)</p></div>""", unsafe_allow_html=True)

here = Path(__file__).parent
res = here

st.markdown('<div class="pb-note"><b>Human review required.</b> This system only flags suspected dumping for an authorised officer. It never accuses anyone, issues a fine, or takes enforcement action.</div>', unsafe_allow_html=True)

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
    fl = int((df["pred_label"] == 1).sum())
    k = st.columns(4)
    k[0].markdown(kpi(total, "Videos analysed (official test set)"), unsafe_allow_html=True)
    k[1].markdown(kpi(fl, "Flagged as dumping", "a"), unsafe_allow_html=True)
    k[2].markdown(kpi(total - fl, "Not flagged", "b"), unsafe_allow_html=True)
    k[3].markdown(kpi(f"{df['confidence'].mean():.1%}", "Average prediction confidence", "c"), unsafe_allow_html=True)
    st.write("")

    pct = fl / total * 100 if total else 0
    dash = 251.3 * pct / 100
    donut = (f'<svg viewBox="0 0 100 100" width="150" height="150"><circle cx="50" cy="50" r="40" fill="none" stroke="#1aab40" stroke-width="16"/>'
             f'<circle cx="50" cy="50" r="40" fill="none" stroke="#d64550" stroke-width="16" stroke-dasharray="{dash:.1f} 251.3" transform="rotate(-90 50 50)"/>'
             f'<text x="50" y="55" text-anchor="middle" font-size="16" font-weight="600" fill="#1f2937">{pct:.0f}%</text></svg>'
             f'<div style="font-size:13px"><div><span style="color:#d64550">&#9632;</span> Flagged: {fl}</div><div><span style="color:#1aab40">&#9632;</span> Not flagged: {total - fl}</div></div>')
    tn = int(((df["true_label"] == 0) & (df["pred_label"] == 0)).sum()); fp = int(((df["true_label"] == 0) & (df["pred_label"] == 1)).sum())
    fn = int(((df["true_label"] == 1) & (df["pred_label"] == 0)).sum()); tp = int(((df["true_label"] == 1) & (df["pred_label"] == 1)).sum())
    cm = (f'<div class="pb-cm"><span></span><span class="h">Pred: no dumping</span><span class="h">Pred: dumping</span>'
          f'<span class="h">Actual: no dumping</span><div class="c" style="background:#3d7dd8">{tn}</div><div class="c" style="background:#8fb6e8;color:#12305e !important">{fp}</div>'
          f'<span class="h">Actual: dumping</span><div class="c" style="background:#c8dbf3;color:#12305e !important">{fn}</div><div class="c" style="background:#12239e">{tp}</div></div>'
          f'<p style="font-size:12.5px;margin:10px 0 0">Missed dumping: {fn}. False alarms: {fp}.</p>')
    perf = "".join(bar(n, metrics[key] * 100, key == "recall") for n, key in
                   [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1")]) if metrics else ""
    c = st.columns(3)
    c[0].markdown(card("Flagged vs not flagged", f'<div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap">{donut}</div>'), unsafe_allow_html=True)
    c[1].markdown(card("Confusion matrix", cm), unsafe_allow_html=True)
    c[2].markdown(card("Model performance", perf + '<p style="font-size:12.5px;margin:10px 0 0">Recall is highest by design: a missed incident costs more than a false alarm.</p>'), unsafe_allow_html=True)
    st.write("")

    pos = df[df["true_label"] == 1]
    sd = ""
    if len(pos) and "event_type" in pos:
        for et, g in pos.groupby("event_type"):
            sd += bar(f"{et.title()} ({int(g['pred_label'].sum())} of {len(g)} caught)", g["pred_label"].mean() * 100)
    d = st.columns(3)
    d[0].markdown(card("Detection rate: static vs dynamic dumping", sd), unsafe_allow_html=True)
    d[1].markdown(card("How the system works", "<ul style='font-size:13.5px;line-height:1.7;padding-left:18px'><li>Surveillance video, sampled frames</li><li>Pretrained MobileNetV2 features</li><li>Bidirectional GRU classifier</li><li>DUMPING / NO DUMPING + confidence</li><li>Human officer reviews every flag</li></ul>"), unsafe_allow_html=True)
    d[2].markdown(card("Training curves", ""), unsafe_allow_html=True)
    with d[2]:
        for name in ["accuracy_curve.png", "loss_curve.png"]:
            if (res / name).exists():
                st.image(str(res / name))
    st.write("")

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
            "Time of day (dataset)": q["time_of_day"].values,
            "Reviewer decision": "Pending",
        })
        st.data_editor(
            view, hide_index=True,
            disabled=[c_ for c_ in view.columns if c_ != "Reviewer decision"],
            column_config={"Reviewer decision": st.column_config.SelectboxColumn(
                options=["Pending", "Confirmed by officer", "Dismissed by officer"])},
        )
        st.caption("Ranked by model probability so reviewers see the most likely incidents first. Decisions here are a demo and are not saved.")

    with st.expander("Error analysis: videos the model got wrong"):
        st.dataframe(df[~df["correct"]][["video_id", "true_label", "pred_label", "dumping_probability", "event_type", "time_of_day"]])
    with st.expander("All test predictions"):
        st.dataframe(df)

st.header("Responsible use")
st.markdown("- Predictions are **suspicions, not evidence**. A trained officer must watch the flagged clip.\n"
            "- False alarms are expected; recall and precision are reported so the trade-off is visible.\n"
            "- No person is identified, accused, or fined by this software.\n"
            "- The model was trained on public MIVIA-IWDD-500 videos, not on Mumbai footage, so it would need local validation before any real deployment.")
