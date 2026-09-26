import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI-Assisted Illegal Waste Dumping Review", layout="wide")

st.title("AI-Assisted Detection and Prioritisation of Illegal Waste Dumping")
st.caption("College prototype | Dataset: public MIVIA-IWDD-500 (NOT Mumbai CCTV footage) | "
           "Mumbai is only the proposed real-world application context")

# ---- locate the results produced by the Colab notebook ----
here = Path(__file__).parent
default_dir = here if (here / "predictions_test.csv").exists() else Path("results")
res = Path(st.sidebar.text_input("Results folder", str(default_dir)))
st.sidebar.markdown("Run the Colab notebook first. It creates `predictions_test.csv`, `metrics.json` and the PNG figures in this folder.")

pred_file = res / "predictions_test.csv"
if not pred_file.exists():
    st.error(f"Could not find {pred_file}. Run the Colab notebook first, then point this dashboard at the results folder.")
    st.stop()

df = pd.read_csv(pred_file)
metrics = json.load(open(res / "metrics.json")) if (res / "metrics.json").exists() else {}

# ---- human-review banner ----
st.warning("**Human review required.** This system only flags *suspected* dumping incidents for an authorised "
           "officer to review. It never accuses a person, issues a fine, or triggers enforcement action by itself.")

# ---- headline numbers ----
total = len(df)
flagged = int((df["pred_label"] == 1).sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total videos analysed", total)
c2.metric("Dumping detected (flagged)", flagged)
c3.metric("No dumping detected", total - flagged)
c4.metric("Average prediction confidence", f"{df['confidence'].mean():.1%}")
st.caption("Videos analysed = the official test set, which the model never saw during training.")

# ---- model performance ----
st.header("Model performance (official test set)")
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

# ---- static vs dynamic ----
st.header("Static vs dynamic dumping")
pos = df[df["true_label"] == 1]
if len(pos) and "event_type" in pos:
    tab = pos.groupby("event_type").agg(videos=("video_id", "count"), detected=("pred_label", "sum"))
    tab["detection_rate"] = (tab["detected"] / tab["videos"]).round(3)
    a, b = st.columns(2)
    a.dataframe(tab)
    b.bar_chart(tab["detection_rate"])
    st.caption("The model only outputs DUMPING / NO DUMPING. Static/dynamic comes from the dataset annotation "
               "and is used here only to analyse where the model succeeds or fails.")

# ---- confidence distribution ----
st.header("Prediction confidence")
counts = pd.cut(df["confidence"], bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True).value_counts().sort_index()
counts.index = counts.index.astype(str)
st.bar_chart(counts)

# ---- review queue ----
st.header("Flagged incidents: priority review queue")
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
               "The 'annotated' columns are ground truth from the dataset, shown for evaluation only; "
               "a live system would not have them. Reviewer decisions here are a demo and are not saved.")

with st.expander("Error analysis: videos the model got wrong"):
    st.dataframe(df[~df["correct"]][["video_id", "true_label", "pred_label", "dumping_probability", "event_type", "time_of_day"]])
with st.expander("All test predictions"):
    st.dataframe(df)

st.header("Responsible use")
st.markdown("- Predictions are **suspicions, not evidence**. A trained officer must watch the flagged clip.\n"
            "- False alarms are expected; recall and precision are reported so the trade-off is visible.\n"
            "- No person is identified, accused, or fined by this software.\n"
            "- The model was trained on public MIVIA-IWDD-500 videos, not on Mumbai footage, so it would need local validation before any real deployment.")
