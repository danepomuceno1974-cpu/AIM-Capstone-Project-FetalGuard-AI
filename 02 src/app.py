#!/usr/bin/env python
# coding: utf-8

# ### **FetalGuard AI — Clinical Decision Support Tool**
# Phase 07: Deployment (Interactive Streamlit App)
# 
# - Run with: streamlit run app.py
# - Requires: deployment_artifact.pkl in the same directory (built from Phases 2–6).

# In[ ]:


import pickle

import pandas as pd
import streamlit as st
import shap
import plotly.graph_objects as go


import json
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from genai_tools import (
    llm_eda_summary,
    llm_data_dictionary,
    llm_explain_prediction,
    llm_recommend_next_action,
    llm_chat_response,
)

# import os
# import sys
# from IPython import get_ipython


# In[ ]:


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FetalGuard AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# In[ ]:

# ── Load artifact (cached — only runs once per session) ───────────────────────
@st.cache_resource
def load_artifact():
    with open("../04 misc/deployment_artifact.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifact = load_artifact()
except FileNotFoundError:
    st.error(
        "**deployment_artifact.pkl not found.** Run `python build_artifact.py` "
        "first to generate this file from the Phase 02 preprocessing output."
    )
    st.stop()

model             = artifact["model"]
scaler            = artifact["scaler"]
FEATURES          = artifact["feature_names"]
TOP5              = artifact["top5_features"]
feature_stats     = artifact["feature_stats"]
clinical_labels   = artifact["clinical_labels"]
risk_direction    = artifact["risk_direction"]
default_threshold = artifact["optimal_threshold"]

# SHAP explainer — cached as a resource (tied to the immutable model object)
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)


# In[ ]:


# ── Core prediction logic (pure functions, no Streamlit calls — testable) ─────
def build_full_feature_vector(user_inputs: dict) -> pd.DataFrame:
    """
    Build a 1-row DataFrame with ALL 22 model features in the correct order.
    Features not in user_inputs default to training-set medians — the UI only
    exposes the top-5 SHAP-ranked features for clinical usability.
    """
    row = {feat: user_inputs.get(feat, feature_stats[feat]["median"]) for feat in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)


def predict_with_explanation(user_inputs: dict, threshold: float) -> dict:
    """
    Full pipeline: build vector → scale with SAVED scaler → predict_proba →
    apply threshold → SHAP-explain this specific prediction.
    Never refits the scaler — always reuses the one fitted in Phase 02.
    """
    X_raw    = build_full_feature_vector(user_inputs)
    X_scaled = scaler.transform(X_raw)

    proba_abnormal = float(model.predict_proba(X_scaled)[0, 1])
    prediction     = "High Risk" if proba_abnormal >= threshold else "Normal"

    shap_vals = explainer.shap_values(X_scaled)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_row = shap_vals[0]

    contributions = [
        (feat, float(shap_row[FEATURES.index(feat)])) for feat in TOP5
    ]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "prediction"   : prediction,
        "probability"  : proba_abnormal,
        "contributions": contributions,
        "shap_all"     : [(f, float(shap_row[FEATURES.index(f)])) for f in FEATURES],
        "threshold_used": threshold,
    }


def describe_direction(feat: str, shap_val: float) -> str:
    """Return 'High' or 'Low' — which value direction drove this SHAP contribution."""
    if risk_direction[feat] == "higher_is_riskier":
        return "High" if shap_val > 0 else "Low"
    return "Low" if shap_val > 0 else "High"


def build_explanation(contributions: list, prediction: str) -> str:
    """Natural-language explanation e.g. 'Risk is high primarily due to ...'"""
    if prediction == "Normal":
        return "Values are within ranges the model associates with normal fetal status."
    top_driver, top_val = contributions[0]
    text = (
        f"Risk is high primarily due to **{describe_direction(top_driver, top_val)} "
        f"{clinical_labels.get(top_driver, top_driver)}**"
    )
    if len(contributions) > 1:
        d2, v2 = contributions[1]
        if abs(v2) > 0.05:
            text += (
                f" and **{describe_direction(d2, v2)} "
                f"{clinical_labels.get(d2, d2)}**"
            )
    return text + "."


# In[ ]:


# ── SIDEBAR: Fairness notice + model info ─────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About this tool")
    st.markdown(
        "**Model:** XGBoost (`scale_pos_weight`)\n\n"
        "**Threshold:** cost-optimised at 0.10 "
        "(Phase 04, minimises FP×1 + FN×10)\n\n"
        "**Dataset:** UCI Cardiotocography "
        "(Campos & Bernardes, 2000)\n\n"
        "**Explainability:** SHAP TreeExplainer\n\n"
        "**Author:** Darwin A. Nepomuceno\n\n"
        "**Post Graduate Diploma in Artificial Intelligence and Machine Learning**"

    )

    st.divider()

    st.warning(
        "⚠️ **Fairness Notice (Phase 06 Audit)**\n\n"
        "This model was audited on proxy demographic subgroups "
        "(baseline FHR and ASTV quartile groups). The audit found:\n\n"
        "- **Disparate Impact = 0.40** for the baseline FHR proxy group "
        "— below the EEOC 0.80 threshold, indicating unequal positive-prediction "
        "rates across subgroups.\n\n"
        "- **Equal Opportunity Difference > 0.10** — Recall differs between "
        "subgroups, meaning some patient profiles are more likely to have an "
        "Abnormal status missed.\n\n"
        "Use this tool as **decision support only**, not as a standalone "
        "diagnostic. Review outputs carefully for patients with unusually "
        "high or low baseline FHR.",
        icon="⚠️",
    )

    st.divider()
    st.caption(
        "Phase 07 — Deployment · "
        "For research and demonstration only · "
        "Not a substitute for clinical judgment."
    )



# In[ ]:


# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🩺 FetalGuard AI")
st.caption(
    "Clinical decision-support tool · XGBoost model trained on Cardiotocography data · "
    "For research/demonstration purposes only — not a substitute for clinical judgment."
)

st.divider()

col_input, col_output = st.columns([1, 1.3], gap="large")


# In[ ]:


# ── INPUT: Top-5 SHAP feature sliders ────────────────────────────────────────
with col_input:
    st.subheader("📋 Patient CTG Readings")
    st.caption("Top 5 features ranked by SHAP global importance (Phase 05 analysis).")

    user_inputs = {}
    for feat in TOP5:
        stats     = feature_stats[feat]
        label     = clinical_labels.get(feat, feat)
        is_count  = feat in ("AC", "FM", "UC", "DL", "DS", "DP", "Nmax", "Nzeros")
        step      = 1.0 if is_count else round((stats["max"] - stats["min"]) / 100, 2) or 0.1

        user_inputs[feat] = st.slider(
            label     = f"{label}  *({feat})*",
            min_value = float(stats["hard_min"]),
            max_value = float(stats["hard_max"]),
            value     = float(stats["median"]),
            step      = float(step),
            help      = (
                f"Training-set range: {stats['hard_min']:.1f} – {stats['hard_max']:.1f} "
                f"(median: {stats['median']:.1f})"
            ),
        )

    st.info(
        "ℹ️ The other 17 model features (not shown) are held at their "
        "training-set median values for this simplified clinical input form.",
        icon="ℹ️",
    )

    st.divider()

    st.subheader("🎚️ Sensitivity Slider")
    st.caption(
        "**Lower** the threshold → flag more cases as High Risk "
        "(higher Recall, more false alarms — demand higher safety). "
        "**Raise** it → reduce false alarms "
        "(lower Recall, may miss true Abnormal cases)."
    )
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05, max_value=0.90,
        value=float(default_threshold), step=0.01,
        help=f"Cost-optimised default (Phase 04): {default_threshold}",
    )
    if abs(threshold - default_threshold) > 1e-6:
        st.caption(
            f"⚠️ Using custom threshold ({threshold:.2f}) instead of the "
            f"cost-optimised default ({default_threshold:.2f})."
        )


# In[ ]:


# ── RUN PREDICTION ────────────────────────────────────────────────────────────
result = predict_with_explanation(user_inputs, threshold)
proba  = result["probability"]


# In[ ]:


# ── OUTPUT ────────────────────────────────────────────────────────────────────
with col_output:
    st.subheader("🔬 Prediction Result")

    if result["prediction"] == "High Risk":
        st.error(f"### ⚠️ {result['prediction']}", icon="🚨")
    else:
        st.success(f"### ✅ {result['prediction']}", icon="✅")


# In[ ]:


# ── Confidence gauge ───────────────────────────────────────────────────────
fig_gauge = go.Figure(go.Indicator(
    mode  = "gauge+number",
    value = proba * 100,
    number= {"suffix": "%", "font": {"size": 36}},
    title = {"text": "Abnormal Probability", "font": {"size": 16}},
    gauge = {
        "axis": {"range": [0, 100], "tickwidth": 1},
        "bar" : {"color": "#1976D2"},
        "steps": [
            {"range": [0, threshold * 100],   "color": "#C8E6C9"},
            {"range": [threshold * 100, 100],  "color": "#FFCDD2"},
        ],
        "threshold": {
            "line": {"color": "black", "width": 3},
            "thickness": 0.85,
            "value": threshold * 100,
        },
    },
))
fig_gauge.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
st.plotly_chart(fig_gauge, width="stretch")

st.metric(
    label      = "Confidence (probability of Abnormal)",
    value      = f"{proba:.1%}",
    delta      = f"{'above' if proba >= threshold else 'below'} threshold ({threshold:.0%})",
    delta_color= "inverse",
)


# In[ ]:


# ── Natural-language explanation ───────────────────────────────────────────
st.markdown("#### 💡 Explanation")
st.write(build_explanation(result["contributions"], result["prediction"]))

# ── SHAP bar chart (per-prediction feature contributions) ─────────────────
st.markdown("#### 📊 Feature Contributions (SHAP)")
st.caption(
    "Red bars push the prediction toward **Abnormal**; "
    "blue bars push toward **Normal**. "
    "Showing top-5 features used as inputs."
)
contrib_feats  = [c[0] for c in result["contributions"]]
contrib_vals   = [c[1] for c in result["contributions"]]
contrib_labels = [
    f"{clinical_labels.get(f, f)} ({f})" for f in contrib_feats
]
bar_colors = ["#D32F2F" if v > 0 else "#1976D2" for v in contrib_vals]

fig_shap = go.Figure(go.Bar(
    x           = contrib_vals,
    y           = contrib_labels,
    orientation = "h",
    marker_color= bar_colors,
    marker_line = dict(color="white", width=0.8),
    text        = [f"{v:+.3f}" for v in contrib_vals],
    textposition= "outside",
))
fig_shap.add_vline(x=0, line_color="black", line_width=1)
fig_shap.update_layout(
    height      = 220,
    margin      = dict(t=10, b=10, l=10, r=60),
    xaxis_title = "SHAP value (log-odds contribution toward Abnormal)",
    yaxis       = dict(autorange="reversed"),
    showlegend  = False,
)
st.plotly_chart(fig_shap, width="stretch")

with st.expander("🔍 Full feature contribution breakdown"):
    df_contrib = pd.DataFrame(
        result["contributions"], columns=["Feature", "SHAP Contribution"]
    )
    df_contrib.insert(1, "Clinical Name",
                        df_contrib["Feature"].map(lambda f: clinical_labels.get(f, f)))
    df_contrib.insert(2, "Direction",
                        df_contrib.apply(
                            lambda r: describe_direction(r["Feature"], r["SHAP Contribution"]),
                            axis=1))
    st.dataframe(
        df_contrib.style.format({"SHAP Contribution": "{:+.4f}"}),
        width="stretch",
        hide_index=True,
    )


# In[ ]:


# ── Threshold sensitivity table ────────────────────────────────────────────
st.markdown("#### 🎯 Threshold Sensitivity")
st.caption(
    "How would this patient's classification change at different thresholds? "
    "Lowering the threshold increases sensitivity (catches more Abnormal cases "
    "but raises false alarms)."
)
sweep = sorted(set(round(t, 2) for t in [0.10, 0.20, default_threshold, 0.35, 0.50, 0.70]))
sweep_rows = [
    {
        "Threshold"      : f"{t:.2f}",
        "Classification" : "High Risk" if proba >= t else "Normal",
        "Note"           : (
            "← current" if abs(t - threshold) < 1e-6
            else ("cost-optimised default" if abs(t - default_threshold) < 1e-6 else "")
        ),
    }
    for t in sweep
]
st.dataframe(pd.DataFrame(sweep_rows), width="stretch", hide_index=True)


# In[ ]:


# ── LIMITATIONS & ETHICS EXPANDER ─────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ Model limitations & ethical considerations (Phase 06 Audit)"):
    st.markdown("""
**Explainability**
- SHAP TreeExplainer provides exact feature attributions for tree models.
- Top driver: *Abnormal Short-Term Variability (ASTV)* — consistent with FIGO intrapartum monitoring guidelines.
- PDP/ICE analysis (Phase 06) confirmed monotonic feature relationships with no unexpected non-linearities.

**Known limitations**
- Trained on 1990s CTG data (UCI, Campos & Bernardes, 2000) — potential concept drift with modern equipment.
- Small test set (96 patients) — 95% confidence interval on Recall ≈ ±0.05.
- This interface exposes only 5 of 22 model features; the remaining 17 default to training-set medians, reducing real-world predictive accuracy.
- Winsorization was applied pre-split in Phase 01 — minor leakage risk in the research pipeline.

**Fairness audit findings (Phase 06)**
- No real demographic labels (age, race, gender, socioeconomic status) exist in the dataset.
- Proxy audit using baseline FHR (LB) and Abnormal Short-Term Variability (ASTV) subgroups found:
  - **Disparate Impact = 0.40** for the LB proxy group — below the EEOC 0.80 adverse-impact threshold.
  - **Equal Opportunity Difference > 0.10** — Recall differs between LB subgroups.
- These findings suggest the model's error rate is not uniform across patient profiles and warrants clinical review.

**Recommended mitigations**
1. Deploy fairness monitoring (monthly Disparate Impact checks).
2. Apply group-specific thresholds to equalise Recall across subgroups.
3. Collect real demographic labels for a proper fairness audit.
4. Revalidate on modern CTG equipment data before any clinical adoption.

**References:** UCI CTG dataset · FIGO 2015 guidelines · AIF360 (Bellamy et al., 2019) · SHAP (Lundberg & Lee, 2017)
""")

st.caption(
    "Model: XGBoost (scale_pos_weight) · "
    "Threshold: Phase 04 cost-optimised (FP×1 + FN×10) · "
    "Explainability: SHAP TreeExplainer · "
    "Fairness audit: Phase 06 (AIF360) · "
    "Dataset: UCI Cardiotocography"
)


# ___
# ___
# ___

# In[ ]:


st.divider()
st.title("GenAI-Helper")

@st.cache_data
def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="xlrd")
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    raise ValueError("Unsupported file type")

@st.cache_data
def cached_eda_summary(df):
    return llm_eda_summary(df)

@st.cache_data
def cached_data_dictionary(df):
    return llm_data_dictionary(df)

def make_pdf(report_text: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40

    for line in report_text.splitlines():
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(40, y, line[:110])
        y -= 14

    c.save()
    buffer.seek(0)
    return buffer.read()

uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xls", "xlsx"])
df = None

if uploaded_file is not None:
    try:
        df = load_file(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Generate EDA Summary"):
                with st.spinner("Generating summary..."):
                    st.session_state["eda_summary"] = cached_eda_summary(df)
                st.markdown(st.session_state["eda_summary"])

        with col2:
            if st.button("Generate Data Dictionary"):
                with st.spinner("Generating dictionary..."):
                    st.session_state["data_dictionary"] = cached_data_dictionary(df)
                st.markdown(st.session_state["data_dictionary"])

        report_text = ""
        if "eda_summary" in st.session_state:
            report_text += "EDA Summary\n" + st.session_state["eda_summary"] + "\n\n"
        if "data_dictionary" in st.session_state:
            report_text += "Data Dictionary\n" + st.session_state["data_dictionary"]

        if report_text.strip():
            st.download_button(
                "Download PDF Report",
                data=make_pdf(report_text),
                file_name="genai_report.pdf",
                mime="application/pdf",
            )

    except Exception as e:
        st.error(f"Failed to load file: {e}")

st.subheader("Model Prediction + LLM Explanation")
feature_text = st.text_area("Paste patient features as JSON", height=180)
prediction = st.selectbox("Prediction", [0, 1])
confidence = st.slider("Confidence", 0.0, 1.0, 0.80)

if st.button("Predict and Explain"):
    try:
        features = json.loads(feature_text) if feature_text.strip() else {}
        st.session_state["prediction_result"] = {
            "prediction": prediction,
            "confidence": confidence,
            "features": features,
        }

        explanation = llm_explain_prediction(features, prediction, confidence)
        recommendation = llm_recommend_next_action(prediction, features)

        st.success(f"Prediction: {prediction}")
        st.write(explanation)
        st.write(recommendation)

    except json.JSONDecodeError:
        st.error("Invalid JSON in feature input.")

st.divider()
st.subheader("Chat Interface")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_msg = st.chat_input("Ask about the dataset, model, or prediction...")
if user_msg:
    st.session_state.chat_messages.append({"role": "user", "content": user_msg})

    context = ""
    if df is not None:
        context += f"Dataset shape: {df.shape}\n"
    if "prediction_result" in st.session_state:
        context += f"Latest prediction: {st.session_state['prediction_result']['prediction']}\n"

    answer = llm_chat_response(user_msg, context)

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

