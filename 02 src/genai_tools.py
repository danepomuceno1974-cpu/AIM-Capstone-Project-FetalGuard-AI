# genai_tools.py
import os
import json
import pandas as pd
from openai import OpenAI

client = OpenAI(
  api_key="your api_key_here"  # Replace with your actual OpenAI API key
)


def _safe_numeric_describe(df: pd.DataFrame):
    df_num = df.select_dtypes(include="number")
    if df_num.empty:
        return {}
    return df_num.describe().round(3).to_dict()


def llm_eda_summary(df: pd.DataFrame) -> str:
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": {col: int(v) for col, v in df.isna().sum().to_dict().items()},
        "numeric_describe": _safe_numeric_describe(df),
        "categorical_top": {
            col: df[col].value_counts(dropna=True).head(5).to_dict()
            for col in df.select_dtypes(include=["object", "category"]).columns
        },
    }

    prompt = f"""
You are a data analyst. Summarize the dataset in plain English.

Dataset summary (JSON):
{json.dumps(summary, indent=2)}

Return:
1. Main patterns
2. Data quality issues
3. Modeling implications
4. 3 concise bullet recommendations
"""
    response = client.responses.create(model="gpt-5.4-mini", input=prompt)
    return response.output_text


def llm_data_dictionary(df: pd.DataFrame) -> str:
    cols = []
    for col in df.columns:
        s = df[col]
        cols.append({
            "name": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
            "sample": s.dropna().astype(str).head(3).tolist(),
        })

    prompt = f"""
Create a professional data dictionary from this metadata.

Metadata (JSON):
{json.dumps(cols, indent=2)}

Return a table with:
- column name
- type
- description
- missingness
- notes
"""
    response = client.responses.create(model="gpt-5.4-mini", input=prompt)
    return response.output_text


def llm_explain_prediction(patient_features: dict, prediction: int, confidence: float = None) -> str:
    prompt = f"""
You are a clinical decision support assistant.
Explain this model prediction in simple, cautious language.

Input features:
{json.dumps(patient_features, indent=2)}

Prediction:
{prediction}

Confidence:
{confidence}

Rules:
- Do not claim certainty.
- Mention this is a model output, not a diagnosis.
- Keep it concise and understandable.
"""
    response = client.responses.create(model="gpt-5.4-mini", input=prompt)
    return response.output_text


def llm_recommend_next_action(prediction: int, features: dict) -> str:
    prompt = f"""
You are a clinical workflow assistant.
Given the model output and features, suggest the next operational action.

Prediction:
{prediction}

Features:
{json.dumps(features, indent=2)}

Return only:
- recommended next action
- short rationale
"""
    response = client.responses.create(model="gpt-5.4-mini", input=prompt)
    return response.output_text


def llm_chat_response(user_message: str, context: str = "") -> str:
    prompt = f"""
You are a helpful assistant for a machine learning Streamlit app.

Context:
{context}

User message:
{user_message}

Respond clearly, briefly, and use the context when relevant.
"""
    response = client.responses.create(model="gpt-5.4-mini", input=prompt)
    return response.output_text
