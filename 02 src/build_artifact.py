"""
build_artifact.py — Phase 06 deployment artifact builder.

Run this AFTER the Phase 02 preprocessing notebook has produced
'ctg_preprocessed.pkl' in the same directory. It trains the Phase 03
best model (XGBoost + scale_pos_weight), computes SHAP feature
importance to identify the top-5 clinical inputs, and bundles
everything the Streamlit app needs into 'deployment_artifact.pkl'.

Usage:
    python build_artifact.py
"""

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

SEED = 42
PREPROCESSED_PATH = "../04 misc/ctg_preprocessed.pkl"
OUTPUT_PATH = "../04 misc/deployment_artifact.pkl"
 

# ── Clinical labels for UI display (extend as needed for other features) ─────
CLINICAL_LABELS = {
    "LB": "Baseline Fetal Heart Rate (bpm)",
    "AC": "Accelerations (count)",
    "FM": "Fetal Movements (count)",
    "UC": "Uterine Contractions (count)",
    "ASTV": "Abnormal Short-Term Variability (%)",
    "MSTV": "Mean Short-Term Variability",
    "ALTV": "Abnormal Long-Term Variability (%)",
    "MLTV": "Mean Long-Term Variability",
    "DL": "Light Decelerations (count)",
    "DS": "Severe Decelerations (count)",
    "DP": "Prolonged Decelerations (count)",
    "DR": "Repetitive Decelerations (count)",
    "Width": "FHR Histogram Width",
    "Min": "FHR Histogram Minimum",
    "Max": "FHR Histogram Maximum",
    "Nmax": "Histogram Peak Count",
    "Nzeros": "Histogram Zero Count",
    "Mode": "FHR Histogram Mode",
    "Mean": "FHR Histogram Mean",
    "Median": "FHR Histogram Median",
    "Variance": "FHR Histogram Variance",
    "Tendency": "FHR Histogram Tendency",
}


def clean_float(x, decimals=4):
    """Round and zero-out floating point noise (e.g. 2.2e-16 -> 0.0)."""
    val = round(float(x), decimals)
    return 0.0 if abs(val) < 1e-6 else val


def main():
    with open(PREPROCESSED_PATH, "rb") as f:
        prep = pickle.load(f)

    X_train_scaled = prep["X_train_scaled"]
    y_train = prep["y_train"]
    X_test_scaled = prep["X_test_scaled"]
    FEATURES = prep["feature_names"]
    scaler = prep["scaler"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("Training best model (XGBoost + scale_pos_weight)...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        verbosity=0,
        random_state=SEED,
    )
    model.fit(X_train_scaled, y_train)

    OPTIMAL_THRESHOLD = 0.1
    # selected in Phase 04 via cost-minimization sweep

    print("Computing SHAP values to rank feature importance...")
    explainer = shap.TreeExplainer(model)
    shap_values_test = explainer.shap_values(X_test_scaled)
    if isinstance(shap_values_test, list):
        shap_values_test = shap_values_test[1]

    mean_abs_shap = np.abs(shap_values_test).mean(axis=0)
    df_importance = pd.DataFrame(
        {"Feature": FEATURES, "Mean |SHAP|": mean_abs_shap}
    ).sort_values("Mean |SHAP|", ascending=False)
    TOP5_FEATURES = df_importance["Feature"].head(5).tolist()
    print(f"Top 5 features: {TOP5_FEATURES}")

    # Raw-scale stats for slider bounds (inverse-transform the scaled training data)
    X_train_raw = scaler.inverse_transform(X_train_scaled)
    df_X_train_raw = pd.DataFrame(X_train_raw, columns=FEATURES)

    feature_stats = {}
    for feat in FEATURES:
        col = df_X_train_raw[feat]
        feature_stats[feat] = {
            "min": clean_float(np.nanpercentile(col, 1)),
            "max": clean_float(np.nanpercentile(col, 99)),
            "median": clean_float(col.median()),
            "hard_min": clean_float(col.min()),
            "hard_max": clean_float(col.max()),
        }

    # Empirically determine whether higher values of each top-5 feature increase
    # or decrease predicted risk, via correlation between feature value and SHAP value
    risk_direction = {}
    for feat in TOP5_FEATURES:
        idx = FEATURES.index(feat)
        corr = np.corrcoef(X_test_scaled[:, idx], shap_values_test[:, idx])[0, 1]
        risk_direction[feat] = "higher_is_riskier" if corr > 0 else "lower_is_riskier"
    print(f"Risk directions: {risk_direction}")

    deployment_artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURES,
        "top5_features": TOP5_FEATURES,
        "feature_stats": feature_stats,
        "clinical_labels": CLINICAL_LABELS,
        "risk_direction": risk_direction,
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "shap_explainer_data": {
            "background_scaled": X_train_scaled[:100],
        },
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(deployment_artifact, f)

    print(f"\nSaved {OUTPUT_PATH}")
    print("Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
