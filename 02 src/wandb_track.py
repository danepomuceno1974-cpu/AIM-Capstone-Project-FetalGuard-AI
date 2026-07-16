# %%
"""
wandb_track.py
Logs a CTG model training run to Weights & Biases.

Usage:
    wandb login          # first time only
    python wandb_track.py
"""
import numpy as np
import pickle
from zipfile import Path
import wandb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,confusion_matrix
)
# PREPROCESSED_PKL = Path("../04 misc/ctg_preprocessed.pkl")
# ARTIFACT_PKL     = Path("../04 misc/deployment_artifact.pkl")

# %%
wandb.login(key="your_wandb_api_key_here")  # Replace with your actual W&B API key

# %%

# ---- Config -----------------------------------------------------------
PREPROCESSED_PATH = "../04 misc/ctg_preprocessed.pkl"
PROJECT_NAME = "FetalGuard AI"
RUN_NAME = "xgboost-scale-pos-weight"
THRESHOLD = 0.10

CONFIG = {
    "model": "XGBClassifier",
    "scale_pos_weight": "auto",
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "threshold": THRESHOLD,
}

# %%

# ---- Load data ----------------------------------------------------------
with open(PREPROCESSED_PATH, "rb") as f:
    data = pickle.load(f)

# ctg_preprocessed.pkl (Phase 02 output) contains:
#   X_train_scaled, y_train                 -- scaled, no resampling
#   X_train_smote,  y_train_smote            -- SMOTE-resampled
#   X_train_adasyn, y_train_adasyn           -- ADASYN-resampled
#   X_test_scaled,  y_test
#   feature_names                            -- list[str], since the X arrays are plain ndarrays
#   class_weight_dict, scaler
#
# RESAMPLING selects which training set to use. "none" matches the
# scale_pos_weight approach referenced in the Phase 08 deployment guide.
RESAMPLING = "none"  # one of: "none", "smote", "adasyn"

_train_map = {
    "none": ("X_train_scaled", "y_train"),
    "smote": ("X_train_smote", "y_train_smote"),
    "adasyn": ("X_train_adasyn", "y_train_adasyn"),
}
x_key, y_key = _train_map[RESAMPLING]

X_train = data[x_key]
y_train = data[y_key]
X_test = data["X_test_scaled"]
y_test = data["y_test"]
feature_names = data["feature_names"]

# Normalize to plain numpy arrays (y_train_smote/adasyn are pandas Series)
y_train = y_train.values if hasattr(y_train, "values") else y_train
y_test = y_test.values if hasattr(y_test, "values") else y_test

CONFIG["resampling"] = RESAMPLING

# %%

# ---- Start W&B run --------------------------------------------------------
run = wandb.init(project=PROJECT_NAME, name=RUN_NAME, config=CONFIG)
cfg = wandb.config


# %%


# ---- Train ----------------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=cfg.n_estimators,
    max_depth=cfg.max_depth,
    learning_rate=cfg.learning_rate,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

# %%
# ---- Evaluate ---------------------------------------------------------
proba = model.predict_proba(X_test)[:, 1]
preds = (proba >= cfg.threshold).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, preds),
    "precision": precision_score(y_test, preds),
    "recall": recall_score(y_test, preds),
    "f1": f1_score(y_test, preds),
    "roc_auc": roc_auc_score(y_test, proba),
}
print("Metrics:", metrics)
wandb.log(metrics)


# %%

# Feature importance
importances = sorted(
    zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True
)
top5 = [f for f, _ in importances[:5]]
print("Top 5 features:", top5)
# Log as a Table, not a raw list — wandb.log() can't chart plain lists of strings
wandb.log({"top5_features": wandb.Table(data=[[f] for f in top5], columns=["feature"])})
wandb.log(
    {
        "feature_importance": wandb.plot.bar(
            wandb.Table(
                data=[[f, float(v)] for f, v in importances[:15]],
                columns=["feature", "importance"],
            ),
            "feature",
            "importance",
            title="Top 15 Feature Importances",
        )
    }
)


# %%

# ROC curve
wandb.log(
    {
        "roc_curve": wandb.plot.roc_curve(
            y_test, model.predict_proba(X_test), labels=["Normal", "High Risk"]
        )
    }
)

# %%
# Precision-Recall curve
wandb.log(
    {
        "pr_curve": wandb.plot.pr_curve(
            y_test, model.predict_proba(X_test), labels=["Normal", "High Risk"]
        )
    }
)


# %%
# Confusion matrix (at the operating threshold, cfg.threshold)
wandb.log(
    {
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(),
            preds=preds.tolist(),
            class_names=["Normal", "High Risk"],
        )
    }
)


# %%
# Cost vs. Decision Threshold
# Clinical cost = FP x 1 + FN x 10 (false negatives — missed abnormal cases — are
# weighted 10x more expensive), matching the Phase 04 threshold-optimization notebook.
FN_WEIGHT = 10
sweep_thresholds = np.arange(0.01, 0.99, 0.01)
sweep_costs, sweep_precisions, sweep_recalls = [], [], []
for t in sweep_thresholds:
    sweep_preds = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, sweep_preds, labels=[0, 1]).ravel()
    sweep_costs.append(int(fp * 1 + fn * FN_WEIGHT))
    sweep_precisions.append(precision_score(y_test, sweep_preds, zero_division=0))
    sweep_recalls.append(recall_score(y_test, sweep_preds, zero_division=0))

best_idx = int(np.argmin(sweep_costs))
best_sweep_threshold = float(sweep_thresholds[best_idx])
wandb.log({"threshold_sweep_best_cost": sweep_costs[best_idx],
           "threshold_sweep_best_threshold": best_sweep_threshold})

cost_table = wandb.Table(
    data=[[float(t), c] for t, c in zip(sweep_thresholds, sweep_costs)],
    columns=["threshold", "cost"],
)
wandb.log(
    {
        "cost_vs_threshold": wandb.plot.line(
            cost_table, "threshold", "cost", title="Clinical Cost vs. Decision Threshold"
        )
    }
)


# %%

# Precision & recall over the same threshold sweep, as a separate chart
# (kept apart from cost since W&B line charts don't support a secondary y-axis,
# and cost/precision/recall are on very different scales).
wandb.log(
    {
        "precision_recall_vs_threshold": wandb.plot.line_series(
            xs=sweep_thresholds.tolist(),
            ys=[sweep_precisions, sweep_recalls],
            keys=["precision", "recall"],
            title="Precision & Recall vs. Decision Threshold",
            xname="threshold",
        )
    }
)


# %%

# ---- Save & log model artifact ----------------------------------------
model_path = "./models/model_artifact.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

artifact = wandb.Artifact(
    name="ctg-fetalguard-ai",
    type="model",
    metadata={"threshold": cfg.threshold, "features": feature_names, **metrics},
)
artifact.add_file(model_path)
run.log_artifact(artifact)

print(f"Run URL: {run.url}")
wandb.finish()


