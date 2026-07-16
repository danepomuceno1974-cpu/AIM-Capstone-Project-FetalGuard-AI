"""
src/model.py
============
Cost-sensitive model training, hyperparameter tuning, and evaluation.

Why cost-sensitive rather than accuracy-optimised
--------------------------------------------------
In fetal monitoring, a False Negative (missed Abnormal) may result in
delayed intervention and irreversible harm. A False Positive (unnecessary
intervention) is recoverable. Standard accuracy treats both error types
equally, which is clinically inappropriate.

We define: Cost = (1 × FP) + (10 × FN)

This function is used as the Optuna objective and the primary comparison
metric across all models. Threshold selection (Phase 04) also minimises
this cost rather than maximising F1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    average_precision_score, make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


SEED = 42
FP_WEIGHT = 1    # cost of a False Positive (false alarm — recoverable)
FN_WEIGHT = 10   # cost of a False Negative (missed Abnormal — potentially harmful)


# ── Cost function ──────────────────────────────────────────────────────────────

def compute_cost(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Compute the asymmetric clinical cost.

    Cost = (FP_WEIGHT × FP) + (FN_WEIGHT × FN)

    FN is weighted 10× because a missed Abnormal case has greater clinical
    consequences than a false alarm. This asymmetry is derived from the
    relative cost of delayed obstetric intervention vs. unnecessary monitoring.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0=Normal, 1=Abnormal).
    y_pred : array-like
        Predicted binary labels.

    Returns
    -------
    int
        Total clinical cost for this prediction set.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(FP_WEIGHT * fp + FN_WEIGHT * fn)


def cost_scorer_fn(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Sklearn-compatible cost scorer (returns raw positive cost).

    Note: make_scorer(..., greater_is_better=False) applies its own
    internal negation so sklearn's "maximise" convention finds the minimum
    cost. This function must NOT pre-negate — doing so would double-flip
    the sign and confuse cross_val_score results.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted labels.

    Returns
    -------
    int
        Raw clinical cost (positive; lower is better).
    """
    return compute_cost(y_true, y_pred)


# Scorer for use with cross_val_score and Optuna CV.
# greater_is_better=False tells sklearn to negate internally for optimisation.
COST_SCORER = make_scorer(cost_scorer_fn, greater_is_better=False)


# ── Model evaluation ───────────────────────────────────────────────────────────

def evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.1,
    name: str = "",
) -> dict:
    """
    Train a model and evaluate it on the test set.

    Uses predict_proba + threshold rather than predict() to allow
    cost-optimised threshold selection (Phase 04). The threshold of 0.1
    was selected in Phase 04 to minimise clinical cost.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model with fit() and predict_proba() methods.
    X_train, y_train : np.ndarray
        Training data (already scaled and optionally resampled).
    X_test, y_test : np.ndarray
        Test data (scaled, never resampled).
    threshold : float
        Decision threshold applied to predicted probabilities.
        Default 0.1 (cost-optimised; see Phase 04).
    name : str
        Label for the result dict.

    Returns
    -------
    dict
        Accuracy, Precision, Recall, F1, Cost, TP, FP, TN, FN, model_name.
    """
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "Model"     : name or type(model).__name__,
        "Accuracy"  : round(accuracy_score(y_test, y_pred), 4),
        "Precision" : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"    : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1"        : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "Cost"      : compute_cost(y_test, y_pred),
        "ROC_AUC"   : round(roc_auc_score(y_test, y_proba), 4),
        "AP"        : round(average_precision_score(y_test, y_proba), 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


# ── Threshold sweep ────────────────────────────────────────────────────────────

def sweep_thresholds(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evaluate model performance across a range of decision thresholds.

    The default threshold of 0.5 is arbitrary for imbalanced problems.
    Sweeping thresholds lets us select the one that minimises clinical cost
    (or achieves Recall ≥ 0.95 with maximum Precision).

    Parameters
    ----------
    model : fitted sklearn estimator
        Must have predict_proba() method.
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        Test labels.
    thresholds : np.ndarray, optional
        Array of threshold values to evaluate. Defaults to 0.10–0.90 step 0.01.

    Returns
    -------
    pd.DataFrame
        One row per threshold with columns:
        Threshold, Cost, Recall, Precision, F1, FP, FN.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.01)

    y_proba = model.predict_proba(X_test)[:, 1]
    rows    = []

    for t in thresholds:
        y_pred        = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "Threshold": round(t, 2),
            "Cost"     : int(fp * FP_WEIGHT + fn * FN_WEIGHT),
            "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "F1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
            "FP"       : int(fp),
            "FN"       : int(fn),
        })

    return pd.DataFrame(rows)


def select_optimal_threshold(df_thresh: pd.DataFrame) -> float:
    """
    Select the threshold that minimises clinical cost.

    Falls back to minimum-cost if no threshold achieves Recall ≥ 0.95.

    Parameters
    ----------
    df_thresh : pd.DataFrame
        Output of sweep_thresholds().

    Returns
    -------
    float
        The selected optimal threshold.
    """
    # Primary strategy: minimum clinical cost
    min_cost_idx  = df_thresh["Cost"].idxmin()
    optimal_cost  = df_thresh.loc[min_cost_idx, "Threshold"]

    # Secondary strategy: Recall ≥ 0.95 with maximum Precision
    high_recall   = df_thresh[df_thresh["Recall"] >= 0.95]
    if not high_recall.empty:
        best_idx  = high_recall["Precision"].idxmax()
        return float(df_thresh.loc[best_idx, "Threshold"])

    return float(optimal_cost)


# ── Optuna tuning ──────────────────────────────────────────────────────────────

def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = SEED,
) -> tuple[dict, optuna.Study]:
    """
    Tune XGBoost hyperparameters using Optuna TPE sampler, minimising
    the 5-fold CV clinical cost (FP×1 + FN×10).

    The objective returns negative cost (Optuna maximises by default).
    Using 5-fold StratifiedKFold ensures class balance is preserved in
    each fold, giving stable cost estimates despite the small dataset.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels.
    n_trials : int
        Number of Optuna trials (50 is sufficient for this search space).
    cv_folds : int
        Number of cross-validation folds.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    best_params : dict
        Hyperparameters for the best trial.
    study : optuna.Study
        Full study object for inspection and visualisation.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 50, 400),
            "max_depth"       : trial.suggest_int("max_depth", 3, 12),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma"           : trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 0.5, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 15.0),
            "eval_metric"     : "logloss",
            "verbosity"       : 0,
            "random_state"    : random_state,
        }
        model  = xgb.XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring=COST_SCORER, n_jobs=-1,
        )
        # scores are negated by make_scorer; mean gives average cost (negated)
        return scores.mean()

    study = optuna.create_study(
        direction = "maximize",   # maximising negated cost = minimising cost
        sampler   = optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, study


def build_best_model(
    scale_pos_weight: float,
    random_state: int = SEED,
) -> xgb.XGBClassifier:
    """
    Return the Phase 03 best model — XGBoost with scale_pos_weight.

    scale_pos_weight directly penalises FN in the XGBoost loss function,
    which proved more effective than SMOTE on this small dataset.
    Value is derived as: (# Normal samples) / (# Abnormal samples).

    Parameters
    ----------
    scale_pos_weight : float
        Ratio of negative to positive training samples.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    xgb.XGBClassifier
        Unfitted model with Phase 03 configuration.
    """
    return xgb.XGBClassifier(
        n_estimators     = 100,
        scale_pos_weight = scale_pos_weight,
        eval_metric      = "logloss",
        verbosity        = 0,
        random_state     = random_state,
    )
