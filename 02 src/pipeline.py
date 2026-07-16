"""
src/pipeline.py
===============
Implements the no-leakage preprocessing pipeline:
  Split → Scale → Resample

Each step is strictly ordered to prevent test-set information from
influencing training, following the principle described in Phase 02.

Order matters
-------------
1. SPLIT first — before any statistical fitting.
   If we scaled the full dataset first and then split, the scaler would
   have seen test-set values during fit(), leaking their distribution.

2. SCALE second — fit the StandardScaler on X_train only.
   The scaler learns mean and standard deviation from training data.
   It then transforms both X_train and X_test using those same statistics.
   "scaler.transform(X_test)" applies the training distribution — it does
   NOT refit. This is the correct no-leakage pattern.

3. RESAMPLE last — apply SMOTE/ADASYN to X_train_scaled only.
   Synthetic samples must never appear in the test set. If SMOTE were
   applied before splitting, synthetic minority-class samples derived
   from training patterns would contaminate the test evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling  import SMOTE, ADASYN

SEED = 42


def split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split.

    Stratification preserves the class ratio in both sets, which is
    critical when the dataset is imbalanced (~57% Normal / ~43% Abnormal).
    Without stratification, one fold could have very few Abnormal cases,
    making evaluation metrics unstable.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target (0=Normal, 1=Abnormal).
    test_size : float
        Fraction of data to hold out for testing.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size    = test_size,
        stratify     = y,       # preserve class ratio in both sets
        random_state = random_state,
    )


def scale(
    X_train: pd.DataFrame | np.ndarray,
    X_test:  pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data only, then transform both sets.

    StandardScaler subtracts the mean and divides by the standard deviation.
    Fitting only on X_train ensures the scaler never 'sees' test-set
    statistics, which would constitute data leakage.

    The scaler object is returned so it can be saved to disk and reused
    at inference time (the SAME scaler must transform new patient data).

    Parameters
    ----------
    X_train : array-like
        Training features (raw scale).
    X_test : array-like
        Test features (raw scale).

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled  : np.ndarray
    scaler         : fitted StandardScaler
    """
    scaler = StandardScaler()

    # fit_transform on train: learns μ and σ from training data, then scales
    X_train_scaled = scaler.fit_transform(X_train)

    # transform only on test: applies the SAME μ and σ learned from training
    # Never call scaler.fit_transform(X_test) — that would leak test distribution
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def resample_smote(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    random_state: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample the minority class using SMOTE on the training set only.

    SMOTE (Synthetic Minority Over-sampling Technique) generates new
    minority-class samples by interpolating between existing ones in
    feature space. It must be applied AFTER scaling so that the synthetic
    samples are created in the same normalised space the model trains on.

    The test set is never resampled — it must represent the real-world
    class distribution to give unbiased evaluation metrics.

    Parameters
    ----------
    X_train_scaled : np.ndarray
        Scaled training features.
    y_train : array-like
        Training labels.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    X_resampled : np.ndarray
    y_resampled : np.ndarray
    """
    smote = SMOTE(sampling_strategy="minority", random_state=random_state)
    return smote.fit_resample(X_train_scaled, y_train)


def resample_adasyn(

    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    random_state: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample using ADASYN — focuses synthetic samples near the decision
    boundary, where the model is most uncertain.

    ADASYN generates more samples for minority-class instances that are
    harder to learn (i.e. surrounded by majority-class neighbours).
    This can improve recall for borderline cases but may slightly increase
    the class imbalance compared to SMOTE's exact 1:1 ratio.

    Parameters
    ----------
    X_train_scaled : np.ndarray
        Scaled training features.
    y_train : array-like
        Training labels.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    X_resampled : np.ndarray
    y_resampled : np.ndarray
    """
    adasyn = ADASYN(sampling_strategy="minority", random_state=random_state)
    return adasyn.fit_resample(X_train_scaled, y_train)


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """
    Compute balanced class weights for cost-sensitive training.

    'balanced' mode calculates weights inversely proportional to class
    frequency: weight_k = n_samples / (n_classes * count_k).
    This penalises misclassification of the minority class (Abnormal)
    more heavily without generating synthetic data.

    Parameters
    ----------
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    dict[int, float]
        {class_label: weight} suitable for sklearn's class_weight parameter.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes.tolist(), weights.tolist()))


def build_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> dict:
    """
    Execute the full no-leakage pipeline in the correct order.

    Returns all train/test variants (raw scaled, SMOTE, ADASYN) and the
    fitted scaler, mirroring the structure of ctg_preprocessed.pkl.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (after winsorization).
    y : pd.Series
        Binary target.
    test_size : float
        Test set fraction.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    dict with keys:
        X_train_scaled, y_train, X_test_scaled, y_test,
        X_train_smote, y_train_smote,
        X_train_adasyn, y_train_adasyn,
        class_weight_dict, scaler, feature_names
    """
    X_train, X_test, y_train, y_test = split(X, y, test_size, random_state)
    X_train_scaled, X_test_scaled, scaler = scale(X_train, X_test)

    X_train_smote,  y_train_smote  = resample_smote(X_train_scaled, y_train, random_state)
    X_train_adasyn, y_train_adasyn = resample_adasyn(X_train_scaled, y_train, random_state)
    class_weight_dict               = compute_class_weights(y_train)

    return {
        "X_train_scaled"   : X_train_scaled,
        "y_train"          : np.array(y_train),
        "X_train_smote"    : X_train_smote,
        "y_train_smote"    : y_train_smote,
        "X_train_adasyn"   : X_train_adasyn,
        "y_train_adasyn"   : y_train_adasyn,
        "class_weight_dict": class_weight_dict,
        "X_test_scaled"    : X_test_scaled,
        "y_test"           : np.array(y_test),
        "scaler"           : scaler,
        "feature_names"    : list(X.columns),
    }
