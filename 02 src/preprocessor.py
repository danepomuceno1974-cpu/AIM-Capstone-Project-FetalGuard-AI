"""
src/preprocessor.py
===================
Handles all data cleaning and feature engineering steps that operate on
the raw loaded DataFrame before the train/test split.

Why these steps come before splitting
--------------------------------------
- Deduplication: must see all rows to identify duplicates.
- Target mapping: NSP → binary label is a schema transformation, not a
  statistical operation, so it cannot leak information.
- Winsorization: in this project it is applied pre-split for EDA
  reproducibility. NOTE: in a strict production pipeline, winsorization
  bounds should be computed on the training set only (see pipeline.py).

Why these steps do NOT include scaling or resampling
-----------------------------------------------------
Scaling (StandardScaler) and resampling (SMOTE) must be applied AFTER
the train/test split to prevent test-set information from influencing
the training process. Those steps live in pipeline.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Physiologically impossible bounds derived from obstetric literature.
# Values outside these ranges are considered artefacts, not real signal.
# Source: FIGO 2015 intrapartum monitoring guidelines.
PHYSIO_BOUNDS: dict[str, tuple[float, float]] = {
    "LB"      : (50,   250),   # Baseline FHR bpm: <50 = asystole, >250 = artefact
    "AC"      : (0,    50),    # Accelerations per second
    "FM"      : (0,    600),   # Fetal movements per second
    "UC"      : (0,    50),    # Uterine contractions per second
    "ASTV"    : (0,    100),   # Percentage (0–100%)
    "MSTV"    : (0,    20),    # Mean STV value
    "ALTV"    : (0,    100),   # Percentage (0–100%)
    "MLTV"    : (0,    60),    # Mean LTV value
    "DL"      : (0,    100),   # Light decelerations count
    "DS"      : (0,    10),    # Severe decelerations count
    "DP"      : (0,    20),    # Prolonged decelerations count
    "Width"   : (0,    300),   # CTG histogram width
    "Min"     : (50,   200),   # CTG histogram minimum FHR
    "Max"     : (100,  300),   # CTG histogram maximum FHR
    "Variance": (0,    1000),  # CTG histogram variance
}

# NSP class mapping: binary reformulation.
# 1 (Normal) → 0 | 2 (Suspect) + 3 (Pathological) → 1 (Abnormal)
# Rationale: both Suspect and Pathological require clinical action,
# making a 3-class model harder to calibrate against a cost function.
NSP_MAP: dict[int, int] = {1: 0, 2: 1, 3: 1}

EXCLUDE_COLS = ("NSP", "target")


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (all columns).

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame with duplicates removed and index reset.
    n_removed : int
        Number of rows removed.
    """
    n_before  = len(df)
    df_clean  = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df_clean)
    return df_clean, n_removed


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary 'target' column derived from the NSP column.

    NSP 1 → 0 (Normal)
    NSP 2 or 3 → 1 (Abnormal)

    The original NSP column is retained for reference but excluded from
    the feature set in get_feature_names().

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing an 'NSP' column.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with an additional 'target' column.

    Raises
    ------
    KeyError
        If 'NSP' column is not present.
    """
    if "NSP" not in df.columns:
        raise KeyError("Column 'NSP' not found. Expected a raw CTG DataFrame.")

    df = df.copy()
    df["target"] = df["NSP"].map(NSP_MAP)

    unmapped = df["target"].isnull().sum()
    if unmapped > 0:
        raise ValueError(
            f"{unmapped} rows have NSP values not in {{1, 2, 3}}. "
            "Check the raw dataset for unexpected class labels."
        )
    return df


def winsorize(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Cap extreme values at the 1st/99th percentile, further constrained
    by physiologically impossible bounds (PHYSIO_BOUNDS).

    Uses numpy directly (not pandas .quantile()) to avoid ArrowDtype
    compatibility issues with older/alternative pandas backends.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the feature columns. Must have numeric dtypes.
    features : list[str]
        Column names to winsorize. Other columns are left unchanged.

    Returns
    -------
    pd.DataFrame
        Copy of df with outlier values capped.

    Notes
    -----
    In this project, winsorization is applied on the full dataset during
    Phase 01 EDA. In a strict no-leakage production pipeline, call this
    function only on the training set and apply the resulting bounds to
    the test set separately.
    """
    df_clean = df.copy()
    for feat in features:
        s      = df_clean[feat].to_numpy(dtype=float, na_value=np.nan)
        lo     = float(np.nanpercentile(s, 1))
        hi     = float(np.nanpercentile(s, 99))
        bounds = PHYSIO_BOUNDS.get(feat)
        if bounds:
            lo = max(lo, bounds[0])
            hi = min(hi, bounds[1])
        df_clean[feat] = np.clip(s, lo, hi)
    return df_clean


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """
    Return the list of numeric feature column names, excluding
    the NSP source column and the derived 'target' label.

    Only numeric columns are returned because non-numeric columns
    (e.g. string labels, object dtype) cannot be fed to sklearn scalers
    or XGBoost without encoding — and this dataset has no categorical
    features that require encoding.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after target mapping.

    Returns
    -------
    list[str]
        Ordered list of numeric feature column names.
    """
    return [
        col for col in df.columns
        if col not in EXCLUDE_COLS
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def preprocess(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Full preprocessing pipeline: deduplicate → map target → winsorize.

    Convenience function that chains the individual steps. Returns the
    cleaned DataFrame and the derived feature name list.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw DataFrame from data_loader.load_raw().

    Returns
    -------
    df_clean : pd.DataFrame
        Fully cleaned DataFrame ready for train/test split.
    features : list[str]
        Numeric feature column names (22 in the standard CTG dataset).
    """
    df, _       = remove_duplicates(df_raw)
    df          = map_target(df)
    features    = get_feature_names(df)
    df          = winsorize(df, features)
    return df, features
