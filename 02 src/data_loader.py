"""
src/data_loader.py
==================
Responsible for loading raw CTG data from Excel and converting it to
standard numpy-backed dtypes.

Why this module exists
----------------------
When pandas is configured with a PyArrow dtype backend (common in newer
environments), numeric columns arrive as ArrowDtype (e.g. int64[pyarrow])
which breaks downstream operations like .quantile(), sklearn scalers, and
scipy statistical tests. This module strips ArrowDtype immediately after
loading so all subsequent code can rely on plain float64/int64 columns.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

# Columns based on standard UCI list
standard_cols = ['LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 
                'DL', 'DS', 'DP', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 
                'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'NSP']

def _safe_to_float(col: pd.Series) -> pd.Series:
    """
    Convert a single Series to float64, silently leaving non-numeric
    columns (e.g. string labels) unchanged.

    Parameters
    ----------
    col : pd.Series
        A column that may have an Arrow-backed or standard numeric dtype.

    Returns
    -------
    pd.Series
        float64 Series if conversion succeeds, otherwise the original Series.

    Notes
    -----
    Using try/except rather than dtype inspection because ArrowDtype objects
    do not have a reliable .kind attribute across all pandas/pyarrow versions.
    """
    try:
        return col.astype("float64")
    except (ValueError, TypeError):
        # Non-numeric column (e.g. large_string[pyarrow]) — leave as-is
        return col


def load_raw(path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    """
    Load the CTG Excel file and convert all numeric columns to float64.

    Parameters
    ----------
    path : str or Path
        Path to the raw test.xlsx file.
    sheet_name : str, int, or None
        Name or index of the sheet to load.

    Returns
    -------
    pd.DataFrame
        DataFrame with all numeric columns as float64, non-numeric untouched.

    Examples
    --------
    >>> df = load_raw("test.xlsx")
    >>> df.dtypes["LB"]
    dtype('float64')
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CTG data file not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)

    
    # Select only the standard columns
    df = df[standard_cols]

    # Strip Arrow extension types immediately after loading.
    # pd.concat reconstructs the DataFrame column-by-column, applying
    # _safe_to_float to each. This is equivalent to df.apply(...) but
    # preserves column order reliably across pandas versions.

    dict_median= df.median()
    df.fillna(value=dict_median, inplace=True)  # Fill missing values with median

    
    df = pd.concat([_safe_to_float(df[col]) for col in df.columns], axis=1)

    return df


def describe_raw(df: pd.DataFrame) -> dict:
    """
    Return a summary dict of raw dataset properties.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loaded DataFrame (before any preprocessing).

    Returns
    -------
    dict
        Keys: shape, n_missing, n_duplicates, dtypes_summary.
    """
    return {
        "shape"          : df.shape,
        "n_missing"      : int(df.isnull().sum().sum()),
        "n_duplicates"   : int(df.duplicated().sum()),
        "dtypes_summary" : df.dtypes.value_counts().to_dict(),
    }
