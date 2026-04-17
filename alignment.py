"""
alignment.py
------------
Step 3: Step-sorted matrix creation
Step 4: Time alignment — lag features and time-delta features
Step 5.2: Forward fill (per-wafer)

Design notes
------------
* The only valid sort key is (wafer_id, seq_num).  Timestamp-based sorting is
  explicitly prohibited because clock skew between tools corrupts process order.

* Lag features are computed with groupby(wafer_id).shift(), so lags NEVER
  cross wafer boundaries.  Rows at the first step of each wafer get NaN lags —
  they are excluded later during causality analysis via dropna().

* Time features (delta_time, queue_time) are derived from tkout_time columns
  and are treated as regular numeric features for downstream analysis.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step 3 — Step-sorted matrix
# ---------------------------------------------------------------------------

def create_step_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format data to a step matrix.

    Index  : (wafer_id, seq_num)
    Columns: unique feature_name values

    When the same (wafer, step, feature) appears more than once (data quality
    issue), the first value is kept and a warning is emitted.
    """
    duplicates = long_df.duplicated(subset=['wafer_id', 'seq_num', 'feature_name'])
    if duplicates.any():
        n_dup = duplicates.sum()
        import warnings
        warnings.warn(
            f"create_step_matrix: {n_dup} duplicate (wafer, step, feature) rows found. "
            "Keeping first occurrence.",
            stacklevel=2,
        )
        long_df = long_df[~duplicates]

    matrix = long_df.pivot(
        index=['wafer_id', 'seq_num'],
        columns='feature_name',
        values='value',
    )
    matrix.columns.name = None

    # to_long_format concatenates mixed types (float, str, datetime) into a
    # single object-dtype 'value' column.  After pivoting, numeric columns
    # retain object dtype unless explicitly coerced.  Convert here so that
    # select_dtypes(include=[np.number]) works correctly downstream.
    def _try_numeric(col: pd.Series) -> pd.Series:
        try:
            return pd.to_numeric(col)
        except (ValueError, TypeError):
            return col

    matrix = matrix.apply(_try_numeric)

    return matrix


def sort_by_process_order(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the step matrix by (wafer_id ASC, seq_num ASC).

    This is mandatory.  seq_num encodes process order, not wall-clock time.
    """
    return matrix_df.sort_index(level=['wafer_id', 'seq_num'])


# ---------------------------------------------------------------------------
# Step 4.2 — Lag features
# ---------------------------------------------------------------------------

def create_lag_features(
    matrix_df: pd.DataFrame,
    feature_cols: List[str],
    lags: List[int] = [1, 2],
) -> pd.DataFrame:
    """
    Create per-wafer lag features for *feature_cols*.

    Lags are shifted within each wafer group — they do NOT cross wafer
    boundaries.  This is critical for valid Granger causality analysis.

    Parameters
    ----------
    matrix_df   : Step matrix sorted by (wafer_id, seq_num)
    feature_cols: Base feature columns to lag (should exclude existing lag cols)
    lags        : Lag orders, e.g. [1, 2] produces ``{col}_lag1``, ``{col}_lag2``

    Returns
    -------
    matrix_df with lag columns appended in-place (copy returned).
    """
    valid_cols = [c for c in feature_cols if c in matrix_df.columns]
    df = matrix_df.copy()

    for lag in lags:
        lagged = (
            df[valid_cols]
            .groupby(level='wafer_id')
            .shift(lag)
        )
        lagged = lagged.rename(columns={c: f'{c}_lag{lag}' for c in valid_cols})
        df = df.join(lagged)

    return df


# ---------------------------------------------------------------------------
# Step 4.3 — Time features
# ---------------------------------------------------------------------------

def create_time_features(
    matrix_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive time-based features from ``eqp_UN*_tkout_time`` columns.

    Features created (per tkout_time column found):
        delta_time_UN{seq} : seconds between consecutive steps within a wafer
        queue_time_UN{seq} : seconds between consecutive wafers at the same step

    Both are added as regular numeric columns and treated as features.
    tkout_time columns are parsed as datetime; numeric representations are
    handled gracefully.
    """
    df = matrix_df.copy()

    tkout_cols = [c for c in df.columns if 'tkout_time' in c]
    if not tkout_cols:
        return df

    for col in tkout_cols:
        # Parse to datetime if stored as string/object
        if df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        # Extract step tag from column name, e.g. 'eqp_UN000010_tkout_time' → 'UN000010'
        step_tag = col.split('_tkout_time')[0].replace('eqp_', '')

        # delta_time: within-wafer time gap between consecutive steps
        delta_col = f'delta_time_{step_tag}'
        raw_delta = df.groupby(level='wafer_id')[col].diff()
        df[delta_col] = _to_seconds(raw_delta)

        # queue_time: across-wafer waiting time at the same step
        # Computed by sorting wafers by tkout_time at each step, then diffing.
        queue_col = f'queue_time_{step_tag}'
        try:
            # Unstack wafer_id → columns so we operate on the seq_num axis
            unstacked = df[col].unstack(level='wafer_id')  # shape: (seq_num, n_wafers)
            queue_series: dict = {}
            for seq in unstacked.index:
                row = unstacked.loc[seq].dropna().sort_values()
                diffs = row.diff()
                for wid, qt in diffs.items():
                    queue_series[(wid, seq)] = _scalar_to_seconds(qt)
            df[queue_col] = pd.Series(queue_series)
        except Exception:
            # queue_time is optional; skip silently on errors
            pass

    return df


def _to_seconds(series: pd.Series) -> pd.Series:
    """Convert a timedelta or numeric series to float seconds."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return series.astype(float, errors='ignore')


def _scalar_to_seconds(value) -> float:
    """Convert a scalar timedelta or float to float seconds."""
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


# ---------------------------------------------------------------------------
# Step 5.2 — Forward fill
# ---------------------------------------------------------------------------

def forward_fill_by_wafer(
    matrix_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Forward-fill missing values within each wafer, along the seq_num axis.

    Physically: a measurement "holds" its last known value until a new
    measurement is taken at a later step.

    Parameters
    ----------
    matrix_df   : Step matrix (index: (wafer_id, seq_num))
    feature_cols: Columns to fill; defaults to all numeric columns
    """
    df = matrix_df.copy()

    cols: List[str] = (
        feature_cols
        if feature_cols is not None
        else df.select_dtypes(include=[np.number]).columns.tolist()
    )
    valid_cols = [c for c in cols if c in df.columns]
    df[valid_cols] = df[valid_cols].groupby(level='wafer_id').ffill()
    return df
