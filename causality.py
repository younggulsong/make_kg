"""
causality.py
------------
Step 6.1: Correlation-based edges  (Pearson + Spearman)
Step 6.2: Granger causality edges  (manual F-test on pre-computed lags)

Design notes — Granger on panel data
--------------------------------------
statsmodels.grangercausalitytests() creates lags internally from a raw array.
If we naively concatenate all wafers, lags at wafer boundaries would be
meaningless (last step of wafer N ↔ first step of wafer N+1).

Instead we:
  1. Use lag columns already created by alignment.create_lag_features()
     (those lags were computed per-wafer and are NaN at wafer boundaries).
  2. Build [Y_current | Y_lag1..k | X_lag1..k] matrices and run OLS F-tests
     directly (dropna() eliminates boundary rows automatically).

This gives us correct panel-aware Granger causality without any external
dependency beyond numpy/scipy.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f as f_dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_pair_data(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    min_samples: int,
) -> Optional[np.ndarray]:
    """Return stacked [col_a, col_b] array for rows where both are non-NaN."""
    valid = df[[col_a, col_b]].dropna()
    if len(valid) < min_samples:
        return None
    return valid.values


def _has_variance(arr: np.ndarray) -> bool:
    """True if every column in *arr* has non-trivial variance."""
    return bool(np.all(arr.std(axis=0) > 1e-10))


# ---------------------------------------------------------------------------
# Step 6.1 — Correlation edges
# ---------------------------------------------------------------------------

def _corr_pair(
    col_a: str,
    col_b: str,
    df: pd.DataFrame,
    min_samples: int,
) -> Optional[Dict]:
    data = _valid_pair_data(df, col_a, col_b, min_samples)
    if data is None:
        return None
    x, y = data[:, 0], data[:, 1]
    if not (_has_variance(x.reshape(-1, 1)) and _has_variance(y.reshape(-1, 1))):
        return None
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    return {
        'pearson': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman': float(spearman_r),
        'spearman_p': float(spearman_p),
        'n': len(x),
    }


def compute_correlation_edges(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.3,
    min_samples: int = 30,
    batch_size: int = 200,
) -> pd.DataFrame:
    """
    Compute undirected correlation edges for all feature pairs.

    Both Pearson and Spearman are computed; the edge weight is
    ``max(|pearson|, |spearman|)``.  Only pairs above *threshold* are kept.

    Parameters
    ----------
    df           : Flat (non-multi-indexed) feature matrix
    feature_cols : Feature names to test
    threshold    : Minimum |correlation| to emit an edge
    min_samples  : Minimum co-observed samples required
    batch_size   : Pairs per logging batch

    Returns
    -------
    DataFrame with columns: src, dst, type, weight, pearson, spearman,
                            pearson_p, spearman_p, n
    """
    valid_cols = [c for c in feature_cols if c in df.columns]
    pairs = list(combinations(valid_cols, 2))
    logger.info(f"Correlation: testing {len(pairs):,} pairs across {len(valid_cols)} features")

    edges: List[Dict] = []
    for i, (col_a, col_b) in enumerate(pairs):
        res = _corr_pair(col_a, col_b, df, min_samples)
        if res is not None:
            weight = max(abs(res['pearson']), abs(res['spearman']))
            if weight >= threshold:
                edges.append({
                    'src': col_a, 'dst': col_b,
                    'type': 'corr', 'weight': weight,
                    **res,
                })
        if (i + 1) % (batch_size * 10) == 0:
            logger.info(f"  Correlation progress: {i+1:,}/{len(pairs):,} pairs, "
                        f"{len(edges)} edges so far")

    logger.info(f"Correlation: {len(edges)} edges found (threshold={threshold})")
    _CORR_COLS = ['src', 'dst', 'type', 'weight', 'pearson', 'spearman',
                  'pearson_p', 'spearman_p', 'n']
    return pd.DataFrame(edges, columns=_CORR_COLS) if edges else pd.DataFrame(columns=_CORR_COLS)


# ---------------------------------------------------------------------------
# Step 6.2 — Granger causality edges
# ---------------------------------------------------------------------------

def _f_test_granger(
    y: np.ndarray,
    y_lags: np.ndarray,
    x_lags: np.ndarray,
) -> float:
    """
    Granger causality F-test using pre-computed lag arrays.

    Restricted   model: Y(t) = intercept + β·Y_lags
    Unrestricted model: Y(t) = intercept + β·Y_lags + γ·X_lags

    H₀: γ = 0  (X does not Granger-cause Y)

    Returns
    -------
    p-value  (lower → stronger evidence of causality)
    """
    n = len(y)
    ones = np.ones((n, 1))
    X_r = np.hstack([ones, y_lags])
    X_u = np.hstack([ones, y_lags, x_lags])

    k_r = X_r.shape[1]
    k_u = X_u.shape[1]
    df2 = n - k_u
    if df2 <= 0:
        return 1.0

    try:
        beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 1.0

    rss_r = float(np.sum((y - X_r @ beta_r) ** 2))
    rss_u = float(np.sum((y - X_u @ beta_u) ** 2))

    if rss_u < 1e-12:
        return 1.0

    df1 = k_u - k_r
    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
    if f_stat < 0:
        return 1.0

    return float(1.0 - f_dist.cdf(f_stat, df1, df2))


def _granger_pair(
    cause: str,
    effect: str,
    df: pd.DataFrame,
    maxlag: int,
    min_samples: int,
    p_threshold: float,
) -> Optional[Dict]:
    """
    Test whether *cause* Granger-causes *effect* using pre-computed lags.

    Looks for columns ``{cause}_lag1``, ``{cause}_lag2``, … and
    ``{effect}_lag1``, ``{effect}_lag2``, … that must already exist in *df*
    (created by alignment.create_lag_features).

    Uses the maximum lag order for which both cause and effect lags exist,
    up to *maxlag*.
    """
    # Collect available lag columns
    cause_lags = [f'{cause}_lag{k}' for k in range(1, maxlag + 1)
                  if f'{cause}_lag{k}' in df.columns]
    effect_lags = [f'{effect}_lag{k}' for k in range(1, maxlag + 1)
                   if f'{effect}_lag{k}' in df.columns]

    if not cause_lags or not effect_lags:
        return None

    # Use the minimum available lag depth
    used_lag = min(len(cause_lags), len(effect_lags))
    cause_lags = cause_lags[:used_lag]
    effect_lags = effect_lags[:used_lag]

    cols_needed = [effect] + effect_lags + cause_lags
    # Drop rows where any required column is NaN
    # (this naturally excludes wafer-boundary rows where lags are NaN)
    valid = df[cols_needed].dropna()
    if len(valid) < min_samples:
        return None

    y = valid[effect].values
    y_lag_arr = valid[effect_lags].values
    x_lag_arr = valid[cause_lags].values

    # Degenerate-data guard
    if y.std() < 1e-10 or not _has_variance(y_lag_arr):
        return None

    p_value = _f_test_granger(y, y_lag_arr, x_lag_arr)
    if p_value >= p_threshold:
        return None

    return {
        'src': cause,
        'dst': effect,
        'type': 'granger',
        'weight': float(1.0 - p_value),
        'p_value': float(p_value),
        'best_lag': used_lag,
        'n': len(valid),
    }


# ProcessPoolExecutor requires top-level picklable callables.
# _granger_pair_worker unpacks a single tuple argument for imap compatibility.
def _granger_pair_worker(args: Tuple) -> Optional[Dict]:
    return _granger_pair(*args)


def compute_granger_edges(
    df: pd.DataFrame,
    feature_cols: List[str],
    maxlag: int = 3,
    min_samples: int = 50,
    p_threshold: float = 0.05,
    n_jobs: int = 1,
    log_every: int = 500,
) -> pd.DataFrame:
    """
    Compute directed Granger causality edges for all ordered feature pairs.

    PREREQUISITE: *df* must contain lag columns (e.g. ``{col}_lag1``)
    created by ``alignment.create_lag_features()``.  Those lags are
    wafer-aware (no cross-wafer contamination).

    Parameters
    ----------
    df           : Flat feature matrix with pre-computed lag columns
    feature_cols : Base feature names (no ``_lag*`` suffix)
    maxlag       : Maximum lag order (must be ≤ lags created in alignment step)
    min_samples  : Minimum valid-row count for a pair to be tested
    p_threshold  : Significance level (p < threshold → directed edge X → Y)
    n_jobs       : Parallel workers (1 = serial)
    log_every    : Log progress every N pairs

    Returns
    -------
    DataFrame with columns: src, dst, type, weight, p_value, best_lag, n
    """
    valid_cols = [c for c in feature_cols if c in df.columns]
    pairs = [(a, b) for a in valid_cols for b in valid_cols if a != b]
    logger.info(f"Granger: testing {len(pairs):,} ordered pairs "
                f"({len(valid_cols)} features, maxlag={maxlag}, n_jobs={n_jobs})")

    args_list = [
        (cause, effect, df, maxlag, min_samples, p_threshold)
        for cause, effect in pairs
    ]

    edges: List[Dict] = []

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_granger_pair_worker, a): i
                       for i, a in enumerate(args_list)}
            for i, future in enumerate(as_completed(futures)):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    res = future.result()
                if res is not None:
                    edges.append(res)
                if (i + 1) % log_every == 0:
                    logger.info(f"  Granger: {i+1:,}/{len(pairs):,} done, "
                                f"{len(edges)} edges")
    else:
        for i, args in enumerate(args_list):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = _granger_pair(*args)
            if res is not None:
                edges.append(res)
            if (i + 1) % log_every == 0:
                logger.info(f"  Granger: {i+1:,}/{len(pairs):,} done, "
                            f"{len(edges)} edges")

    _GRANGER_COLS = ['src', 'dst', 'type', 'weight', 'p_value', 'best_lag', 'n']
    logger.info(f"Granger: {len(edges)} significant edges (p < {p_threshold})")
    return (
        pd.DataFrame(edges, columns=_GRANGER_COLS)
        if edges
        else pd.DataFrame(columns=_GRANGER_COLS)
    )
