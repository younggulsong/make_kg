"""
preprocessing.py
----------------
Step 1: Column parsing  →  Step 2: Long-format conversion
Step 5: Missing-data handling (mask creation, VM-based interpolation)

Design notes
------------
* Column naming convention supported:
    eqp_UN{seq}_eqp_id  / eqp_UN{seq}_tkout_time
    vm_UN{seq}_{item}_{subitem}   (also accepts '*' as separator)
    metro_UN{seq}_{item}_{subitem}
  Limitation: item_id itself must not contain the separator character.

* Missing masks are created BEFORE any fill/imputation so that the
  original missingness pattern is always preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class ColumnInfo:
    col_name: str            # original column name in the DataFrame
    col_type: str            # 'eqp', 'vm', or 'metro'
    seq_num: int             # process step number
    feature_name: str        # normalised name used as graph node label
    item_id: Optional[str] = None
    subitem_id: Optional[str] = None
    subtype: Optional[str] = None   # for eqp: 'eqp_id' or 'tkout_time'


# ---------------------------------------------------------------------------
# Regex patterns  (supports both '_' and '*' as separator)
# ---------------------------------------------------------------------------

_EQP_RE = re.compile(r'^eqp_UN(\d+)_(eqp_id|tkout_time)$')
_VM_RE = re.compile(r'^vm_UN(\d+)[_*](.+?)[_*](.+)$')
_METRO_RE = re.compile(r'^metro_UN(\d+)[_*](.+?)[_*](.+)$')


# ---------------------------------------------------------------------------
# Column parsing
# ---------------------------------------------------------------------------

def parse_column(col: str) -> Optional[ColumnInfo]:
    """Parse one column name into a ColumnInfo, or return None if unrecognised."""
    m = _EQP_RE.match(col)
    if m:
        seq_num, subtype = int(m.group(1)), m.group(2)
        return ColumnInfo(
            col_name=col, col_type='eqp', seq_num=seq_num,
            feature_name=f'eqp_UN{seq_num:06d}_{subtype}',
            subtype=subtype,
        )

    m = _VM_RE.match(col)
    if m:
        seq_num = int(m.group(1))
        item_id, subitem_id = m.group(2), m.group(3)
        return ColumnInfo(
            col_name=col, col_type='vm', seq_num=seq_num,
            feature_name=f'vm_{item_id}_{subitem_id}',
            item_id=item_id, subitem_id=subitem_id,
        )

    m = _METRO_RE.match(col)
    if m:
        seq_num = int(m.group(1))
        item_id, subitem_id = m.group(2), m.group(3)
        return ColumnInfo(
            col_name=col, col_type='metro', seq_num=seq_num,
            feature_name=f'metro_{item_id}_{subitem_id}',
            item_id=item_id, subitem_id=subitem_id,
        )

    return None


def parse_all_columns(
    df: pd.DataFrame,
) -> Tuple[List[ColumnInfo], Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Parse every column in *df* and build lookup dictionaries.

    Returns
    -------
    col_infos : list of ColumnInfo for every recognised column
    vm_dict   : {seq_num: [original_col_names]}
    metro_dict: {seq_num: [original_col_names]}
    """
    col_infos: List[ColumnInfo] = []
    vm_dict: Dict[int, List[str]] = {}
    metro_dict: Dict[int, List[str]] = {}

    for col in df.columns:
        info = parse_column(col)
        if info is None:
            continue
        col_infos.append(info)
        if info.col_type == 'vm':
            vm_dict.setdefault(info.seq_num, []).append(col)
        elif info.col_type == 'metro':
            metro_dict.setdefault(info.seq_num, []).append(col)

    return col_infos, vm_dict, metro_dict


# ---------------------------------------------------------------------------
# Missing-data masks
# ---------------------------------------------------------------------------

def create_missing_masks(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Create binary presence/absence masks for *feature_cols*.

    Returns a DataFrame with columns ``{col}_mask`` (1 = present, 0 = absent).
    The mask DataFrame shares the same index as *df*.

    IMPORTANT: call this BEFORE any fill or imputation.
    """
    mask_df = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            mask_df[f'{col}_mask'] = (~df[col].isna()).astype(np.int8)
    return mask_df


# ---------------------------------------------------------------------------
# Wide → Long conversion
# ---------------------------------------------------------------------------

def to_long_format(
    df: pd.DataFrame,
    col_infos: List[ColumnInfo],
    wafer_id_col: str = 'wafer_id',
) -> pd.DataFrame:
    """
    Convert wide-format dataframe to long format.

    Each output row represents one ``(wafer_id, seq_num, feature_name, value)``
    observation.  The ``feature_name`` strips the seq_num prefix so that the
    same physical feature measured at different steps shares the same name and
    becomes one column in the step-matrix pivot (Step 3).

    Parameters
    ----------
    df          : Wide DataFrame (rows = wafers, columns = step×feature combos)
    col_infos   : Parsed column metadata (from parse_all_columns)
    wafer_id_col: Column or index name that identifies each wafer
    """
    if wafer_id_col in df.columns:
        wafer_ids = df[wafer_id_col].values
    elif df.index.name == wafer_id_col:
        wafer_ids = df.index.values
    else:
        wafer_ids = np.arange(len(df))

    records: List[pd.DataFrame] = []
    for info in col_infos:
        if info.col_name not in df.columns:
            continue
        records.append(pd.DataFrame({
            'wafer_id': wafer_ids,
            'seq_num': info.seq_num,
            'feature_name': info.feature_name,
            'value': df[info.col_name].values,
        }))

    if not records:
        return pd.DataFrame(columns=['wafer_id', 'seq_num', 'feature_name', 'value'])

    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# VM-based interpolation for metro features
# ---------------------------------------------------------------------------

def interpolate_metro_with_vm(
    matrix_df: pd.DataFrame,
    col_infos: List[ColumnInfo],
) -> pd.DataFrame:
    """
    Fill missing metro values using the corresponding VM measurement at the
    same (wafer, step), when available.

    Matching rule: ``metro_{item}_{subitem}`` ← ``vm_{item}_{subitem}``
    (same item_id and subitem_id, different measurement type).

    Only fills cells that are still NaN after forward-fill.
    """
    # Build set of available column names for O(1) lookup
    cols_in_matrix = set(matrix_df.columns)
    fname_to_info = {info.feature_name: info for info in col_infos}

    df = matrix_df.copy()

    for col in list(df.columns):
        if not col.startswith('metro_'):
            continue
        info = fname_to_info.get(col)
        if info is None or info.item_id is None:
            continue
        vm_candidate = f'vm_{info.item_id}_{info.subitem_id}'
        if vm_candidate not in cols_in_matrix:
            continue
        missing = df[col].isna()
        if missing.any():
            df.loc[missing, col] = df.loc[missing, vm_candidate]

    return df
