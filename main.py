"""
main.py
-------
Pipeline orchestration for the semiconductor process knowledge graph.

Usage (CLI)
-----------
    # Demo with synthetic data
    python main.py --demo --output ./output

    # Real data
    python main.py --input data.csv --wafer-id wafer_id --output ./output

    # Tune thresholds
    python main.py --input data.csv --corr-threshold 0.4 --granger-p 0.01 --n-jobs 4

Usage (Python API)
------------------
    from main import run_pipeline, KGConfig
    import pandas as pd

    df = pd.read_csv('your_data.csv')
    config = KGConfig(wafer_id_col='wafer_id', output_dir='./output')
    G, edge_df, node_df = run_pipeline(df, config)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from alignment import (
    create_lag_features,
    create_step_matrix,
    create_time_features,
    forward_fill_by_wafer,
    sort_by_process_order,
)
from causality import compute_correlation_edges, compute_granger_edges
from graph_builder import (
    add_node_attributes,
    build_knowledge_graph,
    compute_node_statistics,
    export_edge_list,
)
from preprocessing import (
    create_missing_masks,
    interpolate_metro_with_vm,
    parse_all_columns,
    to_long_format,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KGConfig:
    """
    All tunable parameters for the knowledge graph pipeline.

    Defaults are sane starting points; adjust based on data characteristics.
    """
    # --- Data ---
    wafer_id_col: str = 'wafer_id'

    # --- Alignment ---
    lags: List[int] = field(default_factory=lambda: [1, 2])
    forward_fill: bool = True
    vm_interpolation: bool = True

    # --- Correlation ---
    corr_threshold: float = 0.3
    corr_min_samples: int = 30

    # --- Granger causality ---
    granger_maxlag: int = 3
    granger_min_samples: int = 50
    granger_p_threshold: float = 0.05
    granger_n_jobs: int = 1

    # --- Graph ---
    add_reverse_corr: bool = True

    # --- Output ---
    output_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    config: Optional[KGConfig] = None,
) -> Tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: raw wide-format DataFrame → Knowledge Graph.

    Steps
    -----
    1.  Parse columns          (preprocessing)
    2.  Create missing masks   (preprocessing)
    3.  Wide → Long format     (preprocessing)
    4.  Pivot to step matrix   (alignment)
    5.  Sort by process order  (alignment)
    6a. Forward fill           (alignment)
    6b. VM interpolation       (preprocessing)
    7a. Lag features           (alignment)
    7b. Time features          (alignment)
    8.  Correlation edges      (causality)
    9.  Granger causality edges(causality)
    10. Build & annotate graph (graph_builder)

    Parameters
    ----------
    df     : Wide DataFrame — one row per wafer, one column per
             (step × feature) measurement.
    config : Pipeline configuration; uses defaults if None.

    Returns
    -------
    G        : NetworkX DiGraph (the knowledge graph)
    edge_df  : Edge list DataFrame
    node_df  : Node statistics DataFrame
    """
    if config is None:
        config = KGConfig()

    # ── Step 1: Column parsing ───────────────────────────────────────────────
    logger.info("Step 1 · Parsing columns")
    col_infos, vm_dict, metro_dict = parse_all_columns(df)

    n_eqp = sum(1 for i in col_infos if i.col_type == 'eqp')
    n_vm = sum(1 for i in col_infos if i.col_type == 'vm')
    n_metro = sum(1 for i in col_infos if i.col_type == 'metro')
    n_steps = len({i.seq_num for i in col_infos})
    logger.info(f"  {len(col_infos)} recognised columns | "
                f"{n_steps} steps | eqp={n_eqp} vm={n_vm} metro={n_metro}")

    if not col_infos:
        raise ValueError(
            "No recognised columns found.  Check that column names follow the "
            "expected format: eqp_UN{seq}_{type}, vm_UN{seq}_{item}_{sub}, "
            "metro_UN{seq}_{item}_{sub}."
        )

    # ── Step 2: Missing masks ────────────────────────────────────────────────
    logger.info("Step 2 · Creating missing-data masks")
    original_col_names = [i.col_name for i in col_infos]
    mask_df = create_missing_masks(df, original_col_names)
    logger.info(f"  {len(mask_df.columns)} mask columns created")

    # ── Step 3: Long format ──────────────────────────────────────────────────
    logger.info("Step 3 · Converting to long format")
    long_df = to_long_format(df, col_infos, wafer_id_col=config.wafer_id_col)
    logger.info(f"  Long format: {len(long_df):,} rows "
                f"({long_df['wafer_id'].nunique()} wafers × "
                f"{long_df['seq_num'].nunique()} steps × "
                f"{long_df['feature_name'].nunique()} features)")

    # ── Step 4: Step matrix ──────────────────────────────────────────────────
    logger.info("Step 4 · Pivoting to step matrix")
    matrix = create_step_matrix(long_df)
    logger.info(f"  Matrix shape: {matrix.shape}  "
                f"(rows=(wafer,step), cols=features)")

    # ── Step 5: Sort by process order ────────────────────────────────────────
    logger.info("Step 5 · Sorting by process order (seq_num, NOT timestamp)")
    matrix = sort_by_process_order(matrix)

    # ── Step 6: Missing-data handling ────────────────────────────────────────
    logger.info("Step 6 · Missing-data handling")

    base_feature_cols = matrix.select_dtypes(include=[np.number]).columns.tolist()

    if config.forward_fill:
        logger.info("  Forward-filling within wafer groups")
        matrix = forward_fill_by_wafer(matrix, base_feature_cols)

    if config.vm_interpolation:
        logger.info("  Interpolating remaining metro gaps with VM values")
        matrix = interpolate_metro_with_vm(matrix, col_infos)

    # ── Step 7: Lag and time features ────────────────────────────────────────
    logger.info(f"Step 7 · Creating lag features (lags={config.lags})")
    matrix = create_lag_features(matrix, base_feature_cols, lags=config.lags)

    logger.info("Step 7b · Deriving time features from tkout_time")
    matrix = create_time_features(matrix)

    # Flatten to a plain DataFrame for statistical tests
    # Lag columns are already present; NaN rows at wafer boundaries are handled
    # inside causality functions via dropna().
    flat_df = matrix.reset_index()

    # Analysis columns: base features only (no lag suffix, no mask suffix)
    analysis_cols = [c for c in base_feature_cols if c in flat_df.columns]
    logger.info(f"  {len(analysis_cols)} features available for causality analysis")

    # ── Step 8: Correlation edges ─────────────────────────────────────────────
    logger.info("Step 8 · Computing correlation edges")
    corr_edges = compute_correlation_edges(
        flat_df,
        feature_cols=analysis_cols,
        threshold=config.corr_threshold,
        min_samples=config.corr_min_samples,
    )

    # ── Step 9: Granger causality edges ──────────────────────────────────────
    logger.info("Step 9 · Computing Granger causality edges")
    granger_edges = compute_granger_edges(
        flat_df,
        feature_cols=analysis_cols,
        maxlag=config.granger_maxlag,
        min_samples=config.granger_min_samples,
        p_threshold=config.granger_p_threshold,
        n_jobs=config.granger_n_jobs,
    )

    # ── Step 10: Build graph ──────────────────────────────────────────────────
    logger.info("Step 10 · Building knowledge graph")
    G = build_knowledge_graph(
        nodes=analysis_cols,
        corr_edges=corr_edges,
        granger_edges=granger_edges,
        add_reverse_corr=config.add_reverse_corr,
    )
    add_node_attributes(G, flat_df, analysis_cols)

    edge_df = export_edge_list(G)
    node_df = compute_node_statistics(G)

    # ── Optional: save outputs ───────────────────────────────────────────────
    if config.output_dir:
        out_path = Path(config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        edge_df.to_csv(out_path / 'edges.csv', index=False)
        node_df.to_csv(out_path / 'nodes.csv', index=False)
        nx.write_graphml(G, str(out_path / 'knowledge_graph.graphml'))
        logger.info(f"Outputs saved to {config.output_dir}/")

    logger.info("Pipeline complete.")
    return G, edge_df, node_df


# ---------------------------------------------------------------------------
# Synthetic demo data
# ---------------------------------------------------------------------------

def make_demo_data(
    n_wafers: int = 100,
    steps: Optional[List[int]] = None,
    seed: int = 42,
    wafer_id_col: str = 'wafer_id',
) -> pd.DataFrame:
    """
    Generate a synthetic wide-format DataFrame that mimics real fab data.

    Structure
    ---------
    * 2 vm features per step (item A/B × sub x/y) — ~95% coverage
    * 2 metro features per step (item P/Q × sub 1/2) — 10–90% missing
    * eqp_id and tkout_time per step

    A mild causal signal is injected: vm_A_x at step t influences
    vm_B_y at step t+1 (with noise), so Granger tests should detect it.
    """
    if steps is None:
        steps = [10, 20, 30, 40, 50]

    rng = np.random.default_rng(seed)
    base_time = pd.Timestamp('2024-01-01')
    rows = []

    # Pre-generate latent process signal for causal injection
    # signal[w, s] = base process value for wafer w at step s
    signal = rng.normal(0, 1, (n_wafers, len(steps)))
    # Add mild AR(1) structure along steps
    for s in range(1, len(steps)):
        signal[:, s] += 0.5 * signal[:, s - 1]

    for wid in range(n_wafers):
        row: dict = {wafer_id_col: f'W{wid:04d}'}
        t = base_time + pd.Timedelta(hours=wid * 0.5)

        for si, step in enumerate(steps):
            row[f'eqp_UN{step:06d}_eqp_id'] = f'EQP_{step % 3 + 1}'
            row[f'eqp_UN{step:06d}_tkout_time'] = t + pd.Timedelta(minutes=step // 5)

            # vm features — moderate signal + noise
            for item, noise_scale in [('A', 0.3), ('B', 0.5)]:
                for sub in ['x', 'y']:
                    val = signal[wid, si] + rng.normal(0, noise_scale)
                    if rng.random() < 0.05:          # ~5% missing
                        val = np.nan
                    row[f'vm_UN{step:06d}_{item}_{sub}'] = val

            # metro features — high missing rate (10–90%)
            for item in ['P', 'Q']:
                for sub in ['1', '2']:
                    missing_rate = rng.uniform(0.1, 0.9)
                    if rng.random() < missing_rate:
                        val = np.nan
                    else:
                        val = signal[wid, si] * 1.2 + rng.normal(0, 0.8)
                    row[f'metro_UN{step:06d}_{item}_{sub}'] = val

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Build a semiconductor process knowledge graph',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input', type=str, help='Input CSV file path')
    p.add_argument('--output', type=str, default='./output', help='Output directory')
    p.add_argument('--wafer-id', type=str, default='wafer_id', help='Wafer ID column')
    p.add_argument('--lags', type=int, nargs='+', default=[1, 2], help='Lag orders')
    p.add_argument('--corr-threshold', type=float, default=0.3)
    p.add_argument('--corr-min-samples', type=int, default=30)
    p.add_argument('--granger-maxlag', type=int, default=3)
    p.add_argument('--granger-min-samples', type=int, default=50)
    p.add_argument('--granger-p', type=float, default=0.05)
    p.add_argument('--n-jobs', type=int, default=1, help='Parallel workers for Granger')
    p.add_argument('--no-ffill', action='store_true', help='Disable forward fill')
    p.add_argument('--no-vm-interp', action='store_true', help='Disable VM interpolation')
    p.add_argument('--demo', action='store_true', help='Run with synthetic demo data')
    p.add_argument('--demo-wafers', type=int, default=100,
                   help='Number of wafers in demo data')
    return p


def main() -> None:
    args = _build_parser().parse_args()

    config = KGConfig(
        wafer_id_col=args.wafer_id,
        lags=args.lags,
        forward_fill=not args.no_ffill,
        vm_interpolation=not args.no_vm_interp,
        corr_threshold=args.corr_threshold,
        corr_min_samples=args.corr_min_samples,
        granger_maxlag=args.granger_maxlag,
        granger_min_samples=args.granger_min_samples,
        granger_p_threshold=args.granger_p,
        granger_n_jobs=args.n_jobs,
        output_dir=args.output,
    )

    if args.demo or args.input is None:
        logger.info(f"Generating demo data ({args.demo_wafers} wafers)...")
        df = make_demo_data(n_wafers=args.demo_wafers, wafer_id_col=args.wafer_id)
    else:
        logger.info(f"Loading data from {args.input} ...")
        df = pd.read_csv(args.input)

    G, edge_df, node_df = run_pipeline(df, config)

    print("\n" + "─" * 60)
    print(f"Knowledge Graph Summary")
    print("─" * 60)
    print(f"  Nodes  : {G.number_of_nodes()}")
    print(f"  Edges  : {G.number_of_edges()}  "
          f"(corr={edge_df[edge_df.type=='corr'].shape[0]}, "
          f"granger={edge_df[edge_df.type=='granger'].shape[0]})")
    print("\nTop 10 nodes by total degree:")
    print(node_df.head(10).to_string(index=False))
    print("\nSample edges (first 10):")
    print(edge_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
