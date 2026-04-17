"""
graph_builder.py
----------------
Step 7: Build directed knowledge graph from correlation + Granger edges.

Graph design
------------
* Nodes  : every base feature (vm, metro, eqp)
* Edges  : two types, stored as edge attribute ``type``
    - 'corr'    : undirected statistical association → added as two directed
                  edges (A→B and B→A) so DiGraph traversal covers both dirs
    - 'granger' : directed causal edge X → Y

Node attributes stored:
    mean, std, missing_rate, coverage, n_valid

Extension hooks
---------------
The graph is a plain NetworkX DiGraph.  To plug in PCMCI / NOTEARS / Temporal
GNN, export the edge list (export_edge_list) or the adjacency matrix via
nx.to_numpy_array(G, weight='weight').
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_knowledge_graph(
    nodes: List[str],
    corr_edges: pd.DataFrame,
    granger_edges: pd.DataFrame,
    add_reverse_corr: bool = True,
) -> nx.DiGraph:
    """
    Build a directed knowledge graph.

    Parameters
    ----------
    nodes           : All feature names (graph nodes)
    corr_edges      : Correlation edge DataFrame (from causality module)
    granger_edges   : Granger edge DataFrame (from causality module)
    add_reverse_corr: Add B→A edge for every A→B correlation edge
                      (makes correlation bidirectional in the directed graph)

    Returns
    -------
    nx.DiGraph with node and edge attributes populated.
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    # --- Correlation edges (symmetric relationship → bidirectional) ---
    if not corr_edges.empty:
        for _, row in corr_edges.iterrows():
            attrs = {
                'type': 'corr',
                'weight': float(row['weight']),
                'pearson': float(row.get('pearson', np.nan)),
                'spearman': float(row.get('spearman', np.nan)),
                'pearson_p': float(row.get('pearson_p', np.nan)),
                'spearman_p': float(row.get('spearman_p', np.nan)),
                'n': int(row.get('n', 0)),
            }
            G.add_edge(row['src'], row['dst'], **attrs)
            if add_reverse_corr:
                G.add_edge(row['dst'], row['src'], **attrs)

    # --- Granger edges (directed causal relationship) ---
    if not granger_edges.empty:
        for _, row in granger_edges.iterrows():
            G.add_edge(
                row['src'], row['dst'],
                type='granger',
                weight=float(row['weight']),
                p_value=float(row.get('p_value', np.nan)),
                best_lag=int(row.get('best_lag', -1)),
                n=int(row.get('n', 0)),
            )

    n_corr = len(corr_edges)
    n_granger = len(granger_edges)
    logger.info(
        f"Graph built: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges "
        f"({n_corr} corr × {'2' if add_reverse_corr else '1'} + {n_granger} granger)"
    )
    return G


# ---------------------------------------------------------------------------
# Node attributes
# ---------------------------------------------------------------------------

def add_node_attributes(
    G: nx.DiGraph,
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> nx.DiGraph:
    """
    Attach per-feature statistics to graph nodes (in-place).

    Statistics: mean, std, missing_rate, coverage, n_valid.
    """
    if feature_cols is None:
        feature_cols = list(G.nodes())

    for col in feature_cols:
        if col not in G.nodes or col not in df.columns:
            continue
        series = df[col]
        valid = series.dropna()
        G.nodes[col].update({
            'mean': float(valid.mean()) if len(valid) > 0 else float('nan'),
            'std': float(valid.std()) if len(valid) > 1 else float('nan'),
            'missing_rate': float(series.isna().mean()),
            'coverage': float(series.notna().mean()),
            'n_valid': int(series.notna().sum()),
        })

    return G


# ---------------------------------------------------------------------------
# Statistics and export
# ---------------------------------------------------------------------------

def compute_node_statistics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Return a DataFrame summarising degree statistics for every node.

    Columns: node, in_degree, out_degree, total_degree,
             granger_in, granger_out, corr_degree,
             coverage, missing_rate
    """
    records = []
    for node in G.nodes():
        in_edges = list(G.in_edges(node, data=True))
        out_edges = list(G.out_edges(node, data=True))

        granger_in = sum(1 for *_, d in in_edges if d.get('type') == 'granger')
        granger_out = sum(1 for *_, d in out_edges if d.get('type') == 'granger')
        # corr edges are bidirectional, so count each unique pair once
        corr_out = sum(1 for *_, d in out_edges if d.get('type') == 'corr')

        attrs = G.nodes[node]
        records.append({
            'node': node,
            'in_degree': len(in_edges),
            'out_degree': len(out_edges),
            'total_degree': len(in_edges) + len(out_edges),
            'granger_in': granger_in,
            'granger_out': granger_out,
            'corr_degree': corr_out,   # undirected correlation degree
            'coverage': attrs.get('coverage', float('nan')),
            'missing_rate': attrs.get('missing_rate', float('nan')),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values('total_degree', ascending=False).reset_index(drop=True)


def export_edge_list(G: nx.DiGraph) -> pd.DataFrame:
    """
    Export all graph edges as a tidy DataFrame.

    Standard columns (always present): src, dst, type, weight
    Additional columns depend on edge type (pearson, p_value, etc.)
    """
    records = []
    for src, dst, data in G.edges(data=True):
        records.append({'src': src, 'dst': dst, **data})

    if not records:
        return pd.DataFrame(columns=['src', 'dst', 'type', 'weight'])

    df = pd.DataFrame(records)
    std_cols = ['src', 'dst', 'type', 'weight']
    extra_cols = [c for c in df.columns if c not in std_cols]
    return df[std_cols + extra_cols].reset_index(drop=True)
