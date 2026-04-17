"""
visualize.py
------------
예제 데이터를 생성하고 Knowledge Graph를 시각화한다.

출력 파일
---------
output/kg_full.png       : 전체 그래프 (corr + granger)
output/kg_granger.png    : Granger 인과 그래프만
output/kg_corr_heat.png  : Correlation 히트맵
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# 프로젝트 모듈 임포트
sys.path.insert(0, str(Path(__file__).parent))
from main import KGConfig, make_demo_data, run_pipeline


# ---------------------------------------------------------------------------
# 인과 신호가 주입된 강화 demo 데이터
# ---------------------------------------------------------------------------

def make_causal_demo(n_wafers: int = 150, seed: int = 0) -> pd.DataFrame:
    """
    명확한 인과 관계가 심어진 예제 데이터를 생성한다.

    심어진 관계 (Granger가 발견해야 함):
        vm_A_x  → metro_P_1  (lag 1, 강한 인과)
        vm_B_y  → metro_Q_2  (lag 2, 중간 인과)
        metro_P_1 → vm_A_y   (lag 1, 약한 인과)

    데이터 구조:
        5개 공정 step  ×  4 vm 피처  ×  4 metro 피처
        vm coverage > 95%,  metro coverage 30~70%
    """
    steps = [10, 20, 30, 40, 50]
    rng = np.random.default_rng(seed)
    base_time = pd.Timestamp('2024-01-01')

    rows = []
    # wafer 당 독립적인 latent signal
    lat = rng.normal(0, 1, (n_wafers, len(steps)))  # (wafer, step)
    for s in range(1, len(steps)):
        lat[:, s] += 0.3 * lat[:, s - 1]            # 약한 AR(1)

    for wid in range(n_wafers):
        row: dict = {'wafer_id': f'W{wid:04d}'}
        t = base_time + pd.Timedelta(hours=wid * 0.5)

        # ── feature values (step 순 생성으로 lag 신호 심기) ──────────────
        vm_A_x = np.zeros(len(steps))
        vm_A_y = np.zeros(len(steps))
        vm_B_x = np.zeros(len(steps))
        vm_B_y = np.zeros(len(steps))
        metro_P_1 = np.full(len(steps), np.nan)
        metro_P_2 = np.full(len(steps), np.nan)
        metro_Q_1 = np.full(len(steps), np.nan)
        metro_Q_2 = np.full(len(steps), np.nan)

        for si in range(len(steps)):
            vm_A_x[si] = lat[wid, si] + rng.normal(0, 0.3)
            vm_B_x[si] = lat[wid, si] * 0.8 + rng.normal(0, 0.4)
            vm_B_y[si] = lat[wid, si] * 0.6 + rng.normal(0, 0.5)

            # metro_P_1 ← vm_A_x(t-1) 인과 신호 (lag 1)
            cause_p1 = vm_A_x[si - 1] * 0.7 if si > 0 else 0.0
            metro_P_1[si] = cause_p1 + rng.normal(0, 0.4)

            # metro_Q_2 ← vm_B_y(t-2) 인과 신호 (lag 2)
            cause_q2 = vm_B_y[si - 2] * 0.6 if si > 1 else 0.0
            metro_Q_2[si] = cause_q2 + rng.normal(0, 0.5)

            # vm_A_y ← metro_P_1(t-1) 인과 신호 (lag 1)
            cause_ay = metro_P_1[si - 1] * 0.4 if si > 0 else 0.0
            vm_A_y[si] = cause_ay + lat[wid, si] * 0.3 + rng.normal(0, 0.5)

            # metro_P_2, metro_Q_1 : latent signal 반영 (인과 없음)
            metro_P_2[si] = lat[wid, si] * 0.5 + rng.normal(0, 0.7)
            metro_Q_1[si] = lat[wid, si] * 0.4 + rng.normal(0, 0.8)

        # ── 결측 마스크 적용 ─────────────────────────────────────────────
        metro_missing_rate = rng.uniform(0.30, 0.70, 4)  # metro 30~70% 결측
        for si, step in enumerate(steps):
            row[f'eqp_UN{step:06d}_eqp_id'] = f'EQP_{step % 3 + 1}'
            row[f'eqp_UN{step:06d}_tkout_time'] = t + pd.Timedelta(minutes=step // 5)

            row[f'vm_UN{step:06d}_A_x'] = vm_A_x[si] if rng.random() > 0.04 else np.nan
            row[f'vm_UN{step:06d}_A_y'] = vm_A_y[si] if rng.random() > 0.04 else np.nan
            row[f'vm_UN{step:06d}_B_x'] = vm_B_x[si] if rng.random() > 0.04 else np.nan
            row[f'vm_UN{step:06d}_B_y'] = vm_B_y[si] if rng.random() > 0.04 else np.nan

            row[f'metro_UN{step:06d}_P_1'] = (
                metro_P_1[si] if rng.random() > metro_missing_rate[0] else np.nan
            )
            row[f'metro_UN{step:06d}_P_2'] = (
                metro_P_2[si] if rng.random() > metro_missing_rate[1] else np.nan
            )
            row[f'metro_UN{step:06d}_Q_1'] = (
                metro_Q_1[si] if rng.random() > metro_missing_rate[2] else np.nan
            )
            row[f'metro_UN{step:06d}_Q_2'] = (
                metro_Q_2[si] if rng.random() > metro_missing_rate[3] else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 시각화 유틸
# ---------------------------------------------------------------------------

NODE_COLOR = {
    'vm': '#4C9BE8',      # 파란색
    'metro': '#E8724C',   # 주황색
    'eqp': '#6ECC6E',     # 초록색
}

def _node_color(name: str) -> str:
    if name.startswith('vm_'):
        return NODE_COLOR['vm']
    if name.startswith('metro_'):
        return NODE_COLOR['metro']
    return NODE_COLOR['eqp']

def _node_size(G: nx.DiGraph, name: str, scale: float = 800) -> float:
    cov = G.nodes[name].get('coverage', 0.5)
    return max(200, cov * scale)


# ---------------------------------------------------------------------------
# 시각화 1: 전체 그래프
# ---------------------------------------------------------------------------

def plot_full_graph(
    G: nx.DiGraph,
    edge_df: pd.DataFrame,
    ax: plt.Axes,
    title: str = 'Knowledge Graph (Correlation + Granger)',
) -> None:
    """전체 노드/엣지 시각화."""
    pos = nx.spring_layout(G, seed=42, k=2.5)

    node_colors = [_node_color(n) for n in G.nodes()]
    node_sizes = [_node_size(G, n) for n in G.nodes()]

    # 엣지 분리
    corr_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'corr']
    granger_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'granger']

    granger_weights = [
        G[u][v]['weight'] for u, v in granger_edges
    ]
    granger_widths = [w * 4 for w in granger_weights]

    # 1. 상관관계 엣지 (회색, 얇음)
    nx.draw_networkx_edges(
        G, pos, edgelist=corr_edges,
        edge_color='#CCCCCC', width=0.8,
        arrows=False, ax=ax, alpha=0.6,
    )

    # 2. Granger 엣지 (빨강, 두꺼움, 화살표)
    nx.draw_networkx_edges(
        G, pos, edgelist=granger_edges,
        edge_color='#E84C4C', width=granger_widths,
        arrows=True, arrowsize=20,
        connectionstyle='arc3,rad=0.15',
        ax=ax, alpha=0.85,
        min_source_margin=18, min_target_margin=18,
    )

    # 3. 노드
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        ax=ax, alpha=0.92,
    )

    # 4. 레이블
    nx.draw_networkx_labels(
        G, pos, font_size=8, font_weight='bold', ax=ax,
    )

    ax.set_title(title, fontsize=13, pad=12)
    ax.axis('off')

    # 범례
    legend_elements = [
        mpatches.Patch(color=NODE_COLOR['vm'], label='VM feature'),
        mpatches.Patch(color=NODE_COLOR['metro'], label='Metro feature'),
        plt.Line2D([0], [0], color='#CCCCCC', lw=1.5, label='Correlation'),
        plt.Line2D([0], [0], color='#E84C4C', lw=3,
                   label='Granger causality', marker='>', markersize=8),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
              framealpha=0.8)


# ---------------------------------------------------------------------------
# 시각화 2: Granger-only 인과 그래프
# ---------------------------------------------------------------------------

def plot_granger_graph(
    G: nx.DiGraph,
    ax: plt.Axes,
    expected: list[tuple[str, str]] | None = None,
) -> None:
    """Granger 인과 엣지만 표시. expected 인과 관계를 별도 표시."""
    granger_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                     if d.get('type') == 'granger']
    if not granger_edges:
        ax.text(0.5, 0.5, 'No Granger edges', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Granger Causal Graph', fontsize=13)
        ax.axis('off')
        return

    Gg = nx.DiGraph()
    for u, v, d in granger_edges:
        Gg.add_edge(u, v, **d)
    # 고립 노드 제거 안 함 — 모든 노드 표시
    for n in G.nodes():
        if n not in Gg.nodes:
            Gg.add_node(n, **G.nodes[n])

    pos = nx.spring_layout(Gg, seed=7, k=3.0)

    node_colors = [_node_color(n) for n in Gg.nodes()]
    node_sizes = [_node_size(G, n, scale=700) for n in Gg.nodes()]

    granger_list = [(u, v) for u, v in Gg.edges()]
    weights = [Gg[u][v]['weight'] for u, v in granger_list]

    # expected vs detected 색상 분리
    if expected:
        expected_set = set(expected)
        edge_colors = [
            '#E84C4C' if (u, v) in expected_set else '#9B59B6'
            for u, v in granger_list
        ]
    else:
        edge_colors = ['#E84C4C'] * len(granger_list)

    nx.draw_networkx_edges(
        Gg, pos, edgelist=granger_list,
        edge_color=edge_colors,
        width=[w * 5 + 1 for w in weights],
        arrows=True, arrowsize=22,
        connectionstyle='arc3,rad=0.12',
        ax=ax, alpha=0.85,
        min_source_margin=20, min_target_margin=20,
    )
    nx.draw_networkx_nodes(
        Gg, pos, node_color=node_colors, node_size=node_sizes,
        ax=ax, alpha=0.92,
    )
    nx.draw_networkx_labels(Gg, pos, font_size=9, font_weight='bold', ax=ax)

    # p-value 레이블
    edge_labels = {
        (u, v): f"p={Gg[u][v]['p_value']:.3f}\nlag={int(Gg[u][v]['best_lag'])}"
        for u, v in Gg.edges()
    }
    nx.draw_networkx_edge_labels(
        Gg, pos, edge_labels=edge_labels,
        font_size=6.5, label_pos=0.35,
        ax=ax, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
    )

    ax.set_title('Granger Causal Graph\n'
                 '(red=expected, purple=additional)', fontsize=12)
    ax.axis('off')

    if expected:
        legend_elements = [
            plt.Line2D([0], [0], color='#E84C4C', lw=3,
                       label='Expected causal edge'),
            plt.Line2D([0], [0], color='#9B59B6', lw=2,
                       label='Additional detected edge'),
            mpatches.Patch(color=NODE_COLOR['vm'], label='VM feature'),
            mpatches.Patch(color=NODE_COLOR['metro'], label='Metro feature'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=7.5)


# ---------------------------------------------------------------------------
# 시각화 3: Correlation 히트맵
# ---------------------------------------------------------------------------

def plot_corr_heatmap(
    flat_df: pd.DataFrame,
    feature_cols: list[str],
    ax: plt.Axes,
) -> None:
    """Feature 간 Pearson correlation 히트맵."""
    corr_matrix = flat_df[feature_cols].corr(method='pearson')
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')

    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(feature_cols, fontsize=8)

    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6.5,
                        color='white' if abs(val) > 0.6 else 'black')

    ax.set_title('Pearson Correlation Heatmap', fontsize=12)


# ---------------------------------------------------------------------------
# 시각화 4: 노드 통계 바 차트
# ---------------------------------------------------------------------------

def plot_node_stats(node_df: pd.DataFrame, ax: plt.Axes) -> None:
    """노드별 coverage와 degree 시각화."""
    n = len(node_df)
    x = np.arange(n)
    width = 0.35

    colors_cov = [_node_color(r['node']) for _, r in node_df.iterrows()]

    bars1 = ax.bar(x - width / 2, node_df['total_degree'], width,
                   color=colors_cov, alpha=0.75, label='Total degree')
    bars2 = ax.bar(x + width / 2, node_df['granger_out'], width,
                   color='#E84C4C', alpha=0.75, label='Granger out-degree')

    ax2 = ax.twinx()
    ax2.plot(x, node_df['coverage'], 'D--', color='#2C3E50', markersize=7,
             linewidth=1.5, label='Coverage')
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('Coverage', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(node_df['node'], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Degree', fontsize=9)
    ax.set_title('Node Statistics', fontsize=12)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, str(int(h)),
                ha='center', va='bottom', fontsize=6.5)


# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path('output')
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("1. 인과 신호가 심어진 예제 데이터 생성 (150 wafers)")
    print("   심어진 관계:")
    print("     vm_A_x  → metro_P_1  (lag 1, 강한)")
    print("     vm_B_y  → metro_Q_2  (lag 2, 중간)")
    print("     metro_P_1 → vm_A_y   (lag 1, 약한)")
    print("=" * 60)

    df = make_causal_demo(n_wafers=150, seed=0)
    print(f"   데이터 shape: {df.shape}")
    print(f"   컬럼 수: {len(df.columns)}  (wafer_id + 5 steps × 10 features)")

    config = KGConfig(
        wafer_id_col='wafer_id',
        lags=[1, 2],
        corr_threshold=0.25,
        corr_min_samples=30,
        granger_maxlag=3,
        granger_min_samples=40,
        granger_p_threshold=0.05,
        output_dir=str(out_dir),
    )

    print("\n2. 파이프라인 실행...")
    G, edge_df, node_df = run_pipeline(df, config)

    # ── 결과 검증 ───────────────────────────────────────────────────────────
    print("\n3. 결과 검증")
    print("-" * 50)
    print(f"   노드 수: {G.number_of_nodes()}")
    print(f"   전체 엣지: {G.number_of_edges()}")
    granger_df = edge_df[edge_df['type'] == 'granger'][['src', 'dst', 'weight', 'p_value', 'best_lag']]
    corr_df = edge_df[edge_df['type'] == 'corr'][['src', 'dst', 'weight']]
    print(f"   Granger 엣지: {len(granger_df)}")
    print(f"   Correlation 엣지: {len(corr_df)} (양방향 포함)")
    print()
    print("   [Granger 엣지 전체]")
    print(granger_df.to_string(index=False))

    EXPECTED = [
        ('vm_A_x', 'metro_P_1'),
        ('vm_B_y', 'metro_Q_2'),
        ('metro_P_1', 'vm_A_y'),
    ]

    print("\n   [심어진 인과 관계 발견 여부]")
    detected = set(zip(granger_df['src'], granger_df['dst']))
    for src, dst in EXPECTED:
        found = (src, dst) in detected
        row = granger_df[(granger_df['src'] == src) & (granger_df['dst'] == dst)]
        if found:
            p = row['p_value'].values[0]
            lag = row['best_lag'].values[0]
            print(f"   ✓  {src:12s} → {dst:12s}  (p={p:.4f}, lag={int(lag)})")
        else:
            print(f"   ✗  {src:12s} → {dst:12s}  (미검출)")

    # ── flat_df for heatmap ──────────────────────────────────────────────────
    from alignment import (create_lag_features, create_step_matrix,
                           create_time_features, forward_fill_by_wafer,
                           sort_by_process_order)
    from preprocessing import (create_missing_masks, interpolate_metro_with_vm,
                                parse_all_columns, to_long_format)

    col_infos, _, _ = parse_all_columns(df)
    long_df = to_long_format(df, col_infos)
    matrix = create_step_matrix(long_df)
    matrix = sort_by_process_order(matrix)
    numeric_cols = matrix.select_dtypes(include=[np.number]).columns.tolist()
    matrix = forward_fill_by_wafer(matrix, numeric_cols)
    matrix = interpolate_metro_with_vm(matrix, col_infos)
    flat_df = matrix.reset_index()
    analysis_cols = [c for c in numeric_cols if c in flat_df.columns]

    # ── 시각화 ───────────────────────────────────────────────────────────────
    print("\n4. 시각화 생성...")

    # (A) 전체 그래프 + Granger 인과 그래프 (나란히)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle('Semiconductor Process Knowledge Graph\n'
                 f'({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)',
                 fontsize=14, y=1.01)

    plot_full_graph(G, edge_df, axes[0], title='Full Graph (Correlation + Granger)')
    plot_granger_graph(G, axes[1], expected=EXPECTED)

    plt.tight_layout()
    fig.savefig(out_dir / 'kg_full.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   저장: {out_dir / 'kg_full.png'}")

    # (B) Correlation 히트맵 + 노드 통계
    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 7))
    fig2.suptitle('Correlation Analysis & Node Statistics', fontsize=13)

    plot_corr_heatmap(flat_df, analysis_cols, axes2[0])
    plot_node_stats(node_df, axes2[1])

    plt.tight_layout()
    fig2.savefig(out_dir / 'kg_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"   저장: {out_dir / 'kg_analysis.png'}")

    print("\n완료. output/ 폴더를 확인하세요.")


if __name__ == '__main__':
    main()
