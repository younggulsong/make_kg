# make_knowledge_graph

반도체 공정 데이터를 입력받아 `NetworkX DiGraph` 형태의 Knowledge Graph를 만드는 파이프라인입니다.  
입력 데이터의 컬럼명을 파싱해 공정 step 기준으로 정렬하고, 상관관계와 Granger causality를 계산해 엣지를 생성합니다.

핵심 파일은 [main.py](/Users/ygsong/MYPROJECTS/ml_practice/make_knowledge_graph/main.py), [preprocessing.py](/Users/ygsong/MYPROJECTS/ml_practice/make_knowledge_graph/preprocessing.py), [alignment.py](/Users/ygsong/MYPROJECTS/ml_practice/make_knowledge_graph/alignment.py), [causality.py](/Users/ygsong/MYPROJECTS/ml_practice/make_knowledge_graph/causality.py), [graph_builder.py](/Users/ygsong/MYPROJECTS/ml_practice/make_knowledge_graph/graph_builder.py), [visualize.py](/Users/ygsong/MYPROJECTS/ml_practice/make_knowledge_graph/visualize.py) 입니다.

## 개요

이 코드는 다음 순서로 동작합니다.

1. 컬럼명을 파싱해 `eqp`, `vm`, `metro` 피처를 식별합니다.
2. wide format 데이터를 `(wafer_id, seq_num, feature_name, value)` long format으로 바꿉니다.
3. `(wafer_id, seq_num)` 인덱스의 step matrix로 pivot 합니다.
4. wafer 내부에서만 forward fill과 lag 생성을 수행합니다.
5. feature pair 간 correlation, ordered pair 간 Granger causality를 계산합니다.
6. 결과를 directed graph, edge list, node statistics로 저장합니다.

설계상 가장 중요한 점은 공정 정렬 기준이 timestamp가 아니라 `seq_num` 이라는 점입니다.  
또한 lag 계산은 반드시 wafer 단위로 분리해서 수행하므로, wafer 경계가 섞인 잘못된 Granger 분석을 피합니다.

## 디렉터리 구조

```text
make_knowledge_graph/
├── README.md
├── main.py
├── preprocessing.py
├── alignment.py
├── causality.py
├── graph_builder.py
├── visualize.py
├── claude.md
└── output/
```

모듈 역할은 다음과 같습니다.

- `main.py`: 전체 파이프라인, CLI, demo 데이터 생성
- `preprocessing.py`: 컬럼 파싱, missing mask 생성, wide→long 변환, metro 결측치의 VM 기반 보간
- `alignment.py`: step matrix 생성, 공정 순서 정렬, wafer별 forward fill, lag/time feature 생성
- `causality.py`: correlation edge, Granger edge 계산
- `graph_builder.py`: `NetworkX DiGraph` 생성, 노드 통계 계산, edge export
- `visualize.py`: 강화된 demo 데이터를 만들고 그래프/히트맵 이미지를 저장
- `claude.md`: 구현 명세 메모

## 입력 데이터 형식

한 행이 wafer 하나를 나타내는 wide-format `DataFrame` 또는 CSV를 기대합니다.

지원하는 컬럼명 패턴:

```text
eqp_UN{seq_num}_eqp_id
eqp_UN{seq_num}_tkout_time
vm_UN{seq_num}_{item_id}_{subitem_id}
metro_UN{seq_num}_{item_id}_{subitem_id}
```

`vm`/`metro` 는 `_` 대신 `*` 구분자도 허용합니다.

예시:

```text
wafer_id
eqp_UN000010_eqp_id
eqp_UN000010_tkout_time
vm_UN000010_A_x
vm_UN000010_B_y
metro_UN000010_P_1
metro_UN000010_Q_2
```

파싱 후 내부 feature 이름은 다음처럼 정규화됩니다.

- `vm_UN000010_A_x` → `vm_A_x`
- `metro_UN000020_P_1` → `metro_P_1`
- `eqp_UN000010_tkout_time` → `eqp_UN000010_tkout_time`

주의할 점:

- `item_id` 자체에 `_` 또는 `*` 가 포함되면 파싱이 모호해질 수 있습니다.
- `wafer_id` 컬럼이 없으면 row index 또는 단순 row 번호를 wafer 식별자로 사용합니다.

## 파이프라인 상세

`run_pipeline()` 의 실제 처리 순서는 아래와 같습니다.

1. `parse_all_columns(df)`
2. `create_missing_masks(df, original_cols)`
3. `to_long_format(df, col_infos, wafer_id_col=...)`
4. `create_step_matrix(long_df)`
5. `sort_by_process_order(matrix)`
6. `forward_fill_by_wafer(...)`
7. `interpolate_metro_with_vm(matrix, col_infos)`
8. `create_lag_features(matrix, base_feature_cols, lags=[...])`
9. `create_time_features(matrix)`
10. `compute_correlation_edges(flat_df, analysis_cols, ...)`
11. `compute_granger_edges(flat_df, analysis_cols, ...)`
12. `build_knowledge_graph(...)`

현재 구현 기준으로 엣지 계산에 사용되는 `analysis_cols` 는 step matrix에서 추린 기본 numeric feature 입니다.  
즉, `create_time_features()` 로 생성한 시간 파생 피처는 만들어지지만 현재 graph edge 계산에는 포함되지 않습니다.

또한 missing mask는 생성되지만 현재는 후속 단계나 출력 파일에 연결되지는 않습니다.

## 그래프 구성 방식

노드:

- 기본 numeric feature 컬럼들
- 보통 `vm_*`, `metro_*` 가 대상

엣지:

- `corr`: Pearson/Spearman 기반 상관관계
- `granger`: 방향성이 있는 Granger causality

세부 규칙:

- correlation weight = `max(|pearson|, |spearman|)`
- correlation threshold 기본값 = `0.3`
- Granger edge는 `p_value < 0.05` 일 때만 유지
- Granger weight = `1 - p_value`
- correlation edge는 기본적으로 양방향으로 추가됩니다

노드 속성:

- `mean`
- `std`
- `missing_rate`
- `coverage`
- `n_valid`

## 설치 의존성

코드에서 직접 사용하는 주요 라이브러리는 다음과 같습니다.

```bash
pip install pandas numpy scipy networkx matplotlib
```

Python 3.10+ 정도를 기준으로 보는 것이 안전합니다.

## 실행 방법

프로젝트 루트(`/Users/ygsong/MYPROJECTS/ml_practice`) 기준 예시입니다.

데모 데이터로 실행:

```bash
python make_knowledge_graph/main.py --demo --output make_knowledge_graph/output
```

데모 wafer 수를 줄여 빠르게 테스트:

```bash
python make_knowledge_graph/main.py --demo --demo-wafers 20 --output make_knowledge_graph/output_quick
```

실제 CSV 입력:

```bash
python make_knowledge_graph/main.py \
  --input path/to/data.csv \
  --wafer-id wafer_id \
  --output make_knowledge_graph/output_real
```

파라미터 조정 예시:

```bash
python make_knowledge_graph/main.py \
  --input path/to/data.csv \
  --corr-threshold 0.4 \
  --corr-min-samples 50 \
  --granger-maxlag 3 \
  --granger-min-samples 80 \
  --granger-p 0.01 \
  --n-jobs 4
```

forward fill 또는 VM interpolation 비활성화:

```bash
python make_knowledge_graph/main.py --input path/to/data.csv --no-ffill --no-vm-interp
```

## Python API 사용 예시

```python
import pandas as pd

from make_knowledge_graph.main import KGConfig, run_pipeline

df = pd.read_csv("data.csv")

config = KGConfig(
    wafer_id_col="wafer_id",
    lags=[1, 2],
    corr_threshold=0.3,
    granger_p_threshold=0.05,
    output_dir="make_knowledge_graph/output_api",
)

G, edge_df, node_df = run_pipeline(df, config)
```

반환값:

- `G`: `networkx.DiGraph`
- `edge_df`: 전체 edge list `DataFrame`
- `node_df`: 노드 통계 `DataFrame`

## 출력 파일

`--output` 디렉터리 아래에 다음 파일이 저장됩니다.

- `edges.csv`: 엣지 목록
- `nodes.csv`: 노드별 degree/coverage 통계
- `knowledge_graph.graphml`: GraphML 포맷 그래프

`edges.csv` 주요 컬럼:

- 공통: `src`, `dst`, `type`, `weight`
- correlation: `pearson`, `spearman`, `pearson_p`, `spearman_p`, `n`
- granger: `p_value`, `best_lag`, `n`

`nodes.csv` 주요 컬럼:

- `node`
- `in_degree`
- `out_degree`
- `total_degree`
- `granger_in`
- `granger_out`
- `corr_degree`
- `coverage`
- `missing_rate`

## 시각화 스크립트

`visualize.py` 는 인과 신호가 일부러 심어진 demo 데이터를 만든 뒤, 실행 결과를 이미지로 저장합니다.

실행:

```bash
cd make_knowledge_graph
python visualize.py
```

생성 파일:

- `output/kg_full.png`
- `output/kg_analysis.png`
- `output/edges.csv`
- `output/nodes.csv`
- `output/knowledge_graph.graphml`

## 주요 구현 포인트

### 1. Panel-aware Granger

이 구현은 `statsmodels.grangercausalitytests()` 를 직접 쓰지 않고, 미리 만들어 둔 lag column으로 수동 F-test를 수행합니다.  
이 방식은 여러 wafer를 단순 concat 했을 때 생기는 wafer 경계 오염을 막기 위한 것입니다.

### 2. Correlation은 사실상 무방향 관계

그래프는 `DiGraph` 이지만 correlation edge는 양방향으로 넣습니다.  
따라서 탐색은 directed graph로 하되, correlation 자체는 대칭 관계로 취급합니다.

### 3. Edge overwrite 가능성

동일한 `(src, dst)` 쌍에 correlation과 Granger가 모두 생기면 `DiGraph` 특성상 나중에 추가된 edge attribute가 덮어써집니다.  
즉, 이 경우 `G.number_of_edges()` 와 `2 * corr_edges + granger_edges` 가 정확히 일치하지 않을 수 있습니다.

## 확인한 실행 예시

다음 명령으로 짧은 데모 실행을 확인했습니다.

```bash
python make_knowledge_graph/main.py --demo --demo-wafers 20 --output make_knowledge_graph/output_readme_check
```

이 실행에서는 정상적으로 graph가 생성됐고, `edges.csv`, `nodes.csv`, `knowledge_graph.graphml` 이 저장되었습니다.

## 개선 후보

- 생성한 missing mask를 graph feature 또는 output으로 실제 활용
- 시간 파생 피처(`delta_time_*`, `queue_time_*`)를 analysis 대상에 포함
- `DiGraph` 대신 멀티엣지 구조를 사용해 corr/granger 동시 보존
- CLI용 `requirements.txt` 또는 `pyproject.toml` 추가
