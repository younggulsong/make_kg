# 반도체 공정 Knowledge Graph 구축 - 구현 명세서 (v2)

## 1. 목적

반도체 공정 데이터를 기반으로 다음을 반영한 **시간 인지형 Knowledge Graph**를 구축한다:

* 통계적 관계 (Correlation: Pearson, Spearman)
* 시간적 인과관계 (Granger Causality — panel-aware F-test)

본 문제는 **공정 순서(seq_num)를 기준으로 정렬된 인과 그래프**를 구축하는 것이 목표이다.

---

## 2. 입력 데이터 구조

### (1) 설비 정보
* `eqp_UN{seq_num}_eqp_id`
* `eqp_UN{seq_num}_tkout_time`

### (2) Virtual Metrology (VM)
* `vm_UN{seq_num}_{item_id}_{subitem_id}` (구분자: `_` 또는 `*`)

### (3) Physical Metrology (METRO)
* `metro_UN{seq_num}_{item_id}_{subitem_id}`

> **제약**: item_id 자체에 구분자 문자(`_` 또는 `*`)가 포함되면 파싱 불명확.

---

## 3. 데이터 특성 및 제약

* seq_num: 000000 ~ 100000 (공정 순서)
* 공정은 seq_num 순서대로 진행됨
* metro: 결측률 1% ~ 99% (희소 데이터)
* vm: 대부분 존재 (coverage > 90%)

---

## 4. 핵심 설계 원칙

### MUST
* 공정 순서(seq_num) 기반 정렬 필수
* lag feature 기반 인과 분석 필수
* wafer 단위 그룹 연산 필수 (cross-wafer lag 오염 방지)
* 결측 데이터 명시적 마스크 처리 필수 (fill 전)

### MUST NOT
* raw timestamp 기반 직접 정렬 금지
* 결측 데이터 자동 drop 금지
* lag 없이 Granger 수행 금지
* statsmodels `grangercausalitytests`에 naive concat 데이터 전달 금지
  → wafer 경계에서 lag 오염 발생 (대신 manual F-test 사용)

---

## 5. 전체 파이프라인

```
Step 1  Column parsing         (preprocessing.parse_all_columns)
Step 2  Missing mask 생성       (preprocessing.create_missing_masks)  ← fill 전에 먼저
Step 3  Wide → Long format     (preprocessing.to_long_format)
Step 4  Step matrix pivot       (alignment.create_step_matrix)
Step 5  Process order sort      (alignment.sort_by_process_order)
Step 6a Forward fill by wafer  (alignment.forward_fill_by_wafer)
Step 6b VM interpolation        (preprocessing.interpolate_metro_with_vm)
Step 7a Lag features by wafer  (alignment.create_lag_features)
Step 7b Time features          (alignment.create_time_features)
Step 8  Correlation edges       (causality.compute_correlation_edges)
Step 9  Granger edges           (causality.compute_granger_edges)
Step 10 Graph 구성              (graph_builder.build_knowledge_graph)
```

---

## 6. 주요 설계 결정 (원본 명세 대비 변경/추가)

### 6.1 Granger Causality — Panel F-test (핵심 변경)

**원본 명세**: `maxlag=3, 샘플≥50, p<0.05`으로만 기술.

**실제 구현**:
- `statsmodels.grangercausalitytests` 미사용.
  이유: 내부에서 lag를 직접 생성하므로, wafer를 concat한 데이터에 적용하면
  **wafer 경계에서 lag 오염** 발생 (W1의 마지막 step ↔ W2의 첫 step).
- 대신: `alignment.create_lag_features`에서 `groupby(wafer_id).shift(lag)`로
  **wafer-aware lag** 미리 생성 → Granger F-test에 pre-computed lag 사용.
- F-test 구현: Restricted model `Y(t) ~ Y(t-1..k)` vs
  Unrestricted model `Y(t) ~ Y(t-1..k) + X(t-1..k)`, F-통계량 직접 계산.
- `dropna()`가 자동으로 wafer 경계 행(lag=NaN) 제거.

### 6.2 Step matrix dtype 처리

`to_long_format`은 float/str/datetime을 하나의 object-dtype 'value' 컬럼으로 concat.
→ pivot 후 vm/metro 컬럼도 object dtype 유지.
→ `create_step_matrix` 내에서 `pd.to_numeric` 적용으로 float64로 변환.

### 6.3 Correlation edges — 양방향 처리

DiGraph에서 correlation(대칭 관계)은 기본적으로 **양방향 edge**로 추가
(`add_reverse_corr=True`).  Granger edge만 단방향.

### 6.4 Feature naming

| 원본 컬럼 | feature_name (graph node) |
|---|---|
| `vm_UN000010_A_x` | `vm_A_x` |
| `metro_UN000020_P_1` | `metro_P_1` |
| `eqp_UN000010_tkout_time` | `eqp_UN000010_tkout_time` |

같은 feature가 다른 step에서 측정되면 동일한 feature_name을 공유.
step matrix의 (wafer_id, seq_num) 축으로 구분됨.

---

## 7. Edge 생성 규칙

### Correlation
* Pearson + Spearman 모두 계산
* `weight = max(|pearson|, |spearman|)`
* 조건: 유효 샘플 ≥ 30, |weight| ≥ threshold (기본 0.3)
* Edge type: `"corr"`, 양방향

### Granger Causality
* X(t-1..k) → Y(t) (pre-computed lag 기반 manual F-test)
* 조건: 유효 샘플 ≥ 50, p-value < 0.05
* `weight = 1 - p_value`
* Edge type: `"granger"`, 단방향

---

## 8. 출력

| 파일 | 내용 |
|---|---|
| `edges.csv` | src, dst, type, weight, pearson, spearman, p_value, best_lag, n |
| `nodes.csv` | node, in_degree, out_degree, granger_in, granger_out, coverage, missing_rate |
| `knowledge_graph.graphml` | NetworkX DiGraph (GraphML 포맷) |

---

## 9. 코드 구조

```
make_knowledge_graph/
├── preprocessing.py   # 컬럼 파싱, wide→long, mask, VM 보간
├── alignment.py       # step matrix, lag (wafer-aware), time feature, ffill
├── causality.py       # 상관관계 + Granger F-test
├── graph_builder.py   # NetworkX DiGraph 구성, export
└── main.py            # 파이프라인, KGConfig, CLI, demo data 생성
```

각 모듈은 독립적으로 임포트 및 테스트 가능.

---

## 10. 성능

* 벡터화 연산 (numpy/pandas groupby)
* pairwise 계산 batch 처리 (batch_size 설정 가능)
* `n_jobs > 1` 시 ProcessPoolExecutor 사용 (Granger)

---

## 11. 확장성

다음 확장을 위한 인터페이스 준비:
* **PCMCI / NOTEARS**: `export_edge_list()` → adjacency matrix
  (`nx.to_numpy_array(G, weight='weight')`)
* **Temporal GNN 입력**: step matrix (alignment.py 결과)를 직접 node feature tensor로 사용

---

## 12. CLI 사용법

```bash
# 합성 데이터로 데모
python main.py --demo --demo-wafers 100 --output ./output

# 실제 데이터
python main.py --input data.csv --wafer-id wafer_id --output ./output

# 파라미터 조정
python main.py --input data.csv --corr-threshold 0.4 --granger-p 0.01 --n-jobs 4
```
