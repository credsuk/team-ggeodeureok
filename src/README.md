# 의정이슈레이더 (Council Issue Radar)

> 팀 **꺼드럭's**가 만든 광역 지방의회 회의록 분석 시스템입니다. 2024~2025년 회의록에서 정책 이슈의 근거 발언을 검색하고, 최초 관측 시점과 다른 지방의회의 후속 논의를 시간순으로 보여 주며, 다음 달 논의 지속 여부와 언급량을 예측합니다.

이 문서는 단순 실행 안내가 아니라 **발표자가 프로젝트 전체를 설명할 수 있도록 만든 기술 해설서**입니다. 1~3장은 전체 개요, 4장부터는 각 단계의 입력·처리·출력·선택 이유와 평가 방법을 설명합니다.

---

## 1. 프로젝트 한눈에 보기

### 1.1 해결하려는 문제

광역 지방의회 회의록은 양이 많고 한 이슈가 여러 회의·위원회·지역에 흩어져 있습니다. 검색어가 정확히 일치하지 않아도 같은 정책을 말할 수 있어 단순 키워드 검색만으로는 다음 질문에 답하기 어렵습니다.

- 어떤 정책 이슈가 실제로 논의되었는가?
- 제공 데이터 범위에서 어느 의회가 가장 먼저 논의했는가?
- 이후 다른 지방의회에서는 언제 유사 논의를 시작했는가?
- 어떤 회의록과 원문 구간이 근거인가?
- 다음 달에도 논의가 지속될 가능성이 높은가?
- 예상 언급량은 어느 정도인가?

의정이슈레이더는 이를 `검색 → 재순위 → 문맥 복원 → 이슈 군집화 → 최초 관측·후속 논의 흐름 → 시계열 예측 → 근거 기반 설명`으로 해결합니다.

### 1.2 확정 구성

| 항목 | 최종 구성 |
|---|---|
| 분석 기간 | 제공 데이터의 2024~2025년 범위 |
| canonical 발화 | 2,057,214행, 34개 지역×연도 파일 |
| 검색용 자식 청크 | 901,358개 |
| 임베딩 입력 | `raw` 표현 |
| 임베딩 모델 | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` |
| 모델 revision | `55ec6e9358a56d56af759bc8372e970caf8c305f` |
| 벡터 | 1,024차원, L2 정규화, float16 저장 |
| 의미 검색 DB | PostgreSQL + pgvector HNSW |
| 키워드 검색 | SQLite FTS5 BM25 |
| 순위 결합 | RRF(Reciprocal Rank Fusion) |
| 재순위 | `BAAI/bge-reranker-v2-m3` |
| 이슈 군집화 | 코사인 유사도 그래프 + Union-Find |
| 예측 | 활동률 기준선 + Poisson Regression |
| 설명 Agent | 로컬 Ollama `exaone3.5:7.8b` |
| 외부 참고 보강 | 법제처 자치법규·열린국회정보·KOSIS (선택, 예측 입력 제외) |
| 화면 | Streamlit + PyDeck |

### 1.3 전체 구조

```mermaid
flowchart TD
    A["회의록 TSV + LLM metadata"] --> B["Canonical 발화 2,057,214행"]
    B --> C["부모·자식 청킹"]
    C --> D["Arctic 임베딩 901,358개"]
    D --> E["pgvector Dense 검색"]
    C --> F["SQLite BM25 검색"]
    E --> G["RRF 결합 + BGE 재순위"]
    F --> G
    G --> H["부모 문맥 복원·이슈 군집화"]
    H --> I["최초 관측·지역별 타임라인"]
    I --> J["월별 특징·Rolling 예측"]
    J --> K["지도·근거·로컬 챗봇"]
```

핵심 원칙은 **원문 근거와 계산 모델을 분리하는 것**입니다. LLM이 날짜·전파·예측 수치를 마음대로 만들지 않습니다. 날짜와 순서는 데이터로 계산하고, 예측은 통계 모델이 계산하며, 로컬 LLM은 검색된 회의록과 계산 결과를 읽기 쉬운 문장으로 설명합니다.

---

## 2. 보안·재현성 원칙

대회 제공 회의록을 OpenAI·Anthropic 같은 외부 LLM API로 전송하지 않습니다.

- Hugging Face 모델은 **모델 파일만 다운로드**한 뒤 로컬에서 실행합니다.
- Ollama의 `exaone3.5:7.8b`도 로컬 PC에서 실행합니다.
- 회의록 원문·벡터·DB는 Git에 올리지 않습니다.
- 공식 이슈 탐지·군집화·예측은 제공 데이터 기간을 넘는 외부 정보를 사용하지 않습니다.
- 선택 기능인 법제처·열린국회정보·KOSIS 조회에는 검증된 짧은 정책어 또는 통계표 ID만 전송하며, 질문 원문·회의록 원문·발화자 정보는 보내지 않습니다.
- 외부 공공데이터는 `reference_only`로 표시하고 공식 예측 모델 입력과 분리합니다. 대회 기본 모드는 `backtest`이며 기준일 이후 정보는 사후 점검에서만 별도로 표시합니다.
- 모델 파일, PostgreSQL 이미지, Python 패키지가 이미 설치돼 있으면 검색·예측·화면·챗봇을 인터넷 없이 로컬에서 실행할 수 있습니다.

첫 설치, Hugging Face 모델 다운로드, Ollama 모델 pull, Docker 이미지 pull과 외부 공공데이터의 실시간 갱신에는 인터넷이 필요합니다. 공공데이터 기능을 끄거나 캐시된 결과를 사용하면 핵심 검색·예측·Agent는 오프라인으로 동작합니다.

---

## 3. 핵심 용어

| 용어 | 쉬운 뜻 | 프로젝트에서의 역할 |
|---|---|---|
| canonical | 여러 입력을 한 스키마로 통일한 기준 데이터 | 이후 모든 코드가 같은 열과 ID 사용 |
| metadata | 본문을 설명하는 부가 정보 | 날짜·의회·회의록명·안건·발화자·정책분야 등 |
| 청크 | 검색을 위해 나눈 텍스트 조각 | 긴 회의록을 모델이 처리할 크기로 변환 |
| 부모 청크 | 넓은 문맥을 가진 큰 조각 | 검색 결과의 앞뒤 문맥 복원 |
| 자식 청크 | 임베딩·검색에 쓰는 작은 조각 | 한 벡터에 여러 주제가 섞이는 것을 방지 |
| 임베딩 | 문장을 의미를 나타내는 숫자 벡터로 변환 | 표현이 달라도 의미가 비슷한 발언 검색 |
| Dense 검색 | 임베딩 벡터 사이 의미 유사도 검색 | 저출산↔출생률·인구감소처럼 다른 표현 탐색 |
| BM25 | 단어 출현 기반 키워드 검색 | 조례명·기관명·고유명사·정확한 정책명 탐색 |
| RRF | 여러 검색 순위를 순위 기반으로 결합 | Dense와 BM25의 다른 점수 척도를 안정적으로 결합 |
| reranker | 질의와 후보 본문을 함께 읽어 재정렬 | 빠른 1차 검색의 오탐 감소 |
| 군집화 | 유사 발언을 같은 이슈 묶음으로 구성 | 하나의 정책 이슈 구성원 확정 |
| HNSW | 가까운 벡터를 빠르게 찾는 그래프형 인덱스 | 90만 벡터의 빠른 pgvector 검색 |
| Rolling Backtest | 과거로 학습해 바로 다음 시점을 반복 예측 | 미래 정보 누출 없이 시계열 성능 측정 |

---

## 4. 입력 데이터와 canonical 생성

### 4.1 입력 데이터

정제 회의록의 핵심 열은 다음과 같습니다.

```text
MTG_DE, RASMBLY_NM, MTGNM, MINTS_NM,
agenda_title, speaker_role, speaker_name, speaker_type, utterance
```

별도 LLM metadata에는 `policy_category`, `core_issue`, `keywords`, `summary`, `is_substantive` 등이 있습니다. 최신 metadata는 최신 회의록과 동일한 `row_id` 기준으로 다시 생성해 과거의 행 불일치 문제를 해결했습니다.

### 4.2 왜 HTML을 바로 임베딩하지 않는가

HTML을 그대로 임베딩하면 태그·메뉴·표·공백·회의 진행 문구가 의미 벡터에 섞입니다. 한 회의록 전체는 모델 입력 한도를 넘고 여러 주제가 한 벡터에 섞입니다. 따라서 다음 순서로 처리합니다.

1. 인코딩과 깨진 문자 확인
2. HTML 태그·불필요한 공백·UI 잔재 제거
3. 날짜와 의회명 표준화
4. 회의록명·위원회·안건·발화자 결합
5. 최신 metadata를 `row_id`로 결합
6. 고정 ID와 해시 생성
7. Parquet으로 저장

Parquet은 원문을 버린 요약 파일이 아닙니다. 정제된 본문과 metadata를 열 단위로 저장한 분석용 파일입니다. JSON/HTML 원본은 별도 보존하고, Parquet은 빠른 조회·필터·청킹을 위해 사용합니다.

### 4.3 canonical 주요 열

| 열 | 의미 |
|---|---|
| `chunk_id` | 발화 또는 청크의 고정 식별자 |
| `document_id` | 원본 회의록 식별자 |
| `parent_chunk_id` | 자식이 속한 부모 청크 식별자 |
| `council` | 광역 지방의회명 |
| `meeting_date` | 회의일자 |
| `committee` | 위원회명 |
| `meeting_name` | 근거로 표시할 회의록명 |
| `agenda_title` | 안건명 |
| `speaker`, `speaker_role` | 발화자와 역할 |
| `text` | 정제 원문 |
| `embedding_text` | 임베딩 모델 입력 문자열 |
| `text_sha256` | 본문 변경·중복 확인용 해시 |
| `metadata_source/version` | metadata 출처와 버전 |

행 번호는 파일을 다시 정렬하면 바뀝니다. 반면 고정 `chunk_id`를 사용하면 임베딩·DB·BM25·검수·웹 결과가 같은 발언을 계속 가리킬 수 있습니다.

---

## 5. 부모·자식 청킹

### 5.1 왜 두 크기로 나누는가

- 너무 긴 청크: 여러 정책이 섞여 벡터 의미가 흐려집니다.
- 너무 짧은 청크: 관련 문장은 찾지만 누가 왜 말했는지 문맥이 사라집니다.

그래서 **작은 자식으로 검색하고 큰 부모로 설명**합니다.

| 구분 | 설정 | 목적 |
|---|---:|---|
| 부모 목표 길이 | 약 2,400자 | 넓은 논의 문맥 보존 |
| 부모 최대 길이 | 3,200자 | 지나치게 긴 문맥 방지 |
| 자식 목표 길이 | 약 750자 | 의미 검색 집중도 확보 |
| 자식 최대 길이 | 950자 | 모델 입력과 품질 안정화 |
| 자식 최소 길이 | 180자 | 지나치게 짧은 조각 억제 |
| overlap | 1개 발화 단위 | 경계 문맥 손실 완화 |

### 5.2 생성 과정

1. 같은 회의·안건의 연속 발화를 모읍니다.
2. 발화 경계를 보존하며 부모 청크를 만듭니다.
3. 부모 안에서 더 작은 자식 청크를 만듭니다.
4. 각 자식에 동일한 `parent_chunk_id`를 저장합니다.
5. 검색은 자식 벡터로 하고 결과 설명은 부모 본문까지 확장합니다.

경계 내용을 일부 겹치게 하므로 비슷한 자식이 생길 수 있습니다. 이는 오류가 아니라 문맥 보존을 위한 overlap입니다. 최종 검색에서는 동일 본문 해시 제거, 동일 부모 결과 수 제한, 군집화 전 부모 중복 제거로 과도한 중복을 줄입니다.

---

## 6. Raw와 Enriched

### 6.1 Raw 표현

`raw`는 단순 발언문만 뜻하지 않습니다. 검색에 필요한 구조 정보와 원문 중심 표현입니다.

```text
[의회] 경상남도의회
[위원회] 본회의
[회의록] 본회의 회의록 제424회(3차)
[안건] 도정에 대한 질문
[발언] 기획조정실장 김기영
실효성 있는 인구 감소 대응을 위해서는 출산, 양육, 청년...
```

### 6.2 Enriched 표현

`enriched`는 raw 앞에 LLM metadata를 추가합니다.

```text
[정책분야] 보건/복지
[핵심이슈] 저출산 대응 정책 강화
[키워드] 출산율, 육아지원, 인구감소
[의회] ...
[안건] ...
[발언] ...
```

### 6.3 metadata는 어디에 사용하는가

`raw 선택 = metadata 미사용`이 아닙니다. metadata는 다음에 계속 사용합니다.

- 지역·연도·위원회·정책분야 필터
- 회의록명·안건·발화자 표시
- 정책성 발언 우선순위
- 이슈 후보 검수 보조
- 집계와 시계열 특징

즉, 벡터에는 raw를 쓰고 구조화 metadata는 필터·표시·분석에 사용합니다.

### 6.4 비교 결과와 선택 이유

LLM 요약이 일반적인 표현을 추가하면 서로 다른 문서가 비슷해져 정밀도가 낮아질 수 있습니다. 반대로 좋은 metadata는 짧은 발언을 보완할 수 있으므로 동일 평가 질의로 비교했습니다.

| 표현 | nDCG@10 | Precision@10 | Recall@20 | MRR |
|---|---:|---:|---:|---:|
| raw | **0.8928** | **0.9750** | **0.5826** | **1.0000** |
| enriched | 0.8871 | 0.9667 | 0.5744 | 0.9896 |

`enriched - raw`의 nDCG@10 차이는 -0.0057, 95% 신뢰구간은 `[-0.0224, 0.0110]`입니다. 큰 차이라고 단정하기 어렵지만 raw가 주요 평균 지표에서 조금 높고 구조가 단순해 최종 입력으로 선택했습니다.

---

## 7. 임베딩 모델 선택

임베딩 모델은 자식 청크를 1,024개 숫자로 바꿉니다. 의미가 비슷하면 벡터 방향도 비슷합니다. 벡터를 L2 정규화했으므로 내적과 코사인 유사도가 같은 순서를 만듭니다.

$$\operatorname{cosine}(q,d)=\frac{q\cdot d}{\lVert q\rVert\lVert d\rVert}$$

### 7.1 Arctic와 KURE 비교

| 모델 | nDCG@10 | P@10 | Direct P@10 | MRR | 처리율 |
|---|---:|---:|---:|---:|---:|
| Arctic-ko | **0.8969** | **0.9833** | **0.8083** | 1.0000 | **58.95행/초** |
| KURE | 0.8657 | 0.9750 | 0.7833 | 1.0000 | 56.25행/초 |

Arctic가 정확도와 처리율에서 우세해 최종 모델로 선택되었습니다.

### 7.2 전체 임베딩 결과

- 행: 901,358
- 차원: 1,024
- 저장 dtype: float16
- GPU 연산: bf16
- 시간: 4.14시간
- 처리율: 60.49행/초
- NaN/Inf: 0
- ID 중복: 0
- 벡터 norm 평균: 약 1.00094

`embeddings.f16.npy`와 `chunk_metadata.parquet`의 행 순서는 반드시 같아야 하며 digest로 순서 일치를 검증합니다.

---

## 8. 검색과 랭킹

### 8.1 Dense: 의미 검색

1. 질의를 Arctic로 1,024차원 벡터로 변환합니다.
2. pgvector HNSW가 코사인 거리가 가까운 자식 청크를 찾습니다.
3. 지역·연도·위원회 metadata 필터를 SQL 조건으로 적용합니다.

표현이 달라도 의미가 유사하면 찾지만, 정확한 조례명·기관명·숫자를 놓치거나 넓은 문장을 과대평가할 수 있습니다.

### 8.2 BM25: 키워드 검색

SQLite FTS5는 질의 단어 빈도, 전체에서의 희귀도, 문서 길이를 사용합니다. 조례명·기관명·고유명사처럼 정확한 문자열에 강합니다. 한국어 조사 일부를 보수적으로 제거하고 `전남↔전라남도↔전라남도의회`, `전북↔전북특별자치도` 같은 별칭을 검색 시 확장합니다.

### 8.3 RRF 결합

Dense와 BM25 점수는 척도가 달라 직접 더하지 않고 순위를 결합합니다.

$$\operatorname{RRF}(d)=\sum_s \frac{w_s}{k+\operatorname{rank}_s(d)}$$

현재 `k=60`, Dense 가중치 1.0, BM25 가중치 1.0입니다. 두 검색에서 모두 상위면 점수가 커지고, 한 검색에서만 강해도 후보에서 바로 사라지지 않습니다.

### 8.4 BGE Cross-Encoder 재순위

BGE reranker는 `(질의, 후보 본문)`을 함께 읽어 관련성을 재계산합니다.

```text
90만 청크 전체
→ Dense/BM25로 수십~수백 후보
→ RRF로 후보 병합
→ BGE가 후보만 정밀 판정
→ Top-K 근거
```

전체 문서를 BGE로 읽으면 너무 느리므로 **빠른 후보 생성 + 정밀한 후보 재순위**의 2단계 구조를 사용합니다.

### 8.5 검색 평가 지표

| 지표 | 뜻 | 좋은 방향 |
|---|---|---|
| Precision@10 | 상위 10개 중 관련 문서 비율 | 1에 가까울수록 좋음 |
| Recall@20 | 전체 관련 문서 중 상위 20개가 찾은 비율 | 1에 가까울수록 좋음 |
| MRR | 첫 관련 문서가 얼마나 빨리 나오는가 | 1이면 1위가 관련 문서 |
| nDCG@10 | 관련도 0/1/2와 순위를 함께 반영 | 1에 가까울수록 좋음 |

관련도 라벨은 `0=무관`, `1=부분 관련`, `2=직접 관련`입니다.

---

## 9. 이슈 군집화

검색 상위 10개가 이슈 전체는 아닙니다. 이를 seed로 전체 DB에서 유사 발언을 확장합니다.

1. BGE 상위 10개를 seed로 정합니다.
2. seed 벡터 평균을 중심 벡터(centroid)로 만듭니다.
3. 중심과 가까운 후보 500개를 pgvector에서 찾습니다.
4. 중심 유사도 0.45 미만을 제거합니다.
5. 동일 부모의 과도한 중복을 줄입니다.
6. 후보끼리 코사인 유사도를 계산합니다.
7. 임계값(기본 0.68, 초기 실험 0.72) 이상이면 그래프 간선으로 연결합니다.
8. Union-Find로 연결요소를 찾습니다.
9. seed를 포함한 연결요소를 이슈로 선택합니다.
10. 사람이 대표 근거를 검수하고 `issue_id`를 고정합니다.

Union-Find는 A-B, B-C가 연결되면 A·B·C를 같은 집합으로 빠르게 합치는 구조입니다. 임계값이 너무 높으면 같은 이슈가 쪼개지고, 너무 낮으면 인접한 별도 이슈가 섞입니다. `src/26_cluster_threshold_sweep.py`로 여러 값에서 seed 보존율·지역 수·구성원 수를 비교합니다.

지도 화살표는 **유사 논의가 관측된 시간 순서**이지 한 의회가 다른 의회에 정책을 전달했다는 인과 증명이 아닙니다.

---

## 10. 최초 관측과 타임라인

1. 이슈 구성원의 `meeting_date`를 확인합니다.
2. 동일 의회의 중복 청크를 근거 단위로 정리합니다.
3. 의회별 가장 이른 날짜를 계산합니다.
4. 전체에서 가장 이른 날짜를 최초 관측으로 정합니다.
5. 다른 의회의 최초 날짜를 오름차순 정렬합니다.
6. 최초 관측일 대비 `+N일`을 계산합니다.
7. 월별 언급량과 활성 의회 수를 집계합니다.

마지막 단계의 지도 `NEXT` 마커는 **다음 도시를 맞힌 표시가 아닙니다**. 마지막 관측 지점에서 예측 상세를 여는 앵커입니다. 클릭하면 지속 확률·예상 언급량·평가 지표·근거 해석이 열립니다.

---

## 11. 예측 방법

### 11.1 예측 대상

1. 분류: 다음 달에도 이 이슈가 논의될 것인가?
2. 회귀: 다음 달 예상 언급량은 몇 건인가?

현재 모델은 다음 전파 도시를 분류하지 않습니다.

### 11.2 월별 특징

| 특징 | 예시 | 의미 |
|---|---|---|
| 규모 | 현재 언급량, 활성 의회 수, 누적 의회 수 | 현재 논의 크기 |
| 최근 추세 | 최근 3개월 평균, 전월 대비 변화 | 단기 상승·하락 |
| 기울기 | 3개월·6개월 선형 기울기 | 중기 추세 |
| 지속성 | 최근 관측 이후 경과 월, 활성 월 비율 | 논의 단절 여부 |
| 확산성 | 신규 의회 수, 지역 커버리지 | 다른 지역으로 넓어지는 정도 |
| 집중도 | HHI, 상위 의회 비중, 엔트로피 | 특정 의회 편중 정도 |
| 의미 안정성 | 월별 중심과 이슈 중심의 유사도 | 같은 이슈 유지 여부 |

목표값은 한 달 앞으로 이동합니다. 예를 들어 2025-05 특징으로 2025-06 실제 활동 여부와 언급량을 예측합니다.

### 11.3 후보와 최종 모델

분류 후보는 Logistic Regression, 최근 활동 비율 기준선, 직전 달 활동 기준선입니다. 언급량 후보는 Poisson Regression, 최근 3개월 평균, 직전 달 유지입니다.

최종 시간순 평가에서 분류는 `active_ratio_baseline`, 언급량은 `poisson_regression`이 선택되었습니다. 고정 이슈와 월 수가 제한적이므로 복잡한 모델이 항상 더 잘 일반화하지는 않습니다.

### 11.4 시간순 Rolling Backtest

무작위 분할을 쓰지 않고 각 검증 월보다 이전 시점만 학습합니다.

```text
Fold 1: 과거 8개월 학습 → 다음 1개월 검증
Fold 2: 과거 9개월 학습 → 다음 1개월 검증
...
Fold 10: 이전 모든 월 학습 → 마지막 검증 월 예측
```

확정 결과는 10 folds, 학습 행 180개, 검증 예측 100개입니다.

---

## 12. 예측 평가 지표

### 12.1 논의 지속 F1

$$F1=\frac{2\times Precision\times Recall}{Precision+Recall}$$

지속한다고 예측한 결과의 정확도와 실제 지속 이슈를 놓치지 않는 능력을 함께 봅니다. 결과는 **0.8814**입니다.

### 12.2 MAE

$$MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|$$

결과는 **7.1224건**입니다. 다음 달 언급량이 실제와 평균 약 7.12건 차이 났다는 뜻입니다.

### 12.3 MSE와 RMSE

$$MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2$$

$$RMSE=\sqrt{MSE}$$

- RMSE: **10.0425건**
- 표시용 MSE: $10.0425^2\approx$ **100.8512건²**

큰 오차에 더 큰 벌점을 줍니다. MSE는 단위가 건²이라 RMSE를 중심으로 해석하되 화면에는 둘 다 표시합니다.

### 12.4 방향성 정확도

전월 대비 다음 달이 증가·유지·감소 중 어느 방향인지 맞힌 비율입니다. 결과는 **0.7600**, 즉 76%입니다.

### 12.5 단순 기준선 비교

기준선은 “다음 달도 이번 달과 같다”입니다.

| 지표 | 결과 |
|---|---:|
| 논의 지속 F1 | 0.8814 |
| 언급량 MAE | 7.1224 |
| 언급량 RMSE | 10.0425 |
| 방향성 정확도 | 0.7600 |
| 기준선 대비 F1 증가 | +0.0621 |
| 기준선 대비 MAE 개선 | 29.83% |
| 기준선 대비 RMSE 개선 | 24.31% |
| 기준선 대비 방향성 정확도 증가 | +0.74 |

예측 문장은 최근 언급량, 3개월 평균, 활성·누적 의회 수, 지속 확률, 예상 언급량, 지역 집중도를 조합해 생성합니다. 숫자는 통계 모델이 계산하고 Qwen은 그 근거를 문장으로 설명합니다.

---

## 13. ChromaDB와 pgvector 비교

동일한 901,358개 벡터와 metadata를 넣고 exact cosine을 기준으로 비교했습니다.

| 조건 | Chroma p50 | pgvector p50 | Chroma Recall@10 | pgvector Recall@10 |
|---|---:|---:|---:|---:|
| 필터 없음 | 1.42ms | 2.75ms | 0.87 | **0.95** |
| 지역+연도 필터 | 586.29ms | **3.97ms** | 1.00 | 1.00 |

서비스에서는 지역·연도·위원회 필터가 중요하고 pgvector가 필터 조건에서 훨씬 빨랐습니다. SQL 집계와 운영도 편리해 최종 DB로 선택했고 Chroma는 비교 기준으로 보존합니다. DB는 모델이 아니라 벡터와 metadata를 저장하고 찾는 검색 엔진입니다.

---

## 14. 로컬 Agent와 오프라인 실행

```text
사용자 질문
→ 대표 이슈 자동 연결 또는 자유검색
→ pgvector + BM25 + RRF + BGE
→ 회의록 원문·날짜·회의록명 확보
→ 타임라인·예측 결합
→ exaone3.5:7.8b가 근거 기반 답변 작성
```

Qwen은 검색 결과가 없는 내용을 사실처럼 만들면 안 됩니다. 답변에는 회의록명·일자·본문을 포함하고, Ollama가 실행되지 않으면 규칙 기반 질문 해석과 검색 근거 요약으로 동작합니다. 질문 파이프라인은 `질문 해석 → 정책어·지역 검증 → 의도별 도구 실행 → 근거 기반 답변 → 숫자·의회명 검증` 순서입니다. 화면 좌측 하단의 로봇 버튼을 누르면 대화창이 열립니다.

### 14.1 외부 공공데이터 후속 흐름

이 기능은 회의록에서 계산한 지역별 최초 관측 결과에 공공데이터를 **참고 정보로만** 결합합니다.

| 출처 | 용도 | 외부 전송 값 |
|---|---|---|
| 법제처 자치법규 | 광역·기초 조례 제정 여부와 날짜 확인 | 검증된 짧은 정책어 |
| 열린국회정보 | 관련 국회 의안 동향 확인 | 검증된 짧은 정책어 |
| KOSIS | 지역별 정책 수요 지표 비교 | 사전 등록된 통계표 ID |

`src/32a_external_followup_flow.py`가 후속 단계와 후보 점수를 계산하고, `src/34_build_final_web_bundle.py`가 화면 JSON에 결합합니다. 후보 점수는 규칙 기반 우선순위이며 확률이나 인과 추론이 아닙니다. 대회 제출 기본값은 `COUNCIL_FOLLOWUP_MODE=backtest`이고, `live`는 별도 사후 탐색에서만 사용합니다.

---

## 15. 화면 구성

### 대표 주제 TOP 10

검수 완료한 이슈를 카드로 보여 주고 최초 관측일·참여 의회·예측 상태를 비교합니다.

### 이슈 레이더

- 대표 이슈 선택 또는 자유 검색
- 지도 타임라인 재생
- 작은 관측 순서 표
- 원문 근거
- 마지막 단계의 `NEXT` 예측 앵커
- `NEXT` 클릭 시 F1·MAE·MSE·RMSE·방향성 정확도와 예측 근거

### 원문 탐색

지역·연도·위원회 조건을 적용해 회의록 근거를 직접 확인합니다.

---

## 16. 활성 코드 01~35 역할

| 단계 | 파일 | 역할 |
|---:|---|---|
| 01 | `src/01_restart_inventory.py` | 34개 입력·인코딩·문자 품질 검사 |
| 02 | `src/02_probe_received_metadata.py` | metadata 스키마와 고정 ID 검사 |
| 03 | `src/03_validate_row_id_join.py` | 회의록·metadata 결합 검증 |
| 04 | `src/04_build_direct_canonical.py` | 2,057,214행 canonical 생성 |
| 05 | `src/05_build_parent_child_chunks.py` | 부모·자식 청크 생성 |
| 05b | `src/05b_validate_chunks.py` | 길이·참조·본문 무결성 검사 |
| 06 | `src/06_make_embedding_eval_sample.py` | 60,000개 평가 표본 생성 |
| 07 | `src/07_embed_eval_sample.py` | raw/enriched 표본 임베딩 |
| 08 | `src/08_build_blind_retrieval_pool.py` | blind 검색 평가 후보 생성 |
| 09 | `src/09_local_llm_relevance_judge.py` | 로컬 Qwen 0/1/2 라벨 초안 |
| 10 | `src/10_compare_embedding_variants.py` | raw/enriched 지표 비교 |
| 11 | `src/11_build_embedding_model_pool.py` | Arctic/KURE 후보 풀 생성 |
| 12 | `src/12_compare_embedding_models.py` | 모델 정확도·속도 비교 |
| 08 full | `src/08_embed_full_corpus.py` | 901,358개 전체 임베딩 |
| 15 | `src/15_load_chroma.py` | Chroma 적재 |
| 16 | `src/16_load_pgvector.py` | pgvector HNSW 적재 |
| 17 | `src/17_benchmark_vector_databases.py` | 두 DB 정확도·속도 비교 |
| 18 | `src/18_pgvector_retrieval.py` | Dense/BM25/RRF 검색·중복 제거 |
| 19 | `src/19_rerank_candidates.py` | BGE 재순위 |
| 20 | `src/20_expand_cluster_issue.py` | seed 확장·그래프 군집화 |
| 21 | `src/21_build_issue_timeline.py` | 최초 관측·후속 관측·월별 집계 |
| 22 | `src/22_build_forecast_features.py` | 단일 이슈 월별 특징 |
| 23 | `src/23_forecast_issue.py` | 단일 이슈 예측 프로토타입 |
| 24 | `src/24_build_web_payload.py` | 지도·근거·예측 JSON 결합 |
| 25 | `src/25_build_issue_candidates.py` | 대표 10개 후보 일괄 생성 |
| 26 | `src/26_cluster_threshold_sweep.py` | 군집 임계값 비교 |
| 27 | `src/27_make_llm_extraction_evalset.py` | 팀 LLM 추출물 비교셋 준비 |
| 29 | `src/29_validate_external_context.py` | 선택적 외부 지표 검증 |
| 30 | `src/30_finalize_reviewed_issues.py` | 검수 반영·issue_id 고정 |
| 31 | `src/31_build_multi_issue_panel.py` | 대표 이슈×월 패널 생성 |
| 32 | `src/32_train_multi_issue_forecast.py` | 시간순 다중 이슈 예측·평가 |
| 32a | `src/32a_external_followup_flow.py` | 공공데이터 기반 후속 단계·후보 참고값 생성 |
| 33 | `src/33_analyze_free_query.py` | 임의 검색어 동적 분석 |
| 34 | `src/34_build_final_web_bundle.py` | 최종 웹 번들 생성 |
| 35 | `src/35_build_bm25_index.py` | 901,358개 BM25 인덱스 생성 |

공용 모듈은 `src/council_vector_runtime.py`, `src/bm25_runtime.py`, `src/search_aliases.py`입니다. 외부 참고 모듈은 `src/external_policy_sources.py`, `src/external_followup.py`, `src/regional_stats.py`이고, Agent 질문 제어는 `agent/query_pipeline.py`, `agent/followup_tools.py`가 담당합니다. 최종 화면은 `demo/final_app.py`입니다.

---

## 17. 폴더 구조

```text
README.md
agent/                      로컬 Agent와 도구
api/                        선택적 검색 API
config/                     데이터 계약·대표 이슈·평가 질의
data/                       원문·canonical·벡터·DB (Git 제외)
demo/                       Streamlit 화면
docs/                       실험 기록·실행서·평가 설명
eval/                       검색 평가 질의
models/                     Hugging Face 캐시 (Git 제외)
outputs/                    실험·이슈·예측 결과 (Git 제외)
scripts/                    실행 스크립트
src/                        전처리·검색·군집·예측 코드
docker-compose.yml          PostgreSQL+pgvector
requirements-*.txt          단계별 의존성
```

---

## 18. 설치와 실행

권장: Ubuntu/WSL2 24.04, Python 3.12, NVIDIA GPU, Docker, Ollama.

```bash
cd ~/projects/council-vector-benchmark
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-app.txt
```

`.env` 예시:

```bash
PGVECTOR_DSN=postgresql://council:council@127.0.0.1:5432/council_vectors
COUNCIL_PG_TABLE=council_arctic_raw_v1
COUNCIL_RETRIEVAL_MODE=hybrid
COUNCIL_BM25_INDEX=data/bm25/arctic_raw_v1/bm25.sqlite3
OLLAMA_URL=http://127.0.0.1:11434
COUNCIL_AGENT_MODEL=exaone3.5:7.8b
COUNCIL_FOLLOWUP_MODE=backtest
COUNCIL_MINUTES_CUTOFF=2025-12-31
COUNCIL_EXTERNAL_CACHE=data/external/cache
# 선택: 실제 값은 .env에만 저장하고 Git에 올리지 않습니다.
LAW_OC=
ASSEMBLY_API_KEY=
KOSIS_API_KEY=
```

기존 산출물이 있는 컴퓨터:

```bash
source .venv/bin/activate
docker compose up -d pgvector
ollama list
bash scripts/run_final_app.sh
```

브라우저는 `http://localhost:8501`입니다. 같은 네트워크의 다른 PC에서 보려면 `HOST=0.0.0.0 bash scripts/run_final_app.sh`로 실행하고 방화벽을 허용합니다.

주요 결과 재생성:

```bash
bash scripts/build_bm25_hybrid.sh
bash scripts/run_representative_issues.sh candidates
bash scripts/run_issue_review.sh
bash scripts/run_representative_issues.sh finalize
bash scripts/run_forecast_evaluation.sh
bash scripts/run_final_app.sh
```

### 팀원에게 pgvector DB 공유

`data/postgres`를 직접 복사하지 않고 PostgreSQL custom dump를 사용합니다.

```bash
bash scripts/export_pgvector_for_share.sh
```

생성된 `exports/*.dump`, `.sha256`, `.info.txt` 세 파일은 Google Drive의 **제한된 팀 전용 폴더**로 공유합니다. 팀원은 다음과 같이 복원합니다.

```bash
bash scripts/restore_pgvector_from_dump.sh ~/Downloads/council_vectors_YYYYMMDD_HHMMSS.dump
```

자세한 내용은 `docs/POSTGRES_DB_TEAM_SHARE.md`를 참고합니다. DB에는 대회 데이터에서 파생된 본문이 포함되므로 공개 링크나 GitHub에 올리지 않습니다.

---

## 19. 주요 산출물

| 산출물 | 위치 |
|---|---|
| canonical | `data/canonical_v3/corpus.parquet` |
| 부모·자식 청크 | `data/canonical_v3/parents.parquet`, `chunks.parquet` |
| 전체 임베딩 | `data/embeddings/arctic_raw_v1/` |
| BM25 | `data/bm25/arctic_raw_v1/bm25.sqlite3` |
| pgvector 테이블 | `council_arctic_raw_v1` |
| 대표 이슈 | `outputs/issues/final_v1/` |
| 예측 | `outputs/forecast/multi_issue_v1/model/` |
| 웹 번들 | `outputs/web/` |
| 외부 후속 흐름 참고 결과 | `outputs/external/followup_v1/` |
| 실험 기록 | `docs/EXPERIMENT_LOG.md` |
| 구조화 레지스트리 | `config/experiment_registry.json` |

---

## 20. Git 제외 대상

```text
.env
.venv/
data/input/
data/canonical_v3/
data/embedding_eval_vectors/
data/embeddings/
data/bm25/
data/chroma/
data/postgres/
exports/
models/huggingface/
outputs/
*.dump
*.npy
*.parquet
*.sqlite3
*.faiss
__pycache__/
```

공개 저장소에는 코드, 데이터 구조 설명, 재현 명령, 집계 평가만 올립니다. API key·비밀번호·원문 회의록은 커밋하지 않습니다.

---

## 21. 한계와 올바른 해석

1. 관측 순서는 인과적 정책 전파의 증명이 아닙니다.
2. 현재 예측은 다음 달 활동 여부와 언급량이며 다음 도시를 예측하지 않습니다.
3. 고정 이슈 수와 월 수가 제한적이므로 단순 모델이 복잡한 모델보다 안정적일 수 있습니다.
4. 자동 LLM 라벨은 초안이며 핵심 근거는 사람이 검수해야 합니다.
5. 군집 임계값에 따라 이슈 범위가 달라져 대표 근거와 함께 평가해야 합니다.
6. 회의록에 없는 사실과 제공 기간 이후 정보는 예측 근거로 사용하지 않습니다.
7. 예상 언급량은 점추정치이므로 MAE·RMSE와 함께 해석해야 합니다.
8. 외부 공공데이터 후보 점수는 정책 후속 가능성을 정렬하는 설명용 규칙이며 확률이 아닙니다.
9. `live` 외부 조회 결과는 대회 공식 예측과 분리해야 하며 제출 기본값은 `backtest`입니다.

---

## 22. 발표용 1분 설명

> 저희는 2024~2025년 광역 지방의회 회의록 205만 발화를 표준화하고, 검색 정확도와 문맥을 동시에 확보하기 위해 부모·자식 구조로 90만 개의 검색 청크를 만들었습니다. 자식 청크는 Arctic 한국어 임베딩으로 의미 검색하고, 정확한 정책명은 BM25로 보완한 뒤 RRF와 BGE 재순위로 최종 근거를 정합니다. 상위 근거를 seed로 유사 발언을 확장하고 코사인 유사도 그래프와 Union-Find로 이슈를 구성합니다. 이후 의회별 최초 관측일을 계산해 후속 논의 흐름을 지도에 표현합니다. 예측은 LLM이 만드는 것이 아니라 과거 월만 학습하는 rolling backtest로 논의 지속 여부와 다음 달 언급량을 계산합니다. 최종 F1은 0.8814, MAE는 7.12건, RMSE는 10.04건, 방향성 정확도는 76%입니다. 로컬 Qwen은 계산 결과와 회의록 원문을 사용해 설명만 생성하므로 외부 API로 데이터를 보내지 않습니다.

---

## 23. 자주 받는 질문

### 왜 부모를 직접 임베딩하지 않았나요?

부모는 여러 논점이 섞일 수 있어 의미가 흐려집니다. 작은 자식을 검색하고 큰 부모를 보여 주면 정밀도와 문맥을 함께 얻습니다.

### metadata를 모두 임베딩하면 더 좋지 않나요?

항상 그렇지 않습니다. LLM 요약이 일반적인 표현을 추가해 문서를 서로 비슷하게 만들 수 있습니다. 실제 비교에서 raw가 근소하게 우세했습니다. metadata는 필터·표시·집계에 계속 사용합니다.

### BM25와 Dense 모두 유사도 검색인가요?

둘 다 관련 문서의 순위를 만들지만 Dense는 의미 벡터, BM25는 단어 일치와 희귀도를 사용합니다. RRF가 두 순위를 합칩니다.

### BGE와 임베딩 모델은 무엇이 다른가요?

임베딩은 문서 벡터를 미리 계산해 전체 DB를 빠르게 찾습니다. BGE는 질의와 후보를 함께 읽어 더 정확하지만 느리므로 소수 후보에만 사용합니다.

### 군집화가 이슈명을 자동 생성하나요?

군집화는 구성원을 묶습니다. 이슈명은 seed·키워드·metadata로 제안할 수 있지만 최종 이름과 `issue_id`는 검수 후 고정합니다.

### 예측 수치는 Qwen이 만든 것인가요?

아닙니다. 활동 여부는 활동률 기준선, 언급량은 Poisson Regression이 계산합니다. Qwen은 수치와 검색 근거를 문장으로 설명합니다.

### NEXT 마커는 다음 논의 도시인가요?

아닙니다. 마지막 관측 지점에서 예측 상세 패널을 여는 앵커입니다.

### 인터넷을 끄면 처음에 안 될 수 있는 이유는?

모델·패키지·Docker 이미지가 로컬에 없으면 다운로드가 필요하기 때문입니다. 한 번 모두 설치하고 Ollama와 PostgreSQL이 로컬에서 실행 중이면 오프라인으로 동작합니다.

---

## 24. 팀과 프로젝트

- 팀명: **꺼드럭's**
- 프로젝트명: **의정이슈레이더**
- 데이터: 대회 제공 광역 지방의회 회의록 2024~2025
- 원칙: 로컬 실행, 원문 근거, 시간 누출 방지, 재현 가능한 ID와 실험 기록

세부 실험은 [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md), 활성 경로는 [`docs/ACTIVE_PIPELINE.md`](docs/ACTIVE_PIPELINE.md), 예측 평가는 [`docs/FORECAST_EVALUATION.md`](docs/FORECAST_EVALUATION.md)를 참고합니다.


### 최종 로컬 LLM과 실험 이력

최종 실행 기본 모델은 LG AI Research의 `exaone3.5:7.8b`입니다. 외부 상용 LLM API로 회의록 원문을 전송하지 않습니다. 초기 검색 관련성 자동 라벨 실험에는 `qwen3:8b`가 사용되었으며, 재현성을 위해 `config/experiment_registry.json`의 과거 기록은 수정하지 않습니다. 최종 제출 전 EXAONE으로 구조화 출력 회귀시험을 별도로 수행합니다.
