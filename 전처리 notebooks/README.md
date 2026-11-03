# 📓 데이터 전처리 및 메타데이터 추출 주피터 노트북 (`notebooks/`)

본 디렉토리는 대회 제출 표준 규격(`forder.txt`)에 맞춰 구성된 주피터 노트북 모음입니다.
`scripts/` 내의 파이썬 스크립트들을 **실행 가능한 대화형 노트북으로 분리 변환**하고,
필요에 따라 단계별 일괄 실행이 가능한 **통합 파이프라인(Pipeline)**으로 통폐합하여 제공합니다.

---

## 📂 디렉토리 구조 및 노트북 목록

```plaintext
notebooks/
├── json_to_text/                      # 회의록 원본 JSON/HTML -> 정제 TSV 변환 파이프라인
│   ├── 01_split_sample.ipynb         # [단계별] 대표 HTML 본문 샘플 추출 (1_split_sample.py)
│   ├── 02_rules_make.ipynb           # [단계별] 로컬 LLM 기반 의회별 CSS Selector 추론 (2_rules_make.py)
│   ├── 03_rules_summary.ipynb        # [단계별] 파싱 규칙 검증 및 통합 JSON 빌드 (3_rules_summary.py)
│   ├── 04_to_txt.ipynb               # [단계별] 통합 규칙 기반 정제 TSV 추출 및 연속발언 병합 (4_to_txt.py)
│   └── json_to_text_pipeline.ipynb   # [통합] 1~4단계를 원클릭으로 일괄 실행하는 통합 파이프라인
│
└── metadata_llm/                      # 정제 회의록 TSV -> LLM 정책 메타데이터 추출 파이프라인
    ├── 01_split_council_tsv.ipynb     # [단계별] 회의록 TSV 소형 배치 분할 및 전역 row_id 부여
    ├── 02_extract_metadata_llm.ipynb  # [단계별] 로컬 LLM 정책분야/핵심이슈/키워드 추출 (체크포인트 지원)
    └── metadata_llm_pipeline.ipynb    # [통합] TSV 분할부터 LLM 정책 메타데이터 추출 및 분석 대시보드
```

---

## 🎯 기능별 상세 안내

### 1. `json_to_text/` (회의록 정제 및 구조화)
1. **`01_split_sample.ipynb`**:
   - `MINTS_HTML`에서 노이즈 태그(스크립트, 사이드바, 푸터 등)를 정제하고 발언 마커 기반으로 본문 영역을 채점하여 선별합니다.
   - LLM 입력 한도(토큰)를 초과하지 않도록 문맥 태그 구조를 보존하면서 장문을 안전하게 축약합니다.
2. **`02_rules_make.ipynb`**:
   - 추출된 샘플 HTML을 로컬 LLM(Ollama)에 전달하여 안건(`agenda`), 발언자(`speaker`), 발언문(`utterance`)의 최적 CSS Selector를 탐색합니다.
   - 모델 추론 실패 시 DOM 구조 기반 휴리스틱 탐색 및 텍스트 노드 특수 구조(`__TEXT_NODE_*__`) 자동 감지를 수행합니다.
3. **`03_rules_summary.ipynb`**:
   - 개별 의회별 규칙 파일들을 수집하고, 한글/자연어 오염이 없는 순수 CSS 선택자인지 엄격 검증합니다.
   - 결측이나 오류가 있는 의회는 기본 Fallback(`div`, `span.spkrbld`, `p`)으로 안전하게 보정하여 단일 통합 JSON으로 생성합니다.
4. **`04_to_txt.ipynb`**:
   - 통합 규칙을 적용하여 의전 멘트(`(개의)`, `(인사)`, `(박수)` 등)를 제거하고, 자간 분절 오류(`의 원 여 러 분`)를 복원합니다.
   - 동일 화자의 연속된 발언을 자연스럽게 1개 레코드로 병합하여 표준 TSV(`cleaned_*.txt`)로 저장합니다.
5. **`json_to_text_pipeline.ipynb`**:
   - 1단계부터 4단계까지의 전체 파이프라인을 단일 노트북에서 일괄 실행할 수 있는 통합 워크플로우입니다.

### 2. `metadata_llm/` (LLM 기반 정책 메타데이터 구축)
1. **`01_split_council_tsv.ipynb`**:
   - 대규모 회의록 TSV 파일을 행 수(예: 20행) 및 문자 수(예: 10,000자) 기준으로 소형 배치 파일(`part_*.tsv`)로 분할합니다.
   - 각 발언에 원본 기준 1번부터 시작하는 전역 고유 `row_id`를 부여하여 추후 메타데이터 결합 시 무결성을 보장합니다.
2. **`02_extract_metadata_llm.ipynb`**:
   - 로컬 LLM(Ollama)을 호출하여 **8대 정책 카테고리(`policy_category`)**, **핵심 쟁점(`core_issue`)**, **2~4어절 복합 정책 키워드(`keywords`)**, **발언 요약(`summary`)**을 구조화된 JSONL로 추출합니다.
   - '사업', '예산', '조례' 등 포괄적 1단어 명사를 엄격히 배제하는 Negative Constraint 프롬프트를 적용합니다.
   - **체크포인트(이어하기) 기능**: 이미 처리된 `row_id`는 자동으로 건너뛰며, 중단 후 재실행 시 기존 파일에 실시간 추가(append)됩니다.
3. **`metadata_llm_pipeline.ipynb`**:
   - 회의록 TSV 분할부터 로컬 LLM 메타데이터 추출, 정책 분야별 빈도 분석 대시보드까지 전 과정을 일괄 실행합니다.

---

## 🛠️ 실행 요구 사항
- **Python 환경**: Python 3.11+
- **필수 패키지**: `beautifulsoup4`, `pandas`, `openai`, `requests`, `nbformat`
- **로컬 LLM (Ollama)**:
  ```bash
  ollama pull qwen2.5-coder:14b   # 또는 7b (json_to_text CSS 규칙 추론용)
  ollama pull gemma4:e2b           # 또는 gemma4:e4b (메타데이터 추출용)
  ```
