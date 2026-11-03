# 🏛️ 지방의회 회의록 JSON to Text 전처리 파이프라인 (`scripts/json_to_text`)

원본 지방의회 회의록 JSON(내부에 복잡한 HTML 포함)에서 의회별 상이한 구조를 극복하고, 발언 단위 텍스트(TSV)를 정제·추출하는 4단계 파이프라인입니다.

---

## 📂 파일 구성 및 역할

| 파일명 | 기능 및 설명 | 주요 입력 및 출력 |
| :--- | :--- | :--- |
| `_constant.py` | 파이프라인 공통 상수(대상 연도, 폴더 경로, Ollama 설정 등) 정의 | 모든 스크립트에서 import |
| `pipeline_timing.py` | 단계별/파일별 소요 시간 측정 및 `pipeline_timing.csv` 기록 모듈 | 시간 측정 및 통계 |
| `1_split_sample.py` | 원본 JSON의 `MINTS_HTML`에서 LLM 분석용 대표 본문 HTML 샘플 추출 | 입력: `origin/*.json`<br>출력: `data/sample/*.txt` |
| `2_rules_make.py` | 로컬 LLM을 호출하여 안건/화자/발언 CSS Selector 추론 및 유효성 검증 | 입력: `data/sample/*.txt`<br>출력: `data/llm_rules_outputs/rule_*.json` |
| `3_rules_summary.py` | 개별 규칙들의 CSS Selector 유효성을 검증하고 단일 통합 JSON으로 병합 | 입력: `rule_*.json`<br>출력: `consolidated_parsing_rules_*.json` |
| `4_to_txt.py` | 통합 규칙을 적용하여 의전 멘트 제거, 자간 복원, 연속 발언 병합 후 TSV 변환 | 입력: `origin/*.json`, 통합규칙<br>출력: `data/to_text2/*.txt` (TSV) |

---

## 🚀 파이프라인 실행 순서

```bash
# 1단계: 대표 HTML 샘플 추출
python scripts/json_to_text/1_split_sample.py

# 2단계: 로컬 LLM 기반 의회별 CSS Selector 규칙 생성
python scripts/json_to_text/2_rules_make.py

# 3단계: 파싱 규칙 통합 및 엄격한 유효성 검증
python scripts/json_to_text/3_rules_summary.py

# 4단계: 전체 회의록 TSV 정제 데이터 추출
python scripts/json_to_text/4_to_txt.py
```

> **주피터 노트북 환경**: 대화형 실행 및 단계별 시각화를 원하시는 경우 `notebooks/json_to_text/` 내의 `.ipynb` 파일들을 활용하세요.
