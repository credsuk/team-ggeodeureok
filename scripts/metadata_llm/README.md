# 🔍 회의록 LLM 정책 메타데이터 추출 파이프라인 (`scripts/metadata_llm`)

정제된 회의록 TSV 파일에서 로컬 LLM(Ollama)을 활용하여 정책 분야(`policy_category`), 핵심 쟁점(`core_issue`), 2~4어절 복합 정책 키워드(`keywords`), 요약(`summary`)을 추출하고 구조화된 JSONL 데이터로 저장하는 파이프라인입니다.

---

## 📂 파일 구성 및 역할

| 파일명 | 기능 및 설명 | 주요 입력 및 출력 |
| :--- | :--- | :--- |
| `_constant.py` | 대상 의회, 연도, 기본 입출력 경로, 배치 크기, Ollama 모델 설정 | 공통 환경 설정 |
| `split_council_tsv.py` | 대용량 회의록 TSV를 행/문자 수 기준으로 분할하고 전역 고유 `row_id` 부여 | 입력: `cleaned_*.txt`<br>출력: `split/{의회명}/part_*.tsv` |
| `extract_metadata_llm.py` | 로컬 LLM을 호출하여 정책 메타데이터를 추출하고 실시간 체크포인트(이어하기) 지원 | 입력: `part_*.tsv` 또는 TSV 파일<br>출력: `metadata/*.jsonl` |

---

## 🚀 파이프라인 실행 순서

```bash
# 1단계: 회의록 TSV를 소형 배치 파일들로 분할
python scripts/metadata_llm/split_council_tsv.py

# 2단계: 로컬 LLM을 통한 정책 메타데이터 추출 (100건 테스트 예시)
python scripts/metadata_llm/extract_metadata_llm.py --sample 100

# 전체 데이터 끝까지 이어서 추출 (체크포인트 지원)
python scripts/metadata_llm/extract_metadata_llm.py
```

> **주피터 노트북 환경**: 대화형 실행 및 단계별 검증을 원하시는 경우 `notebooks/metadata_llm/` 내의 `.ipynb` 파일들을 활용하세요.
