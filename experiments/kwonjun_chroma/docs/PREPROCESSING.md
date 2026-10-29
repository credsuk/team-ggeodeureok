# 데이터 전처리 및 청킹 재현 문서

## 1. 목적

서로 다른 2024~2025년 광역 지방의회 회의록과 LLM 메타데이터를 하나의
기준 스키마로 통일하고, 검색 정확도와 원문 문맥을 동시에 확보하는
부모·자식 청크를 생성한다. 전처리 결과는 요약본이 아니라 정제 원문과
출처를 함께 보존한 Parquet 데이터다.

## 2. 입력

회의록 TSV의 기준 열은 다음과 같다.

```text
MTG_DE, RASMBLY_NM, MTGNM, MINTS_NM,
agenda_title, speaker_role, speaker_name, speaker_type, utterance
```

메타데이터 JSONL은 같은 원본에서 생성한 `row_id`와 다음 필드를 가진다.

```text
policy_category, core_issue, keywords, summary, is_substantive,
agenda, speaker, date, assembly
```

원본 파일과 메타데이터는 대회 이용조건 때문에 공개 Git 저장소에 포함하지
않는다. 실행 환경에서는 `data/input/` 아래에 배치한다.

## 3. 처리 단계

| 순서 | 코드 | 입력 | 주요 처리 | 출력 |
|---:|---|---|---|---|
| 1 | `01_restart_inventory.py` | 전체 입력 폴더 | 파일 역할·인코딩·깨진 문자·스키마 검사 | inventory 보고서 |
| 2 | `02_probe_received_metadata.py` | 회의록·JSONL | 파일 쌍과 필드·고정 ID 검사 | metadata probe 보고서 |
| 3 | `03_validate_row_id_join.py` | 회의록·JSONL | `source_key + row_id` 일대일 대응 검증 | 결합 검증 보고서 |
| 4 | `04_build_direct_canonical.py` | 검증된 파일 쌍 | 날짜·의회명 표준화, 고정 ID·해시 생성 | `corpus.parquet` |
| 5 | `05_build_parent_child_chunks.py` | canonical corpus | 발화 경계를 보존한 부모·자식 청킹 | `parents.parquet`, `chunks.parquet` |
| 6 | `05b_validate_chunks.py` | 부모·자식 청크 | ID·필수값·길이·참조·중복 검증 | PASS/FAIL 보고서 |

## 4. Canonical 스키마

주요 열은 다음과 같다.

| 열 | 의미 |
|---|---|
| `source_key`, `source_file`, `source_line_no` | 원본 파일·행 추적 정보 |
| `chunk_id` | 발화 또는 검색 청크의 고정 ID |
| `document_id` | 원본 회의록 ID |
| `parent_chunk_id` | 자식이 속한 부모 문맥 ID |
| `council`, `meeting_date`, `year` | 의회·회의일자·연도 |
| `committee`, `meeting_name`, `agenda_title` | 위원회·회의록명·안건 |
| `speaker`, `speaker_role`, `speaker_type` | 발화자 정보 |
| `text` | 정제된 원문 |
| `embedding_text` | 임베딩 모델에 입력할 표현 |
| `text_sha256` | 본문 변경·중복 검증용 해시 |
| `metadata_source`, `metadata_version` | 메타데이터 출처·버전 |

행 번호만 사용하면 정렬이나 재처리 때 연결이 깨진다. 따라서 출처와 본문을
기반으로 생성한 고정 ID를 임베딩, pgvector, BM25, 검수 결과와 웹 화면까지
공통 키로 사용한다.

## 5. 부모·자식 청킹

- 부모 청크: 목표 약 2,400자, 최대 3,200자. 앞뒤 발화와 논의 배경 보존.
- 자식 청크: 목표 약 750자, 최대 950자, 최소 180자. 임베딩·검색 대상.
- 경계 중첩: 발화 한 단위를 겹쳐 문맥 단절을 줄인다.

검색은 작은 자식 청크로 수행하고, 최종 근거를 제시할 때
`parent_chunk_id`로 큰 부모 문맥을 복원한다. 중첩 때문에 생기는 유사 결과는
동일 본문 해시와 동일 부모 제한으로 제거한다.

## 6. Raw와 Enriched

`raw`는 원문만 던지는 방식이 아니라 의회·회의록·안건·발화자 같은 공식
구조 정보와 정제 원문을 결합한 입력이다. `enriched`는 여기에 LLM이 만든
정책분야·핵심이슈·키워드·요약을 추가한다.

동일 평가셋에서 Raw의 nDCG@10이 0.8928, Enriched가 0.8871로 통계적으로
유의한 차이가 없었다. 최종 임베딩은 원문 의미를 덜 왜곡하고 재현이 단순한
Raw를 사용했다. LLM 메타데이터는 버린 것이 아니라 필터, 표시, 검수, 집계에
구조화 열로 사용한다.

## 7. 무결성 기준

- `chunk_id`, `parent_chunk_id`, `document_id` 참조가 유효해야 한다.
- 검색 청크 ID는 모두 고유해야 한다.
- 의회명·회의일자·회의록명·본문이 비어 있으면 안 된다.
- 깨진 대체문자와 빈 발언이 없어야 한다.
- 청크 본문 길이와 저장된 `char_length`가 일치해야 한다.
- 자식이 가리키는 부모가 반드시 존재해야 한다.
- 전체 34개 지역×연도 입력이 포함되어야 한다.

최종 생성 결과는 canonical 발화 2,057,214행, 검색용 자식 청크
901,358개다.

## 8. 실행

각 스크립트의 경로 옵션은 `--help`로 확인한다. 전체 재구축은 대용량 입력이
필요하므로 심사자는 제공된 샘플로 실행 순서를 확인하고, 전체 산출물은 별도
DB dump와 체크섬으로 복원한다.

```bash
python src/01_restart_inventory.py --help
python src/02_probe_received_metadata.py --help
python src/03_validate_row_id_join.py --help
python src/04_build_direct_canonical.py --help
python src/05_build_parent_child_chunks.py --help
python src/05b_validate_chunks.py --help
```

