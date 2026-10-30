# 최종 로컬 LLM: LG EXAONE

## 선택 모델

- 공식 모델: `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`
- Ollama 태그: `exaone3.5:7.8b`
- 용도: 검색 근거를 바탕으로 한 답변 생성, 질문 의도 해석, 구조화된 이슈 설명
- 보안: 회의록 원문과 검색 근거는 로컬 Ollama에만 전달하며 외부 상용 LLM API에는 전송하지 않음

## 설치와 확인

```bash
ollama pull exaone3.5:7.8b
ollama run exaone3.5:7.8b
export COUNCIL_AGENT_MODEL=exaone3.5:7.8b
```

## 교체 범위

임베딩 모델 Arctic, pgvector/BM25 검색, 군집화, 시계열 예측은 변경하지 않습니다. 생성형 LLM 계층만 EXAONE으로 교체합니다. 따라서 901,358개 벡터를 다시 만들 필요는 없지만, JSON 형식 준수율·근거 인용·한국어 답변 품질은 회귀시험해야 합니다.

## 실험 이력 고지

초기 검색 관련성 자동 라벨 실험에는 `qwen3:8b`가 사용되었습니다. 이 사실은 재현성을 위해 `config/experiment_registry.json`에 그대로 보존합니다. 최종 운영 기본값과 과거 평가용 judge 모델을 혼동하지 않습니다.

## 라이선스

EXAONE 모델의 배포·이용 조건은 EXAONE AI Model License를 확인해야 합니다. 모델 가중치는 Git 저장소에 포함하지 않고 Ollama 설치 명령과 정확한 모델명을 문서화합니다.
